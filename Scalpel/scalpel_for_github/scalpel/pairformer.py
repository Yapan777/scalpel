"""
pairformer.py — Pairformer Logic v10.4.5 (Inherent Structure).

Принцип AlphaFold 3 — «Взаимосвязи»:
  AlphaFold 3 заменяет тяжёлый EvoFormer на PairFormer:
  вместо глобальных множественных выравниваний —
  попарные взаимодействия (pairwise biases) между «токенами».

Здесь: вместо полной матрицы корреляций (O(p²) RAM) вычисляем
  «Энергию взаимодействия» только для топ-k пар признаков.

RAM экономия:
  Классический подход: np.corrcoef(X.T)  → O(p² × 8 байт)
    При p=100 признаках: 80 КБ — терпимо.
    При p=500:  2 МБ.  При p=2000: 32 МБ.
  Pairformer: только топ-k пар (k=50 по умолчанию) → ~500 МБ RAM saved
    при больших датасетах + исключаются тяжёлые глобальные корреляции.

Модель взаимодействия:
  E(i,j) = |cov(xᵢ, xⱼ)| / (σᵢ · σⱼ + ε)  ×  |corr(xᵢ·xⱼ, y)|
  Первый множитель — сила связи признаков.
  Второй — насколько их совместное произведение объясняет цель.
  Пары с высоким E(i,j) — «взаимодействующие атомы».
"""
from __future__ import annotations

import gc
import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

log = logging.getLogger("scalpel")

# ── Константы ─────────────────────────────────────────────────────────────
PAIRFORMER_TOP_K:       int   = 50    # сколько пар вычислять
PAIRFORMER_MIN_CORR:    float = 0.05  # минимальная |корр| для включения пары
PAIRFORMER_MAX_FEAT:    int   = 200   # если признаков > 200 — семплируем


# ══════════════════════════════════════════════════════════════════════════════
# Dataclass
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class PairformerResult:
    """Результат Pairformer-анализа."""
    top_pairs:      List[Tuple[str, str, float]] = field(default_factory=list)
    # (feat_i, feat_j, energy)
    pair_ops:       List[str]  = field(default_factory=list)  # рекомендованные операторы для пар
    selected_feats: List[str]  = field(default_factory=list)  # уникальные признаки из топ-пар
    ram_saved_mb:   float      = 0.0
    n_pairs_total:  int        = 0
    n_pairs_scored: int        = 0
    report_lines:   List[str]  = field(default_factory=list)


# ══════════════════════════════════════════════════════════════════════════════
# Sparse pairwise energy
# ══════════════════════════════════════════════════════════════════════════════

def _pair_energy(
    xi:  np.ndarray,
    xj:  np.ndarray,
    y:   np.ndarray,
    eps: float = 1e-12,
) -> float:
    """
    E(i,j) = |corr(xi, xj)| × |corr(xi·xj, y)|

    Первый множитель: структурная связь признаков.
    Второй: взаимодействие объясняет цель.
    Произведение xi·xj — «ковалентная связь» признаков.
    """
    si = float(np.std(xi))
    sj = float(np.std(xj))
    if si < eps or sj < eps:
        return 0.0

    # Корреляция между признаками
    c_ij = float(np.corrcoef(xi, xj)[0, 1])
    if np.isnan(c_ij):
        return 0.0

    # Взаимодействие xi·xj с целью
    prod = xi * xj
    sy = float(np.std(y))
    if sy < eps or float(np.std(prod)) < eps:
        return 0.0
    c_prod_y = float(np.corrcoef(prod, y)[0, 1])
    if np.isnan(c_prod_y):
        return 0.0

    return abs(c_ij) * abs(c_prod_y)


# ══════════════════════════════════════════════════════════════════════════════
# Главная функция
# ══════════════════════════════════════════════════════════════════════════════

def pairformer_select(
    X:            np.ndarray,
    y:            np.ndarray,
    shadow_names: List[str],
    top_k:        int   = PAIRFORMER_TOP_K,
    min_corr:     float = PAIRFORMER_MIN_CORR,
    max_feat:     int   = PAIRFORMER_MAX_FEAT,
) -> PairformerResult:
    """
    Вычисляет попарную энергию взаимодействия признаков.
    Возвращает топ-k пар и уникальные признаки из них.

    RAM Guard:
      - Не строим полную матрицу p×p.
      - Если p > max_feat — семплируем max_feat признаков с весами по corr(xi, y).
      - Все операции numpy без лишних копий.
    """
    result = PairformerResult()
    n, p   = X.shape

    if p < 2 or n < 5:
        result.report_lines = ["[Pairformer] Пропуск: недостаточно признаков/точек"]
        return result

    # ── Оценка RAM экономии ───────────────────────────────────────────────
    full_matrix_mb   = (p * p * 8) / 1024 / 1024
    pairformer_mb    = (min(top_k, p * (p - 1) // 2) * 3 * 8) / 1024 / 1024
    result.ram_saved_mb = max(0.0, full_matrix_mb - pairformer_mb)

    # ── Семплирование признаков при большом p ────────────────────────────
    if p > max_feat:
        corr_with_y = np.array([
            abs(float(np.corrcoef(X[:, j], y)[0, 1]))
            if np.std(X[:, j]) > 1e-12 else 0.0
            for j in range(p)
        ])
        weights = corr_with_y / (corr_with_y.sum() + 1e-12)
        sel_idx = np.random.choice(p, size=max_feat, replace=False, p=weights)
        sel_idx = np.sort(sel_idx)
        X_use   = X[:, sel_idx]
        names   = [shadow_names[i] for i in sel_idx]
        log.info("[Pairformer] Семплирование: %d → %d признаков", p, max_feat)
    else:
        X_use = X
        names = list(shadow_names)

    p_use = X_use.shape[1]
    total_pairs = p_use * (p_use - 1) // 2
    result.n_pairs_total = total_pairs

    # ── Sparse scoring: вычисляем только верхний треугольник ─────────────
    scored: List[Tuple[str, str, float]] = []

    # Для экономии RAM сначала фильтруем по индивидуальным |corr(xi, y)|
    corr_i = np.array([
        abs(float(np.corrcoef(X_use[:, j], y)[0, 1]))
        if np.std(X_use[:, j]) > 1e-12 else 0.0
        for j in range(p_use)
    ])
    # Берём только признаки с достаточной индивидуальной корреляцией
    good_idx = np.where(corr_i >= min_corr)[0]
    if len(good_idx) < 2:
        good_idx = np.argsort(corr_i)[::-1][:min(10, p_use)]

    count = 0
    for ki, gi in enumerate(good_idx):
        for kj in range(ki + 1, len(good_idx)):
            gj = good_idx[kj]
            e  = _pair_energy(X_use[:, gi], X_use[:, gj], y)
            if e >= min_corr:
                scored.append((names[gi], names[gj], e))
            count += 1

    result.n_pairs_scored = count

    if not scored:
        result.report_lines = [
            f"[Pairformer] Нет значимых пар (n_scored={count}, min_corr={min_corr})"
        ]
        return result

    # Сортируем и берём топ-k
    scored.sort(key=lambda x: x[2], reverse=True)
    top = scored[:top_k]
    result.top_pairs = top

    # Уникальные признаки из топ-пар
    seen: List[str] = []
    for fi, fj, _ in top:
        for f in (fi, fj):
            if f not in seen:
                seen.append(f)
    result.selected_feats = seen

    # Операторы: пары с высокой энергией → рекомендуем произведение
    ops: List[str] = ["+", "-", "*", "/"]
    if len(top) >= 3 and top[0][2] > 0.3:
        ops += ["square"]
    if len(top) >= 5 and top[2][2] > 0.2:
        ops += ["sqrt", "log"]
    result.pair_ops = ops

    result.report_lines = [
        f"[Pairformer] Пар всего: {total_pairs} | оценено: {count} | RAM saved ≈{result.ram_saved_mb:.0f} МБ",
        f"[Pairformer] Топ-1 пара: {top[0][0]} × {top[0][1]}  E={top[0][2]:.4f}",
        f"[Pairformer] Уникальных признаков из пар: {len(seen)}",
        f"[Pairformer] Рекомендованные операторы: {ops}",
    ]

    gc.collect()
    log.info("[Pairformer v10.4.5] top_pairs=%d feats=%s",
             len(top), seen[:4])
    return result


# ══════════════════════════════════════════════════════════════════════════════
# Форматирование отчёта
# ══════════════════════════════════════════════════════════════════════════════

def format_pairformer_report(result: PairformerResult) -> str:
    if not result.top_pairs:
        return "[Pairformer] Нет значимых взаимодействий"

    lines = [
        "═" * 62,
        "  PAIRFORMER INTERACTION REPORT v10.4.5",
        f"  Топ-пар: {len(result.top_pairs)}  |  RAM saved ≈{result.ram_saved_mb:.0f} МБ",
        "─" * 62,
    ]
    for i, (fi, fj, e) in enumerate(result.top_pairs[:5], 1):
        lines.append(f"  {i}. {fi} × {fj}  →  E={e:.4f}")
    lines += [
        "─" * 62,
        f"  Признаки: {result.selected_feats[:8]}",
        f"  Операторы: {result.pair_ops}",
        "═" * 62,
    ]
    return "\n".join(lines)
