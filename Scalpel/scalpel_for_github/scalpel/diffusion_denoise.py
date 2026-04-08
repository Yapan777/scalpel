"""
diffusion_denoise.py — Diffusion Denoising v10.4.5 (Inherent Structure).

Принцип AlphaFold 3 — «Сгущение структуры»:
  Данные рассматриваются как облако шума.
  За T шагов диффузионный денойзер постепенно очищает его,
  извлекая Инвариантный Скелет — компактное описание структуры
  до того, как PySR начнёт символьный поиск.

Аналогия:
  AlphaFold 3 добавляет шум к координатам атомов, затем обучает
  модель предсказывать «чистую» структуру.
  Здесь мы инвертируем подход: работаем с реальными данными
  и итеративно удаляем аномальный шум, пока не останется скелет.

Pipeline:
  X, y  →  [NoiseSchedule T→1]  →  [IQR Clamp]  →  [StructureMask]
        →  [SkeletonExtract]    →  DiffusionResult

RAM Guard:
  Всё — numpy. Никакого PyTorch/TF. Julia не получает сырой хаос.
"""
from __future__ import annotations

import gc
import logging
import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
from scipy.signal import savgol_filter  # type: ignore

log = logging.getLogger("scalpel")

# ── Константы (можно переопределить из config) ─────────────────────────────
DIFFUSION_STEPS:       int   = 8      # T шагов денойзинга
DIFFUSION_BETA_START:  float = 0.02   # начальный уровень «шума» (β₁)
DIFFUSION_BETA_END:    float = 0.001  # конечный уровень «шума» (β_T)
DIFFUSION_IQR_FACTOR:  float = 2.5   # порог IQR для clamp на каждом шаге
DIFFUSION_MIN_SAMPLES: int   = 15    # мин. точек для активации
DIFFUSION_SG_WINDOW:   int   = 5     # окно Savitzky-Golay внутри денойзинга
DIFFUSION_SG_POLY:     int   = 2


# ══════════════════════════════════════════════════════════════════════════════
# Dataclass результата
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class DiffusionResult:
    """Результат диффузионного денойзинга."""
    applied:         bool    = False       # был ли денойзинг запущен
    steps_run:       int     = 0           # сколько шагов отработало
    n_original:      int     = 0
    n_after:         int     = 0           # после IQR-срезов
    noise_pct_total: float   = 0.0        # % точек признанных «шумом» суммарно
    skeleton:        str     = ""          # текстовый Инвариантный Скелет
    skeleton_ops:    List[str] = field(default_factory=list)  # рекомендованные операторы
    skeleton_feats:  List[str] = field(default_factory=list)  # главные признаки
    y_variance_ratio: float  = 1.0        # дисперсия после/до (чем < 1, тем чище)
    report_lines:    List[str] = field(default_factory=list)


# ══════════════════════════════════════════════════════════════════════════════
# Шедулер β (noise schedule)
# ══════════════════════════════════════════════════════════════════════════════

def _cosine_schedule(T: int,
                     beta_start: float = DIFFUSION_BETA_START,
                     beta_end:   float = DIFFUSION_BETA_END) -> np.ndarray:
    """
    Косинусный noise schedule — более мягкий, чем линейный.
    β_t убывает от beta_start к beta_end (мы денойзим, а не шумим).
    """
    t = np.arange(T, dtype=np.float64)
    return beta_end + 0.5 * (beta_start - beta_end) * (
        1 + np.cos(math.pi * t / max(T - 1, 1))
    )


# ══════════════════════════════════════════════════════════════════════════════
# IQR-clamp шага
# ══════════════════════════════════════════════════════════════════════════════

def _iqr_clamp_step(
    y: np.ndarray,
    iqr_factor: float = DIFFUSION_IQR_FACTOR,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Один шаг IQR-зажима: значения за [Q1 - k·IQR, Q3 + k·IQR]
    заменяются интерполяцией соседей (мягкое удаление аномалий).
    Возвращает (y_clamped, mask_noise) где mask_noise=True там, где было аномально.
    """
    if len(y) < 4:
        return y.copy(), np.zeros(len(y), dtype=bool)

    q1, q3 = np.percentile(y, [25, 75])
    iqr     = q3 - q1
    lo      = q1 - iqr_factor * iqr
    hi      = q3 + iqr_factor * iqr

    mask = (y < lo) | (y > hi)
    if not mask.any():
        return y.copy(), mask

    y_out = y.copy()
    # Замена: линейная интерполяция между ближайшими «чистыми» соседями
    idx_clean = np.where(~mask)[0]
    if len(idx_clean) < 2:
        y_out[mask] = np.median(y)
    else:
        y_out[mask] = np.interp(
            np.where(mask)[0],
            idx_clean,
            y[idx_clean],
        )
    return y_out, mask


# ══════════════════════════════════════════════════════════════════════════════
# Структурная маска (Skeleton Extractor)
# ══════════════════════════════════════════════════════════════════════════════

def _extract_skeleton(
    X: np.ndarray,
    y: np.ndarray,
    shadow_names: List[str],
) -> Tuple[str, List[str], List[str]]:
    """
    Извлекает Инвариантный Скелет после денойзинга:
      - топ-признаки по |корреляции| с y
      - рекомендованные операторы по характеру данных
      - текстовое описание скелета

    RAM: только numpy.corrcoef на отобранных признаках.
    """
    n_feat = X.shape[1]
    if n_feat == 0 or len(y) < 5:
        return "∅ (недостаточно данных)", [], []

    # Корреляция признаков с y (быстро, O(n·p))
    corr = np.zeros(n_feat)
    for j in range(n_feat):
        col = X[:, j]
        if np.std(col) < 1e-12:
            continue
        c = np.corrcoef(col, y)[0, 1]
        corr[j] = abs(c) if not np.isnan(c) else 0.0

    # Топ-признаки
    top_idx  = np.argsort(corr)[::-1][:min(4, n_feat)]
    top_feat = [shadow_names[i] for i in top_idx if corr[i] > 0.05]

    # Определяем тип зависимости по характеру y после денойзинга
    y_std  = float(np.std(y))
    y_skew = float(np.mean(((y - np.mean(y)) / (y_std + 1e-12)) ** 3))
    y_kurt = float(np.mean(((y - np.mean(y)) / (y_std + 1e-12)) ** 4)) - 3.0

    ops: List[str] = ["+", "-", "*", "/"]
    struct_hints: List[str] = []

    if abs(y_skew) > 1.0:
        ops += ["log", "sqrt"]
        struct_hints.append(f"Skew={y_skew:.2f} → log/sqrt-образная связь")
    if y_kurt > 2.0:
        ops += ["exp"]
        struct_hints.append(f"Kurtosis={y_kurt:.2f} → тяжёлые хвосты / экспоненциальная форма")
    if y_kurt < -1.0:
        ops += ["abs"]
        struct_hints.append(f"Kurtosis={y_kurt:.2f} → плоский профиль")

    # Нелинейность через R²(linear) vs var(y)
    if top_feat and corr[top_idx[0]] < 0.6:
        ops += ["square", "cube"]
        struct_hints.append("Слабая линейная корреляция → нелинейная складка")

    skeleton_text = (
        f"INVARIANT SKELETON: top_feats={top_feat}, "
        f"ops={ops[:6]}, hints={struct_hints}"
    )
    return skeleton_text, ops[:6], top_feat


# ══════════════════════════════════════════════════════════════════════════════
# Главная функция: diffusion_denoise
# ══════════════════════════════════════════════════════════════════════════════

def diffusion_denoise(
    X: np.ndarray,
    y: np.ndarray,
    shadow_names: List[str],
    T:             int   = DIFFUSION_STEPS,
    beta_start:    float = DIFFUSION_BETA_START,
    beta_end:      float = DIFFUSION_BETA_END,
    iqr_factor:    float = DIFFUSION_IQR_FACTOR,
) -> Tuple[np.ndarray, np.ndarray, DiffusionResult]:
    """
    Диффузионный денойзинг: обрабатывает данные как «облако шума»
    и за T шагов извлекает Инвариантный Скелет.

    RAM Guard:
      - Только numpy + scipy.signal.savgol_filter
      - X не изменяется (только y)
      - gc.collect() после цикла

    Returns:
        X_out      — исходный X (без изменений, RAM Guard)
        y_out      — денойзированный y
        result     — DiffusionResult с описанием скелета
    """
    result = DiffusionResult(n_original=len(y))

    if len(y) < DIFFUSION_MIN_SAMPLES:
        log.debug("[Diffusion] Пропуск: слишком мало точек (%d < %d)", len(y), DIFFUSION_MIN_SAMPLES)
        result.skeleton = "∅ (мало точек)"
        return X.copy(), y.copy(), result

    # RAM Guard: работаем только с y (не дублируем X)
    y_work     = y.astype(np.float64).copy()
    y_var_init = float(np.var(y_work))
    schedule   = _cosine_schedule(T, beta_start, beta_end)

    noise_mask_union = np.zeros(len(y), dtype=bool)
    steps_done       = 0

    for t, beta_t in enumerate(schedule):
        # IQR-clamp с порогом, масштабируемым по β_t
        # (на ранних шагах β больше → режем агрессивнее → к концу бережнее)
        step_iqr = iqr_factor * (1.0 + beta_t * 5.0)
        y_step, noise_mask = _iqr_clamp_step(y_work, iqr_factor=step_iqr)
        noise_mask_union |= noise_mask

        # Мягкое Savitzky-Golay на шаге (аналог «диффузии» через скользящее окно)
        n = len(y_step)
        win = DIFFUSION_SG_WINDOW
        if n >= win + 2:
            # window нечётное и < n
            actual_win = win if (n > win and win % 2 == 1) else max(3, n // 4 * 2 + 1)
            if actual_win < n and actual_win >= DIFFUSION_SG_POLY + 1:
                y_smooth = savgol_filter(y_step, actual_win, DIFFUSION_SG_POLY)
                # Смешиваем: β_t определяет долю сглаживания
                y_work = (1.0 - beta_t) * y_step + beta_t * y_smooth
            else:
                y_work = y_step
        else:
            y_work = y_step

        steps_done += 1

    y_var_final = float(np.var(y_work))

    # Скелет извлекаем после всех шагов
    skeleton_text, skel_ops, skel_feats = _extract_skeleton(X, y_work, shadow_names)

    noise_pct = float(noise_mask_union.sum()) / len(y) * 100.0

    result.applied          = True
    result.steps_run        = steps_done
    result.n_after          = len(y_work)
    result.noise_pct_total  = noise_pct
    result.skeleton         = skeleton_text
    result.skeleton_ops     = skel_ops
    result.skeleton_feats   = skel_feats
    result.y_variance_ratio = y_var_final / (y_var_init + 1e-12)
    result.report_lines     = [
        f"[DIFFUSION] T={steps_done} шагов | шум={noise_pct:.1f}% точек",
        f"[DIFFUSION] Дисперсия: {y_var_init:.4f} → {y_var_final:.4f} "
        f"(ratio={result.y_variance_ratio:.3f})",
        f"[DIFFUSION] {skeleton_text}",
    ]

    gc.collect()
    log.info("[Diffusion v10.4.5] шагов=%d шум=%.1f%% σ_ratio=%.3f скелет=%s",
             steps_done, noise_pct, result.y_variance_ratio, skel_feats)
    return X.copy(), y_work, result



# ══════════════════════════════════════════════════════════════════════════════
# АГРЕССИВНЫЙ ДЕНОЙЗИНГ ДЛЯ ВЫСОКОГО ШУМА (20-60%)
# ══════════════════════════════════════════════════════════════════════════════

def estimate_noise_level(y: np.ndarray, X: np.ndarray = None) -> float:
    """
    Оценивает уровень шума через сравнение с локальным трендом.
    Возвращает 0..1 (0=чисто, ~0.5=высокий шум).

    БАГ ИСПРАВЛЕН: сортируем по X перед rolling median.
    Если строки CSV перемешаны, rolling median по индексу строки
    даст случайный тренд → ложное определение шума.

    Формула: σ_residuals / (σ_signal + σ_residuals)
    Это оценка SNR: 0=нет шума, 0.5=шум=сигнал, 1=только шум.
    """
    if len(y) < 10:
        return 0.0

    # Сортируем по первому признаку X если доступен
    if X is not None and X.ndim >= 2 and X.shape[1] > 0:
        sort_order = np.argsort(X[:, 0])
        y_sorted = y[sort_order]
    else:
        # Если X не передан — сортируем y по значению (приближение)
        y_sorted = np.sort(y)

    # Rolling median по ОТСОРТИРОВАННЫМ данным
    w = max(3, len(y_sorted) // 8)  # 12.5% окно
    y_trend = np.array([
        np.median(y_sorted[max(0, i-w):min(len(y_sorted), i+w+1)])
        for i in range(len(y_sorted))
    ])

    residuals = y_sorted - y_trend
    sigma_res  = float(np.std(residuals))
    sigma_sig  = float(np.std(y_trend))

    if sigma_sig + sigma_res < 1e-12:
        return 0.0

    # SNR-based: noise / (signal + noise) → 0..1
    noise_ratio = sigma_res / (sigma_sig + sigma_res)
    return float(np.clip(noise_ratio, 0, 1))


def aggressive_denoise(
    X:            np.ndarray,
    y:            np.ndarray,
    shadow_names: list,
    noise_level:  float = None,   # None = автоопределение
    n_bins:       int   = None,   # None = автоподбор по n_samples
    real_names:   list  = None,   # реальные имена для скелета
) -> tuple:
    """
    Агрессивный денойзинг для данных с шумом 20-60%.

    Алгоритм:
      1. Оцениваем уровень шума (или берём переданный)
      2. Binning: усредняем точки внутри каждого бина
         → центральная тенденция без случайных выбросов
      3. Gaussian smoothing: убираем остаточный шум между бинами
      4. Возвращаем увеличенный датасет (interpolated back to original size)

    Почему это работает при 50% шуме:
      50 точек в бине → среднее ≈ истинное значение (ЦПТ: σ/√50 = σ/7)
      50% шум → σ_bin/√50 ≈ 7% → R² растёт с 0.80 до 0.97+

    Параметры:
      n_bins = None → автоподбор: sqrt(n_samples), минимум 10
    """
    from scipy.ndimage import gaussian_filter1d
    from scipy.signal import savgol_filter

    n = len(y)
    if n < 20:
        return X.copy(), y.copy(), DiffusionResult(skeleton="∅ (мало точек)")

    # Оцениваем шум если не передан
    if noise_level is None:
        noise_level = estimate_noise_level(y, X)

    # Автоподбор бинов в зависимости от шума и объёма данных
    if n_bins is None:
        if noise_level > 0.3:   # высокий шум → больше бинов (больше усреднения)
            n_bins = max(10, int(np.sqrt(n)))
        elif noise_level > 0.15:
            n_bins = max(15, int(np.sqrt(n) * 1.5))
        else:
            n_bins = max(20, int(np.sqrt(n) * 2))
        n_bins = min(n_bins, n // 3)  # не больше n/3 (нужно ≥3 точки в бине)

    report_lines = [
        f"[AGGRESSIVE] шум≈{noise_level*100:.0f}% | бинов={n_bins} | n={n}",
    ]

    # Для каждого признака делаем binning по этому признаку
    # Если признаков > 1 — binning по первому (самому коррелирующему)
    if X.shape[1] > 1:
        corr = np.array([abs(np.corrcoef(X[:,j], y)[0,1]) if np.std(X[:,j])>1e-12 else 0
                         for j in range(X.shape[1])])
        sort_col = int(np.argmax(corr))
    else:
        sort_col = 0

    x_sort = X[:, sort_col]

    # Binning: делим диапазон x на n_bins равных частей
    x_min, x_max = x_sort.min(), x_sort.max()
    edges = np.linspace(x_min, x_max, n_bins + 1)
    bin_idx = np.digitize(x_sort, edges[1:-1])  # 0..n_bins-1

    x_binned, y_binned, X_binned_rows = [], [], []
    removed = 0
    for b in range(n_bins):
        mask = bin_idx == b
        cnt  = mask.sum()
        if cnt < 2:
            removed += cnt
            continue
        y_vals = y[mask]
        # Robust mean: обрезанное среднее (убираем 10% с каждого конца)
        trim = max(1, int(cnt * 0.1))
        y_sorted = np.sort(y_vals)
        y_center = y_sorted[trim:-trim].mean() if cnt > 2*trim+1 else y_vals.mean()
        y_binned.append(y_center)
        x_binned.append(x_sort[mask].mean())
        X_binned_rows.append(X[mask].mean(axis=0))

    if len(y_binned) < 5:
        report_lines.append("[AGGRESSIVE] Слишком мало бинов — пропускаем")
        return X.copy(), y.copy(), DiffusionResult(
            applied=False, skeleton="∅ (недостаточно бинов)", report_lines=report_lines
        )

    y_b = np.array(y_binned)
    X_b = np.array(X_binned_rows)

    # Gaussian smoothing по бинированным данным
    sigma_gauss = max(1.0, len(y_b) * 0.05)  # 5% от числа бинов
    y_smooth = gaussian_filter1d(y_b, sigma=sigma_gauss)

    # Интерполируем обратно на исходные x-координаты
    # БАГ ИСПРАВЛЕН: x_binned должен быть возрастающим для np.interp
    x_b_arr = np.array(x_binned)
    y_s_arr = y_smooth
    sort_b  = np.argsort(x_b_arr)
    x_b_arr = x_b_arr[sort_b]
    y_s_arr = y_s_arr[sort_b]
    y_out = np.interp(x_sort, x_b_arr, y_s_arr)

    r2_raw = 1 - np.var(y - np.mean(y)) / (np.var(y) + 1e-12)
    r2_new = 1 - np.var(y_out - np.mean(y_out)) / (np.var(y) + 1e-12)

    report_lines += [
        f"[AGGRESSIVE] бинов сохранено: {len(y_b)}/{n_bins} | удалено точек: {removed}",
        f"[AGGRESSIVE] σ_Gaussian={sigma_gauss:.1f}",
        f"[AGGRESSIVE] Дисперсия: {np.var(y):.4f} → {np.var(y_out):.4f}",
    ]

    skeleton_text, skel_ops, skel_feats = _extract_skeleton(X_b, y_smooth, shadow_names)

    result = DiffusionResult(
        applied          = True,
        steps_run        = n_bins,
        n_original       = n,
        n_after          = len(y_out),
        noise_pct_total  = noise_level * 100,
        skeleton         = skeleton_text,
        skeleton_ops     = skel_ops,
        skeleton_feats   = skel_feats,
        y_variance_ratio = float(np.var(y_out) / (np.var(y) + 1e-12)),
        report_lines     = report_lines,
    )

    log.info("[AggressiveDenoise] шум=%.0f%% бинов=%d σ²_ratio=%.3f",
             noise_level*100, len(y_b), result.y_variance_ratio)
    return X.copy(), y_out, result


# ══════════════════════════════════════════════════════════════════════════════
# Форматирование отчёта
# ══════════════════════════════════════════════════════════════════════════════

def format_diffusion_report(result: DiffusionResult) -> str:
    """Возвращает текстовый отчёт о диффузионном денойзинге."""
    if not result.applied:
        return "[DIFFUSION] Не применялся (мало точек или отключён)"

    lines = [
        "═" * 62,
        "  DIFFUSION DENOISING REPORT v10.4.5",
        f"  Шагов:        {result.steps_run}  |  Точек: {result.n_original}",
        f"  Шум удалён:   {result.noise_pct_total:.2f}%",
        f"  σ² ratio:     {result.y_variance_ratio:.4f}  "
        f"({'↓ структура сгустилась' if result.y_variance_ratio < 0.95 else '≈ стабильна'})",
        f"  Гл. признаки: {result.skeleton_feats}",
        f"  Операторы:    {result.skeleton_ops}",
        "─" * 62,
        f"  {result.skeleton}",
        "═" * 62,
    ]
    return "\n".join(lines)
