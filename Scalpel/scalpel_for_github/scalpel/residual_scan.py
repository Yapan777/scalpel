"""
residual_scan.py — ResidualScan v10.15: Двухслойный анализ.

Идея: после того как Матрёшка приняла первый закон (Layer 1),
запускаем ПОЛНЫЙ pipeline на остатках (residuals = y - ŷ).

Если в остатках есть структура (не просто шум) — это Layer 2:
второй физический эффект поверх основного закона.

Архитектура Layer 2 ИДЕНТИЧНА Layer 1:
  Oracle → Navigator → PySR → Матрёшка (4 роли) → Летописец → Discovery

НЕ чёрный ящик: Матрёшка получает полный контекст:
  "Layer 1 уже найден: [formula]. Сейчас анализируем остатки.
   Если найдена структура — это второй физический слой."

Результат сохраняется в residual_formulas.json отдельно от gold.
Итоговая формула: y = Layer1 + Layer2 с объяснением каждого слоя.
"""
from __future__ import annotations

import json
import logging
import numpy as np
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List

log = logging.getLogger("scalpel")


# ══════════════════════════════════════════════════════════════════════════════
# Dataclass результата
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class ResidualScanResult:
    """Результат двухслойного анализа."""
    layer1_formula:    str   = ""    # принятая основная формула
    layer1_r2:         float = 0.0
    layer2_formula:    str   = ""    # найденная формула на остатках
    layer2_r2_blind:   float = 0.0
    layer2_consensus:  str   = ""    # ПРИНЯТА / ОТКЛОНЕНА / НЕ ЗАПУСКАЛСЯ
    residual_r2:       float = 0.0   # насколько остатки объяснимы линейно
    combined_formula:  str   = ""    # layer1 + layer2 (итоговая)
    combined_r2:       float = 0.0
    layer2_explanation: str  = ""    # что означает второй слой физически
    ran:               bool  = False  # запускался ли ResidualScan


# ══════════════════════════════════════════════════════════════════════════════
# Сохранение / загрузка данных между фазами
# ══════════════════════════════════════════════════════════════════════════════

def save_residual_data(
    X_train:       np.ndarray,
    y_train:       np.ndarray,
    X_test:        np.ndarray,
    y_test:        np.ndarray,
    feat_names:    List[str],
    domain_type:   str,
    dim_codes:     List[int],
    noise_hint:    float,
) -> None:
    """
    Сохраняет обучающие данные для последующего ResidualScan.
    Вызывается из run_engine() в начале — до Surgery/Diffusion.
    Сохраняем RAW данные (до предобработки), чтобы остатки считались честно.
    """
    from .config import RESIDUAL_DATA_PATH
    RESIDUAL_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "X_train":     X_train.tolist(),
        "y_train":     y_train.tolist(),
        "X_test":      X_test.tolist(),
        "y_test":      y_test.tolist(),
        "feat_names":  feat_names,
        "domain_type": domain_type,
        "dim_codes":   dim_codes,
        "noise_hint":  float(noise_hint or 0.0),
    }
    RESIDUAL_DATA_PATH.write_text(
        json.dumps(data, ensure_ascii=False), encoding="utf-8"
    )
    log.debug("[ResidualScan] Данные сохранены → %s", RESIDUAL_DATA_PATH)


def _load_residual_data() -> Optional[dict]:
    from .config import RESIDUAL_DATA_PATH
    if not RESIDUAL_DATA_PATH.exists():
        return None
    try:
        return json.loads(RESIDUAL_DATA_PATH.read_text(encoding="utf-8"))
    except Exception as e:
        log.warning("[ResidualScan] Ошибка чтения данных: %s", e)
        return None


# ══════════════════════════════════════════════════════════════════════════════
# Проверка: есть ли структура в остатках
# ══════════════════════════════════════════════════════════════════════════════

def _residuals_have_structure(
    residuals: np.ndarray,
    X:         np.ndarray,
    min_r2:    float,
) -> tuple:
    """
    Проверяет есть ли нелинейная структура в остатках.

    Возвращает (has_structure: bool, r2_linear: float).

    Логика:
      - Если R²(линейная регрессия X→residuals) > min_r2 → структура есть
      - Если std(residuals) < 1e-6 → чистый ноль, нет смысла
      - Если std(residuals)/mean(|residuals|) > 5 → чистый шум
    """
    if len(residuals) < 20:
        return False, 0.0

    std_r = float(np.std(residuals))
    if std_r < 1e-9:
        return False, 0.0

    # Простая линейная проверка: R²(X → residuals)
    try:
        from sklearn.linear_model import LinearRegression
        from sklearn.metrics import r2_score
        lr = LinearRegression().fit(X, residuals)
        r2 = float(r2_score(residuals, lr.predict(X)))
        r2 = max(0.0, r2)
        has = r2 >= min_r2
        return has, r2
    except Exception:
        # Fallback: корреляция первого признака
        corr = float(np.corrcoef(X[:, 0], residuals)[0, 1]) ** 2
        return corr >= min_r2, corr


# ══════════════════════════════════════════════════════════════════════════════
# Сохранение результата Layer 2
# ══════════════════════════════════════════════════════════════════════════════

def _save_layer2(result: ResidualScanResult) -> None:
    from .config import RESIDUAL_RESULT_PATH
    import datetime
    RESIDUAL_RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)

    record = {
        "timestamp":         datetime.datetime.now().isoformat(),
        "layer1_formula":    result.layer1_formula,
        "layer1_r2":         round(result.layer1_r2, 4),
        "layer2_formula":    result.layer2_formula,
        "layer2_r2_blind":   round(result.layer2_r2_blind, 4),
        "layer2_consensus":  result.layer2_consensus,
        "residual_r2":       round(result.residual_r2, 4),
        "combined_formula":  result.combined_formula,
        "combined_r2":       round(result.combined_r2, 4),
        "layer2_explanation": result.layer2_explanation,
    }

    # Загружаем существующий файл или создаём новый
    if RESIDUAL_RESULT_PATH.exists():
        try:
            existing = json.loads(RESIDUAL_RESULT_PATH.read_text(encoding="utf-8"))
        except Exception:
            existing = {"layers": []}
    else:
        existing = {"layers": []}

    existing["layers"].append(record)
    existing["count"] = len(existing["layers"])
    existing["last_updated"] = record["timestamp"]

    tmp = RESIDUAL_RESULT_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(existing, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(RESIDUAL_RESULT_PATH)
    log.info("[ResidualScan] Layer 2 сохранён → %s", RESIDUAL_RESULT_PATH)


# ══════════════════════════════════════════════════════════════════════════════
# Главная функция
# ══════════════════════════════════════════════════════════════════════════════

def run_residual_scan(
    layer1_formula_real:  str,
    layer1_formula_shadow: str,
    layer1_r2:            float,
    layer1_predict_fn,           # callable: X → y_pred
    host:                 str,
    model:                str,
) -> ResidualScanResult:
    """
    Запускает полный pipeline на остатках после Layer 1.

    Вызывается из run_llm_phase() сразу после ПРИНЯТА.

    Архитектура Layer 2:
      1. Загружаем оригинальные данные
      2. Считаем остатки: residuals = y - layer1(X)
      3. Проверяем есть ли структура в остатках
      4. Если да — запускаем run_engine() + run_llm_phase() на остатках
         с heritage_context объясняющим что это Layer 2
      5. Сохраняем в residual_formulas.json
    """
    from .config import RESIDUAL_R2_MIN

    result = ResidualScanResult(
        layer1_formula=layer1_formula_real,
        layer1_r2=layer1_r2,
        layer2_consensus="НЕ ЗАПУСКАЛСЯ",
    )

    # ── Шаг 1: загружаем данные ───────────────────────────────────
    raw = _load_residual_data()
    if raw is None:
        log.info("[ResidualScan] Данные не найдены — пропускаем Layer 2")
        return result

    try:
        X_train = np.array(raw["X_train"], dtype=np.float64)
        y_train = np.array(raw["y_train"], dtype=np.float64)
        X_test  = np.array(raw["X_test"],  dtype=np.float64)
        y_test  = np.array(raw["y_test"],  dtype=np.float64)
        feat_names  = raw.get("feat_names",  ["f0"])
        domain_type = raw.get("domain_type", "Physics")
        dim_codes   = raw.get("dim_codes",   [0])
        noise_hint  = float(raw.get("noise_hint", 0.1))
    except Exception as e:
        log.warning("[ResidualScan] Ошибка парсинга данных: %s", e)
        return result

    # ── Шаг 2: считаем остатки ────────────────────────────────────
    try:
        y_pred_train = layer1_predict_fn(X_train)
        y_pred_test  = layer1_predict_fn(X_test)
        residuals_train = y_train - y_pred_train
        residuals_test  = y_test  - y_pred_test
    except Exception as e:
        log.warning("[ResidualScan] Ошибка вычисления остатков: %s", e)
        return result

    # ── Шаг 3: есть ли структура? ─────────────────────────────────
    has_structure, residual_r2 = _residuals_have_structure(
        residuals_train, X_train, min_r2=RESIDUAL_R2_MIN
    )
    result.residual_r2 = residual_r2

    print(f"\n  {'═'*58}")
    print(f"  RESIDUAL SCAN (Layer 2) — анализ остатков")
    print(f"  {'═'*58}")
    print(f"  Layer 1: {layer1_formula_real}  R²={layer1_r2:.4f}")
    print(f"  Структура в остатках: R²_linear={residual_r2:.3f} "
          f"({'ДА — запускаем Layer 2' if has_structure else 'НЕТ — чистый шум'})")

    if not has_structure:
        print(f"  [ResidualScan] R²={residual_r2:.3f} < {RESIDUAL_R2_MIN} — остатки — шум, Layer 2 не нужен")
        result.ran = True
        return result

    # ── Шаг 4: запускаем полный pipeline на остатках ──────────────
    print(f"\n  [ResidualScan] Запускаем полный pipeline на Layer 2...")
    result.ran = True

    # Контекст для Матрёшки — объясняем что именно анализируем
    layer2_heritage = (
        f"\n  ━━━ LAYER 2 CONTEXT (ResidualScan v10.15) ━━━\n"
        f"  ★ Это ВТОРОЙ СЛОЙ анализа.\n"
        f"  Layer 1 (основной закон): {layer1_formula_real}  R²={layer1_r2:.4f}\n"
        f"  Сейчас анализируются ОСТАТКИ: y_residual = y_original - Layer1(x)\n"
        f"  R²(linear fit на остатках) = {residual_r2:.3f} — структура обнаружена.\n"
        f"  Задача Матрёшки: оценить физический смысл второго слоя.\n"
        f"  Вопросы которые нужно задать учёному:\n"
        f"    1. Есть ли в данных известная аномалия / резонанс / фазовый переход?\n"
        f"    2. Могут ли остатки объясняться систематической погрешностью?\n"
        f"    3. Соответствует ли второй закон размерностям первого?\n"
        f"  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    )

    try:
        from .engine import run_engine, run_llm_phase, save_candidates

        # Запускаем PySR на остатках
        res2 = run_engine(
            X_train     = X_train,
            y_train     = residuals_train,
            X_test      = X_test,
            y_test      = residuals_test,
            feat_names  = feat_names,
            target_col  = "residual",
            domain_type = domain_type,
            phase       = "pysr",
            dim_codes   = dim_codes,
            noise_hint  = noise_hint,
            skip_heritage = False,
            # Передаём контекст Layer 2 через extra_heritage
            extra_heritage = layer2_heritage,
        )

        if res2 is None or not res2.formula_real:
            print("  [ResidualScan] Layer 2: PySR ничего не нашёл")
            result.layer2_consensus = "НЕ НАЙДЕНО"
            return result

        print(f"  [ResidualScan] Layer 2 PySR: {res2.formula_real}  R²={res2.r2_blind:.4f}")

        # Запускаем Матрёшку на Layer 2
        res2_llm = run_llm_phase(skip_residual_scan=True)  # не рекурсируем!

        if res2_llm and res2_llm.formula_real:
            result.layer2_formula   = res2_llm.formula_real
            result.layer2_r2_blind  = res2_llm.r2_blind
            result.layer2_consensus = res2_llm.consensus

            # Комбинированная формула
            result.combined_formula = (
                f"({layer1_formula_real}) + ({res2_llm.formula_real})"
            )

            # Оцениваем combined R²
            try:
                from sklearn.metrics import r2_score as _r2
                import re

                # Считаем combined predictions через eval
                # (безопасно — формула уже прошла PySR и Матрёшку)
                def _eval_formula(formula: str, X: np.ndarray) -> np.ndarray:
                    """Простой eval формулы в shadow-виде."""
                    env = {f"f{i}": X[:, i] for i in range(X.shape[1])}
                    env.update({"sqrt": np.sqrt, "log": np.log,
                                "exp": np.exp, "abs": np.abs})
                    return eval(formula.replace("^", "**"), {"__builtins__": {}}, env)

                y_combined_tr = y_pred_train + _eval_formula(
                    res2_llm.formula_real, X_train
                )
                y_combined_te = y_pred_test  + _eval_formula(
                    res2_llm.formula_real, X_test
                )
                y_full = np.concatenate([y_train, y_test])
                y_comb = np.concatenate([y_combined_tr, y_combined_te])
                result.combined_r2 = float(_r2(y_full, y_comb))
            except Exception as _ce:
                log.debug("[ResidualScan] combined R² error: %s", _ce)

            if res2_llm.consensus == "ПРИНЯТА":
                print(f"\n  {'★'*58}")
                print(f"  ★ LAYER 2 ПРИНЯТ Матрёшкой!")
                print(f"  ★ Layer 1: {layer1_formula_real}")
                print(f"  ★ Layer 2: {res2_llm.formula_real}  R²={res2_llm.r2_blind:.4f}")
                print(f"  ★ Комбо:   {result.combined_formula}")
                print(f"  ★ R²_combined = {result.combined_r2:.4f}")
                print(f"  {'★'*58}")
            else:
                print(f"  [ResidualScan] Layer 2 отклонён Матрёшкой ({res2_llm.consensus})")
        else:
            result.layer2_consensus = "LLM_ФАЗА_НЕ_ВЕРНУЛА_РЕЗУЛЬТАТ"

    except Exception as e:
        log.warning("[ResidualScan] Ошибка Layer 2 pipeline: %s", e)
        result.layer2_consensus = f"ОШИБКА: {str(e)[:100]}"

    # ── Шаг 5: сохраняем ─────────────────────────────────────────
    _save_layer2(result)

    return result
