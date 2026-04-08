"""
preparator.py — Препаратор v2.1 (LLM + итерационный режим)

Определяет пространство поиска перед PySR.
Может работать в двух режимах:
  - Автоматический: LLM с учебником принимает решение
  - Итерационный: config.PREPARATOR_FORCE_TRANSFORM задаёт трансформацию

Итерационный режим используется в run_feynman.py:
  Если 4 HADI итерации не нашли закон → пробуем другую трансформацию.
"""
from __future__ import annotations

import json
import logging
import time
import numpy as np
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

log = logging.getLogger("scalpel")

try:
    from .config import SYNTHESIS_MODEL, OLLAMA_HOST, VAULT_DIR
    # PREPARATOR_FORCE_TRANSFORM может отсутствовать в старых конфигах
    try:
        from .config import PREPARATOR_FORCE_TRANSFORM
    except ImportError:
        PREPARATOR_FORCE_TRANSFORM = None
except Exception:
    SYNTHESIS_MODEL = "gemma2:9b"
    OLLAMA_HOST = "http://localhost:11434"
    VAULT_DIR = Path("scalpel_vault")
    PREPARATOR_FORCE_TRANSFORM = None

PREPARATOR_LOG_PATH = VAULT_DIR / "preparator_log.jsonl"
LOG_THRESHOLD = 100.0

# Учебник Препаратора — явные правила
PREPARATOR_GUIDE = """
ПРАВИЛО 1: ratio > 100 И positive% > 90% → log
  Пример: температура 200-6000К, давление в широком диапазоне.
  Степенные и экспоненциальные законы лучше ищутся в log-пространстве.

ПРАВИЛО 2: 20 < ratio <= 100 И positive% > 95% И negative% < 5% → sqrt
  Пример: умеренный разброс, данные почти полностью положительные.
  ВАЖНО: если negative% >= 5% — sqrt не применяй, используй none или standardize.

ПРАВИЛО 2б: 20 < ratio <= 100 И positive% > 90% И negative% >= 5% → none
  Пример: данные в основном положительные но есть значимая доля отрицательных.
  sqrt обнулит отрицательные точки и исказит данные — лучше без трансформации.

ПРАВИЛО 3: negative% > 10% → standardize
  Пример: скорость (может быть отрицательной), температура в Цельсиях.

ПРАВИЛО 4: ratio <= 20 → none
  Данные в норме, преобразование не нужно.

ВАЖНО: Если данные содержат константный признак (все значения одинаковы) —
это физическая константа (g=9.81, c=3e8). Не масштабируй из-за неё.
"""


@dataclass
class PreparationResult:
    y_transformed:    np.ndarray
    transform_name:   str   = "none"
    transform_params: dict  = field(default_factory=dict)
    ratio_before:     float = 1.0
    ratio_after:      float = 1.0
    applied:          bool  = False
    llm_reasoning:    str   = ""
    llm_used:         bool  = False


def analyze_and_prepare(
    y: np.ndarray,
    domain_type: str = "",
    host:  str = OLLAMA_HOST,
    model: str = SYNTHESIS_MODEL,
    verbose: bool = True,
    force_transform: str = None,   # переопределяет config и LLM
    ctx=None,                      # v10.24: SharedContext для координации
) -> PreparationResult:
    """
    Анализирует y и применяет трансформацию.
    force_transform: None=авто, "none"/"log"/"sqrt"/"standardize"=принудительно
    """
    import scalpel.config as _cfg
    # Проверяем config-уровень force transform
    _cfg_force = getattr(_cfg, "PREPARATOR_FORCE_TRANSFORM", None)
    effective_force = force_transform or _cfg_force

    y = np.asarray(y, dtype=float)
    y_min  = float(np.min(y))
    y_max  = float(np.max(y))
    y_mean = float(np.mean(y))
    y_std  = float(np.std(y))
    y_med  = float(np.median(y))
    frac_positive = float(np.mean(y > 0))
    frac_negative = float(np.mean(y < 0))

    # ratio по положительным если их большинство
    if frac_positive > 0.90:
        y_pos = y[y > 0]
        p05 = float(np.percentile(y_pos, 5))
        p95 = float(np.percentile(y_pos, 95))
        ratio_mm = p95 / (p05 + 1e-10)
    else:
        p05 = float(np.percentile(y, 5))
        p95 = float(np.percentile(y, 95))
        ratio_mm = (p95 - p05) / (abs(y_med) + 1e-10)

    if verbose:
        print(f"\n  [Препаратор v2.1] Анализ y:")
        print(f"    min={y_min:.3e}  max={y_max:.3e}  mean={y_mean:.3e}")
        print(f"    p05={p05:.3e}    p95={p95:.3e}    ratio={ratio_mm:.1f}")
        print(f"    positive={frac_positive*100:.0f}%  negative={frac_negative*100:.0f}%")

    # ── Режим: рекомендация МетаПаттернов ───────────────────────
    # v10.25: паттерны имеют приоритет ниже force но выше LLM/fallback
    _pattern_transform = None
    _pattern_hint_str  = ""
    if ctx is not None:
        _pattern_transform = ctx.get_pattern_action("preparator", "transform")
        _pattern_hint_str  = ctx.get_pattern_hint("preparator")
        if _pattern_transform and verbose:
            print(f"  [МетаПаттерны/Препаратор] Рекомендация: {_pattern_transform}")

    # ── Режим: принудительная трансформация ─────────────────────
    if effective_force is not None:
        transform_name = effective_force
        llm_reasoning  = f"Итерационный режим: '{effective_force}'"
        llm_used       = False
        if verbose:
            print(f"  [Препаратор] 🔄 Итерационный режим: {effective_force}")

    # ── Режим: LLM с учебником ───────────────────────────────────
    else:
        transform_name = "none"
        llm_reasoning  = ""
        llm_used       = False

        try:
            from .navigator import ollama_chat

            prompt = (
                f"Ты — Препаратор данных. Используй ТОЛЬКО правила из учебника.\n\n"
                f"УЧЕБНИК:\n{PREPARATOR_GUIDE}\n\n"
                f"СТАТИСТИКА y:\n"
                f"  min={y_min:.4g}  max={y_max:.4g}\n"
                f"  p05={p05:.4g}    p95={p95:.4g}\n"
                f"  ratio={ratio_mm:.1f}\n"
                f"  positive={frac_positive*100:.0f}%  negative={frac_negative*100:.0f}%\n"
                + (f"  Домен: {domain_type}\n" if domain_type else "")
                + f"\nВыбери правило и ответь СТРОГО:\n"
                f"ПРЕОБРАЗОВАНИЕ: <none|log|sqrt|standardize>\n"
                f"ПРИЧИНА: [Правило N] <одно предложение>\n"
                f"Отвечай только на русском языке."
            )

            t0 = time.time()
            response = ollama_chat(
                prompt, model=model, host=host,
                temperature=0.1, num_predict=100,
            ).strip()
            elapsed = time.time() - t0

            if response and not response.startswith("[OLLAMA_ERROR]"):
                llm_used = True
                for line in response.splitlines():
                    line = line.strip()
                    if line.upper().startswith("ПРЕОБРАЗОВАНИЕ:"):
                        raw = line.split(":", 1)[1].strip().lower()
                        if raw in ("none", "log", "sqrt", "standardize"):
                            transform_name = raw
                    elif line.upper().startswith("ПРИЧИНА:"):
                        llm_reasoning = line.split(":", 1)[1].strip()

                # FIX v10.23: пост-валидация — LLM не должна выбирать sqrt
                # если есть значимая доля отрицательных (sqrt обнулит их)
                if transform_name == "sqrt" and frac_negative >= 0.05:
                    old_name = transform_name
                    transform_name = "none"
                    llm_reasoning += f" [OVERRIDE: sqrt→none, negative={frac_negative*100:.0f}% >= 5%]"
                    if verbose:
                        print(f"  [Препаратор] ⚠️  Переопределение: {old_name}→none (negative={frac_negative*100:.0f}%)")

                if verbose:
                    print(f"  [Препаратор] 🧠 LLM решение ({elapsed:.1f}с): {transform_name}")
                    print(f"  [Препаратор] {llm_reasoning}")
            else:
                raise ValueError("LLM недоступна")

        except Exception as e:
            # Математический fallback
            log.debug("[Препаратор] Fallback: %s", e)
            if frac_positive > 0.90 and ratio_mm > LOG_THRESHOLD:
                transform_name = "log"
                llm_reasoning  = f"Fallback [Правило 1]: ratio={ratio_mm:.1f} > {LOG_THRESHOLD}"
            elif frac_positive > 0.90 and ratio_mm > 20:
                # FIX v10.23: sqrt только если < 5% отрицательных
                if frac_negative < 0.05:
                    transform_name = "sqrt"
                    llm_reasoning  = f"Fallback [Правило 2]: ratio={ratio_mm:.1f}"
                else:
                    transform_name = "none"
                    llm_reasoning  = f"Fallback [Правило 2б]: ratio={ratio_mm:.1f} но negative={frac_negative*100:.0f}% >= 5%"
            elif frac_negative > 0.10:
                transform_name = "standardize"
                llm_reasoning  = "Fallback [Правило 3]: есть отрицательные"
            else:
                transform_name = "none"
                llm_reasoning  = "Fallback [Правило 4]: данные в норме"
            # v10.25: паттерн перекрывает fallback если надёжный
            if _pattern_transform and transform_name == "none":
                transform_name = _pattern_transform
                llm_reasoning  = f"МетаПаттерн: {_pattern_transform}"
            if verbose:
                print(f"  [Препаратор] ⚠ LLM недоступна → {transform_name}: {llm_reasoning}")

    # ── Сигнализируем намерение в SharedContext ДО применения ────
    # v10.24: хирург/координатор может ещё вмешаться на этом этапе
    if ctx is not None:
        ctx.prep_signal_intent(transform_name)

    # ── Применяем трансформацию ──────────────────────────────────
    if transform_name == "log":
        y_out = np.log(np.clip(y, 1e-10, None))
        ratio_after = float(np.max(y_out) - np.min(y_out))
        if verbose:
            print(f"  [Препаратор] ⚡ log(y) → диапазон [{np.min(y_out):.2f}, {np.max(y_out):.2f}]")
            print(f"  [Препаратор] → Navigator: добавить операторы ['log', 'exp']")
        result = PreparationResult(
            y_transformed=y_out, transform_name="log",
            transform_params={"y_min_orig": float(y_min), "y_max_orig": float(y_max)},
            ratio_before=ratio_mm, ratio_after=ratio_after,
            applied=True, llm_reasoning=llm_reasoning, llm_used=llm_used,
        )

    elif transform_name == "sqrt":
        y_out = np.sqrt(np.clip(y, 0, None))
        ratio_after = float(np.max(y_out) / (np.min(y_out) + 1e-10))
        if verbose:
            print(f"  [Препаратор] ⚡ sqrt(y)")
            print(f"  [Препаратор] → Navigator: добавить операторы ['sqrt']")
        result = PreparationResult(
            y_transformed=y_out, transform_name="sqrt",
            ratio_before=ratio_mm, ratio_after=ratio_after,
            applied=True, llm_reasoning=llm_reasoning, llm_used=llm_used,
        )

    elif transform_name == "standardize":
        m = float(np.mean(y)); s = float(np.std(y)) + 1e-10
        y_out = (y - m) / s
        if verbose:
            print(f"  [Препаратор] ⚡ standardize (mean={m:.3g} std={s:.3g})")
        result = PreparationResult(
            y_transformed=y_out, transform_name="standardize",
            transform_params={"mean": m, "std": s},
            ratio_before=ratio_mm, ratio_after=float(np.max(y_out)-np.min(y_out)),
            applied=True, llm_reasoning=llm_reasoning, llm_used=llm_used,
        )

    else:  # none
        if verbose:
            print(f"  [Препаратор] ✓ Без преобразования")
        result = PreparationResult(
            y_transformed=y, transform_name="none",
            ratio_before=ratio_mm, ratio_after=ratio_mm,
            applied=False, llm_reasoning=llm_reasoning, llm_used=llm_used,
        )

    # v10.24: записываем что реально применили
    if ctx is not None:
        ctx.prep_write_applied(result.transform_name)

    # Сохраняем решение
    _save_log(result, y_min, y_max, ratio_mm, frac_positive, domain_type, effective_force)
    return result


def _save_log(result, y_min, y_max, ratio, frac_pos, domain, force):
    try:
        PREPARATOR_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        entry = {
            "ts": datetime.now().isoformat(timespec="seconds"),
            "domain": domain, "y_min": round(y_min, 6), "y_max": round(y_max, 6),
            "ratio": round(ratio, 1), "frac_positive": round(frac_pos, 3),
            "transform": result.transform_name, "applied": result.applied,
            "llm_used": result.llm_used, "force": force,
            "reasoning": result.llm_reasoning,
        }
        with open(PREPARATOR_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception as e:
        log.debug("[Препаратор] Лог: %s", e)


def inverse_transform(formula_str: str, prep: PreparationResult) -> str:
    """Переводит формулу обратно в оригинальное пространство."""
    if not prep.applied:
        return formula_str
    if prep.transform_name == "log":
        return f"exp({formula_str})"
    if prep.transform_name == "sqrt":
        return f"({formula_str})**2"
    if prep.transform_name == "standardize":
        m = prep.transform_params.get("mean", 0)
        s = prep.transform_params.get("std", 1)
        return f"({formula_str}) * {s:.6g} + {m:.6g}"
    return formula_str


def prepare_report(prep: PreparationResult) -> str:
    src = "LLM" if prep.llm_used else ("force" if prep.transform_name != "none" else "auto")
    if not prep.applied:
        return f"Препаратор: без преобразования. {prep.llm_reasoning}"
    return (
        f"Препаратор ({src}): '{prep.transform_name}' "
        f"(ratio {prep.ratio_before:.1f}→{prep.ratio_after:.1f}). "
        f"{prep.llm_reasoning}"
    )
