"""
surgeon.py — LLM-Хирург v1.0 (yi:9b / 01.AI)

Независимый ЛЛМ-хирург для предобработки данных перед PySR.

Принципы:
  - Видит ТОЛЬКО статистику данных (не сырые точки, не формулы)
  - Принимает решения о предобработке: cut_fraction, Ricci окно
  - Не участвует в поиске формулы — изолирован от Матрёшки
  - Учится от прогона к прогону через surgeon_log.jsonl

Семья: 01.AI (yi:9b) — не пересекается ни с одной ролью в системе.

Цикл обучения:
  Прогон N → решение хирурга → R² результат →
  surgeon_log.jsonl → Прогон N+1 умнее
"""
from __future__ import annotations

import gc
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, TYPE_CHECKING
if TYPE_CHECKING:
    from .shared_context import SharedContext

import numpy as np

log = logging.getLogger("scalpel")

# ─────────────────────────────────────────────────────────────────
# КОНСТАНТЫ
# ─────────────────────────────────────────────────────────────────

try:
    from .config import SURGEON_MODEL, VAULT_DIR
    _SURGEON_MODEL = SURGEON_MODEL
    _VAULT_DIR     = VAULT_DIR
except ImportError:
    _SURGEON_MODEL = "yi:9b"
    _VAULT_DIR     = Path("scalpel_vault")

SURGEON_LOG_PATH = _VAULT_DIR / "surgeon_log.jsonl"
OLLAMA_HOST      = "http://localhost:11434"
SURGEON_TIMEOUT  = 120   # yi:9b быстрее чем deepseek, 120с достаточно

# Дефолтные значения если LLM недоступен
DEFAULT_CUT_FRACTION  = 0.050  # REAL MODE: 10% шум — увеличен с 0.025
DEFAULT_RICCI_WINDOW  = 11     # REAL MODE: 10% шум — увеличен с 7
DEFAULT_IQR_MULT      = 2.5    # REAL MODE: 10% шум — снижен с 3.0 (агрессивнее)


# ─────────────────────────────────────────────────────────────────
# DATACLASS РЕШЕНИЯ ХИРУРГА
# ─────────────────────────────────────────────────────────────────

@dataclass
class SurgeonDecision:
    """Решение хирурга по предобработке данных."""
    cut_fraction:   float  = DEFAULT_CUT_FRACTION   # доля точек для удаления
    iqr_multiplier: float  = DEFAULT_IQR_MULT        # порог IQR детектора
    ricci_window:   int    = DEFAULT_RICCI_WINDOW    # окно Savitzky-Golay
    apply_surgery:  bool   = True                    # применять ли хирургию
    apply_ricci:    bool   = True                    # применять ли Ricci Flow
    reasoning:      str    = ""                      # объяснение решения
    llm_used:       bool   = False                   # использовался ли LLM
    fallback:       bool   = False                   # True = IQR fallback


# ─────────────────────────────────────────────────────────────────
# ЛОГ ХИРУРГА — память между прогонами
# ─────────────────────────────────────────────────────────────────

def _load_surgeon_experience(limit: int = 10) -> str:
    """
    Читает последние N записей из surgeon_log.jsonl.
    Формирует текст опыта для промпта хирурга.
    """
    if not SURGEON_LOG_PATH.exists():
        return "No previous experience yet."

    try:
        lines = SURGEON_LOG_PATH.read_text(encoding="utf-8").strip().splitlines()
        records = []
        for line in lines[-limit:]:
            try:
                records.append(json.loads(line))
            except Exception:
                pass

        if not records:
            return "No previous experience yet."

        parts = []
        for r in records:
            stats   = r.get("data_stats", {})
            dec     = r.get("decision", {})
            outcome = r.get("outcome", {})
            r2_b    = outcome.get("r2_before_surgery", "?")
            r2_a    = outcome.get("r2_after_pysr",     "?")
            cut     = dec.get("cut_fraction", "?")
            ricci_w = dec.get("ricci_window",  "?")
            ratio   = stats.get("ratio", "?")
            pct_out = stats.get("pct_iqr_outliers", "?")
            surgery = dec.get("apply_surgery", "?")
            parts.append(
                f"- ratio={ratio}, outliers={pct_out}% → "
                f"surgery={surgery}, cut={cut}, ricci_w={ricci_w} → "
                f"R²={r2_a} (was ~{r2_b})"
            )

        return "\n".join(parts)

    except Exception as e:
        log.warning("[Surgeon] Не удалось загрузить опыт: %s", e)
        return "Experience unavailable."


def _save_surgeon_log(
    data_stats: dict,
    decision:   SurgeonDecision,
    r2_before:  Optional[float],
    r2_after:   Optional[float],
) -> None:
    """Записывает решение и результат в лог для следующих прогонов."""
    try:
        SURGEON_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        entry = {
            "ts":          time.strftime("%Y-%m-%dT%H:%M:%S"),
            "data_stats":  data_stats,
            "decision": {
                "cut_fraction":   decision.cut_fraction,
                "iqr_multiplier": decision.iqr_multiplier,
                "ricci_window":   decision.ricci_window,
                "apply_surgery":  decision.apply_surgery,
                "apply_ricci":    decision.apply_ricci,
                "reasoning":      decision.reasoning[:200],
                "llm_used":       decision.llm_used,
            },
            "outcome": {
                "r2_before_surgery": round(r2_before, 4) if r2_before is not None else None,
                "r2_after_pysr":     round(r2_after,  4) if r2_after  is not None else None,
            },
        }
        with open(SURGEON_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception as e:
        log.warning("[Surgeon] Ошибка записи лога: %s", e)


# ─────────────────────────────────────────────────────────────────
# СТАТИСТИКА ДАННЫХ
# ─────────────────────────────────────────────────────────────────

def _compute_data_stats(y: np.ndarray) -> dict:
    """Вычисляет статистику данных для промпта хирурга."""
    y_flat = y.ravel()
    n      = len(y_flat)

    Q1  = float(np.percentile(y_flat, 25))
    Q3  = float(np.percentile(y_flat, 75))
    IQR = Q3 - Q1

    # % выбросов по стандартному IQR критерию (k=3)
    if IQR > 1e-12:
        lower = Q1 - 3.0 * IQR
        upper = Q3 + 3.0 * IQR
        pct_out = round(100.0 * np.sum((y_flat < lower) | (y_flat > upper)) / n, 1)
    else:
        pct_out = 0.0

    y_min = float(np.min(y_flat))
    y_max = float(np.max(y_flat))
    ratio = round(abs(y_max / y_min), 1) if abs(y_min) > 1e-12 else round(y_max, 2)

    return {
        "n":                n,
        "y_min":            round(y_min, 4),
        "y_max":            round(y_max, 4),
        "y_mean":           round(float(np.mean(y_flat)), 4),
        "y_std":            round(float(np.std(y_flat)),  4),
        "Q1":               round(Q1,  4),
        "Q3":               round(Q3,  4),
        "IQR":              round(IQR, 4),
        "ratio":            ratio,
        "pct_negative":     round(100.0 * np.sum(y_flat < 0) / n, 1),
        "pct_iqr_outliers": pct_out,
    }


# ─────────────────────────────────────────────────────────────────
# LLM ВЫЗОВ
# ─────────────────────────────────────────────────────────────────

def _call_surgeon_llm(stats: dict, experience: str) -> Optional[dict]:
    """
    Вызывает yi:9b для принятия решения о предобработке.
    Возвращает dict с параметрами или None при ошибке.
    """
    import urllib.request

    prompt = f"""You are a data preprocessing surgeon for symbolic regression.
You see ONLY data statistics — never the raw data or the formula.
Your job: decide how to clean the data so PySR can find the formula.

DATA STATISTICS:
- n={stats['n']} points
- min={stats['y_min']}, max={stats['y_max']}, mean={stats['y_mean']}
- std={stats['y_std']}, IQR={stats['IQR']}
- ratio max/min = {stats['ratio']}
- negative values: {stats['pct_negative']}%
- IQR outliers (k=3): {stats['pct_iqr_outliers']}%

YOUR PAST EXPERIENCE (what worked before):
{experience}

DECISION RULES:
- apply_surgery=true if pct_iqr_outliers > 1%  # REAL MODE: низкий порог
- cut_fraction: between 0.03 and 0.08 (real data has more noise)
- iqr_multiplier: 2.0 if many outliers (>5%), 3.0 if few (<3%)
- ricci_window: 11 for real data (10% noise), 15 if very noisy (std/IQR > 3)
- apply_ricci=true always unless n < 30

Respond ONLY with valid JSON, no explanation outside JSON:
{{
  "apply_surgery": true or false,
  "cut_fraction": 0.025,
  "iqr_multiplier": 3.0,
  "apply_ricci": true or false,
  "ricci_window": 7,
  "reasoning": "one sentence why"
}}"""

    try:
        payload = json.dumps({
            "model":  _SURGEON_MODEL,
            "prompt": prompt,
            "stream": False,
        }).encode()

        req = urllib.request.Request(
            f"{OLLAMA_HOST}/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=SURGEON_TIMEOUT) as resp:
            raw  = json.loads(resp.read())
            text = raw.get("response", "").strip()

        # Парсим JSON из ответа
        start = text.find("{")
        end   = text.rfind("}") + 1
        if start >= 0 and end > start:
            parsed = json.loads(text[start:end])
            return parsed

    except Exception as e:
        log.warning("[Surgeon] LLM ошибка: %s", e)

    return None


# ─────────────────────────────────────────────────────────────────
# ГЛАВНАЯ ФУНКЦИЯ
# ─────────────────────────────────────────────────────────────────

def surgeon_decide(y: np.ndarray, ctx=None) -> tuple[SurgeonDecision, dict]:
    """
    Хирург принимает решение о предобработке данных.

    Возвращает:
        (SurgeonDecision, data_stats)

    Если LLM недоступен — fallback на IQR defaults.
    """
    stats      = _compute_data_stats(y)
    experience = _load_surgeon_experience(limit=10)
    decision   = SurgeonDecision()

    print(f"  [Хирург yi:9b] Анализирую данные "
          f"(n={stats['n']}, outliers={stats['pct_iqr_outliers']}%, "
          f"ratio={stats['ratio']})…")

    # v10.25: паттерны могут рекомендовать iqr_k до LLM вызова
    _pattern_iqr = None
    if ctx is not None:
        _pattern_iqr = ctx.get_pattern_action("surgeon", "iqr_adjust")
        if _pattern_iqr:
            print(f"  [МетаПаттерны/Хирург] Рекомендованный iqr_k={_pattern_iqr}")

    t0     = time.time()
    result = _call_surgeon_llm(stats, experience)
    elapsed = round(time.time() - t0, 1)

    if result:
        # FIX v10.23: yi:9b возвращает null вместо числа —
        # .get(key, default) не помогает если ключ есть но значение None.
        _s = result.get
        decision.apply_surgery  = bool(_s("apply_surgery")  if _s("apply_surgery")  is not None else True)
        decision.cut_fraction   = float(_s("cut_fraction")   or DEFAULT_CUT_FRACTION)
        decision.iqr_multiplier = float(_s("iqr_multiplier") or DEFAULT_IQR_MULT)
        decision.apply_ricci    = bool(_s("apply_ricci")     if _s("apply_ricci")   is not None else True)
        decision.ricci_window   = int(_s("ricci_window")     or DEFAULT_RICCI_WINDOW)
        decision.reasoning      = str(_s("reasoning")        or "")
        decision.llm_used       = True

        # Защита от экстремальных значений
        decision.cut_fraction   = max(0.005, min(0.05, decision.cut_fraction))
        decision.iqr_multiplier = max(1.5,   min(5.0,  decision.iqr_multiplier))
        decision.ricci_window   = max(5,      min(21,   decision.ricci_window))
        if decision.ricci_window % 2 == 0:
            decision.ricci_window += 1

        print(f"  [Хирург yi:9b] ✓ Решение ({elapsed}с): "
              f"surgery={decision.apply_surgery}, "
              f"cut={decision.cut_fraction:.3f}, "
              f"iqr_k={decision.iqr_multiplier}, "
              f"ricci_w={decision.ricci_window}")
        print(f"  [Хирург yi:9b] Обоснование: {decision.reasoning}")
    else:
        decision.fallback  = True
        # v10.25: паттерн заменяет fallback если есть рекомендация
        if _pattern_iqr:
            decision.iqr_multiplier = float(_pattern_iqr)
            decision.reasoning = f"LLM недоступен → паттерн: iqr_k={_pattern_iqr}"
            print(f"  [МетаПаттерны/Хирург] IQR defaults → паттерн iqr_k={_pattern_iqr}")
        else:
            decision.reasoning = "LLM недоступен — IQR defaults"
        print(f"  [Хирург yi:9b] ⚠️  Fallback на IQR defaults ({elapsed}с)")

    # v10.24: записываем намерение в SharedContext
    if ctx is not None:
        intent = "skip" if not decision.apply_surgery else (
            "aggressive" if decision.iqr_multiplier < 2.5 else "gentle"
        )
        ctx.surgeon_write(
            intent        = intent,
            outlier_ratio = stats["pct_iqr_outliers"] / 100.0,
            iqr_mult      = decision.iqr_multiplier,
            cut_fraction  = decision.cut_fraction,
        )

    gc.collect()
    return decision, stats


def surgeon_record_outcome(
    stats:     dict,
    decision:  SurgeonDecision,
    r2_before: Optional[float],
    r2_after:  Optional[float],
) -> None:
    """
    Записывает исход в лог после того как PySR отработал.
    Это и есть механизм обучения — следующий прогон прочитает этот опыт.
    """
    _save_surgeon_log(stats, decision, r2_before, r2_after)
    log.info(
        "[Surgeon] Опыт записан: r2_before=%.4f r2_after=%.4f",
        r2_before or 0.0, r2_after or 0.0,
    )
