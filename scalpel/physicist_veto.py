"""
physicist_veto.py — Право вето Физика v1.1 (уровень 4)

FIX v10.28: Модель вето изменена с phi4:14b → llama3.1:8b (Meta).

Причина: phi4:14b используется в Матрёшке и Диалектике ПОСЛЕ PySR.
Если та же модель одобрила гипотезу в вето ДО PySR — она предвзята
при верификации. Это нарушение принципа антиинцеста по истории
взаимодействия. llama3.1:8b (Meta) — независимая семья, не участвует
в вето ранее, нет конфликта интересов.

Вето проверяет предложение Navigator ДО запуска PySR.
Если операторы физически бессмысленны для данных размерностей — ВЕТО.

Зачем это нужно:
  Без вето: Navigator предлагает log(f0) где f0 содержит отрицательные →
  PySR тратит часы → Матрёшка отклоняет → ещё одна HADI-итерация.
  С вето: Летописец блокирует за секунды ДО PySR, хирург и препаратор
  пересматривают данные, Navigator получает новый шанс.

Вето проверяет:
  1. Нет ли операторов несовместимых с размерностями признаков
  2. Нет ли физически бессмысленных гипотез (log от отрицательного и т.д.)
  3. Соответствуют ли гипотезы числу признаков

Ответ: ВЕТО (с причиной) или OK (с кратким обоснованием).
"""
from __future__ import annotations

import json
import logging
import urllib.request
from typing import List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .shared_context import SharedContext

log = logging.getLogger("scalpel")

try:
    from .config import OLLAMA_HOST, ROLE_MODELS
    # FIX v10.28: вето использует Летописца (llama3.1:8b, Meta) — не phi4:14b.
    # phi4:14b стоит в Матрёшке и Диалектике ПОСЛЕ PySR → конфликт интересов.
    # llama3.1:8b независим от Физика, нет истории взаимодействия с гипотезой.
    _PHYSICIST_MODEL = ROLE_MODELS.get("Летописец", "llama3.1:8b")
except ImportError:
    OLLAMA_HOST = "http://localhost:11434"
    _PHYSICIST_MODEL = "llama3.1:8b"

VETO_TIMEOUT = 120   # llama3.1:8b быстрее phi4:14b на CPU (~40-60с) → 120с с запасом
MAX_VETOES   = 2    # максимум вето за один закон


# ─────────────────────────────────────────────────────────────────
# РЕЗУЛЬТАТ ПРОВЕРКИ
# ─────────────────────────────────────────────────────────────────

class VetoResult:
    def __init__(self, vetoed: bool, reason: str, suggestion: str = ""):
        self.vetoed     = vetoed
        self.reason     = reason
        self.suggestion = suggestion   # что изменить Navigator-у

    def __bool__(self):
        return self.vetoed

    def __str__(self):
        if self.vetoed:
            return f"ВЕТО: {self.reason}" + (f" | Совет: {self.suggestion}" if self.suggestion else "")
        return f"OK: {self.reason}"


# ─────────────────────────────────────────────────────────────────
# БЫСТРАЯ ЛОКАЛЬНАЯ ПРОВЕРКА (без LLM)
# ─────────────────────────────────────────────────────────────────

_DIM_NAMES = {
    0: "dimensionless", 1: "unknown", 2: "length",
    3: "mass", 4: "temperature", 5: "count",
    6: "force/energy", 8: "time", 10: "price",
}

def _fast_local_check(
    operators:    List[str],
    hypotheses:   List[str],
    shadow_names: List[str],
    dim_codes:    List[int],
    y_negative:   float,
) -> Optional[VetoResult]:
    """
    Мгновенная проверка без LLM — ловит очевидные ошибки.
    Возвращает VetoResult если нашёл проблему, None если всё OK.
    """
    # Правило 1: log/sqrt при отрицательных данных
    if y_negative > 0.02:
        if "log" in operators:
            return VetoResult(
                vetoed=True,
                reason=f"Оператор log несовместим: {y_negative:.1%} значений отрицательные",
                suggestion="Убрать log из операторов или применить standardize к y",
            )
        if "sqrt" in operators:
            return VetoResult(
                vetoed=True,
                reason=f"Оператор sqrt несовместим: {y_negative:.1%} значений отрицательные",
                suggestion="Убрать sqrt из операторов или применить standardize к y",
            )

    # Правило 2: гипотезы ссылаются на несуществующие признаки
    import re
    valid = set(shadow_names)
    for hyp in hypotheses:
        refs = set(re.findall(r'f\d+', hyp))
        bad  = refs - valid
        if bad:
            return VetoResult(
                vetoed=True,
                reason=f"Гипотеза '{hyp}' ссылается на несуществующие признаки: {bad}",
                suggestion=f"Использовать только: {sorted(valid)}",
            )

    # Правило 3: sin/cos для нефизических размерностей (деньги, количество)
    non_angular = {10, 5}   # цена и количество — sin/cos бессмысленны
    if ("sin" in operators or "cos" in operators):
        if all(d in non_angular for d in dim_codes):
            return VetoResult(
                vetoed=True,
                reason="Операторы sin/cos бессмысленны для финансовых/счётных данных",
                suggestion="Заменить sin/cos на *, /, pow",
            )

    return None   # всё OK


# ─────────────────────────────────────────────────────────────────
# LLM ПРОВЕРКА (phi4:14b)
# ─────────────────────────────────────────────────────────────────

def _llm_veto_check(
    operators:    List[str],
    hypotheses:   List[str],
    shadow_names: List[str],
    dim_codes:    List[int],
    y_stats:      dict,
) -> Optional[VetoResult]:
    """
    Физик проверяет предложение Navigator через phi4:14b.
    Только если локальная проверка не нашла проблем.
    """
    dim_desc = ", ".join(
        f"{n}({_DIM_NAMES.get(d, str(d))})"
        for n, d in zip(shadow_names, dim_codes)
    )

    prompt = f"""You are a physicist reviewing a symbolic regression proposal.
Evaluate if the proposed operators and hypotheses are physically valid.

DATA STATISTICS:
  features: {dim_desc}
  y: min={y_stats.get('y_min', '?'):.3g}, max={y_stats.get('y_max', '?'):.3g}
  negative_fraction: {y_stats.get('negative_fraction', 0):.1%}

NAVIGATOR PROPOSAL:
  operators: {operators}
  hypotheses: {hypotheses[:3]}

VETO if ANY of these apply:
  - log or sqrt proposed but data has negative values
  - operators physically incompatible with given dimensions
  - hypotheses reference features that don't exist
  - sin/cos on non-angular, non-periodic quantities

Respond in JSON only:
{{
  "veto": true or false,
  "reason": "one sentence",
  "suggestion": "what to change (or empty string)"
}}"""

    try:
        payload = json.dumps({
            "model": _PHYSICIST_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.1, "num_predict": 150},
        }).encode()
        req = urllib.request.Request(
            f"{OLLAMA_HOST}/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=VETO_TIMEOUT) as resp:
            text = json.loads(resp.read()).get("response", "").strip()

        start = text.find("{")
        end   = text.rfind("}") + 1
        if start >= 0 and end > start:
            data = json.loads(text[start:end])
            return VetoResult(
                vetoed     = bool(data.get("veto", False)),
                reason     = str(data.get("reason", "")),
                suggestion = str(data.get("suggestion", "")),
            )
    except Exception as e:
        log.warning("[Физик/Вето] LLM ошибка: %s", e)

    return None


# ─────────────────────────────────────────────────────────────────
# ГЛАВНАЯ ФУНКЦИЯ
# ─────────────────────────────────────────────────────────────────

def physicist_veto_check(
    operators:    List[str],
    hypotheses:   List[str],
    shadow_names: List[str],
    dim_codes:    List[int],
    y_stats:      dict,
    ctx:          "SharedContext" = None,
    use_llm:      bool = True,
) -> VetoResult:
    """
    Физик проверяет предложение Navigator ДО запуска PySR.

    Порядок проверок:
      1. Быстрая локальная (мгновенно, без LLM)
      2. LLM phi4:14b (если локальная ничего не нашла)

    Если ctx передан — записывает результат в SharedContext.
    """
    # Проверяем лимит вето
    if ctx is not None and ctx.veto_count >= MAX_VETOES:
        result = VetoResult(False, f"Лимит вето достигнут ({MAX_VETOES})")
        log.info("[Физик/Вето] Пропущено — лимит вето достигнут")
        return result

    print(f"  [Физик/Вето] Проверяю предложение Navigator…")

    # 1. Локальная проверка
    y_neg = y_stats.get("negative_fraction", 0.0)
    local = _fast_local_check(operators, hypotheses, shadow_names, dim_codes, y_neg)
    if local is not None:
        print(f"  [Физик/Вето] ⚡ Локально: {local}")
        if ctx is not None and local.vetoed:
            ctx.issue_veto(local.reason, local.suggestion)
        return local

    # 2. LLM проверка
    if use_llm:
        llm_result = _llm_veto_check(operators, hypotheses, shadow_names, dim_codes, y_stats)
        if llm_result is not None:
            print(f"  [Физик/Вето] phi4: {llm_result}")
            if ctx is not None and llm_result.vetoed:
                ctx.issue_veto(llm_result.reason, llm_result.suggestion)
            return llm_result

    result = VetoResult(False, "Физически корректно")
    print(f"  [Физик/Вето] ✓ OK")
    return result
