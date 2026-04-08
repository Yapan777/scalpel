"""
navigator.py — Штурман v9.9.

Приоритет:
  1. DSPy NavModule (скомпилированный, если есть)
  2. Legacy Ollama JSON (как в v9.8)
  3. Hardcoded fallback

DSPyOrchestrator вызывается из engine.py.
Этот модуль отвечает только за парсинг и legacy-путь.
"""
from __future__ import annotations

import json
import logging
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Dict, List, Optional

from .config import OLLAMA_HOST, OLLAMA_MODEL

log = logging.getLogger("scalpel")

# FIX v10.16: не хардкодим f1/f2 — генерируем под реальные признаки
_FALLBACK_HYPOTHESES_1 = [            # 1 признак
    # FIX v10.21: убран f0^1.5 — прямая подсказка для теста Кеплера (нечестно)
    # Fallback должен содержать только общие структуры, не конкретные законы
    "f0 ^ 2", "sqrt(f0)", "log(f0)",
    "exp(f0)", "f0 * f0",
]
_FALLBACK_HYPOTHESES_2 = [            # 2+ признака
    "f0 / f1", "f0 * f1",
    "sqrt(f0) * f1", "f0 ^ 2 + f1",
    "f0 + f1 * f0",
]
_FALLBACK_HYPOTHESES_3 = [            # 3+ признака
    "f0 / f1", "f0 * f1 + f2",
    "sqrt(f0) * f1", "f0 * f2",
]
_DEFAULT_OPS = ["+", "-", "*", "/", "sqrt", "log", "exp", "abs"]


def _fallback_hypotheses(shadow_names: list) -> list:
    """Возвращает fallback гипотезы, совместимые с реальным набором признаков."""
    n = len(shadow_names)
    if n >= 3:
        return _FALLBACK_HYPOTHESES_3[:]
    elif n == 2:
        return _FALLBACK_HYPOTHESES_2[:]
    else:
        return _FALLBACK_HYPOTHESES_1[:]


def _filter_hypotheses(hyps: list, shadow_names: list) -> list:
    """
    Убирает гипотезы, ссылающиеся на несуществующие признаки.
    FIX v10.16: fallback генерировал f1/f2 при shadow_names=['f0'].
    FIX v10.20: отсекаем голые имена признаков (f0, f1) — это не выражения.
    """
    import re as _re
    valid = set(shadow_names)
    result = []
    for h in hyps:
        h = h.strip()
        if not h:
            continue
        # Отсекаем голые имена признаков — они бесполезны как seed для PySR
        if h in valid:
            continue
        # Отсекаем однослойные: только одна fN-ссылка и никаких операторов
        refs = set(_re.findall(r'f\d+', h))
        has_operator = bool(_re.search(r'[+\-*/^]|sqrt|log|exp|abs|sin|cos', h))
        if refs and not has_operator and len(refs) == 1 and h == list(refs)[0]:
            continue
        if refs.issubset(valid):
            result.append(h)
    return result or _fallback_hypotheses(shadow_names)

NAVIGATOR_LEGACY_PROMPT = """\
Ты — Штурман символьной регрессии. Дай стратегическое решение в JSON.
Применяй OODA (стабильность), Double-Diamond (признаки), ICE-rank (гипотезы).
Формат (строгий JSON, без markdown):
{
  "selected_features": ["f0","f2"],
  "selected_operators": ["+","-","*","/","sqrt","log"],
  "hypotheses": ["f0/f2","sqrt(f0)*f2","f0*f2","f0+f2","f0*f0/f2"],
  "ooda_stable": true,
  "reasoning": "..."
}
Поля selected_features, selected_operators, hypotheses ОБЯЗАТЕЛЬНЫ.
"""


@dataclass
class NavDecision:
    selected_features:  List[str]
    selected_operators: List[str]
    hypotheses:         List[str]
    ooda_stable:        bool = True
    reasoning:          str  = ""


def _assert_localhost(host: str) -> None:
    netloc = urllib.parse.urlparse(host).netloc.split(":")[0].lower()
    if netloc not in ("localhost", "127.0.0.1", "::1"):
        raise PermissionError(f"[BLACK BOX] Внешний URL заблокирован: {host}")


def ollama_chat(
    user_msg:    str,
    system_msg:  str = "",
    model:       str = OLLAMA_MODEL,
    host:        str = OLLAMA_HOST,
    timeout:     int = 120,    # FIX v10.27: было 60 — 14B модели (phi4, qwen14b) на CPU нужно ~80-100с
    temperature: float = 0.2,
    num_predict: int = 500,
) -> str:
    _assert_localhost(host)
    options = {"temperature": temperature, "num_predict": num_predict}
    if system_msg:
        try:
            payload = json.dumps({
                "model": model, "stream": False,
                "messages": [
                    {"role": "system", "content": system_msg},
                    {"role": "user",   "content": user_msg},
                ],
                "options": options,
            }).encode()
            req = urllib.request.Request(
                f"{host.rstrip('/')}/api/chat", data=payload,
                headers={"Content-Type": "application/json"}, method="POST",
            )
            with urllib.request.urlopen(req, timeout=timeout) as r:
                val = json.loads(r.read()).get("message", {}).get("content", "")
                if val:
                    return val.strip()
        except Exception:
            pass
    try:
        combined = f"[System]\n{system_msg}\n\n[User]\n{user_msg}" if system_msg else user_msg
        payload  = json.dumps({
            "model": model, "prompt": combined,
            "stream": False, "options": options,
        }).encode()
        req = urllib.request.Request(
            f"{host.rstrip('/')}/api/generate", data=payload,
            headers={"Content-Type": "application/json"}, method="POST",
        )
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read()).get("response", "").strip()
    except urllib.error.URLError as e:
        return f"[OLLAMA_ERROR] URLError: {e.reason}"
    except Exception as e:
        return f"[OLLAMA_ERROR] {type(e).__name__}: {e}"


def navigator_ask_legacy(
    shadow_names:  List[str],
    dim_codes:     List[int],
    n_samples:     int,
    prev_hyps:     Optional[List[str]] = None,
    death_context: str = "",
    host:  str = OLLAMA_HOST,
    model: str = OLLAMA_MODEL,
    extra_ops:     Optional[List[str]] = None,  # v2.1: операторы от Препаратора
) -> NavDecision:
    vars_block = "\n".join(
        f"  {n}: dim_code={d}" for n, d in zip(shadow_names, dim_codes)
    )
    # Подсказка от Препаратора — только ЧТО применено, не ПОЧЕМУ
    _prep_hint = ""
    if extra_ops:
        _prep_hint = (
            f"\nДАННЫЕ МАСШТАБИРОВАНЫ. Рекомендуемые операторы: {', '.join(extra_ops)}."
        )
    user_msg = (
        f"ДАННЫЕ: {n_samples} наблюдений, {len(shadow_names)} признаков\n"
        f"ПРИЗНАКИ:\n{vars_block}\n"
        + _prep_hint
    )
    if death_context:
        user_msg += f"\nПРЕДЫДУЩАЯ ПОПЫТКА ПРОВАЛИЛАСЬ:\n{death_context}\n"
    if prev_hyps:
        user_msg += "\nУЖЕ ПРОВЕРЕНЫ:\n" + "\n".join(f"  - {h}" for h in prev_hyps) + "\n"
    user_msg += "\nВыдай JSON-решение."
    raw = ollama_chat(
        user_msg, system_msg=NAVIGATOR_LEGACY_PROMPT,
        model=model, host=host, timeout=360,   # FIX v10.27b: deepseek-r1:7b думает 165с, 360с = запас ×2.2
        temperature=0.3,  # FIX v10.19: было 0.2 — поднято для разнообразных гипотез
        num_predict=500,
    )
    return _parse_nav(raw, shadow_names)


def nav_decision_from_dspy(dspy_result: Dict, shadow_names: List[str]) -> NavDecision:
    feats_raw = dspy_result.get("selected_features", "")
    feats = [f.strip() for f in feats_raw.split(",") if f.strip() in shadow_names]
    if not feats:                          # FIX v10.16: was len<2, broke single-feature datasets
        feats = list(shadow_names)
    ops_raw = dspy_result.get("selected_operators", "")
    allowed = {"+","-","*","/","sqrt","log","exp","abs","sin","cos","tanh"}
    ops = [o.strip() for o in ops_raw.split(",") if o.strip() in allowed] or _DEFAULT_OPS[:]
    hyps_raw = dspy_result.get("hypotheses", "")
    hyps = [h.strip() for h in hyps_raw.split(";") if h.strip()]
    if not hyps:
        hyps = [h.strip() for h in hyps_raw.split(",") if h.strip()] or _fallback_hypotheses(shadow_names)
    hyps = _filter_hypotheses(hyps, shadow_names)  # FIX v10.16 + v10.20
    # FIX v10.20: если после фильтрации остались только fallback-гипотезы
    # и они все голые имена признаков — принудительно берём fallback
    if not hyps or all(h in shadow_names for h in hyps):
        hyps = _fallback_hypotheses(shadow_names)
    ooda = dspy_result.get("ooda_stable", "true").lower() != "false"
    # FIX v10.20: reasoning не должен начинаться с ] или мусора
    raw_reasoning = dspy_result.get("reasoning", "")
    clean_reasoning = raw_reasoning.lstrip("]},: \n").strip()[:200]
    return NavDecision(
        selected_features  = feats,
        selected_operators = ops,
        hypotheses         = hyps[:5],
        ooda_stable        = ooda,
        reasoning          = clean_reasoning,
    )


def _parse_nav(raw: str, shadow_names: List[str]) -> NavDecision:
    if not raw or raw.startswith("[OLLAMA_ERROR]"):
        return NavDecision(list(shadow_names), _DEFAULT_OPS[:], _fallback_hypotheses(shadow_names))
    clean = raw.strip()
    if clean.startswith("```"):
        clean = "\n".join(l for l in clean.splitlines() if not l.startswith("```")).strip()
    start, end = clean.find("{"), clean.rfind("}")
    if start == -1 or end <= start:
        return NavDecision(list(shadow_names), _DEFAULT_OPS[:], _fallback_hypotheses(shadow_names))
    try:
        data = json.loads(clean[start:end + 1])
    except json.JSONDecodeError:
        return NavDecision(list(shadow_names), _DEFAULT_OPS[:], _fallback_hypotheses(shadow_names))
    feats = [f for f in data.get("selected_features", []) if f in shadow_names]
    if not feats:                          # FIX v10.16: was len<2, broke single-feature datasets
        feats = list(shadow_names)
    allowed = {"+","-","*","/","sqrt","log","exp","abs","sin","cos","tanh"}
    ops = [o for o in data.get("selected_operators", []) if o in allowed] or _DEFAULT_OPS[:]
    raw_hyps = data.get("hypotheses", []) or _fallback_hypotheses(shadow_names)
    hyps = _filter_hypotheses(raw_hyps, shadow_names)  # FIX v10.16 + v10.20
    if not hyps or all(h in shadow_names for h in hyps):
        hyps = _fallback_hypotheses(shadow_names)
    raw_reasoning = str(data.get("reasoning", ""))
    clean_reasoning = raw_reasoning.lstrip("]},: \n").strip()[:120]
    return NavDecision(
        selected_features  = feats,
        selected_operators = ops,
        hypotheses         = hyps[:5],
        ooda_stable        = bool(data.get("ooda_stable", True)),
        reasoning          = clean_reasoning,
    )
