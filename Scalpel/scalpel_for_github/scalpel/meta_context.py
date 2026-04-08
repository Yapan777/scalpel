"""
meta_context.py — Инжекция мета-знаний во все LLM компоненты v10.14.

После мета-рефлексии система накопила выводы:
  - Паттерны ошибок (что не работает)
  - Паттерны успеха (что стабильно работает)
  - Правила навигации (если X → делай Y)
  - Открытия (новые законы)

Этот модуль раздаёт эти знания ВСЕМ LLM компонентам
чтобы каждый думал умнее на основе истории.

ПРИНЦИП: каждый компонент получает РЕЛЕВАНТНУЮ часть знаний:
  Navigator    → правила навигации + успешные операторы
  Матрёшка     → паттерны ошибок + контекст домена
  Delphi       → успешные советы + открытия
  Летописец    → все + discovery context
  HADI         → паттерны ошибок + что не сработало
"""
from __future__ import annotations

import json
import logging
from functools import lru_cache
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .config import VAULT_DIR

log = logging.getLogger("scalpel")

META_REFLECTION_LOG = VAULT_DIR / "meta_reflection.jsonl"


# ══════════════════════════════════════════════════════════════════
# ЗАГРУЗКА ПОСЛЕДНЕЙ МЕТА-РЕФЛЕКСИИ
# ══════════════════════════════════════════════════════════════════

@lru_cache(maxsize=1)
def _load_latest_meta() -> Optional[Dict]:
    """Загружает последнюю запись meta_reflection.jsonl. Кэш сбрасывается при новой рефлексии."""
    if not META_REFLECTION_LOG.exists():
        return None
    try:
        lines = META_REFLECTION_LOG.read_text(encoding="utf-8").strip().splitlines()
        if not lines:
            return None
        return json.loads(lines[-1])
    except Exception as e:
        log.debug("[MetaContext] Ошибка чтения: %s", e)
        return None


def _load_all_meta(last_n: int = 3) -> List[Dict]:
    """Загружает последние N рефлексий для агрегации."""
    if not META_REFLECTION_LOG.exists():
        return []
    try:
        lines = META_REFLECTION_LOG.read_text(encoding="utf-8").strip().splitlines()
        result = []
        for line in lines[-last_n:]:
            try:
                result.append(json.loads(line))
            except Exception:
                continue
        return result
    except Exception:
        return []


# ══════════════════════════════════════════════════════════════════
# КОНТЕКСТ ДЛЯ КАЖДОГО КОМПОНЕНТА
# ══════════════════════════════════════════════════════════════════

def get_navigator_context(domain_type: str = "") -> str:
    """
    Контекст для Navigator: правила навигации + успешные операторы.
    Инжектируется в failure_logs или data_meta.
    """
    meta = _load_latest_meta()
    if not meta:
        return ""

    analysis = meta.get("analysis", {})
    patterns = meta.get("patterns", {})
    parts    = []

    nav_rules = analysis.get("nav_rules", "")
    if nav_rules:
        parts.append(f"[META-ПРАВИЛА НАВИГАЦИИ]\n{nav_rules[:300]}")

    # Успешные операторы из паттернов
    ops = patterns.get("successful_operators", [])
    if ops:
        ops_str = ", ".join(f"{op}({c})" for op, c in ops[:5])
        parts.append(f"[УСПЕШНЫЕ ОПЕРАТОРЫ из истории]: {ops_str}")

    # Советы Delphi которые работали
    delphi_good = patterns.get("delphi_hints", {}).get("successful", [])
    if delphi_good:
        hints = ", ".join(f"{h}({c})" for h, c in delphi_good[:3])
        parts.append(f"[СОВЕТЫ DELPHI которые работали]: {hints}")

    # Открытия в этом домене
    disc_summary = _get_domain_discoveries(domain_type)
    if disc_summary:
        parts.append(disc_summary)

    return "\n".join(parts) if parts else ""


def get_matryoshka_context(
    formula: str,
    domain_type: str = "",
    role: str = "",
) -> str:
    """
    Контекст для ролей Матрёшки: паттерны ошибок + роль-специфичные выводы.
    """
    meta = _load_latest_meta()
    if not meta:
        return ""

    analysis = meta.get("analysis", {})
    patterns = meta.get("patterns", {})
    parts    = []

    errors = analysis.get("errors", "")
    if errors:
        parts.append(f"[ПАТТЕРНЫ ОШИБОК из истории]\n{errors[:250]}")

    # Кто чаще всего отклоняет — полезно для баланса
    rejectors = patterns.get("top_rejectors", [])
    if rejectors and role:
        for r in rejectors:
            if r.get("role") == role:
                reasons = ", ".join(r.get("top_reasons", [])[:2])
                parts.append(
                    f"[Ты ({role}) ранее отклонял {r['count']}× по причинам: {reasons}]"
                )

    successes = analysis.get("successes", "")
    if successes:
        parts.append(f"[ЧТО РАБОТАЛО]: {successes[:150]}")

    # Открытия
    disc_summary = _get_domain_discoveries(domain_type)
    if disc_summary:
        parts.append(disc_summary)

    return "\n".join(parts) if parts else ""


def get_delphi_context(domain_type: str = "") -> str:
    """
    Контекст для Delphi: успешные паттерны + что советовать.
    """
    meta = _load_latest_meta()
    if not meta:
        return ""

    analysis = meta.get("analysis", {})
    patterns = meta.get("patterns", {})
    parts    = []

    successes = analysis.get("successes", "")
    if successes:
        parts.append(f"[СИСТЕМА ЗНАЕТ что работает]\n{successes[:250]}")

    # Советы которые реально помогали
    sci = patterns.get("scientific_insights", {})
    sci_ops = sci.get("top_operators", [])
    sci_vars = sci.get("top_variables", [])
    if sci_ops:
        ops_str = ", ".join(f"{o}({c})" for o, c in sci_ops[:4])
        parts.append(f"[НАУЧНЫЕ ОПЕРАТОРЫ из прошлых циклов]: {ops_str}")
    if sci_vars:
        vars_str = ", ".join(f"{v}({c})" for v, c in sci_vars[:3])
        parts.append(f"[ПОЛЕЗНЫЕ ПЕРЕМЕННЫЕ из прошлых циклов]: {vars_str}")

    return "\n".join(parts) if parts else ""


def get_chronicle_context(
    domain_type: str = "",
    discovery_result = None,
) -> str:
    """
    Контекст для Летописца: всё + статус открытия.
    """
    parts = []

    # Мета-рефлексия
    meta = _load_latest_meta()
    if meta:
        analysis = meta.get("analysis", {})
        summary  = analysis.get("summary", "")
        if summary:
            parts.append(f"[ЭВОЛЮЦИЯ СИСТЕМЫ]\n{summary[:300]}")

    # Discovery статус
    if discovery_result is not None:
        status = getattr(discovery_result, "status", "")
        title  = getattr(discovery_result, "discovery_title", "")
        field  = getattr(discovery_result, "field", "")
        expl   = getattr(discovery_result, "explanation", "")

        if status == "discovery":
            parts.append(
                f"[★ ПОТЕНЦИАЛЬНОЕ ОТКРЫТИЕ]\n"
                f"Домен: {field} | {title}\n"
                + (f"Объяснение: {expl}" if expl else "")
            )
        elif status == "known_law":
            heritage = getattr(discovery_result, "heritage_label", "")
            parts.append(f"[ИЗВЕСТНЫЙ ЗАКОН: {heritage}]")
        elif status == "similar_to_law":
            parts.append(f"[ПОХОЖА НА ИЗВЕСТНЫЙ ЗАКОН в домене: {field}]")

    return "\n".join(parts) if parts else ""


def get_hadi_context() -> str:
    """
    Контекст для HADI (рефлексия смерти): паттерны ошибок.
    """
    all_meta = _load_all_meta(last_n=3)
    if not all_meta:
        return ""

    # Агрегируем паттерны ошибок из нескольких рефлексий
    error_parts = []
    for meta in all_meta:
        errors = meta.get("analysis", {}).get("errors", "")
        if errors and errors not in error_parts:
            error_parts.append(errors[:150])

    if not error_parts:
        return ""

    combined = " | ".join(error_parts[:2])
    return f"[ИСТОРИЧЕСКИЕ ПАТТЕРНЫ ОШИБОК]\n{combined}"


# ══════════════════════════════════════════════════════════════════
# ИНЖЕКЦИЯ В DSPy ПОДПИСИ
# ══════════════════════════════════════════════════════════════════

def enrich_data_meta(
    data_meta:   str,
    domain_type: str = "",
    role:        str = "",
) -> str:
    """
    Добавляет мета-контекст в data_meta строку.
    Используется в navigate() и audit_role() перед вызовом LLM.
    """
    ctx = get_navigator_context(domain_type) if not role else \
          get_matryoshka_context("", domain_type, role)

    if not ctx:
        return data_meta

    return data_meta + f"\n\n{ctx}"


def enrich_failure_logs(
    failure_logs: str,
    domain_type:  str = "",
) -> str:
    """
    Добавляет мета-правила навигации в failure_logs JSON.
    Это позволяет Navigator учитывать исторический опыт.
    """
    nav_ctx = get_navigator_context(domain_type)
    if not nav_ctx:
        return failure_logs

    try:
        logs_list = json.loads(failure_logs) if failure_logs.strip() else []
    except Exception:
        logs_list = []

    # Добавляем мета-правило как отдельную запись
    logs_list.append({
        "hypothesis":   "[META-REFLECTION]",
        "death_reason": nav_ctx[:300],
        "source":       "meta_reflection",
    })
    return json.dumps(logs_list, ensure_ascii=False)


# ══════════════════════════════════════════════════════════════════
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ══════════════════════════════════════════════════════════════════

def _get_domain_discoveries(domain_type: str) -> str:
    """Загружает открытия релевантного домена."""
    try:
        from .discovery import load_discoveries_summary
        summary = load_discoveries_summary()
        if summary and domain_type:
            # Фильтруем по домену
            relevant_lines = [
                l for l in summary.splitlines()
                if domain_type.lower() in l.lower() or "Physics" in l or l.startswith("ИЗВЕСТНЫЕ")
            ]
            return "\n".join(relevant_lines[:4]) if relevant_lines else ""
        return summary[:300] if summary else ""
    except Exception:
        return ""


def get_full_meta_summary() -> str:
    """
    Полный мета-контекст для инжекции в FINAL_REPORT.
    """
    meta = _load_latest_meta()
    if not meta:
        return ""

    analysis  = meta.get("analysis", {})
    ts        = meta.get("ts", "")[:10]
    counts    = meta.get("data_counts", {})

    lines = [
        f"[МЕТА-РЕФЛЕКСИЯ от {ts}]",
        f"База: хроника={counts.get('chronicle_steps',0)}, "
        f"gold={counts.get('gold_formulas',0)}, "
        f"curriculum={counts.get('curriculum',0)}",
    ]
    for key, label in [("successes","Успехи"), ("errors","Ошибки"), ("nav_rules","Правила")]:
        val = analysis.get(key, "")
        if val:
            lines.append(f"{label}: {val[:100]}")

    return "\n".join(lines)
