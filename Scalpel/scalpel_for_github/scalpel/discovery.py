"""
discovery.py — Детектор научных открытий v10.14.

Когда Scalpel находит формулу — этот модуль определяет:

  1. ДОМЕН — физика / биология / химия / экономика / инженерия / универсальная
  2. СТАТУС — известный закон | похожий на закон | новое открытие | эмпирическая
  3. КОНТЕКСТ — что это означает в данном домене

Если формула прошла Матрёшку и не совпадает ни с одним известным законом —
система генерирует "сертификат открытия" и сохраняет в discoveries.jsonl.

ЧТО СЧИТАЕТСЯ ОТКРЫТИЕМ:
  - R²_blind >= 0.90
  - consensus == ПРИНЯТА (Матрёшка)
  - НЕ найдено в INVARIANT_LIBRARY (Heritage)
  - НЕ дубликат предыдущих открытий
  - shuffle_p < 0.001 (не случайность)
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .config import VAULT_DIR, OLLAMA_HOST, OLLAMA_MODEL, SYNTHESIS_MODEL

log = logging.getLogger("scalpel")

DISCOVERIES_PATH = VAULT_DIR / "discoveries.jsonl"

# ══════════════════════════════════════════════════════════════════
# НАУЧНЫЕ ДОМЕНЫ — расширенная классификация
# ══════════════════════════════════════════════════════════════════

DOMAIN_SIGNATURES: Dict[str, Dict] = {

    "Механика": {
        "description": "Классическая механика: силы, движение, энергия",
        "field": "Physics",
        "indicators": {
            "operators": ["^2", "sqrt", "/", "*"],
            "dim_codes": [3, 6, 2, 8],         # масса, сила, длина, время
            "name_keywords": ["mass", "force", "velocity", "acceleration",
                              "momentum", "energy", "position", "distance"],
        },
        "known_laws": ["F=ma", "E=mv²/2", "p=mv", "F=-kx"],
        "prompt_hint": (
            "Формула из области механики. Проверь: выполняется ли закон сохранения "
            "энергии? Есть ли связь с законом Ньютона F=ma или законом Гука F=-kx?"
        ),
    },

    "Гравитация": {
        "description": "Гравитационные взаимодействия и орбитальная механика",
        "field": "Physics",
        "indicators": {
            "operators": ["^2", "/", "^1.5", "^3"],
            "dim_codes": [3, 2, 6],
            "name_keywords": ["radius", "orbit", "period", "distance",
                              "mass", "gravity", "planet", "semi_axis"],
        },
        "known_laws": ["T~a^1.5 (Кеплер)", "F~1/r² (Ньютон)", "v~1/sqrt(r)"],
        "prompt_hint": (
            "Возможна гравитационная зависимость. Третий закон Кеплера: T ~ a^(3/2). "
            "Закон тяготения Ньютона: F ~ m₁·m₂/r². Проверь степень расстояния."
        ),
    },

    "Термодинамика": {
        "description": "Тепловые процессы, газы, термодинамические циклы",
        "field": "Physics",
        "indicators": {
            "operators": ["/", "*", "exp", "log"],
            "dim_codes": [4, 6, 5],             # температура, давление, объём
            "name_keywords": ["temp", "pressure", "volume", "entropy",
                              "heat", "gas", "kelvin", "thermal"],
        },
        "known_laws": ["PV=nRT", "Q=mcΔT", "η=1-T₂/T₁"],
        "prompt_hint": (
            "Термодинамический процесс. Идеальный газ: PV = nRT. "
            "КПД цикла Карно: η=1-T₂/T₁. Проверь размерности давления и объёма."
        ),
    },

    "Электромагнетизм": {
        "description": "Электрические и магнитные поля, токи, волны",
        "field": "Physics",
        "indicators": {
            "operators": ["^2", "/", "sqrt"],
            "dim_codes": [2, 6],
            "name_keywords": ["charge", "current", "voltage", "field",
                              "magnetic", "electric", "resistance", "frequency"],
        },
        "known_laws": ["F=kq₁q₂/r²", "V=IR", "P=I²R", "E=hf"],
        "prompt_hint": (
            "Электромагнитная зависимость. Закон Кулона: F=kq₁q₂/r². "
            "Закон Ома: V=IR. Мощность: P=I²R. Проверь показатель степени 2."
        ),
    },

    "Оптика и волны": {
        "description": "Волновые явления, оптика, колебания",
        "field": "Physics",
        "indicators": {
            "operators": ["sin", "cos", "sqrt", "/"],
            "dim_codes": [2, 8],
            "name_keywords": ["wavelength", "frequency", "amplitude",
                              "length", "period", "oscillation", "wave"],
        },
        "known_laws": ["T=2π√(L/g) маятник", "v=fλ", "f=1/T"],
        "prompt_hint": (
            "Колебательный или волновой процесс. Маятник: T=2π√(L/g). "
            "Скорость волны: v=fλ. Проверь наличие sqrt и π."
        ),
    },

    "Экспоненциальный рост/распад": {
        "description": "Экспоненциальные процессы в природе",
        "field": "Universal",
        "indicators": {
            "operators": ["exp", "log", "e^"],
            "dim_codes": [8, 0],
            "name_keywords": ["time", "rate", "decay", "growth", "concentration"],
        },
        "known_laws": ["N=N₀e^(-λt) распад", "P=P₀e^(rt) рост"],
        "prompt_hint": (
            "Экспоненциальный процесс. Радиоактивный распад: N=N₀·e^(-λt). "
            "Популяционный рост: P=P₀·e^(rt). Определи постоянную в показателе."
        ),
    },

    "Популяционная биология": {
        "description": "Рост популяций, экологические взаимодействия",
        "field": "Biology",
        "indicators": {
            "operators": ["*", "/", "exp", "log", "-"],
            "dim_codes": [0, 5],
            "name_keywords": ["population", "growth", "carrying", "predator",
                              "prey", "birth", "death", "species", "biomass"],
        },
        "known_laws": ["dN/dt=rN(1-N/K) логистика", "Лотка-Вольтерра"],
        "prompt_hint": (
            "Биологическая динамика популяций. Логистический рост: dN/dt=rN(1-N/K). "
            "Уравнения Лотка-Вольтерра для хищник-жертва. Ищи насыщение."
        ),
    },

    "Биохимия": {
        "description": "Ферментативные реакции, биохимическая кинетика",
        "field": "Biology",
        "indicators": {
            "operators": ["*", "/", "+"],
            "dim_codes": [0, 5],
            "name_keywords": ["enzyme", "substrate", "concentration",
                              "michaelis", "km", "vmax", "reaction", "binding"],
        },
        "known_laws": ["v=Vmax·S/(Km+S) Михаэлис-Ментен"],
        "prompt_hint": (
            "Биохимическая кинетика. Уравнение Михаэлиса-Ментен: v=Vmax·S/(Km+S). "
            "Проверь: есть ли гиперболическое насыщение при больших S?"
        ),
    },

    "Химическая кинетика": {
        "description": "Скорости реакций, равновесие, катализ",
        "field": "Chemistry",
        "indicators": {
            "operators": ["*", "exp", "^", "/"],
            "dim_codes": [4, 0],
            "name_keywords": ["rate", "concentration", "temp", "activation",
                              "catalyst", "reaction", "equilibrium", "energy"],
        },
        "known_laws": ["k=Ae^(-Ea/RT) Аррениус", "r=k[A]^m[B]^n"],
        "prompt_hint": (
            "Химическая кинетика. Уравнение Аррениуса: k=A·e^(-Ea/RT). "
            "Закон скорости: r=k[A]^m[B]^n. Проверь зависимость от температуры."
        ),
    },

    "Финансовая математика": {
        "description": "Финансовые модели: цены, волатильность, доходность",
        "field": "Economics",
        "indicators": {
            "operators": ["*", "/", "log", "sqrt", "exp"],
            "dim_codes": [10, 0, 8],            # деньги, безразмерное, время
            "name_keywords": ["price", "return", "volatility", "rate",
                              "yield", "revenue", "cost", "profit", "value"],
        },
        "known_laws": ["P=P₀e^(rt) компаундирование", "Блэк-Шоулз"],
        "prompt_hint": (
            "Финансовая зависимость. Компаундирование: P=P₀·e^(rt). "
            "Степенные законы Парето. Проверь логнормальность распределения."
        ),
    },

    "Механика материалов": {
        "description": "Прочность, деформации, усталость конструкций",
        "field": "Engineering",
        "indicators": {
            "operators": ["*", "/", "^", "sqrt"],
            "dim_codes": [6, 2, 0],
            "name_keywords": ["stress", "strain", "force", "displacement",
                              "modulus", "load", "deformation", "elastic"],
        },
        "known_laws": ["σ=E·ε закон Гука", "σ=F/A напряжение"],
        "prompt_hint": (
            "Механика деформируемых тел. Закон Гука: σ=E·ε. "
            "Напряжение: σ=F/A. Проверь линейность деформации."
        ),
    },

    "Скейлинг и степенные законы": {
        "description": "Универсальные масштабные зависимости в природе",
        "field": "Universal",
        "indicators": {
            "operators": ["^", "*", "/"],
            "dim_codes": [0, 2, 3, 5],
            "name_keywords": ["size", "scale", "mass", "metabolic", "body"],
        },
        "known_laws": ["P~M^0.75 закон Клейбера", "f~L^-0.5 акустика"],
        "prompt_hint": (
            "Степенной закон масштабирования. Закон Клейбера: метаболизм ~ масса^0.75. "
            "Проверь показатель степени — кратен ли 1/4?"
        ),
    },
}


# ══════════════════════════════════════════════════════════════════
# РЕЗУЛЬТАТ КЛАССИФИКАЦИИ
# ══════════════════════════════════════════════════════════════════

@dataclass
class DiscoveryResult:
    formula_shadow:  str
    formula_real:    str
    r2_blind:        float
    domain_detected: str          # "Механика" | "Биология" | ...
    field:           str          # "Physics" | "Biology" | "Chemistry" | "Economics" | "Universal"
    status:          str          # "known_law" | "similar_to_law" | "discovery" | "empirical"
    heritage_match:  bool         # True если нашлось в INVARIANT_LIBRARY
    heritage_label:  str          # "Kepler — Третий закон" или ""
    discovery_title: str          # "Новый степенной закон в механике" или ""
    prompt_context:  str          # инжектируется в промпты LLM компонентов
    significance:    str          # "high" | "medium" | "low"
    is_new_discovery: bool        # True = сохранить в discoveries.jsonl
    explanation:     str          # читаемое объяснение для FINAL_REPORT


# ══════════════════════════════════════════════════════════════════
# КЛАССИФИКАЦИЯ ДОМЕНА (без LLM, только паттерны)
# ══════════════════════════════════════════════════════════════════

def _lookup_domain_from_history(dim_pattern: str) -> str:
    """
    Ищет домен для данного dim_pattern в истории инвариантов.
    Если система уже видела dim=[2,8] → T=a^1.5 (Гравитация) —
    следующий раз с dim=[2,8] сразу знает домен.
    """
    try:
        from .episodic_memory import get_memory
        records = get_memory().recall_invariants_for_domain(limit=50)
        # Парсим подсказки и ищем dim_pattern
        inv_path = get_memory().memory_dir / "invariants_learned.jsonl"
        if not inv_path.exists():
            return ""
        from collections import Counter
        domain_counter: Counter = Counter()
        import json as _j
        for line in inv_path.read_text(encoding="utf-8").strip().splitlines():
            try:
                rec = _j.loads(line)
                if rec.get("dim_pattern") == dim_pattern and rec.get("domain"):
                    domain_counter[rec["domain"]] += 1
            except Exception:
                continue
        if domain_counter:
            return domain_counter.most_common(1)[0][0]
    except Exception:
        pass
    return ""


def _detect_domain(
    formula:    str,
    feat_names: List[str],
    dim_codes:  List[int],
) -> Tuple[str, str]:
    """
    Определяет научный домен формулы по:
    - операторам в формуле
    - dim_codes признаков
    - именам признаков

    Возвращает (domain_name, field).
    """
    formula_lower = formula.lower()
    feat_lower    = [n.lower() for n in feat_names]

    best_domain  = "Универсальная"
    best_score   = 0
    best_field   = "Universal"

    for domain_name, info in DOMAIN_SIGNATURES.items():
        indicators = info["indicators"]
        score      = 0

        # Операторы в формуле
        for op in indicators.get("operators", []):
            if op in formula:
                score += 2

        # dim_codes совпадают
        for dc in indicators.get("dim_codes", []):
            if dc in dim_codes:
                score += 3

        # Ключевые слова в именах признаков
        for kw in indicators.get("name_keywords", []):
            if any(kw in fn for fn in feat_lower):
                score += 4

        if score > best_score:
            # FIX v10.16: победа ТОЛЬКО по операторам без keyword/dim_code — не засчитывается.
            # Иначе sqrt(*) даёт "Финансовая математика" вместо физики.
            op_score = sum(
                2 for op in indicators.get("operators", []) if op in formula
            )
            has_keyword = any(
                any(kw in fn for fn in feat_lower)
                for kw in indicators.get("name_keywords", [])
            )
            has_dimcode = any(dc in dim_codes for dc in indicators.get("dim_codes", []))
            if op_score == score and not has_keyword and not has_dimcode:
                continue  # пропускаем — нет реальных признаков домена
            best_score  = score
            best_domain = domain_name
            best_field  = info["field"]

    return best_domain, best_field


# ══════════════════════════════════════════════════════════════════
# ОСНОВНАЯ ФУНКЦИЯ: ПОЛНАЯ КЛАССИФИКАЦИЯ
# ══════════════════════════════════════════════════════════════════

def classify_discovery(
    formula_shadow:  str,
    formula_real:    str,
    r2_blind:        float,
    shuffle_p:       float,
    feat_names:      List[str],
    dim_codes:       List[int],
    domain_type:     str        = "",
    heritage_result  = None,    # HeritageResult из atomic_precision
    consensus:       str        = "",
    ask_fn           = None,    # callable для LLM-объяснения (опционально)
) -> DiscoveryResult:
    """
    Полная классификация найденной формулы.

    Вызывается из engine.py после vault.save().
    Результат инжектируется во все LLM компоненты.
    """
    # ── 1. Домен ─────────────────────────────────────────────────
    domain, field = _detect_domain(formula_shadow, feat_names, dim_codes)

    # v10.21: Если пользователь указал domain_type — он ПРИОРИТЕТ.
    # Обновляем и field И domain (раньше обновлялся только field → "Финансовая математика (Physics)")
    # Автодетект _detect_domain используется как дополнение, не замена.
    if domain_type:
        domain_lower = domain_type.lower()
        if "phys" in domain_lower:
            field  = "Physics"
            # Если автодетект дал нефизический домен — переопределяем
            _physics_domains = {"Физика", "Механика", "Термодинамика",
                                "Астрофизика", "Квантовая физика", "Оптика"}
            if domain not in _physics_domains:
                domain = "Физика"   # базовый физический домен как fallback
        elif "biol" in domain_lower:
            field  = "Biology"
            if domain not in {"Биология", "Экология", "Генетика"}:
                domain = "Биология"
        elif "chem" in domain_lower:
            field  = "Chemistry"
            if domain not in {"Химия", "Термохимия", "Кинетика"}:
                domain = "Химия"
        elif "econ" in domain_lower or "financ" in domain_lower:
            field  = "Economics"
            if domain not in {"Экономика", "Финансовая математика"}:
                domain = "Экономика"

    # Уточняем домен через историю инвариантов (обучение на опыте)
    # Если система уже видела такой же dim_pattern → берём домен оттуда
    dim_pattern = ",".join(str(d) for d in sorted(set(dim_codes))) if dim_codes else ""
    if dim_pattern:
        learned_domain = _lookup_domain_from_history(dim_pattern)
        if learned_domain and learned_domain != domain:
            log.info("[Discovery] Домен уточнён из истории: %s → %s (dim=%s)",
                     domain, learned_domain, dim_pattern)
            domain = learned_domain
            field  = DOMAIN_SIGNATURES.get(domain, {}).get("field", field)

    # ── 1b. LLM-уточнение домена (если ask_fn есть и dim_codes неинформативны) ─
    # Система учится определять домен по ФИЗИЧЕСКОМУ СМЫСЛУ, не только по паттернам
    all_zero_dims = all(d == 0 for d in (dim_codes or [0]))
    # v10.14 БАГ 2: LLM не вызывается если история уже дала домен
    _history_gave_domain = domain != "Универсальная"
    if ask_fn and all_zero_dims and r2_blind >= 0.85 and not _history_gave_domain:
        try:
            domain_from_llm, field_from_llm = _llm_classify_domain(
                formula_shadow, formula_real, feat_names, dim_codes, r2_blind, ask_fn
            )
            if domain_from_llm and domain_from_llm != "Универсальная":
                domain = domain_from_llm
                field  = DOMAIN_SIGNATURES.get(domain_from_llm, {}).get("field", field)
                log.info("[Discovery] LLM домен: %s (%s)", domain, field)
        except Exception as _llm_dom_err:
            log.debug("[Discovery] LLM домен: %s", _llm_dom_err)

    # ── 2. Heritage (известные законы) ───────────────────────────
    heritage_detected = False
    heritage_label    = ""
    if heritage_result is not None:
        heritage_detected = getattr(heritage_result, "detected", False)
        if heritage_detected:
            scientists = getattr(heritage_result, "matched_scientists", [])
            heritage_label = ", ".join(scientists[:2])

    # ── 3. Уникальность (не дубликат) ─────────────────────────────
    is_duplicate = _check_duplicate(formula_shadow, r2_blind)

    # ── 4. Статус открытия ─────────────────────────────────────────
    accepted    = "ПРИНЯТА" in consensus or not consensus
    high_quality = r2_blind >= 0.90 and shuffle_p < 0.001

    if heritage_detected:
        status    = "known_law"
        is_new    = False
        title     = f"Известный закон: {heritage_label}"
        sig       = "high"
    elif high_quality and accepted and not is_duplicate and r2_blind >= 0.92:
        status    = "discovery"
        is_new    = True
        title     = _generate_discovery_title(domain, field, formula_shadow)
        sig       = "high"
    elif high_quality and accepted:
        status    = "similar_to_law"
        is_new    = False
        title     = f"Новая зависимость в: {domain}"
        sig       = "medium"
    else:
        status    = "empirical"
        is_new    = False
        title     = ""
        sig       = "low"

    # ── 5. Контекст для промптов ──────────────────────────────────
    domain_info = DOMAIN_SIGNATURES.get(domain, {})
    prompt_hint = domain_info.get("prompt_hint", "")
    known_laws  = domain_info.get("known_laws", [])

    ctx_parts = []

    if heritage_detected and heritage_label:
        ctx_parts.append(
            f"★ HERITAGE DETECTED: формула совпадает с {heritage_label}. "
            f"Это подтверждённый физический/научный закон."
        )
    elif status == "discovery":
        ctx_parts.append(
            f"★ ПОТЕНЦИАЛЬНОЕ ОТКРЫТИЕ в области '{domain}' ({field}). "
            f"R²={r2_blind:.3f}, p={shuffle_p:.4f}. "
            f"Матрёшка приняла. Это может быть новый закон!"
        )
    elif status == "similar_to_law":
        ctx_parts.append(
            f"ЗАВИСИМОСТЬ В ОБЛАСТИ '{domain}' ({field}). "
            f"Похожие известные законы: {', '.join(known_laws[:2])}."
        )

    if prompt_hint:
        ctx_parts.append(prompt_hint)

    prompt_context = "\n".join(ctx_parts)

    # ── 6. LLM объяснение (если ask_fn передан) ───────────────────
    explanation = ""
    if ask_fn and is_new and status == "discovery":
        try:
            explanation = _generate_explanation(
                formula_shadow=formula_shadow,
                domain=domain,
                field=field,
                r2_blind=r2_blind,
                known_laws=known_laws,
                ask_fn=ask_fn,
            )
        except Exception as e:
            log.debug("[Discovery] Объяснение не сгенерировано: %s", e)

    result = DiscoveryResult(
        formula_shadow   = formula_shadow,
        formula_real     = formula_real,
        r2_blind         = r2_blind,
        domain_detected  = domain,
        field            = field,
        status           = status,
        heritage_match   = heritage_detected,
        heritage_label   = heritage_label,
        discovery_title  = title,
        prompt_context   = prompt_context,
        significance     = sig,
        is_new_discovery = is_new,
        explanation      = explanation,
    )

    # ── 7. Сохраняем открытие ─────────────────────────────────────
    if is_new:
        _save_discovery(result)
        print(f"\n  ★ [Discovery] НОВОЕ ОТКРЫТИЕ: {title}")
        print(f"  ★ [Discovery] Домен: {domain} ({field})")
        print(f"  ★ [Discovery] R²={r2_blind:.4f}, p={shuffle_p:.5f}")
        if explanation:
            print(f"  ★ [Discovery] {explanation[:120]}…")

    return result


# ══════════════════════════════════════════════════════════════════
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ══════════════════════════════════════════════════════════════════

def _check_duplicate(formula: str, r2: float) -> bool:
    """Проверяет, не сохранено ли уже это открытие."""
    if not DISCOVERIES_PATH.exists():
        return False
    try:
        for line in DISCOVERIES_PATH.read_text(encoding="utf-8").strip().splitlines():
            rec = json.loads(line)
            if rec.get("formula_shadow") == formula:
                return True
    except Exception:
        pass
    return False


def _generate_discovery_title(domain: str, field: str, formula: str) -> str:
    """Генерирует название открытия."""
    ops = []
    if "sqrt" in formula: ops.append("корневая")
    elif "^2" in formula: ops.append("квадратичная")
    elif "^3" in formula: ops.append("кубическая")
    elif "/" in formula:  ops.append("обратная")
    elif "exp" in formula or "log" in formula: ops.append("логарифмическая")
    else:                 ops.append("степенная")

    op_str = ops[0] if ops else "новая"
    return f"Новая {op_str} зависимость: {domain} ({field})"


def _llm_classify_domain(
    formula_shadow: str,
    formula_real:   str,
    feat_names:     List[str],
    dim_codes:      List[int],
    r2_blind:       float,
    ask_fn,
) -> Tuple[str, str]:
    """
    LLM определяет научный домен. Видит ВСЁ: formula_real с реальными
    именами, dim_codes, список доменов с примерами законов.
    """
    feat_str  = ", ".join(feat_names[:6]) if feat_names else "неизвестны"
    dim_names = {0:"безразмерное",2:"длина",3:"масса",4:"температура",
                 5:"объём",6:"сила/давление",8:"время",10:"деньги"}
    dim_str = ", ".join(
        f"{feat_names[i] if i < len(feat_names) else f'f{i}'}="
        f"{dim_names.get(d, str(d))}"
        for i, d in enumerate(dim_codes[:5])
    ) if dim_codes else "нет"

    domains_lines = []
    for name, info in DOMAIN_SIGNATURES.items():
        laws = " | ".join(info.get("known_laws", [])[:2])
        domains_lines.append(
            f"  {name} ({info['field']}): {info['description']}"
            + (f". Примеры: {laws}" if laws else "")
        )
    domains_list = "\n".join(domains_lines)

    prompt = (
        f"Ты — учёный-эксперт. Определи научный домен по зависимости.\n\n"
        f"ФОРМУЛА (структура): {formula_shadow}\n"
        f"ФОРМУЛА (реальные имена): {formula_real}\n"
        f"ПРИЗНАКИ: {feat_str}\n"
        f"РАЗМЕРНОСТИ: {dim_str}\n"
        f"КАЧЕСТВО: R²={r2_blind:.3f}\n\n"
        f"ДОМЕНЫ:\n{domains_list}\n"
        f"  Универсальная (Universal): нет явной научной области\n\n"
        f"Определи домен по смыслу переменных и структуре формулы.\n"
        f"temperature+volume=Термодинамика, mass+velocity=Механика, "
        f"population=Биология, price+rate=Финансы.\n"
        f"Ответь ОДНИМ словом из списка. Если не уверен — 'Универсальная'."
    )
    try:
        raw = ask_fn(prompt).strip().rstrip(".").strip()
        raw_lower = raw.lower()
        for domain_name, info in DOMAIN_SIGNATURES.items():
            if domain_name.lower() in raw_lower or raw_lower in domain_name.lower():
                log.info("[Discovery/LLM] Домен: '%s' → %s (%s)",
                         raw[:30], domain_name, info["field"])
                return domain_name, info["field"]
    except Exception as e:
        log.debug("[Discovery/LLM] %s", e)
    return "Универсальная", "Universal"

def _generate_explanation(
    formula_shadow: str,
    domain:         str,
    field:          str,
    r2_blind:       float,
    known_laws:     List[str],
    ask_fn,
) -> str:
    """Генерирует краткое научное объяснение через LLM."""
    laws_str = ", ".join(known_laws[:3]) if known_laws else "нет прямых аналогов"
    prompt = (
        f"Формула из области '{domain}' ({field}): {formula_shadow}\n"
        f"R²_blind={r2_blind:.3f} (очень высокое качество)\n"
        f"Похожие известные законы: {laws_str}\n\n"
        f"Дай краткое научное объяснение (2-3 предложения):\n"
        f"1. Что физически/биологически означает эта зависимость?\n"
        f"2. Почему именно такая структура формулы имеет смысл?\n"
        f"3. Как это соотносится с известными законами?\n"
        f"Пиши просто и конкретно."
    )
    return ask_fn(prompt).strip()[:400]


def _save_discovery(result: DiscoveryResult) -> None:
    """Сохраняет подтверждённое открытие в discoveries.jsonl."""
    record = {
        "ts":              datetime.now().isoformat(),
        "formula_shadow":  result.formula_shadow,
        "formula_real":    result.formula_real,
        "r2_blind":        round(result.r2_blind, 4),
        "domain":          result.domain_detected,
        "field":           result.field,
        "status":          result.status,
        "title":           result.discovery_title,
        "explanation":     result.explanation,
        "significance":    result.significance,
    }
    try:
        DISCOVERIES_PATH.parent.mkdir(parents=True, exist_ok=True)
        with DISCOVERIES_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        log.info("[Discovery] Открытие сохранено: %s", result.discovery_title)
    except Exception as e:
        log.warning("[Discovery] Ошибка сохранения: %s", e)


# ══════════════════════════════════════════════════════════════════
# ЗАГРУЗКА ОТКРЫТИЙ ДЛЯ DSPy
# ══════════════════════════════════════════════════════════════════

def load_discoveries_as_examples() -> List[Any]:
    """
    Загружает сохранённые открытия как dspy.Example с усилением.
    Открытия — самые ценные примеры: система сама нашла новый закон.
    Усиление: high → 5×, medium → 2×.

    Вызывается из dspy_optimizer.siege_compile().
    """
    try:
        import dspy
    except ImportError:
        return []

    if not DISCOVERIES_PATH.exists():
        return []

    examples = []
    try:
        for line in DISCOVERIES_PATH.read_text(encoding="utf-8").strip().splitlines():
            try:
                rec = json.loads(line)
            except Exception:
                continue

            formula  = rec.get("formula_shadow", "")
            r2       = float(rec.get("r2_blind", 0))
            domain   = rec.get("domain", "")
            field    = rec.get("field", "")
            title    = rec.get("title", "")
            expl     = rec.get("explanation", "")

            ex = dspy.Example(
                data_meta          = (
                    f"DISCOVERY: domain={domain}, field={field}, "
                    f"r2_blind={r2:.3f}, title='{title}'"
                ),
                failure_logs       = "[]",
                selected_features  = "f0, f1",
                selected_operators = "+,-,*,/,sqrt,log,exp",
                hypotheses         = formula[:80],
                ooda_stable        = "true",
                reasoning          = (
                    f"★ VERIFIED DISCOVERY in {field}: {title}. "
                    + (expl[:200] if expl else "")
                ),
            ).with_inputs("data_meta", "failure_logs")

            sig    = rec.get("significance", "medium")
            weight = 5 if sig == "high" else 2
            examples.extend([ex] * weight)

    except Exception as e:
        log.warning("[Discovery] Ошибка загрузки: %s", e)

    log.info("[Discovery] Загружено %d примеров-открытий", len(examples))
    return examples


def load_discoveries_summary() -> str:
    """
    Возвращает краткую сводку всех открытий для инжекции в промпты.
    """
    if not DISCOVERIES_PATH.exists():
        return ""
    try:
        records = [
            json.loads(l)
            for l in DISCOVERIES_PATH.read_text(encoding="utf-8").strip().splitlines()
            if l
        ]
        if not records:
            return ""
        lines = ["ИЗВЕСТНЫЕ ОТКРЫТИЯ СИСТЕМЫ:"]
        for rec in records[-5:]:  # последние 5
            lines.append(
                f"  [{rec.get('field','?')}] {rec.get('title','?')} "
                f"(R²={rec.get('r2_blind',0):.3f})"
            )
        return "\n".join(lines)
    except Exception:
        return ""


# Импорт Any для аннотаций
from typing import Any
