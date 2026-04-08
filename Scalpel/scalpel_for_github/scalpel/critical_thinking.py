"""
critical_thinking.py — Гуманитарные фильтры v10.2.7.

Четыре блока критического мышления:

  1. Deep Root   (5 Почему)        — Root Cause Analysis, 5 уровней вглубь.
  2. Dialectic   (Сократ)          — Перекрёстный допрос Скептик ↔ Физик,
                                     3 вопроса каждый, встроен в Delphi.
  3. Sinquain    (Синквейн)        — Семантическое сжатие инварианта в 5 строк.
  4. Lasso       (Бритва Оккама)   — Стягивание аргументов к единому центру.

RAM Guard: все блоки — текстовые, не загружают дополнительных моделей.
Они работают поверх уже запущенного ollama_chat (или DSPy).
"""
from __future__ import annotations

import logging
from typing import List, Optional, Tuple

log = logging.getLogger("scalpel")


# ══════════════════════════════════════════════════════════════════
# БЛОК 1: DEEP ROOT — «5 Почему» (Root Cause Analysis)
# ══════════════════════════════════════════════════════════════════

def deep_root_analysis(
    invariant:   str,
    dependency:  str,
    r2_train:    float,
    domain_type: str,
    ask_fn,                         # callable(prompt: str) -> str
    levels:      int = 5,
) -> List[str]:
    """
    Пропускает найденную зависимость через 5 уровней «Почему?».

    Каждый уровень берёт ответ предыдущего как новое «явление»
    и снова спрашивает «Почему это так?» — пока не достигнет
    фундаментального принципа или не исчерпает levels.

    Возвращает список строк длиной <= levels: каждая — один уровень RCA.
    """
    chain: List[str] = []
    phenomenon = (
        f"Инвариант: {invariant}. "
        f"Зависимость: {dependency}. "
        f"R²={r2_train:.4f}. Домен: {domain_type or 'неизвестен'}."
    )

    print(f"\n  {'─'*58}")
    print(f"  DEEP ROOT ANALYSIS (5 Почему) — {levels} уровней")
    print(f"  {'─'*58}")

    for level in range(1, levels + 1):
        prompt = (
            f"Отвечай СТРОГО на русском языке. Никакого другого языка.\n"
            f"[RCA Уровень {level}/{levels}]\n"
            f"Явление: «{phenomenon}»\n\n"
            f"Почему это так? Дай одно конкретное причинное объяснение "
            f"(1-2 предложения). Не повторяй само явление. "
            f"Иди вглубь: от симптома к механизму, от механизма к закону."
        )
        answer = ask_fn(prompt)
        answer = answer.strip() if answer else ""
        # Если ответ пустой или слишком короткий — не ломаем цепочку
        # v10.37: при пустом ответе сохраняем предыдущий phenomenon
        if len(answer) < 10:
            answer = f"[Модель не ответила на уровне {level}]"
            # НЕ обновляем phenomenon — следующий уровень идёт от предыдущего
        else:
            phenomenon = answer   # следующий уровень идёт от этого ответа
        chain.append(f"Уровень {level}: {answer}")
        print(f"  Why {level}: {answer[:120]}")

    print(f"  {'─'*58}\n")
    return chain


def format_root_cause_section(chain: List[str], formula_real: str) -> str:
    """Форматирует RCA-секцию для финального отчёта."""
    lines = [
        "══════════════════════════════════════════════════════════════",
        "  ROOT CAUSE ANALYSIS — «5 Почему» (Deep Root)",
        f"  Формула: {formula_real}",
        "──────────────────────────────────────────────────────────────",
    ]
    for entry in chain:
        lines.append(f"  {entry}")
    lines += [
        "──────────────────────────────────────────────────────────────",
        f"  Фундаментальная причина: {chain[-1] if chain else '—'}",
        "══════════════════════════════════════════════════════════════",
    ]
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════
# БЛОК 2: DIALECTIC — Сократовская перекрёстная проверка
#         (Скептик ↔ Физик, 3 вопроса каждый)
# ══════════════════════════════════════════════════════════════════

def socratic_cross_examination(
    formula_shadow:   str,
    shadow_names:     List[str],
    r2_train:         float,
    complexity:       int,
    domain_type:      str,
    ask_fn,                           # callable(prompt: str) -> str
    n_questions:      int = 3,
    interround_fn     = None,         # v10.3.9: callable() → ollama_stop+gc между раундами
    heritage_context: str = "",       # v10.5: инъекция Heritage из AtomicPrecisionResult
) -> List[dict]:
    """
    Этап «Сократовского допроса» в рамках Delphi.

    v10.5: heritage_context из AtomicPrecisionResult.heritage_context.
    Если передан — Скептик и Физик знают, какое именно Наследие они обсуждают:
        «HERITAGE: формула содержит структуру Кеплера T~a^(3/2).
         Скептик: проверь масштабный эффект. Физик: какая сила за ним стоит?»

    Возвращает список dicts:
        {"speaker": str, "type": "question"|"answer", "text": str}
    """
    dialogue: List[dict] = []
    context = (
        f"Формула: {formula_shadow}\n"
        f"Признаки: {', '.join(shadow_names[:6])}\n"
        f"R²={r2_train:.4f}, сложность={complexity}\n"
        f"Домен: {domain_type or 'неизвестен'}\n"
    )
    if heritage_context:
        context += heritage_context + "\n"

    print(f"\n  {'─'*58}")
    print(f"  SOCRATIC CROSS-EXAMINATION (Dialectic Block)")
    print(f"  Скептик → Физик → встречный допрос")
    print(f"  {'─'*58}")

    # ── Раунд 1: Скептик задаёт вопросы ──────────────────────────
    skeptic_questions: List[str] = []
    for i in range(1, n_questions + 1):
        prev_q = "; ".join(skeptic_questions) if skeptic_questions else "нет"
        prev_q = (prev_q[:497] + "…") if len(prev_q) > 500 else prev_q  # v10.3.9-patch: RAM-Save 500 chars
        prompt = (
            f"Отвечай СТРОГО на русском языке. Никакого другого языка.\n"
            f"Ты — Скептик. Контекст:\n{context}\n"
            f"Уже спросил: «{prev_q}»\n\n"
            f"Вопрос {i}/{n_questions}: сформулируй острый вопрос "
            f"о границах применимости или возможных контрпримерах для этой формулы. "
            f"Только вопрос, без ответа. 1 предложение."
        )
        q = ask_fn(prompt).strip()
        skeptic_questions.append(q)
        dialogue.append({"speaker": "Скептик", "type": "question", "text": q})
        print(f"  Скептик Q{i}: {q[:100]}")

    # v10.3.9 RAM FIX: между раундами Delphi выгружаем модель из VRAM.
    # Без этого при 7.7 ГБ вторая модель накладывается на первую → OOM.
    if interround_fn is not None:
        log.info("[Delphi] interround_fn: ollama_stop + gc между раундами.")
        try:
            interround_fn()
        except Exception as _irf_err:
            log.warning("[Delphi] interround_fn ошибка: %s", _irf_err)

    # ── Раунд 2: Физик отвечает и задаёт встречные вопросы ───────
    physicist_questions: List[str] = []
    for i, sq in enumerate(skeptic_questions, 1):
        # Физик отвечает на вопрос Скептика
        sq_safe = (sq[:497] + "…") if len(sq) > 500 else sq  # v10.3.9-patch: RAM-Save 500 chars
        ans_prompt = (
            f"Отвечай СТРОГО на русском языке. Никакого другого языка.\n"
            f"Ты — Физик. Контекст:\n{context}\n"
            f"Скептик спросил: «{sq_safe}»\n\n"
            f"Ответь чётко (1-2 предложения) с позиции физической корректности."
        )
        ans = ask_fn(ans_prompt).strip()
        dialogue.append({"speaker": "Физик", "type": "answer", "text": ans})
        print(f"  Физик A{i}: {ans[:100]}")

    for i in range(1, n_questions + 1):
        prev_pq = "; ".join(physicist_questions) if physicist_questions else "нет"
        prev_pq = (prev_pq[:497] + "…") if len(prev_pq) > 500 else prev_pq  # v10.3.9-patch: RAM-Save 500 chars
        prompt = (
            f"Отвечай СТРОГО на русском языке. Никакого другого языка.\n"
            f"Ты — Физик. Контекст:\n{context}\n"
            f"Уже спросил: «{prev_pq}»\n\n"
            f"Встречный вопрос {i}/{n_questions}: задай вопрос Скептику "
            f"о том, какие именно данные он считает контрпримером "
            f"и почему его сомнение валидно. Только вопрос, 1 предложение."
        )
        pq = ask_fn(prompt).strip()
        physicist_questions.append(pq)
        dialogue.append({"speaker": "Физик", "type": "question", "text": pq})
        print(f"  Физик Q{i}: {pq[:100]}")

    # ── Раунд 3: Скептик отвечает на встречные вопросы ───────────
    for i, pq in enumerate(physicist_questions, 1):
        pq_safe = (pq[:497] + "…") if len(pq) > 500 else pq  # v10.3.9-patch: RAM-Save 500 chars
        ans_prompt = (
            f"Отвечай СТРОГО на русском языке. Никакого другого языка.\n"
            f"Ты — Скептик. Контекст:\n{context}\n"
            f"Физик спросил: «{pq_safe}»\n\n"
            f"Ответь честно: конкретизируй своё сомнение (1-2 предложения)."
        )
        ans = ask_fn(ans_prompt).strip()
        dialogue.append({"speaker": "Скептик", "type": "answer", "text": ans})
        print(f"  Скептик A{i}: {ans[:100]}")

    print(f"  {'─'*58}\n")
    return dialogue


def format_dialectic_section(dialogue: List[dict]) -> str:
    """Форматирует диалог для включения в отчёт Delphi / Матрёшки."""
    lines = [
        "══════════════════════════════════════════════════════════════",
        "  SOCRATIC CROSS-EXAMINATION — Диалектический блок",
        "  (Скептик ↔ Физик · 3 вопроса каждый · границы применимости)",
        "──────────────────────────────────────────────────────────────",
    ]
    for entry in dialogue:
        icon = "?" if entry["type"] == "question" else "→"
        lines.append(f"  [{entry['speaker']}] {icon} {entry['text']}")
    lines.append("══════════════════════════════════════════════════════════════")
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════
# БЛОК 3: SINQUAIN — Синквейн инварианта (Мистик / Антрополог)
# ══════════════════════════════════════════════════════════════════

def generate_sinquain(
    formula_real: str,
    consensus:    str,
    r2_train:     float,
    domain_type:  str,
    ask_fn,                         # callable(prompt: str) -> str
    role:         str = "Мистик",
) -> str:
    """
    Генерирует синквейн инварианта.

    Структура классического синквейна:
      Строка 1: 1 существительное (тема)
      Строка 2: 2 прилагательных (описание)
      Строка 3: 3 глагола (действие)
      Строка 4: Фраза из 4 слов (суть)
      Строка 5: 1 слово-резюме (синоним темы)

    Возвращает строку с 5 строками синквейна.
    """
    prompt = (
        f"Ты — {role}. Ты только что рассмотрел физический инвариант.\n"
        f"Формула: {formula_real}\n"
        f"Домен: {domain_type or 'неизвестен'}\n"
        f"R²={r2_train:.4f}, консенсус={consensus}\n\n"
        f"Напиши СИНКВЕЙН этого инварианта для быстрой расшифровки смысла.\n"
        f"Строгий формат (5 строк, без нумерации):\n"
        f"  Строка 1: 1 существительное (ключевое понятие)\n"
        f"  Строка 2: 2 прилагательных через пробел\n"
        f"  Строка 3: 3 глагола через пробел\n"
        f"  Строка 4: фраза-суть из 4 слов\n"
        f"  Строка 5: 1 слово-итог\n\n"
        f"СТРОГО 5 строк и ничего больше. Никаких заголовков, нумерации, пояснений.\n"
        f"Отвечай СТРОГО на русском языке."
    )
    raw = ask_fn(prompt).strip()
    # FIX v10.21: обрезаем до 5 строк если модель написала больше
    lines = [l.strip() for l in raw.splitlines() if l.strip()]
    if len(lines) > 5:
        lines = lines[:5]
    raw = "\n".join(lines)

    # v10.38: фильтр токсичного контента — llama3:8b галлюцинирует
    _toxic_words = [
        "наци", "нацист", "фашист", "расист", "террорист", "убий",
        "nazi", "fascist", "racist", "terrorist", "kill", "murder",
        "ненависть", "уничтожить", "ненавижу",
    ]
    _raw_lower = raw.lower()
    if any(w in _raw_lower for w in _toxic_words):
        log.warning("[SINQUAIN] Токсичный контент обнаружен — заменяю на заглушку")
        raw = "Инвариант\nСтабильный точный\nВычисляет описывает предсказывает\nФизический закон найден\nЗакономерность"

    log.info("[SINQUAIN/%s] %s", role, raw.replace("\n", " | "))
    return raw


def format_sinquain_section(sinquain: str, role: str, formula_real: str) -> str:
    """Форматирует секцию Синквейна для финального отчёта."""
    lines = [
        "══════════════════════════════════════════════════════════════",
        f"  СИНКВЕЙН ИНВАРИАНТА — {role} (Semantic Compression)",
        f"  Формула: {formula_real}",
        "──────────────────────────────────────────────────────────────",
    ]
    for i, sline in enumerate(sinquain.splitlines(), 1):
        sline = sline.strip()
        if sline:
            labels = ["", "Существительное", "Прилагательные", "Глаголы",
                      "Фраза-суть", "Итог"]
            label = labels[i] if i < len(labels) else f"Строка {i}"
            lines.append(f"  {i}. [{label:16s}] {sline}")
    lines.append("══════════════════════════════════════════════════════════════")
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════
# БЛОК 4: LASSO — Топологическое стягивание (Бритва Оккама)
# ══════════════════════════════════════════════════════════════════

LASSO_SYSTEM_INSTRUCTION = (
    "LASSO (Топологическое стягивание): "
    "Стяни все аргументы в одну точку-причину. "
    "Если аргумент не ведёт к центру — отсекай его (Бритва Оккама). "
    "Итог: одно предложение-ядро, из которого выводятся все остальные."
)


def lasso_pull(
    arguments:   List[str],
    formula_shadow: str,
    ask_fn,
) -> Tuple[str, List[str]]:
    """
    Применяет топологическое стягивание к списку аргументов.

    Возвращает:
      core_point  — центральная точка-причина (одно предложение)
      kept_args   — аргументы, признанные ведущими к центру
    """
    if not arguments:
        return "Нет аргументов для стягивания.", []

    args_block = "\n".join(f"  [{i+1}] {a}" for i, a in enumerate(arguments))
    prompt = (
        f"Отвечай СТРОГО на русском языке. Никакого другого языка.\n"
        f"{LASSO_SYSTEM_INSTRUCTION}\n\n"
        f"Формула: {formula_shadow}\n"
        f"Аргументы (всего {len(arguments)}):\n{args_block}\n\n"
        f"Задача:\n"
        f"  1. Определи центральную точку-причину (1 предложение).\n"
        f"  2. Перечисли номера аргументов, которые ВЕДУТ к центру (через запятую).\n"
        f"  3. Остальные — отсечены Бритвой Оккама.\n\n"
        f"Формат ответа (строго 2 строки):\n"
        f"ЯДРО: <одно предложение>\n"
        f"СОХРАНИТЬ: <номера через запятую, или 'все', или 'ни одного'>"
    )
    raw = ask_fn(prompt).strip()

    core_point = formula_shadow
    kept_indices: List[int] = list(range(len(arguments)))

    for line in raw.splitlines():
        line = line.strip()
        if line.upper().startswith("ЯДРО:"):
            core_point = line[5:].strip()
        elif line.upper().startswith("СОХРАНИТЬ:"):
            val = line[10:].strip().lower()
            if val == "все":
                kept_indices = list(range(len(arguments)))
            elif val in ("ни одного", "никаких", "none"):
                kept_indices = []
            else:
                kept_indices = []
                for part in val.replace(";", ",").split(","):
                    part = part.strip()
                    if part.isdigit():
                        idx = int(part) - 1
                        if 0 <= idx < len(arguments):
                            kept_indices.append(idx)

    kept_args = [arguments[i] for i in sorted(set(kept_indices))]
    cut_count = len(arguments) - len(kept_args)

    log.info(
        "[LASSO] Ядро: «%s…» Оставлено %d/%d, отсечено %d",
        core_point[:60], len(kept_args), len(arguments), cut_count,
    )
    print(f"  [Lasso] Ядро: {core_point[:100]}")
    print(f"  [Lasso] Оставлено {len(kept_args)}/{len(arguments)}, "
          f"отсечено {cut_count} (Бритва Оккама)")

    return core_point, kept_args


def format_lasso_section(core_point: str, kept_args: List[str],
                          cut_count: int) -> str:
    """Форматирует Lasso-секцию для отчёта."""
    lines = [
        "══════════════════════════════════════════════════════════════",
        "  LASSO (Топологическое стягивание · Бритва Оккама)",
        "──────────────────────────────────────────────────────────────",
        f"  Ядро-причина: {core_point}",
        f"  Отсечено аргументов: {cut_count}",
        "  Аргументы, ведущие к центру:",
    ]
    for i, arg in enumerate(kept_args, 1):
        lines.append(f"    {i}. {arg[:120]}")
    lines.append("══════════════════════════════════════════════════════════════")
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════
# v10.10: DELPHI CONSILIUM — пятый агент-судья
# Собирает мнения всех 4 ролей → единый структурный совет для PySR
# ══════════════════════════════════════════════════════════════════

def delphi_consilium(
    formula_shadow: str,
    shadow_names:   list,
    role_results:   list,       # List[RoleResult] — мнения 4 ролей
    r2_train:       float,
    domain_type:    str,
    ask_fn,                     # callable(prompt) -> str
) -> dict:
    """
    Пятый агент — Delphi Судья.
    Читает анализы всех 4 ролей и формирует ЕДИНЫЙ структурный совет.

    Возвращает dict:
        consensus_advice:   str   — текстовый итог консилиума
        forced_features:    list  — признаки которые ОБЯЗАТЕЛЬНО добавить
        forced_operators:   list  — операторы которые ОБЯЗАТЕЛЬНО добавить
        suggested_exponent: float — рекомендованная степень (0 = не задана)
        confidence:         float — 0..1 насколько роли согласны
    """
    import re as _re

    # Собираем анализы всех ролей
    role_block = ""
    for rr in role_results:
        verdict   = getattr(rr, "verdict",               "")
        analysis  = getattr(rr, "analysis",              "")
        critique  = getattr(rr, "structural_critique",   "")
        suggest   = getattr(rr, "improvement_suggestion","")
        role_block += (
            f"\n[{rr.role_name}] Вердикт: {verdict}\n"
            f"  Анализ: {analysis[:200]}\n"
            f"  Проблема: {critique[:150]}\n"
            f"  Совет: {suggest[:150]}\n"
        )

    prompt = (
        f"Отвечай СТРОГО на русском языке. Никакого другого языка.\n"
        f"Ты — Delphi Судья. Прочитай мнения 4 экспертов о формуле.\n\n"
        f"Формула: {formula_shadow}\n"
        f"Признаки: {', '.join(shadow_names)}\n"
        f"R²={r2_train:.4f}  Домен: {domain_type or 'неизвестен'}\n"
        f"\nМНЕНИЯ ЭКСПЕРТОВ:{role_block}\n"
        f"\nТвоя задача: найти то в чём эксперты СОГЛАСНЫ и сформулировать "
        f"единый структурный совет для следующего поиска формулы.\n"
        f"\nОтветь СТРОГО в формате JSON (без markdown):\n"
        f"{{\n"
        f'  "consensus_advice": "краткий итог на что обратить внимание",\n'
        f'  "forced_features": ["f0", "f2"],\n'
        f'  "forced_operators": ["sqrt", "/"],\n'
        f'  "suggested_exponent": 1.5,\n'
        f'  "confidence": 0.75\n'
        f"}}\n"
        f"Если эксперты не согласны по какому-то пункту — не включай его.\n"
        f'forced_features и forced_operators — только то что упомянули 2+ экспертов.\n'
        f"suggested_exponent = 0 если не упоминался."
    )

    print(f"\n  [Delphi Consilium] Судья формирует итоговый совет…")
    raw = ask_fn(prompt)

    # FIX v10.16: retry если Ollama не была готова к моменту вызова Delphi
    # (ollama_stop между раундами иногда не успевает перезагрузить модель)
    import time as _time
    if "недоступна" in raw.lower() or "unavailable" in raw.lower() or not raw.strip():
        import logging as _log_retry
        _log_retry.getLogger("scalpel").warning(
            "[Delphi] Ollama недоступна — ждём 8 сек и повторяем…"
        )
        _time.sleep(8)
        raw = ask_fn(prompt)

    # Парсим JSON ответ
    result = {
        "consensus_advice":   "",
        "forced_features":    [],
        "forced_operators":   [],
        "suggested_exponent": 0.0,
        "confidence":         0.0,
    }

    try:
        import json as _json
        clean = raw.strip()
        if clean.startswith("```"):
            clean = "\n".join(
                l for l in clean.splitlines() if not l.startswith("```")
            ).strip()
        s, e = clean.find("{"), clean.rfind("}")
        if s != -1 and e > s:
            data = _json.loads(clean[s:e+1])
            result["consensus_advice"]   = str(data.get("consensus_advice", ""))[:300]
            result["forced_features"]    = [
                f for f in data.get("forced_features", [])
                if f in shadow_names
            ]
            allowed_ops = {"+","-","*","/","sqrt","log","exp","abs","sin","cos","^"}
            result["forced_operators"]   = [
                op for op in data.get("forced_operators", [])
                if op in allowed_ops
            ]
            result["suggested_exponent"] = float(data.get("suggested_exponent", 0.0))
            result["confidence"]         = float(data.get("confidence", 0.0))
    except Exception as _e:
        import logging as _log
        _log.getLogger("scalpel").warning("[Delphi] Ошибка парсинга: %s", _e)
        # Fallback: эвристический парсинг из текста
        raw_lower = raw.lower()
        for fname in shadow_names:
            # Считаем сколько ролей упомянули этот признак
            mentions = sum(
                1 for rr in role_results
                if fname in getattr(rr, "structural_critique", "") or
                   fname in getattr(rr, "improvement_suggestion", "")
            )
            if mentions >= 2:
                result["forced_features"].append(fname)
        for kw, op in [("sqrt","sqrt"),("log","log"),("denominator","/"),
                       ("divide","/"),("power","^"),("exponent","^")]:
            mentions = sum(
                1 for rr in role_results
                if kw in getattr(rr, "improvement_suggestion","").lower()
            )
            if mentions >= 2:
                result["forced_operators"].append(op)

    # Выводим результат
    print(f"  [Delphi] Совет: {result['consensus_advice'][:100]}")
    print(f"  [Delphi] Принудительные признаки: {result['forced_features']}")
    print(f"  [Delphi] Принудительные операторы: {result['forced_operators']}")
    if result["suggested_exponent"]:
        print(f"  [Delphi] Рекомендованная степень: {result['suggested_exponent']}")
    print(f"  [Delphi] Уверенность консилиума: {result['confidence']:.0%}")

    return result


def format_consilium_section(consilium: dict) -> str:
    """Форматирует итог консилиума для FINAL REPORT."""
    lines = [
        "═" * 62,
        "  DELPHI CONSILIUM — ЕДИНЫЙ СОВЕТ ДЛЯ СЛЕДУЮЩЕГО PySR",
        "═" * 62,
        f"  Итог: {consilium.get('consensus_advice', '—')}",
        "",
    ]
    ff = consilium.get("forced_features", [])
    fo = consilium.get("forced_operators", [])
    se = consilium.get("suggested_exponent", 0.0)
    conf = consilium.get("confidence", 0.0)

    if ff:
        lines.append(f"  Признаки (обязательно):  {', '.join(ff)}")
    if fo:
        lines.append(f"  Операторы (обязательно): {', '.join(fo)}")
    if se:
        lines.append(f"  Степень (рекомендована): {se}")
    lines.append(f"  Уверенность консилиума:  {conf:.0%}")
    lines.append("═" * 62)
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════
# v10.12: SCIENTIFIC CYCLE — Замкнутый цикл научного открытия
# ══════════════════════════════════════════════════════════════════

def generate_scientific_question(
    formula_shadow:             str,
    formula_real:               str,
    heritage_result:            object,       # HeritageResult из atomic_precision
    domain_type:                str,
    ask_fn,                                   # callable(prompt) -> str
    previous_cycle_conclusions: list = None,  # v10.13: история предыдущих циклов
) -> str:
    """
    Генерирует ОДИН научный вопрос для следующего эксперимента.

    Это не технический совет для PySR.
    Это вопрос учёного: "что нужно измерить дальше?"

    Пример:
      Heritage: "похоже на Кеплера T~a^(3/2)"
      Вопрос:   "Если это гравитационная зависимость —
                 что произойдёт с показателем степени
                 если измерить массу центрального тела?"
    """
    # Контекст Heritage
    heritage_block = ""
    if heritage_result is not None:
        detected = getattr(heritage_result, "detected", False)
        scientists = getattr(heritage_result, "matched_scientists", [])
        verdict_lines = getattr(heritage_result, "verdict_lines", [])
        if detected and scientists:
            heritage_block = (
                f"\nHERITAGE MATCH: формула напоминает законы учёных: "
                f"{', '.join(scientists)}\n"
            )
            if verdict_lines:
                heritage_block += "\n".join(verdict_lines[:2]) + "\n"
    else:
        heritage_block = "\nHeritage: совпадений с известными законами не найдено.\n"

    # v10.13: блок памяти предыдущих циклов
    memory_block = ""
    if previous_cycle_conclusions:
        memory_block = "\nЧТО УЖЕ ПРОВЕРЯЛИ В ПРОШЛЫХ ЦИКЛАХ:\n"
        for conclusion in previous_cycle_conclusions[-5:]:  # последние 5 выводов
            memory_block += f"  — {conclusion}\n"
        memory_block += "Не повторяй эти направления — задай новый вопрос.\n"

    prompt = (
        f"Ты — научный руководитель. Перед тобой найденная формула.\n\n"
        f"Формула (shadow): {formula_shadow}\n"
        f"Формула (real):   {formula_real}\n"
        f"Домен: {domain_type or 'неизвестен'}\n"
        f"{heritage_block}\n"
        f"{memory_block}"
        f"Твоя задача: сформулировать ОДИН конкретный научный вопрос\n"
        f"для следующего эксперимента или сбора данных.\n\n"
        f"Требования к вопросу:\n"
        f"  — Конкретный: называет переменную или условие\n"
        f"  — Проверяемый: можно ответить экспериментом\n"
        f"  — Развивающий: расширяет найденную формулу\n\n"
        f"Примеры хороших вопросов:\n"
        f"  'Что происходит с показателем степени при изменении температуры?'\n"
        f"  'Есть ли порог значения f1 после которого формула перестаёт работать?'\n"
        f"  'Какая третья переменная может объяснить остаточную ошибку?'\n\n"
        f"Ответь ОДНИМ предложением — только вопрос, без пояснений."
    )

    print(f"\n  [Scientific Cycle] Генерируем научный вопрос…")
    question = ask_fn(prompt).strip()

    # Очищаем от кавычек и лишнего
    question = question.replace(chr(34), "").replace(chr(39), "").strip()
    if len(question) > 300:
        question = question[:300] + "…"

    print(f"  [Scientific Cycle] Вопрос: {question}")
    return question


def scientific_matryoshka_round(
    scientific_question: str,
    formula_shadow:      str,
    shadow_names:        list,
    domain_type:         str,
    ask_fn,              # callable(prompt) -> str
) -> dict:
    """
    Новый раунд Матрёшки — роли отвечают на научный вопрос.

    Каждая роль смотрит на вопрос со своей точки зрения:
      Скептик:   корректен ли вопрос вообще?
      Физик:     какая переменная или константа нужна физически?
      Прагматик: реально ли получить эти данные?
      Мистик:    с какими законами это может быть связано?

    Возвращает dict для Delphi Scientific.
    """
    roles = {
        "Скептик": (
            "Критик. Оцени: этот научный вопрос корректен? "
            "Есть ли в нём логические ошибки или неопределённости? "
            "Что может пойти не так при его проверке? "
            "1-2 предложения."
        ),
        "Физик": (
            "Физик. Ответь: какая конкретная переменная, константа "
            "или условие нужна чтобы ответить на этот вопрос? "
            "Назови её явно. 1 предложение."
        ),
        "Прагматик": (
            "Прагматик. Оцени: реально ли собрать данные "
            "для проверки этого вопроса? Что конкретно нужно измерить? "
            "1-2 предложения."
        ),
        "Мистик": (
            "Мистик. Найди аналогию: с какими известными законами природы "
            "или физическими принципами этот вопрос может быть связан? "
            "1 предложение."
        ),
    }

    context = (
        f"Найденная формула: {formula_shadow}\n"
        f"Признаки: {', '.join(shadow_names[:6])}\n"
        f"Домен: {domain_type or 'неизвестен'}\n"
        f"Научный вопрос: {scientific_question}\n"
    )

    responses = {}
    print(f"\n  [Scientific Cycle] Матрёшка отвечает на научный вопрос…")

    for role_name, task in roles.items():
        prompt = (
            # FIX v10.16: явное требование языка — mistral:7b переключался на китайский
            f"Отвечай СТРОГО на русском языке. Никакого другого языка.\n\n"
            f"Роль: {role_name}\n"
            f"Контекст:\n{context}\n"
            f"Задание: {task}"
        )
        response = ask_fn(prompt).strip()

        # FIX v10.36: если пустой ответ — retry с ультракоротким промптом
        if not response or len(response) < 5:
            short_prompt = (
                f"Отвечай на русском. Роль: {role_name}. "
                f"Вопрос: {scientific_question[:100]}. "
                f"Формула: {formula_shadow[:60]}. "
                f"Один короткий ответ (1 предложение)."
            )
            response = ask_fn(short_prompt).strip()

        if not response or len(response) < 5:
            response = f"[{role_name} не ответил]"

        responses[role_name] = response[:300]
        print(f"  [{role_name}] {response[:80]}…" if len(response) > 80 else f"  [{role_name}] {response}")

    return responses


def delphi_scientific(
    scientific_question:  str,
    role_responses:       dict,
    shadow_names:         list,
    ask_fn,               # callable(prompt) -> str
) -> dict:
    """
    Delphi Scientific — синтез ответов ролей в actionable совет.

    Возвращает:
        new_variable_hint:  str   — что нужно добавить в данные
        new_operator_hint:  list  — операторы для поиска
        new_exponent_hint:  float — рекомендованная степень
        next_question:      str   — уточнённый вопрос для следующего цикла
        confidence:         float — уверенность
    """
    import json as _json

    roles_block = ""
    for role, resp in role_responses.items():
        roles_block += f"\n[{role}]: {resp}\n"

    prompt = (
        f"Ты — Delphi Научный Синтезатор.\n\n"
        f"Научный вопрос который задали:\n{scientific_question}\n\n"
        f"Ответы 4 экспертов:{roles_block}\n"
        f"Признаки в данных: {', '.join(shadow_names[:8])}\n\n"
        f"Синтезируй ответы в ACTIONABLE совет для следующего раунда поиска.\n"
        f"Ответь СТРОГО в JSON без markdown:\n"
        f"{{\n"
        f'  "new_variable_hint": "название переменной которую нужно добавить или измерить",\n'
        f'  "new_operator_hint": ["+", "sqrt"],\n'
        f'  "new_exponent_hint": 1.5,\n'
        f'  "next_question": "уточнённый вопрос для следующего цикла",\n'
        f'  "confidence": 0.75\n'
        f"}}\n"
        f"new_exponent_hint = 0 если степень не упоминалась.\n"
        f"new_variable_hint = '' если нет конкретной переменной."
    )

    print(f"\n  [Scientific Cycle] Delphi синтезирует…")
    raw = ask_fn(prompt)

    result = {
        "new_variable_hint": "",
        "new_operator_hint": [],
        "new_exponent_hint": 0.0,
        "next_question":     scientific_question,  # fallback
        "confidence":        0.0,
    }

    try:
        clean = raw.strip()
        if clean.startswith("```"):
            clean = "\n".join(
                l for l in clean.splitlines() if not l.startswith("```")
            ).strip()
        s, e = clean.find("{"), clean.rfind("}")
        if s != -1 and e > s:
            data = _json.loads(clean[s:e+1])
            _var_hint = str(data.get("new_variable_hint", ""))[:100]
            # FIX v10.16: фильтр системных/зарезервированных слов.
            # LLM видит "shadow" в контексте формулы и возвращает его как переменную.
            _RESERVED_VARS = {
                "shadow", "shadowmapper", "shadow_names",
                "shuffle_test", "cross_blind",
                "f0", "f1", "f2", "f3", "f4", "f5",  # shadow-имена признаков
                "formula", "result", "score", "metric",
            }
            if _var_hint.strip().lower() in _RESERVED_VARS:
                import logging as _log_sci
                _log_sci.getLogger("scalpel").warning(
                    "[Delphi Scientific] new_variable_hint='%s' — зарезервированное слово, сброс.", _var_hint
                )
                _var_hint = ""
            result["new_variable_hint"] = _var_hint
            allowed = {"+","-","*","/","sqrt","log","exp","abs","^","sin","cos"}
            result["new_operator_hint"] = [
                op for op in data.get("new_operator_hint", [])
                if op in allowed
            ]
            result["new_exponent_hint"] = float(data.get("new_exponent_hint", 0.0))
            result["next_question"]     = str(data.get("next_question", scientific_question))[:300]
            result["confidence"]        = float(data.get("confidence", 0.0))
    except Exception as _e:
        import logging as _log
        _log.getLogger("scalpel").warning("[Delphi Scientific] Ошибка парсинга: %s", _e)

    print(f"  [Scientific Cycle] Переменная: '{result['new_variable_hint']}'")
    print(f"  [Scientific Cycle] Операторы: {result['new_operator_hint']}")
    if result["new_exponent_hint"]:
        print(f"  [Scientific Cycle] Степень: {result['new_exponent_hint']}")
    print(f"  [Scientific Cycle] Следующий вопрос: {result['next_question'][:80]}")
    print(f"  [Scientific Cycle] Уверенность: {result['confidence']:.0%}")

    return result


def format_scientific_frontier(
    scientific_question: str,
    delphi_sci:          dict,
    heritage_detected:   bool,
    scientists:          list,
) -> str:
    """Форматирует секцию SCIENTIFIC FRONTIER для FINAL REPORT."""
    lines = [
        "═" * 62,
        "  SCIENTIFIC FRONTIER — СЛЕДУЮЩИЙ ШАГ",
        "═" * 62,
    ]
    if heritage_detected and scientists:
        lines.append(f"  Heritage: {', '.join(scientists)}")
        lines.append("")

    lines.append(f"  Вопрос: {scientific_question}")
    lines.append("")

    vh = delphi_sci.get("new_variable_hint", "")
    oh = delphi_sci.get("new_operator_hint", [])
    eh = delphi_sci.get("new_exponent_hint", 0.0)
    nq = delphi_sci.get("next_question", "")
    cf = delphi_sci.get("confidence", 0.0)

    if vh:
        lines.append(f"  Добавить переменную: {vh}")
    if oh:
        lines.append(f"  Операторы поиска:    {', '.join(oh)}")
    if eh:
        lines.append(f"  Степень (hint):      {eh}")
    if nq and nq != scientific_question:
        lines.append(f"  Уточнённый вопрос:   {nq[:80]}")
    lines.append(f"  Уверенность:         {cf:.0%}")
    lines.append("═" * 62)
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════
# ЛЕТОПИСЕЦ — generate_chronicle()  v10.14
# ══════════════════════════════════════════════════════════════════

def generate_chronicle(
    hadi_history:   list,    # список кандидатов со всеми полями
    consilium:      dict,    # советы Delphi финального кандидата
    heritage:       str,     # Heritage-строка (на что похоже)
    domain:         str,     # домен
    formula_final:  str,     # финальная принятая формула
    ask_fn,                  # callable(prompt) → str
) -> str:
    """
    Летописец: читает историю HADI цикла и пишет понятный человеку
    рассказ о пути поиска.

    Вызывается после matryoshka_audit перед сохранением в GoldVault.
    temperature=0.4  (ask_fn должен быть создан с этим параметром).

    Возвращает строку-рассказ для поля "chronicle" в gold_formulas.json
    и секции ИСТОРИЯ ПОИСКА в FINAL_REPORT.
    """
    if not hadi_history:
        log.debug("[Летописец] Пустая история — пропускаем")
        return ""

    # ── Строим структурированный контекст из hadi_history ─────────
    attempts_lines = []
    for idx, cand in enumerate(hadi_history, 1):
        formula   = cand.get("formula_shadow", "?")
        r2        = float(cand.get("r2_blind", cand.get("r2_train", 0.0)))
        feedback  = cand.get("matryoshka_feedback") or []
        cand_con  = cand.get("delphi_consilium")  or {}

        # Кто отклонил и краткая причина
        rejectors = []
        for fb in feedback:
            if fb.get("verdict") in ("ОТКЛОНЕНА", "УСЛОВНО"):
                role     = fb.get("role", "?")
                critique = (fb.get("critique") or "")[:80]
                rejectors.append(f"{role}: {critique}")
        rejection_str = "; ".join(rejectors) if rejectors else "—"

        # Совет Delphi для этого кандидата
        delphi_ops = cand_con.get("forced_operators") or []
        delphi_exp = cand_con.get("suggested_exponent") or 0.0
        delphi_str = ""
        if delphi_ops:
            delphi_str = f"добавить {', '.join(str(o) for o in delphi_ops[:3])}"
        if delphi_exp:
            delphi_str += f" степень {delphi_exp}"
        delphi_str = delphi_str.strip() or "—"

        is_final = (formula == formula_final)
        status   = "→ ПРИНЯТА" if is_final else f"R²={r2:.3f} → отклонена"

        attempts_lines.append(
            f"Попытка {idx}: {formula}\n"
            f"  Результат: {status}\n"
            f"  Причина отклонения: {rejection_str}\n"
            f"  Совет Delphi: {delphi_str}"
        )

    attempts_block = "\n".join(attempts_lines)
    n_attempts     = len(hadi_history)

    # Контекст домена и Heritage
    domain_ctx   = f"Домен: {domain}." if domain else ""
    heritage_ctx = f"Heritage (похоже на): {heritage}." if heritage else ""
    style_hint   = "физическим языком" if domain else "простым языком"

    prompt = (
        f"Ты — Летописец символьной регрессии.\n"
        f"Задача: написать краткий, понятный рассказ о пути поиска формулы.\n"
        f"{domain_ctx} {heritage_ctx}\n\n"
        f"История попыток ({n_attempts} итераций):\n"
        f"{'─' * 50}\n"
        f"{attempts_block}\n"
        f"{'─' * 50}\n\n"
        f"Финальная формула: {formula_final}\n\n"
        f"Требования к рассказу:\n"
        f"1. Для каждой попытки: что пробовала, R², кто и почему отклонил\n"
        f"2. Как советы Delphi изменяли направление поиска\n"
        f"3. Итог: как система пришла к финальной формуле\n"
        f"Пиши {style_hint}. Длина: 8–15 строк. "
        f"Начни с 'ИСТОРИЯ ПОИСКА'."
    )

    print(f"\n  [Летописец] Составляю историю поиска ({n_attempts} попыток)…")
    try:
        chronicle = ask_fn(prompt)
        chronicle = chronicle.strip()
        if not chronicle:
            raise ValueError("пустой ответ")
        print(f"  [Летописец] ✓ {len(chronicle)} символов")
        return chronicle
    except Exception as exc:
        log.warning("[Летописец] Ошибка генерации: %s", exc)
        # Минимальный fallback без LLM
        fallback_lines = ["ИСТОРИЯ ПОИСКА"]
        for idx, cand in enumerate(hadi_history, 1):
            f_   = cand.get("formula_shadow", "?")
            r2_  = float(cand.get("r2_blind", 0))
            fb_  = cand.get("matryoshka_feedback") or []
            rej_ = next(
                (f"{x.get('role','?')}: {(x.get('critique') or '')[:50]}"
                 for x in fb_ if x.get("verdict") == "ОТКЛОНЕНА"), "—"
            )
            status_ = "ПРИНЯТА" if f_ == formula_final else f"отклонена ({rej_})"
            fallback_lines.append(f"  Попытка {idx}: {f_} | R²={r2_:.3f} | {status_}")
        return "\n".join(fallback_lines)
