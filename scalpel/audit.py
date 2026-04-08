"""
audit.py — Матрёшка v10.2.7 (RAM Queue + Critical Thinking).

Полная логика перенесена в ram_queue.MatryoshkaQueue.
Этот модуль — тонкий фасад для обратной совместимости с engine.py.

Протокол:
  Роль 1 (Скептик)   → RAM-слот → ollama_stop → gc
  Роль 2 (Физик)     → RAM-слот → ollama_stop → gc
    ↳ [Dialectic] Socratic Cross-Examination: Скептик ↔ Физик (3 вопроса)
  Роль 3 (Прагматик) → RAM-слот → ollama_stop → gc
  Роль 4 (Мистик)    → RAM-слот → ollama_stop → gc
    ↳ [Sinquain] Семантическое сжатие инварианта (5 строк)

Никогда два модуля в RAM одновременно.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, List, Tuple

from .config import (
    OLLAMA_HOST, OLLAMA_MODEL, SCRIPT_DIR, ROLE_MODELS,
    DIALECTIC_QUESTIONS, SINQUAIN_ROLE,
    SYNTHESIS_MODEL, CHRONICLE_MODEL,  # FIX: антиинцест для Dialectic/Delphi/Sinquain
)
from .ram_queue import MatryoshkaQueue, ram_status_report
from .critical_thinking import (
    socratic_cross_examination, format_dialectic_section,
    generate_sinquain, format_sinquain_section,
    delphi_consilium, format_consilium_section,   # v10.10
)
from .navigator import ollama_chat  # hoisted для patch('scalpel.audit.ollama_chat')

log = logging.getLogger("scalpel")

# Экспортируем для тестов
MATRYOSHKA_ROLES = [
    ("Скептик",   (
        "Найди конкретные физические слабости формулы. "
        "ВАЖНО: если не можешь назвать КОНКРЕТНОЕ физическое нарушение "
        "(деление на ноль, нарушение размерности, физическая невозможность) — "
        "голосуй ВОЗДЕРЖАЛАСЬ. Никогда не голосуй ОТКЛОНЕНА только из-за "
        "незнания домена или недостатка информации."
    )),
    ("Физик",     "Проверь размерностную корректность. Соответствует известным законам?"),
    ("Прагматик", "Оцени практическую применимость. В каких условиях ломается?"),
    ("Мистик",    "Найди аналогии с известными физическими законами или структурами."),
]


def matryoshka_audit(
    formula_shadow:  str,
    shadow_names:    List[str],
    r2_train:        float,
    complexity:      int,
    domain_type:     str = "",
    host:            str = OLLAMA_HOST,
    model:           str = OLLAMA_MODEL,
    dspy_orch:       Any = None,
    heritage_context: str = "",
    r2_blind:        float = 0.0,   # FIX: передаётся в episodic_memory
    # ── v9.9.2-dataaware: реальные данные для LLM ─────────────────
    X_samples:       Any = None,
    y_samples:       Any = None,
    y_pred_samples:  Any = None,
    real_names:      Any = None,
    ctx:             Any = None,    # v10.24: SharedContext — роли видят историю
) -> Tuple[str, str, List, dict]:
    """
    Запускает 4 роли строго по одной через MatryoshkaQueue.
    v10.8: возвращает (consensus, extended_report, role_results).
    v10.10: возвращает (consensus, extended_report, role_results, consilium).

    v10.2.7 расширения:
      - После ролей Скептик + Физик: Socratic Cross-Examination (Dialectic).
      - После роли Мистик: Синквейн инварианта (Sinquain).

    dspy_orch используется только для определения dspy_active.
    """
    dspy_active = dspy_orch is not None and getattr(dspy_orch, "is_active", False)

    queue = MatryoshkaQueue(
        model       = model,
        host        = host,
        dspy_active = dspy_active,
        chat_fn     = ollama_chat,
        role_models = ROLE_MODELS,   # v10.9: разные модели для разных ролей
    )

    consensus, report, role_results = queue.run(
        formula_shadow = formula_shadow,
        shadow_names   = shadow_names,
        r2_train       = r2_train,
        r2_blind       = r2_blind,   # FIX: передаём правильное значение
        complexity     = complexity,
        domain_type    = domain_type,
        X_samples      = X_samples,
        y_samples      = y_samples,
        y_pred_samples = y_pred_samples,
        real_names     = real_names,
        ctx            = ctx,         # v10.24: SharedContext
    )

    # Выводим RAM-отчёт
    print(ram_status_report(role_results))

    # v10.18: пишем полный reasoning_log.jsonl
    try:
        import json as _json, datetime as _dt
        from .config import VAULT_DIR as _VD
        _log_path = _VD / "reasoning_log.jsonl"
        _ts = _dt.datetime.now().isoformat(timespec="seconds")
        for _rr in role_results:
            _entry = {
                "ts":            _ts,
                "role":          _rr.role_name,
                "formula":       formula_shadow,
                "verdict":       _rr.verdict,
                "used_dspy":     _rr.used_dspy,
                "elapsed_sec":   round(_rr.elapsed_sec, 1),
                "full_response": _rr.full_response,
                "analysis":      _rr.analysis,
            }
            with open(_log_path, "a", encoding="utf-8") as _lf:
                _lf.write(_json.dumps(_entry, ensure_ascii=False) + "\n")
    except Exception as _rlog_err:
        import logging as _rlog_log
        _rlog_log.getLogger("scalpel").debug("[ReasoningLog] Не записан: %s", _rlog_err)

    # ── [Dialectic] Socratic Cross-Examination ────────────────────
    # Запускаем после основного audit-прогона (RAM уже освобождена)
    _chat = ollama_chat  # module-level import; patch('scalpel.audit.ollama_chat') работает

    # FIX: разные модели для разных блоков (антиинцест)
    # _ask      — аналитика (Qwen через OLLAMA_MODEL): RCA, Lasso
    # _ask_syn  — синтез (Gemma через SYNTHESIS_MODEL): Dialectic, Delphi
    # _ask_chr  — нарратив (LLaMA через CHRONICLE_MODEL): Sinquain
    def _ask(prompt: str) -> str:
        return _chat(prompt, model=model, host=host, temperature=0.4, num_predict=300)

    def _ask_syn(prompt: str) -> str:
        return _chat(prompt, model=SYNTHESIS_MODEL, host=host, temperature=0.4, num_predict=400)

    def _ask_chr(prompt: str) -> str:
        return _chat(prompt, model=CHRONICLE_MODEL, host=host, temperature=0.6, num_predict=300)

    dialectic_section = ""
    try:
        print("\n  [Dialectic v10.2.7] Socratic Cross-Examination…")
        # v10.3.9 RAM FIX: передаём interround_fn для выгрузки Ollama между раундами.
        # Это критично при 7.7 ГБ — без этого второй раунд «Физика» попытается
        # загрузить модель поверх незавершённого «Скептика».
        import gc as _gc
        from .engine import ollama_stop as _ollama_stop
        def _interround():
            _ollama_stop(SYNTHESIS_MODEL)   # FIX: останавливаем ту модель что работала
            _gc.collect()

        dialogue = socratic_cross_examination(
            formula_shadow   = formula_shadow,
            shadow_names     = shadow_names,
            r2_train         = r2_train,
            complexity       = complexity,
            domain_type      = domain_type,
            ask_fn           = _ask_syn,        # FIX: SYNTHESIS_MODEL (Gemma)
            n_questions      = DIALECTIC_QUESTIONS,
            interround_fn    = _interround,
            heritage_context = heritage_context,
        )
        dialectic_section = format_dialectic_section(dialogue)
    except Exception as _de:
        log.warning("[Dialectic] Ошибка: %s", _de)
        dialectic_section = f"[Dialectic] Недоступен: {_de}"

    # ── [Sinquain] Семантическое сжатие ──────────────────────────
    sinquain_section = ""
    try:
        print(f"\n  [Sinquain v10.2.7] {SINQUAIN_ROLE} генерирует синквейн…")
        sinquain_text = generate_sinquain(
            formula_real = formula_shadow,
            consensus    = consensus,
            r2_train     = r2_train,
            domain_type  = domain_type,
            ask_fn       = _ask_chr,    # FIX: CHRONICLE_MODEL (LLaMA) — нарратив
            role         = SINQUAIN_ROLE,
        )
        sinquain_section = format_sinquain_section(
            sinquain_text, SINQUAIN_ROLE, formula_shadow
        )
        print(sinquain_section)
        # FIX: синквейн сохраняется в память Мистика → становится DSPy few-shot примером
        try:
            from .episodic_memory import get_memory as _sinq_mem
            _sinq_mem().remember(
                role_name  = SINQUAIN_ROLE,
                formula    = formula_shadow,
                verdict    = "СИНКВЕЙН",
                analysis   = sinquain_text[:500],
                r2_train   = r2_train,
                domain     = domain_type,
            )
            log.debug("[Sinquain] Сохранён в episodic_memory роли %s", SINQUAIN_ROLE)
        except Exception as _sinq_mem_err:
            log.debug("[Sinquain/Memory] %s", _sinq_mem_err)
    except Exception as _se:
        log.warning("[Sinquain] Ошибка: %s", _se)
        sinquain_section = f"[Sinquain] Недоступен: {_se}"

    # ── [Delphi Consilium v10.10] Пятый агент-судья ─────────────
    # v10.14 БАГ 6: Delphi получает мета-контекст — что советовалось и работало
    def _ask_delphi(prompt: str) -> str:
        try:
            from .meta_context import get_delphi_context as _dctx
            meta_hint = _dctx(domain_type=domain_type)
            if meta_hint:
                prompt = meta_hint + "\n\n" + prompt
        except Exception:
            pass
        return _ask_syn(prompt)   # FIX: SYNTHESIS_MODEL (Gemma) для синтеза

    consilium: dict = {}
    consilium_section = ""   # FIX БАГ 2: инициализация до try, иначе NameError если format_consilium_section упадёт
    try:
        print(f"\n  [Delphi Consilium v10.10] Формирует итоговый совет…")
        consilium = delphi_consilium(
            formula_shadow = formula_shadow,
            shadow_names   = shadow_names,
            role_results   = role_results,
            r2_train       = r2_train,
            domain_type    = domain_type,
            ask_fn         = _ask_delphi,  # v10.14: с мета-контекстом
        )
        consilium_section = format_consilium_section(consilium)
        print(consilium_section)
    except Exception as _ce:
        log.warning("[Consilium] Ошибка: %s", _ce)
        consilium_section = f"[Consilium] Недоступен: {_ce}"

    # Собираем расширенный отчёт
    extended_report = "\n\n".join(filter(None, [
        report,
        dialectic_section,
        sinquain_section,
        consilium_section,
    ]))

    # Сохраняем отчёт
    report_path = SCRIPT_DIR / "scalpel_vault" / "CONSENSUS_REPORT.txt"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(extended_report, encoding="utf-8")
    log.info("[МАТРЁШКА v10.2.7] Консенсус: %s", consensus)

    return consensus, extended_report, role_results, consilium
