"""
meta_reflection.py — Мета-обучение по критической массе v10.14.

Когда накапливается достаточно данных — система делает полный
самоанализ по всем источникам памяти и генерирует:

  1. ПАТТЕРНЫ УСПЕХА  — что стабильно работает
  2. ПАТТЕРНЫ ОШИБОК  — что стабильно не работает и почему
  3. ПРАВИЛА НАВИГАЦИИ — что Navigator должен делать иначе
  4. ИТОГОВЫЕ ВЫВОДЫ  — читаемый отчёт Летописца по всей истории

Результат сохраняется как:
  - meta_reflection.jsonl      — структурированные выводы
  - meta_report_<ts>.txt       — читаемый отчёт
  - Высококачественные dspy.Example для siege_compile (5×-усиление)

═══════════════════════════════════════════════════════════════
ТРИГГЕРЫ КРИТИЧЕСКОЙ МАССЫ
═══════════════════════════════════════════════════════════════

Рефлексия запускается когда ЛЮБОЙ из порогов превышен:

  THRESHOLD_CHRONICLE   = 50   хроника шагов
  THRESHOLD_CURRICULUM  = 30   curriculum датасетов
  THRESHOLD_GOLD        = 10   принятых формул
  THRESHOLD_FAILURES    = 20   смертей

И не чаще чем раз в COOLDOWN_RUNS запусков.
"""
from __future__ import annotations

import gc
import json
import logging
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .config import VAULT_DIR, OLLAMA_HOST, OLLAMA_MODEL, SYNTHESIS_MODEL

log = logging.getLogger("scalpel")

# ── Пути ──────────────────────────────────────────────────────────
META_REFLECTION_LOG  = VAULT_DIR / "meta_reflection.jsonl"
META_REPORT_DIR      = VAULT_DIR / "meta_reports"
META_STATE_PATH      = VAULT_DIR / "meta_reflection_state.json"

# ── Пороги критической массы ──────────────────────────────────────
THRESHOLD_CHRONICLE   = 50    # chronicle_step записей
THRESHOLD_CURRICULUM  = 30    # curriculum датасетов
THRESHOLD_GOLD        = 10    # принятых формул
THRESHOLD_FAILURES    = 20    # записей в dspy_failure_log

COOLDOWN_RUNS         = 5     # минимум запусков между рефлексиями


# ══════════════════════════════════════════════════════════════════
# СОСТОЯНИЕ: когда последний раз была рефлексия
# ══════════════════════════════════════════════════════════════════

def _load_state() -> Dict:
    if not META_STATE_PATH.exists():
        return {"runs_since_last": 0, "total_reflections": 0,
                "last_ts": "", "last_counts": {}}
    try:
        return json.loads(META_STATE_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {"runs_since_last": 0, "total_reflections": 0,
                "last_ts": "", "last_counts": {}}


def _save_state(state: Dict) -> None:
    try:
        META_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        META_STATE_PATH.write_text(
            json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    except Exception as e:
        log.warning("[MetaReflection] Ошибка сохранения состояния: %s", e)


def _increment_run_counter() -> None:
    """Вызывается при каждом запуске run_engine."""
    state = _load_state()
    state["runs_since_last"] = state.get("runs_since_last", 0) + 1
    _save_state(state)


# ══════════════════════════════════════════════════════════════════
# СБОР ДАННЫХ: читаем все источники памяти
# ══════════════════════════════════════════════════════════════════

def _collect_all_data() -> Dict:
    """
    Читает все источники памяти и возвращает сводку.
    Не падает — возвращает пустые структуры при ошибках.
    """
    from .episodic_memory import get_memory, MEMORY_DIR
    from .config import GOLD_PATH, DSPY_FAILURE_LOG

    data: Dict[str, Any] = {
        "chronicle_steps":   [],
        "chronicle_finals":  [],
        "curriculum":        [],
        "gold_formulas":     [],
        "failures":          [],
        "role_memory":       {},
        "scientific_cycles": [],
    }

    # chronicle_steps + chronicle_finals
    chron_path = MEMORY_DIR / "chronicle_steps.jsonl"
    if chron_path.exists():
        for line in chron_path.read_text(encoding="utf-8").strip().splitlines():
            try:
                rec = json.loads(line)
                if rec.get("event") == "chronicle_step":
                    data["chronicle_steps"].append(rec)
                elif rec.get("event") == "chronicle_final":
                    data["chronicle_finals"].append(rec)
            except Exception:
                continue

    # curriculum_memory
    curr_path = MEMORY_DIR / "curriculum_memory.jsonl"
    if curr_path.exists():
        seen = set()  # убираем дубли от усиления
        for line in curr_path.read_text(encoding="utf-8").strip().splitlines():
            try:
                rec = json.loads(line)
                key = (rec.get("formula_true", ""), rec.get("level", 0))
                if key not in seen:
                    seen.add(key)
                    data["curriculum"].append(rec)
            except Exception:
                continue

    # gold_formulas
    if GOLD_PATH.exists():
        try:
            gdata = json.loads(GOLD_PATH.read_text(encoding="utf-8"))
            data["gold_formulas"] = gdata.get("formulas", [])
        except Exception:
            pass

    # dspy_failure_log
    if DSPY_FAILURE_LOG.exists():
        for line in DSPY_FAILURE_LOG.read_text(encoding="utf-8").strip().splitlines():
            try:
                data["failures"].append(json.loads(line))
            except Exception:
                continue

    # Роли (топ-паттерны)
    mem = get_memory()
    from .config import ROLE_NAMES
    for role in ROLE_NAMES:
        try:
            data["role_memory"][role] = mem.top_patterns(role, verdict="ПРИНЯТА", top_n=5)
        except Exception:
            data["role_memory"][role] = []

    # Scientific cycles
    sci_path = MEMORY_DIR / "scientific_cycles.jsonl"
    if sci_path.exists():
        for line in sci_path.read_text(encoding="utf-8").strip().splitlines():
            try:
                data["scientific_cycles"].append(json.loads(line))
            except Exception:
                continue

    return data


# ══════════════════════════════════════════════════════════════════
# АНАЛИТИКА: извлекаем паттерны из данных
# ══════════════════════════════════════════════════════════════════

def _extract_patterns(data: Dict) -> Dict:
    """
    Чисто аналитическая функция — никакого LLM.
    Извлекает паттерны из накопленных данных.
    """
    patterns: Dict[str, Any] = {}

    # ── 1. Паттерны отклонений (кто чаще всего отклоняет и за что) ─
    rejection_counter: Counter = Counter()
    rejection_reasons: Dict[str, List[str]] = defaultdict(list)
    for step in data["chronicle_steps"]:
        rej = step.get("rejected_by", "")
        reason = step.get("reason", "")[:80]
        if rej:
            rejection_counter[rej] += 1
            if reason:
                rejection_reasons[rej].append(reason)

    patterns["top_rejectors"] = [
        {"role": role, "count": count,
         "top_reasons": list(set(rejection_reasons[role]))[:3]}
        for role, count in rejection_counter.most_common(4)
    ]

    # ── 2. Паттерны Delphi — что советовал и что помогало ──────────
    delphi_hints: Counter = Counter()
    successful_hints: Counter = Counter()
    for step in data["chronicle_steps"]:
        hint = step.get("delphi_hint", "").strip()
        if hint:
            delphi_hints[hint] += 1
            # Если led_to имеет высокий R² — этот совет работал
            if step.get("r2", 0) >= 0.85:
                successful_hints[hint] += 1

    patterns["delphi_hints"] = {
        "all":        list(delphi_hints.most_common(5)),
        "successful": list(successful_hints.most_common(5)),
    }

    # ── 3. Паттерны успешных формул ────────────────────────────────
    success_operators: Counter = Counter()
    for final in data["chronicle_finals"]:
        if final.get("passed") and final.get("r2_blind", 0) >= 0.85:
            formula = final.get("formula_final", "")
            for op in ["sqrt", "log", "exp", "/", "*", "^", "+", "-"]:
                if op in formula:
                    success_operators[op] += 1

    for rec in data["gold_formulas"]:
        formula = rec.get("formula", "")
        for op in ["sqrt", "log", "exp", "/", "*", "^", "+", "-"]:
            if op in formula:
                success_operators[op] += 1

    patterns["successful_operators"] = list(success_operators.most_common(6))

    # ── 4. Curriculum статистика ────────────────────────────────────
    curr_by_level: Dict[int, Dict] = defaultdict(lambda: {"total": 0, "good": 0, "bad": 0, "r2_sum": 0.0})
    for rec in data["curriculum"]:
        lv = int(rec.get("level", 1))
        curr_by_level[lv]["total"] += 1
        rt = rec.get("result_type", "")
        if rt == "good":
            curr_by_level[lv]["good"] += 1
        elif rt == "bad":
            curr_by_level[lv]["bad"] += 1
        curr_by_level[lv]["r2_sum"] += float(rec.get("r2_blind", 0))

    patterns["curriculum_by_level"] = {
        lv: {
            "total":     v["total"],
            "pass_rate": round(v["good"] / max(v["total"], 1), 3),
            "avg_r2":    round(v["r2_sum"] / max(v["total"], 1), 3),
        }
        for lv, v in curr_by_level.items()
    }

    # ── 5. Самые частые паттерны ошибок ────────────────────────────
    failure_types: Counter = Counter()
    for fail in data["failures"]:
        ft = fail.get("failure_type", "UNKNOWN")
        failure_types[ft] += 1

    patterns["top_failure_types"] = list(failure_types.most_common(5))

    # ── 6. Что советовал Физик/Скептик в scientific cycles ─────────
    sci_variables: Counter = Counter()
    sci_operators: Counter = Counter()
    for cycle in data["scientific_cycles"]:
        var = cycle.get("variable", "").strip()
        if var:
            sci_variables[var] += 1
        delphi = cycle.get("delphi", "")
        for op in ["sqrt", "log", "/", "^", "exp"]:
            if op in delphi:
                sci_operators[op] += 1

    patterns["scientific_insights"] = {
        "top_variables": list(sci_variables.most_common(3)),
        "top_operators": list(sci_operators.most_common(4)),
    }

    return patterns


# ══════════════════════════════════════════════════════════════════
# LLM-СИНТЕЗ: генерируем выводы через Ollama
# ══════════════════════════════════════════════════════════════════

def _generate_meta_analysis(
    patterns:   Dict,
    data:       Dict,
    ask_fn,
) -> Dict[str, str]:
    """
    Отправляет паттерны в LLM и получает структурированные выводы.
    Возвращает dict с ключами: errors, successes, nav_rules, summary.
    """
    n_steps    = len(data["chronicle_steps"])
    n_gold     = len(data["gold_formulas"])
    n_curr     = len(data["curriculum"])
    n_failures = len(data["failures"])

    rejectors_str = "\n".join(
        f"  {r['role']}: {r['count']} раз — причины: {', '.join(r['top_reasons'][:2]) or 'разные'}"
        for r in patterns.get("top_rejectors", [])
    ) or "  (нет данных)"

    delphi_good = patterns.get("delphi_hints", {}).get("successful", [])
    delphi_str  = ", ".join(f"{h}({c})" for h, c in delphi_good[:4]) or "нет данных"

    ops_str = ", ".join(f"{op}({c})" for op, c in patterns.get("successful_operators", [])[:5])

    curr_str = "\n".join(
        f"  Уровень {lv}: pass_rate={v['pass_rate']:.0%}, avg_R²={v['avg_r2']:.3f}"
        for lv, v in sorted(patterns.get("curriculum_by_level", {}).items())
    ) or "  (нет данных)"

    fail_str = ", ".join(f"{ft}({c})" for ft, c in patterns.get("top_failure_types", [])[:4]) or "нет"

    sci = patterns.get("scientific_insights", {})
    sci_var = ", ".join(f"{v}({c})" for v, c in sci.get("top_variables", [])[:3]) or "нет"
    sci_op  = ", ".join(f"{o}({c})" for o, c in sci.get("top_operators", [])[:4]) or "нет"

    prompt = (
        f"Ты — Мета-аналитик системы символьной регрессии Scalpel.\n"
        f"Проанализируй накопленный опыт системы и дай структурированные выводы.\n\n"
        f"СТАТИСТИКА БАЗЫ ЗНАНИЙ:\n"
        f"  Шагов поиска в хронике: {n_steps}\n"
        f"  Принятых формул (Gold): {n_gold}\n"
        f"  Датасетов curriculum: {n_curr}\n"
        f"  Зафиксированных ошибок: {n_failures}\n\n"
        f"ТОП ОТКЛОНЕНИЙ:\n{rejectors_str}\n\n"
        f"УСПЕШНЫЕ СОВЕТЫ DELPHI (оператор → кол-во успешных применений):\n  {delphi_str}\n\n"
        f"ОПЕРАТОРЫ В УСПЕШНЫХ ФОРМУЛАХ:\n  {ops_str}\n\n"
        f"CURRICULUM ПО УРОВНЯМ:\n{curr_str}\n\n"
        f"ТОП ТИПОВ ОШИБОК: {fail_str}\n\n"
        f"НАУЧНЫЕ ИНСАЙТЫ — переменные: {sci_var}, операторы: {sci_op}\n\n"
        f"Дай КРАТКИЕ (2-4 предложения каждый) выводы в JSON без markdown:\n"
        f"{{\n"
        f'  "errors":    "Главные паттерны ошибок: что не работает и почему",\n'
        f'  "successes": "Главные паттерны успеха: что стабильно работает",\n'
        f'  "nav_rules": "3-5 конкретных правил для Navigator: если X → делай Y",\n'
        f'  "summary":   "Итоговый вывод: как система эволюционировала"\n'
        f"}}"
    )

    print(f"\n  [MetaReflection] LLM синтезирует выводы по {n_steps} шагам + {n_gold} Gold…")
    raw = ask_fn(prompt)

    # Парсим JSON
    result = {
        "errors":    "",
        "successes": "",
        "nav_rules": "",
        "summary":   "",
    }
    try:
        clean = raw.strip()
        if clean.startswith("```"):
            clean = "\n".join(l for l in clean.splitlines() if not l.startswith("```")).strip()
        s, e = clean.find("{"), clean.rfind("}")
        if s != -1 and e > s:
            parsed = json.loads(clean[s:e+1])
            for key in result:
                if parsed.get(key):
                    result[key] = str(parsed[key])[:500]
    except Exception as ex:
        log.warning("[MetaReflection] Ошибка парсинга LLM: %s", ex)
        # Fallback: кладём raw в summary
        result["summary"] = raw.strip()[:500]

    return result


# ══════════════════════════════════════════════════════════════════
# КОНВЕРТАЦИЯ В DSPy ПРИМЕРЫ
# ══════════════════════════════════════════════════════════════════

def _meta_to_dspy_examples(
    analysis: Dict[str, str],
    patterns: Dict,
) -> List[Any]:
    """
    Превращает выводы мета-анализа в высококачественные dspy.Example.
    Эти примеры идут с усилением 5× — они самые ценные в trainset.
    """
    try:
        import dspy
    except ImportError:
        return []

    examples = []

    nav_rules = analysis.get("nav_rules", "")
    successes = analysis.get("successes", "")
    errors    = analysis.get("errors", "")

    if not nav_rules:
        return []

    # Пример 1: правила навигации из мета-анализа
    ex1 = dspy.Example(
        data_meta          = "meta_reflection: distilled from all historical data",
        failure_logs       = json.dumps([{
            "hypothesis":   "meta_pattern",
            "death_reason": errors[:200] if errors else "see nav_rules",
            "source":       "meta_reflection",
        }]),
        selected_features  = "f0, f1, f2",
        selected_operators = _extract_best_operators(patterns),
        hypotheses         = _extract_best_hypothesis_pattern(patterns),
        ooda_stable        = "true",
        reasoning          = nav_rules[:400],
    ).with_inputs("data_meta", "failure_logs")

    # Пример 2: паттерн успеха
    if successes:
        ex2 = dspy.Example(
            data_meta          = "meta_reflection: success patterns",
            failure_logs       = "[]",
            selected_features  = "f0, f1",
            selected_operators = _extract_best_operators(patterns),
            hypotheses         = _extract_best_hypothesis_pattern(patterns),
            ooda_stable        = "true",
            reasoning          = successes[:400],
        ).with_inputs("data_meta", "failure_logs")
        examples.extend([ex2] * 5)  # 5× усиление

    examples.extend([ex1] * 5)  # 5× усиление
    return examples


def _extract_best_operators(patterns: Dict) -> str:
    ops = [op for op, _ in patterns.get("successful_operators", [])[:5]]
    if not ops:
        return "+,-,*,/,sqrt,log"
    return ",".join(ops)


def _extract_best_hypothesis_pattern(patterns: Dict) -> str:
    delphi_good = patterns.get("delphi_hints", {}).get("successful", [])
    if delphi_good:
        return f"try_{delphi_good[0][0]}_first"
    curr = patterns.get("curriculum_by_level", {})
    if curr:
        best_lv = max(curr.items(), key=lambda x: x[1].get("avg_r2", 0))
        return f"f0^2 or sqrt(f0) (L{best_lv[0]} avg_R²={best_lv[1]['avg_r2']:.2f})"
    return "f0/f1 or sqrt(f0)*f1"


# ══════════════════════════════════════════════════════════════════
# СОХРАНЕНИЕ РЕЗУЛЬТАТОВ
# ══════════════════════════════════════════════════════════════════

def _save_reflection(
    patterns:     Dict,
    analysis:     Dict[str, str],
    data_counts:  Dict[str, int],
) -> Path:
    """
    Сохраняет мета-рефлексию в двух форматах:
    1. meta_reflection.jsonl — машиночитаемый
    2. meta_reports/report_<ts>.txt — читаемый человеком
    """
    ts = datetime.now().isoformat()
    ts_file = datetime.now().strftime("%Y%m%d_%H%M%S")

    # ── JSONL запись ──────────────────────────────────────────────
    record = {
        "ts":           ts,
        "data_counts":  data_counts,
        "patterns":     patterns,
        "analysis":     analysis,
    }
    try:
        META_REFLECTION_LOG.parent.mkdir(parents=True, exist_ok=True)
        with META_REFLECTION_LOG.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as e:
        log.warning("[MetaReflection] Ошибка записи JSONL: %s", e)

    # ── Читаемый отчёт ────────────────────────────────────────────
    sep = "═" * 62
    report_lines = [
        sep,
        f"  МЕТА-РЕФЛЕКСИЯ SCALPEL — {ts[:19]}",
        sep,
        "",
        f"  БАЗА ЗНАНИЙ НА МОМЕНТ АНАЛИЗА:",
        *[f"    {k}: {v}" for k, v in data_counts.items()],
        "",
        sep,
        "  ПАТТЕРНЫ ОШИБОК",
        sep,
        analysis.get("errors", "(нет данных)"),
        "",
        sep,
        "  ПАТТЕРНЫ УСПЕХА",
        sep,
        analysis.get("successes", "(нет данных)"),
        "",
        sep,
        "  ПРАВИЛА НАВИГАЦИИ (для Navigator)",
        sep,
        analysis.get("nav_rules", "(нет данных)"),
        "",
        sep,
        "  ИТОГОВЫЙ ВЫВОД",
        sep,
        analysis.get("summary", "(нет данных)"),
        "",
        sep,
        "  ДЕТАЛИ: ТОП ОТКЛОНИТЕЛЕЙ",
        sep,
    ]

    for r in patterns.get("top_rejectors", []):
        report_lines.append(
            f"  {r['role']}: {r['count']}× — {', '.join(r['top_reasons'][:2])}"
        )

    report_lines += [
        "",
        sep,
        "  УСПЕШНЫЕ ОПЕРАТОРЫ",
        sep,
        ", ".join(f"{op}({c})" for op, c in patterns.get("successful_operators", [])[:6]),
        "",
        sep,
        "  CURRICULUM — РЕЗУЛЬТАТЫ ПО УРОВНЯМ",
        sep,
    ]

    for lv, v in sorted(patterns.get("curriculum_by_level", {}).items()):
        report_lines.append(
            f"  Уровень {lv}: {v['total']} датасетов | "
            f"pass_rate={v['pass_rate']:.0%} | avg_R²={v['avg_r2']:.3f}"
        )

    report_lines += ["", sep]
    report_text = "\n".join(report_lines)

    META_REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = META_REPORT_DIR / f"meta_report_{ts_file}.txt"
    try:
        report_path.write_text(report_text, encoding="utf-8")
    except Exception as e:
        log.warning("[MetaReflection] Ошибка записи отчёта: %s", e)

    return report_path


# ══════════════════════════════════════════════════════════════════
# ЗАГРУЗКА МЕТА-ПРИМЕРОВ ДЛЯ DSPy
# ══════════════════════════════════════════════════════════════════

def load_meta_examples() -> List[Any]:
    """
    Загружает последний мета-анализ и конвертирует в dspy.Example.
    Вызывается из DSPyOrchestrator.siege_compile().
    """
    if not META_REFLECTION_LOG.exists():
        return []
    try:
        lines = META_REFLECTION_LOG.read_text(encoding="utf-8").strip().splitlines()
        if not lines:
            return []
        # Берём последние 3 рефлексии (они накапливают мудрость)
        recent = []
        for line in lines[-3:]:
            try:
                recent.append(json.loads(line))
            except Exception:
                continue
        examples = []
        for rec in recent:
            analysis = rec.get("analysis", {})
            patterns = rec.get("patterns", {})
            exs = _meta_to_dspy_examples(analysis, patterns)
            examples.extend(exs)
        log.info("[MetaReflection] Загружено %d мета-примеров из %d рефлексий",
                 len(examples), len(recent))
        return examples
    except Exception as e:
        log.warning("[MetaReflection] Ошибка загрузки мета-примеров: %s", e)
        return []


# ══════════════════════════════════════════════════════════════════
# ГЛАВНАЯ ФУНКЦИЯ: проверка порога + запуск рефлексии
# ══════════════════════════════════════════════════════════════════

def check_and_reflect(
    model:       str = OLLAMA_MODEL,
    host:        str = OLLAMA_HOST,
    force:       bool = False,
) -> Optional[Path]:
    """
    Главная точка входа. Вызывается из engine.py после каждого запуска.

    1. Подсчитывает накопленные данные
    2. Проверяет достижение критической массы
    3. Если достигнута (и cooldown пройден) — запускает рефлексию
    4. Сохраняет выводы + генерирует DSPy примеры

    Возвращает путь к отчёту или None если рефлексия не нужна.
    """
    _increment_run_counter()
    state = _load_state()

    # Собираем счётчики
    data         = _collect_all_data()
    data_counts  = {
        "chronicle_steps":  len(data["chronicle_steps"]),
        "chronicle_finals": len(data["chronicle_finals"]),
        "curriculum":       len(data["curriculum"]),
        "gold_formulas":    len(data["gold_formulas"]),
        "failures":         len(data["failures"]),
        "scientific_cycles":len(data["scientific_cycles"]),
    }

    # Проверяем пороги
    thresholds_hit = (
        data_counts["chronicle_steps"]  >= THRESHOLD_CHRONICLE
        or data_counts["curriculum"]    >= THRESHOLD_CURRICULUM
        or data_counts["gold_formulas"] >= THRESHOLD_GOLD
        or data_counts["failures"]      >= THRESHOLD_FAILURES
    )

    runs_since = state.get("runs_since_last", 0)
    cooldown_ok = runs_since >= COOLDOWN_RUNS

    # Сравниваем с прошлым состоянием — нет смысла делать рефлексию
    # если данные не выросли значимо
    last_counts = state.get("last_counts", {})
    grew_enough = (
        data_counts["chronicle_steps"] > last_counts.get("chronicle_steps", 0) + 10
        or data_counts["gold_formulas"] > last_counts.get("gold_formulas", 0) + 2
        or data_counts["curriculum"]   > last_counts.get("curriculum", 0) + 5
    )

    should_reflect = force or (thresholds_hit and cooldown_ok and grew_enough)

    print(f"\n  [MetaReflection] Данные: хроника={data_counts['chronicle_steps']} "
          f"gold={data_counts['gold_formulas']} curriculum={data_counts['curriculum']}")
    print(f"  [MetaReflection] Порог: {'✓' if thresholds_hit else '✗'} | "
          f"Cooldown: {'✓' if cooldown_ok else f'✗ ({COOLDOWN_RUNS-runs_since} запусков)'} | "
          f"Рост: {'✓' if grew_enough else '✗'}")

    if not should_reflect:
        return None

    # ── Запускаем рефлексию ───────────────────────────────────────
    print(f"\n  [MetaReflection] ═══ КРИТИЧЕСКАЯ МАССА ДОСТИГНУТА ═══")
    print(f"  [MetaReflection] Запускаю мета-анализ по всей базе знаний…")
    t_start = time.time()

    try:
        from .navigator import ollama_chat

        def _ask(prompt: str) -> str:
            # FIX: SYNTHESIS_MODEL (Gemma) для аналитического синтеза паттернов
            # CHRONICLE_MODEL передаётся снаружи для нарратива (итоговый текст)
            _analysis_model = SYNTHESIS_MODEL if SYNTHESIS_MODEL else model
            return ollama_chat(
                prompt, model=_analysis_model, host=host,
                temperature=0.3, num_predict=1000,
            )

        # Аналитика (без LLM)
        patterns = _extract_patterns(data)

        # LLM синтез (с Ollama)
        analysis = _generate_meta_analysis(patterns, data, _ask)

        # Сохранение
        report_path = _save_reflection(patterns, analysis, data_counts)

        elapsed = time.time() - t_start
        print(f"\n  [MetaReflection] ✓ Завершено за {elapsed:.0f}с")
        print(f"  [MetaReflection] Отчёт: {report_path}")
        print(f"  [MetaReflection] Выводы:")
        for key, label in [("errors","Ошибки"), ("successes","Успехи"), ("nav_rules","Правила")]:
            val = analysis.get(key, "")
            if val:
                print(f"    {label}: {val[:100]}…")

        # Обновляем состояние
        state["runs_since_last"]  = 0
        state["total_reflections"] = state.get("total_reflections", 0) + 1
        state["last_ts"]          = datetime.now().isoformat()
        state["last_counts"]      = data_counts
        _save_state(state)

        # Инвалидируем DSPy кэш — новые мета-примеры должны попасть немедленно
        try:
            from .config import DSPY_COMPILED_PATH
            count_path = DSPY_COMPILED_PATH.parent / "chronicle_count_at_compile.json"
            if count_path.exists():
                count_path.unlink()
                log.info("[MetaReflection] DSPy кэш инвалидирован")
        except Exception:
            pass

        gc.collect()
        return report_path

    except Exception as e:
        log.warning("[MetaReflection] Ошибка рефлексии: %s", e)
        return None


# ══════════════════════════════════════════════════════════════════
# CLI: принудительный запуск через --reflect
# ══════════════════════════════════════════════════════════════════

def run_forced_reflection(
    model: str = OLLAMA_MODEL,
    host:  str = OLLAMA_HOST,
) -> Optional[Path]:
    """Принудительный запуск мета-рефлексии (--reflect флаг в cli.py)."""
    print("\n" + "═" * 62)
    print("  SCALPEL META-REFLECTION — Принудительный анализ")
    print("═" * 62)
    return check_and_reflect(model=model, host=host, force=True)
