"""
pre_pysr_debate.py — Дебаты перед PySR v1.0 (уровень 4, пункт 4)

Один общий раунд где роли договариваются ДО запуска PySR.
Navigator видит итог дебатов и обновляет гипотезы/операторы.

RAM-безопасность:
  Роли загружаются ПОСЛЕДОВАТЕЛЬНО — никогда одновременно.
  Пик RAM в любой момент: ~9 ГБ (phi4:14b) + ~5 ГБ система = ~14 ГБ.
  На 32 ГБ остаётся ~18 ГБ для Julia/PySR → безопасно.

Hard timeout 240с на роль → максимум ~15 минут на всё (3 роли + синтез).

Структура:
  Round 1: Прагматик → Мистик → Скептик (каждый видит предыдущих)
  Round 2: только если Скептик поднял реальный конфликт
  Синтез:  gemma2:9b собирает консенсус → рекомендации Navigator
  Navigator обновляет hypotheses + selected_operators
"""
from __future__ import annotations

import gc
import json
import logging
import re
import time
import urllib.request
import urllib.error
from dataclasses import dataclass, field
from typing import List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .shared_context import SharedContext

log = logging.getLogger("scalpel")

try:
    from .config import OLLAMA_HOST, ROLE_MODELS, SYNTHESIS_MODEL
    _HOST = OLLAMA_HOST
    _ROLES = {
        "Прагматик": ROLE_MODELS.get("Прагматик", "mistral:7b"),
        "Мистик":    ROLE_MODELS.get("Мистик",    "llama3:8b"),
        "Скептик":   ROLE_MODELS.get("Скептик",   "qwen2.5:14b"),
    }
    _SYNTH_MODEL = SYNTHESIS_MODEL
except ImportError:
    # FIX v10.27: Прагматик granite3-moe:3b → mistral:7b (слишком мал, галлюцинировал)
    _HOST = "http://localhost:11434"
    _ROLES = {
        "Прагматик": "mistral:7b",
        "Мистик":    "llama3:8b",
        "Скептик":   "qwen2.5:14b",
    }
    _SYNTH_MODEL = "gemma2:9b"

ROLE_TIMEOUT    = 240   # FIX v10.29: solar:10.7b (Скептик) на CPU ~80-130с + загрузка ~15с → 240с запас ×1.7
SYNTH_TIMEOUT   = 120   # gemma2:9b (Delphi) ~34-38с → 120с с запасом
MAX_ROUNDS      = 2     # максимум раундов дебатов
CONFLICT_WORDS  = (     # признаки реального конфликта у Скептика
    "не согласен", "ошибк", "неверн", "проблем", "сомнева",
    "disagree", "wrong", "incorrect", "flawed", "concern",
)

# ═══════════════════════════════════════════════════════════════════
# ВАЛИДАТОР ФОРМУЛ (FIX v10.31)
# ═══════════════════════════════════════════════════════════════════

def _is_valid_formula(h: str) -> bool:
    """
    Проверяет что строка является математической формулой а не текстом.

    Правила:
      1. Должна содержать хотя бы одну переменную f0..f9
      2. Должна содержать математический оператор или функцию
      3. Не должна быть чистым текстом (без цифр и операторов)

    Примеры:
      "exp(-f0**2)"        → True  ✅
      "f0/sqrt(f1**2+f2**2)"→ True ✅
      "hypothesis based on Planck's equation" → False ❌
      "thermodynamic phase transition"        → False ❌
    """
    h = h.strip()
    if not h:
        return False
    # Должна содержать переменную f0..f9
    has_feature = bool(re.search(r'\bf\d+\b', h))
    if not has_feature:
        return False
    # Должна содержать математический оператор или функцию
    has_math = bool(re.search(
        r'[+\-*/^]|sqrt|log|exp|abs|sin|cos|tan|f\d+\s*\*\*\s*\d',
        h
    ))
    if not has_math:
        return False
    # Не должна содержать слова (текст без мат. контекста)
    word_count = len(re.findall(r'[a-zA-Zа-яА-Я]{4,}', h))
    math_tokens = len(re.findall(r'f\d+|[+\-*/()^]|sqrt|log|exp|abs|sin|cos', h))
    if word_count > math_tokens:
        return False
    return True


# ═══════════════════════════════════════════════════════════════════
# СТРУКТУРЫ ДАННЫХ
# ═══════════════════════════════════════════════════════════════════

@dataclass
class DebateStatement:
    role:    str
    round_n: int
    text:    str
    elapsed: float

    def short(self) -> str:
        return self.text[:200].replace("\n", " ").strip()


@dataclass
class DebateResult:
    statements:      List[DebateStatement] = field(default_factory=list)
    synthesis:       str                   = ""
    nav_ops_add:     List[str]             = field(default_factory=list)
    nav_hyps_update: List[str]             = field(default_factory=list)
    conflict_found:  bool                  = False
    rounds_run:      int                   = 0
    total_elapsed:   float                 = 0.0
    skipped:         bool                  = False

    def as_failure_log_entry(self) -> dict:
        """Форматирует итог для failure_logs Navigator."""
        if self.skipped or not self.synthesis:
            return {}
        return {
            "hypothesis":   "[ДЕБАТЫ]",
            "death_reason": f"Предебатный консенсус: {self.synthesis[:300]}",
            "source":       "pre_pysr_debate",
        }

    def summary_lines(self) -> List[str]:
        lines = [f"  Дебаты: {self.rounds_run} раунд(а), {self.total_elapsed:.0f}с"]
        for s in self.statements:
            lines.append(f"    [{s.role} R{s.round_n}] {s.short()[:120]}")
        if self.synthesis:
            lines.append(f"  Синтез: {self.synthesis[:200]}")
        if self.nav_ops_add:
            lines.append(f"  +Операторы: {self.nav_ops_add}")
        if self.nav_hyps_update:
            lines.append(f"  +Гипотезы: {self.nav_hyps_update}")
        return lines


# ═══════════════════════════════════════════════════════════════════
# LLM ВЫЗОВ (один, с таймаутом)
# ═══════════════════════════════════════════════════════════════════

def _call_role(model: str, prompt: str, timeout: int = ROLE_TIMEOUT) -> str:
    """Вызывает Ollama модель с жёстким таймаутом."""
    try:
        payload = json.dumps({
            "model":  model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.4,
                "num_predict": 200,   # коротко — это дебаты, не эссе
            },
        }).encode()
        req = urllib.request.Request(
            f"{_HOST}/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read()).get("response", "").strip()
    except Exception as e:
        log.warning("[Дебаты/%s] Таймаут/ошибка: %s", model, e)
        return ""


def _unload_model(model: str) -> None:
    """Выгружает модель из RAM после использования."""
    try:
        payload = json.dumps({
            "model": model, "keep_alive": 0, "prompt": "",
        }).encode()
        req = urllib.request.Request(
            f"{_HOST}/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        urllib.request.urlopen(req, timeout=5)
    except Exception:
        pass
    gc.collect()


# ═══════════════════════════════════════════════════════════════════
# ПРОМПТЫ РОЛЕЙ
# ═══════════════════════════════════════════════════════════════════

def _prompt_pragmatist(nav_proposal: dict, data_info: str, prev: str) -> str:
    return f"""You are the Pragmatist in a symbolic regression team.
Navigator proposes this before running PySR:
  Hypotheses: {nav_proposal.get('hypotheses', [])[:3]}
  Operators:  {nav_proposal.get('operators', [])}
  Features:   {nav_proposal.get('features', [])}

Data context: {data_info}
{("Previous statements:\n" + prev) if prev else ""}

Your role: evaluate PRACTICAL feasibility.
- Will these hypotheses converge in PySR?
- Are the operators computationally stable?
- Any obvious improvements?

Reply in 2-3 sentences in Russian. Be concrete, not abstract."""


def _prompt_mystic(nav_proposal: dict, data_info: str, prev: str) -> str:
    return f"""You are the Mystic in a symbolic regression team.
Navigator proposes:
  Hypotheses: {nav_proposal.get('hypotheses', [])[:3]}
  Operators:  {nav_proposal.get('operators', [])}

Data context: {data_info}
{("Previous statements:\n" + prev) if prev else ""}

Your role: find NON-OBVIOUS analogies or structures.
- What physical law does this remind you of?
- What alternative structure might fit?
- Any symmetry or invariant worth exploring?

Reply in 2-3 sentences in Russian. Suggest a specific hypothesis if you have one."""


def _prompt_skeptic(nav_proposal: dict, data_info: str, prev: str) -> str:
    return f"""You are the Skeptic in a symbolic regression team.
Navigator proposes:
  Hypotheses: {nav_proposal.get('hypotheses', [])[:3]}
  Operators:  {nav_proposal.get('operators', [])}

Data context: {data_info}
Previous statements:
{prev}

Your role: find WEAKNESSES in the proposal and the previous statements.
- What assumption is wrong?
- Which hypothesis is most likely to fail and why?
- Do you disagree with Pragmatist or Mystic? Say so explicitly.

Reply in 2-3 sentences in Russian. Be critical but constructive."""


def _prompt_synthesis(nav_proposal: dict, statements: List[DebateStatement]) -> str:
    debate_text = "\n".join(
        f"  {s.role} (R{s.round_n}): {s.short()}"
        for s in statements
    )
    return f"""You are Delphi, the synthesis role.
The team just debated Navigator's proposal before PySR:

NAVIGATOR PROPOSAL:
  Hypotheses: {nav_proposal.get('hypotheses', [])[:3]}
  Operators:  {nav_proposal.get('operators', [])}

DEBATE:
{debate_text}

Synthesize the debate into actionable recommendations for Navigator.
Reply ONLY with JSON:
{{
  "consensus": "one sentence summary of what the team agreed on",
  "add_operators": ["op1", "op2"],
  "add_hypotheses": ["h1", "h2"],
  "remove_operators": ["op3"],
  "confidence": "high/medium/low"
}}
Only include add_operators/add_hypotheses if team SPECIFICALLY suggested them.
Operators must be from: +, -, *, /, sqrt, log, exp, abs, sin, cos, tanh"""


# ═══════════════════════════════════════════════════════════════════
# ГЛАВНАЯ ФУНКЦИЯ
# ═══════════════════════════════════════════════════════════════════

def run_pre_pysr_debate(
    nav_hypotheses:   List[str],
    nav_operators:    List[str],
    nav_features:     List[str],
    shadow_names:     List[str],
    dim_codes:        List[int],
    y_stats:          dict,
    ctx:              "SharedContext" = None,
    host:             str             = _HOST,
) -> DebateResult:
    """
    Запускает раунд дебатов ДО PySR.
    Возвращает DebateResult с рекомендациями для Navigator.

    RAM-безопасность: каждая модель загружается и выгружается по очереди.
    """
    result = DebateResult()
    t_total = time.time()

    nav_proposal = {
        "hypotheses": nav_hypotheses,
        "operators":  nav_operators,
        "features":   nav_features,
    }

    # Контекст данных для всех ролей
    _DIM_NAMES = {0:"dimensionless",1:"unknown",2:"length",3:"mass",
                  4:"temperature",5:"count",6:"force",8:"time",10:"price"}
    dim_desc = ", ".join(
        f"{n}({_DIM_NAMES.get(d,'?')})"
        for n, d in zip(shadow_names, dim_codes or [0]*len(shadow_names))
    )
    _has_inf = y_stats.get("has_inf", False)
    _inf_warn = " ⚠️ DATA HAS INF/NaN — DO NOT suggest sin/cos/tanh" if _has_inf else ""
    data_info = (
        f"features=[{dim_desc}], "
        f"ratio={y_stats.get('ratio', '?'):.1f}, "
        f"negative={y_stats.get('negative_fraction', 0):.1%}, "
        f"n={y_stats.get('n_samples', '?')}"
        f"{_inf_warn}"
    )

    print(f"\n  {'─'*58}")
    print(f"  ПРЕДЕБАТНЫЙ РАУНД (перед PySR)")
    print(f"  {'─'*58}")

    # ── РАУНДЫ ДЕБАТОВ ───────────────────────────────────────────
    for round_n in range(1, MAX_ROUNDS + 1):
        result.rounds_run = round_n

        # Собираем предыдущие высказывания для контекста
        prev_text = "\n".join(
            f"  {s.role}: {s.short()}"
            for s in result.statements
            if s.round_n == round_n - 1
        ) if round_n > 1 else ""

        round_had_output = False

        for role_name, prompt_fn in [
            ("Прагматик", _prompt_pragmatist),
            ("Мистик",    _prompt_mystic),
            ("Скептик",   _prompt_skeptic),
        ]:
            model = _ROLES[role_name]

            # Текущий контекст = все высказывания этого раунда до сих пор
            current_prev = "\n".join(
                f"  {s.role}: {s.short()}"
                for s in result.statements
                if s.round_n == round_n
            )

            prompt = prompt_fn(nav_proposal, data_info, current_prev or prev_text)

            print(f"  [{role_name}] думает…", end="", flush=True)
            t0 = time.time()
            text = _call_role(model, prompt, timeout=ROLE_TIMEOUT)
            elapsed = time.time() - t0

            _unload_model(model)   # выгружаем сразу после ответа

            if text:
                stmt = DebateStatement(
                    role=role_name, round_n=round_n,
                    text=text, elapsed=elapsed,
                )
                result.statements.append(stmt)
                round_had_output = True
                print(f" {elapsed:.0f}с ✓")
                print(f"  [{role_name}]: {stmt.short()[:140]}")

                # Пишем в SharedContext
                if ctx is not None:
                    ctx._log("debate", f"{role_name}_R{round_n}", stmt.short()[:200])
            else:
                print(f" таймаут ({ROLE_TIMEOUT}с)")

        # Проверяем нужен ли раунд 2
        if round_n == 1:
            skeptic_stmts = [s for s in result.statements if s.role == "Скептик"]
            if skeptic_stmts:
                last_skeptic = skeptic_stmts[-1].text.lower()
                result.conflict_found = any(w in last_skeptic for w in CONFLICT_WORDS)
            if not result.conflict_found:
                log.info("[Дебаты] Конфликта нет — второй раунд пропущен")
                break
            else:
                print(f"\n  [Дебаты] Конфликт обнаружен → раунд 2")
        
        if not round_had_output:
            break

    # ── СИНТЕЗ ───────────────────────────────────────────────────
    if result.statements:
        print(f"\n  [Синтез/Delphi] Формирую рекомендации…", end="", flush=True)
        synth_prompt = _prompt_synthesis(nav_proposal, result.statements)
        t0 = time.time()
        synth_raw = _call_role(_SYNTH_MODEL, synth_prompt, timeout=SYNTH_TIMEOUT)
        synth_elapsed = time.time() - t0
        _unload_model(_SYNTH_MODEL)
        print(f" {synth_elapsed:.0f}с")

        if synth_raw:
            # Парсим JSON из синтеза
            try:
                start = synth_raw.find("{")
                end   = synth_raw.rfind("}") + 1
                if start >= 0 and end > start:
                    synth_data = json.loads(synth_raw[start:end])
                    result.synthesis = synth_data.get("consensus", "")

                    # Фильтруем операторы — только допустимые
                    allowed_ops = {"+","-","*","/","sqrt","log","exp","abs","sin","cos","tanh"}
                    raw_add_ops = synth_data.get("add_operators", [])
                    result.nav_ops_add = [o for o in raw_add_ops if o in allowed_ops]

                    # Фильтруем гипотезы — только ссылающиеся на реальные признаки
                    import re
                    valid_features = set(shadow_names)
                    raw_hyps = synth_data.get("add_hypotheses", [])
                    for h in raw_hyps:
                        refs = set(re.findall(r'f\d+', h))
                        if refs.issubset(valid_features) or not refs:
                            if _is_valid_formula(h):  # FIX v10.31: только валидные формулы
                                result.nav_hyps_update.append(h)
                            else:
                                log.debug("[Дебаты/Синтез] Отброшена невалидная гипотеза: '%s'", h[:80])

                    # Убираем операторы которые Скептик хотел убрать
                    remove_ops = set(synth_data.get("remove_operators", []))
                    if remove_ops:
                        result.nav_ops_add = [o for o in result.nav_ops_add
                                              if o not in remove_ops]
            except Exception as e:
                log.debug("[Дебаты/Синтез] JSON parse: %s", e)
                result.synthesis = synth_raw[:300]

        if result.synthesis:
            print(f"  [Синтез] {result.synthesis[:180]}")

    result.total_elapsed = time.time() - t_total

    # Итог в SharedContext
    if ctx is not None:
        ctx._log("debate", "complete",
                 f"rounds={result.rounds_run} ops+={result.nav_ops_add} "
                 f"hyps+={result.nav_hyps_update[:2]} t={result.total_elapsed:.0f}s")

    print(f"  {'─'*58}")
    print(f"  Дебаты завершены за {result.total_elapsed:.0f}с")
    if result.nav_ops_add:
        print(f"  +Операторы: {result.nav_ops_add}")
    if result.nav_hyps_update:
        print(f"  +Гипотезы: {result.nav_hyps_update}")

    gc.collect()
    return result
