"""
oracle.py — Oracle: постоянный мозг v10.14.

Oracle создаётся один раз в начале run_engine() и живёт всю сессию.
Он видит КАЖДУЮ итерацию HADI, помнит ВСЁ что произошло в текущем прогоне,
и подсказывает Navigator что попробовать дальше.

ОТЛИЧИЕ от других компонентов:
  Navigator      — знает итерацию + прошлые сессии (DSPy)
  Meta-context   — пассивно добавляет текст в промпты
  Meta-reflection — запускается редко, по критической массе
  Oracle         — ЖИВЁТ весь прогон, АКТИВНО думает, помнит СЕЙЧАС

ПРИНЦИП РАБОТЫ:
  1. observe(attempt, tried, r2, rejected_by, reason)
     → накапливает картину текущей сессии
  2. suggest(data_meta, dim_codes, domain)
     → смотрит историю + память + инварианты
     → если нужно: 1 LLM вызов через SYNTHESIS_MODEL (Gemma)
     → возвращает строку-подсказку для failure_logs
  3. learn(formula, verdict, r2, domain)
     → запоминает исход для следующей итерации
  4. finalize()
     → записывает выводы сессии в episodic_memory

RAM: использует ORACLE_MODEL (Qwen) — ДРУГАЯ модель чем Delphi (Gemma).
     Это намеренно: Oracle советует, Delphi синтезирует — разные семьи.
     Вызывается не чаще 1 раза за HADI итерацию.
     1 LLM вызов ≈ 400 токенов ≈ 2-3 секунды.
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

from .config import SYNTHESIS_MODEL, ORACLE_MODEL, OLLAMA_HOST, VAULT_DIR

log = logging.getLogger("scalpel")


def _load_preparator_stats() -> dict:
    """
    Читает preparator_log.jsonl и строит статистику:
    при каких параметрах (ratio, frac_positive) какая трансформация
    давала успех (R² > 0.5) или провал.

    Возвращает словарь с рекомендациями для Препаратора.
    """
    try:
        log_path = VAULT_DIR / "preparator_log.jsonl"
        if not log_path.exists():
            return {}

        records = []
        with open(log_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except Exception:
                        pass

        if len(records) < 3:
            return {"total": len(records), "note": "мало данных"}

        # Группируем по трансформации и результату
        stats: Dict[str, Dict] = {}
        for r in records:
            t = r.get("transform", "none")
            result = r.get("result", "unknown")
            ratio = r.get("ratio", 0)
            pos = r.get("frac_positive", 1)

            if t not in stats:
                stats[t] = {"success": 0, "fail": 0, "ratios": []}
            if result == "SUCCESS":
                stats[t]["success"] += 1
            elif result in ("FAIL", "REJECTED"):
                stats[t]["fail"] += 1
            stats[t]["ratios"].append(ratio)

        # Формируем рекомендации
        recommendations = []
        for t, s in stats.items():
            total = s["success"] + s["fail"]
            if total > 0:
                rate = s["success"] / total * 100
                avg_ratio = sum(s["ratios"]) / len(s["ratios"]) if s["ratios"] else 0
                recommendations.append(
                    f"{t}: успех {rate:.0f}% из {total} попыток "
                    f"(средний ratio={avg_ratio:.1f})"
                )

        return {
            "total": len(records),
            "stats": stats,
            "recommendations": recommendations,
        }
    except Exception as e:
        log.debug("[Oracle] preparator_stats: %s", e)
        return {}


# Порог: Oracle советует через LLM только если достаточно попыток
ORACLE_LLM_THRESHOLD = 2     # после 2-й неудачной попытки включаем LLM
ORACLE_MAX_LLM_CALLS = 3     # не более 3 LLM вызовов за сессию


# ══════════════════════════════════════════════════════════════════
# СОСТОЯНИЕ ОДНОЙ ПОПЫТКИ
# ══════════════════════════════════════════════════════════════════

@dataclass
class AttemptRecord:
    attempt:     int
    tried:       str
    r2:          float
    rejected_by: str  = ""
    reason:      str  = ""
    verdict:     str  = ""  # ПРИНЯТА / ОТКЛОНЕНА / DEATH


# ══════════════════════════════════════════════════════════════════
# ORACLE — ПОСТОЯННЫЙ МОЗГ
# ══════════════════════════════════════════════════════════════════

class Oracle:
    """
    Постоянный мозг сессии. Живёт от первого до последнего HADI шага.

    Пример использования в engine.py:
        oracle = Oracle(model=SYNTHESIS_MODEL, host=host)
        oracle.load_context(data_meta, dim_codes, domain_type)

        for attempt in range(HADI_MAX_RETRIES + 1):
            hint = oracle.suggest(attempt, failure_logs)
            if hint:
                failure_logs = _inject_oracle_hint(failure_logs, hint)
            ...
            oracle.observe(attempt, formula, r2, rejected_by, reason)

        oracle.finalize(best_formula, best_r2, domain)
    """

    def __init__(
        self,
        model: str = SYNTHESIS_MODEL,
        host:  str = OLLAMA_HOST,
    ):
        self.model  = model
        self.host   = host
        self._llm_calls = 0

        # Контекст текущей сессии
        self._data_meta:   str        = ""
        self._dim_codes:   List[int]  = []
        self._domain:      str        = ""
        self._feat_names:  List[str]  = []

        # История попыток в текущей сессии
        self._attempts:    List[AttemptRecord] = []

        # Знания из прошлых сессий (загружаются при инициализации)
        self._inv_hints:        List[str] = []  # из invariants_learned
        self._meta_rules:       str       = ""  # из meta_reflection
        self._disc_summary:     str       = ""  # из discoveries
        self._chronicle_paths:  List[str] = []  # нарративы Летописца
        self._rejected_warnings: List[str] = []  # v10.15: СТОП-паттерны Матрёшки
        self._disputed_warnings: List[str] = []  # v10.15: спорные (хороший R², но LLM усомнилась)

        # Стратегические выводы Oracle за сессию
        self._oracle_log:  List[Dict] = []

        # Флаги
        self._initialized  = False
        self._session_start = time.time()

    # ── Инициализация ─────────────────────────────────────────────

    def load_context(
        self,
        data_meta:  str,
        dim_codes:  List[int],
        domain:     str       = "",
        feat_names: List[str] = None,
    ) -> None:
        """
        Загружает контекст задачи + знания из памяти.
        Вызывается один раз в начале run_engine().
        """
        self._data_meta  = data_meta
        self._dim_codes  = list(dim_codes)
        self._domain     = domain
        self._feat_names = list(feat_names or [])

        # Загружаем знания из прошлых сессий
        try:
            from .episodic_memory import get_memory
            self._inv_hints = get_memory().recall_invariants_for_domain(
                dim_codes=dim_codes, domain=domain, limit=5
            )
        except Exception as e:
            log.debug("[Oracle] invariants: %s", e)

        try:
            from .meta_context import get_navigator_context
            self._meta_rules = get_navigator_context(domain)[:300]
        except Exception as e:
            log.debug("[Oracle] meta_context: %s", e)

        # Читаем нарративы Летописца — пути к успешным формулам
        self._chronicle_paths: list = []
        try:
            from .episodic_memory import get_memory as _cm
            self._chronicle_paths = _cm().recall_chronicle_paths(
                domain=domain, limit=3
            )
        except Exception as e:
            log.debug("[Oracle] chronicle_paths: %s", e)

        try:
            from .discovery import load_discoveries_summary
            self._disc_summary = load_discoveries_summary()[:200]
        except Exception as e:
            log.debug("[Oracle] discoveries: %s", e)

        # v10.22: Загружаем статистику Препаратора для обучения
        self._preparator_stats = _load_preparator_stats()
        if self._preparator_stats:
            log.info("[Oracle] Препаратор: загружено %d записей опыта",
                     self._preparator_stats.get("total", 0))

        # v10.20: Загружаем историю решений Navigator по этому домену
        self._nav_decision_history: list = []
        try:
            import json as _ndl_j
            _ndl_path = VAULT_DIR / "navigator_decisions_log.jsonl"
            if _ndl_path.exists():
                _ndl_lines = _ndl_path.read_text(encoding="utf-8").strip().splitlines()
                _same_dim = set(dim_codes)
                for _line in _ndl_lines[-50:]:   # последние 50 записей
                    try:
                        _entry = _ndl_j.loads(_line)
                        # Берём только успешные (ПРИНЯТА) для этого домена
                        if (_entry.get("matryoshka_consensus") == "ПРИНЯТА"
                                and _entry.get("r2_blind", 0) >= 0.80):
                            self._nav_decision_history.append(_entry)
                    except Exception:
                        continue
                # Сортируем по R² — лучшие первые
                self._nav_decision_history.sort(
                    key=lambda e: e.get("r2_blind", 0), reverse=True
                )
                self._nav_decision_history = self._nav_decision_history[:10]
            if self._nav_decision_history:
                log.info("[Oracle] Загружено %d успешных решений Navigator из лога",
                         len(self._nav_decision_history))
        except Exception as _ndl_err:
            log.debug("[Oracle] navigator_decisions_log: %s", _ndl_err)

        # Загружаем СВОЁ прошлое — что советовал раньше для тех же dim_codes
        try:
            past = Oracle.load_past_sessions(limit=20)
            same_dim = ",".join(str(d) for d in sorted(set(dim_codes)))
            relevant = [s for s in past
                        if ",".join(str(d) for d in sorted(set(s.get("dim_codes",[]))))
                           == same_dim and s.get("passed")]
            if relevant:
                best = max(relevant, key=lambda s: s.get("final_r2", 0))
                hint = (f"Для dim={dim_codes} прошлая успешная сессия: "
                        f"формула={best.get('formula_final','?')[:40]} "
                        f"R²={best.get('final_r2',0):.3f} "
                        f"за {best.get('n_attempts','?')} попыток")
                if hint not in self._inv_hints:
                    self._inv_hints.insert(0, hint)  # ставим первым
        except Exception as e:
            log.debug("[Oracle] past sessions: %s", e)

        # v10.15: Загружаем СТОП-паттерны — отклонённые Матрёшкой формулы
        # Oracle предупреждает Navigator: эти структуры проверены и отвергнуты
        self._rejected_warnings: list = []
        self._disputed_warnings: list = []  # мягкие — хороший R², но LLM усомнилась
        try:
            from .config import REJECTED_PATH as _rp, DISPUTED_PATH as _dp
            import json as _json
            # Жёсткие стопы (плохой R²)
            if _rp.exists():
                with _rp.open(encoding="utf-8") as _rf:
                    _rdata = _json.load(_rf)
                for _r in _rdata.get("formulas", [])[-15:]:
                    _skel  = _r.get("skeleton", "")
                    _by    = _r.get("rejected_by", "Матрёшка")
                    _rsn   = _r.get("reason", "")[:80]
                    _r2t   = _r.get("r2_train", 0.0)
                    _r2b   = _r.get("r2_blind", 0.0)
                    if _skel:
                        self._rejected_warnings.append(
                            f"⛔ skeleton '{_skel}' отклонён {_by}: {_rsn} "
                            f"(train={_r2t:.3f} vs blind={_r2b:.3f} — переобучение)"
                        )
            # Мягкие (спорные — хороший R², но LLM усомнилась)
            if _dp.exists():
                with _dp.open(encoding="utf-8") as _df:
                    _ddata = _json.load(_df)
                for _d in _ddata.get("formulas", [])[-10:]:
                    _skel  = _d.get("skeleton", "")
                    _rsn   = _d.get("reason", "")[:80]
                    _r2b   = _d.get("r2_blind", 0.0)
                    _unani = _d.get("unanimous", False)
                    if _skel:
                        self._disputed_warnings.append(
                            f"⚠️  skeleton '{_skel}' спорная (R²_blind={_r2b:.3f} хороший, "
                            f"но LLM отклонила{'единогласно' if _unani else ''}): {_rsn}. "
                            f"Можно переформулировать."
                        )
        except Exception as e:
            log.debug("[Oracle] rejected/disputed patterns: %s", e)

        if self._rejected_warnings:
            print(f"  [Oracle] ⛔ {len(self._rejected_warnings)} СТОП-паттернов"
                  f" (плохой R², отклонены Матрёшкой)")
        if self._disputed_warnings:
            print(f"  [Oracle] ⚠️  {len(self._disputed_warnings)} спорных"
                  f" (хороший R², но LLM усомнилась — не банить)")

        self._initialized = True
        hints_n = len(self._inv_hints)
        log.info("[Oracle] Инициализирован: domain=%s dim=%s hints=%d",
                 domain, dim_codes, hints_n)

        if hints_n:
            print(f"  [Oracle] Знаю {hints_n} подсказок для dim={dim_codes}")
        if self._meta_rules:
            print(f"  [Oracle] Загружены правила навигации из прошлых сессий")

    # ── Наблюдение за попыткой ────────────────────────────────────

    def observe(
        self,
        attempt:     int,
        tried:       str,
        r2:          float,
        rejected_by: str = "",
        reason:      str = "",
        verdict:     str = "",
    ) -> None:
        """
        Запоминает результат попытки.
        Вызывается ПОСЛЕ каждого HADI шага.
        """
        rec = AttemptRecord(
            attempt     = attempt,
            tried       = tried,
            r2          = r2,
            rejected_by = rejected_by,
            reason      = reason,
            verdict     = verdict,
        )
        self._attempts.append(rec)
        log.debug("[Oracle] Попытка %d: %s R²=%.3f вердикт=%s",
                  attempt, tried[:40], r2, verdict or "?")

    # ── Совет Navigator ───────────────────────────────────────────

    def suggest(
        self,
        attempt: int,
        failure_logs: str = "[]",
    ) -> Optional[str]:
        """
        Возвращает стратегический совет для текущей итерации.
        Добавляется в failure_logs → Navigator видит его.

        Уровни мышления:
          Уровень 1 (быстро, без LLM): инвариантные подсказки из памяти
          Уровень 2 (медленно, LLM):   стратегический анализ через Gemma

        Возвращает строку-подсказку или None.
        """
        if not self._initialized:
            return None

        # Уровень 1: быстрые подсказки из инвариантов (без LLM)
        fast_hint = self._fast_hint(attempt)
        if fast_hint:
            return fast_hint

        # Уровень 2: LLM если достаточно попыток и лимит не исчерпан
        if (attempt >= ORACLE_LLM_THRESHOLD
                and self._llm_calls < ORACLE_MAX_LLM_CALLS
                and len(self._attempts) >= 1):
            return self._llm_hint(attempt)

        return None

    def _fast_hint(self, attempt: int) -> Optional[str]:
        """Подсказки из памяти — без LLM, мгновенно."""

        # v10.22: Статистика Препаратора — что работало раньше
        if attempt == 0 and self._preparator_stats.get("recommendations"):
            recs = self._preparator_stats["recommendations"]
            total = self._preparator_stats.get("total", 0)
            if total >= 3:  # достаточно данных для рекомендации
                rec_str = " | ".join(recs[:3])
                return (
                    f"[ORACLE/preparator] Опыт Препаратора ({total} законов): {rec_str}. "
                    f"Учитывай при выборе гипотез."
                )

        # Если есть подсказки из инвариантов — отдаём первую на attempt=0
        if attempt == 0 and self._inv_hints:
            hint = self._inv_hints[0]
            return f"[ORACLE/memory] Из прошлых сессий: {hint[:120]}"

        # Если первая попытка была с высоким R² но отклонили — подсказываем
        if self._attempts and attempt == 1:
            last = self._attempts[-1]
            if last.r2 > 0.70 and last.rejected_by:
                return (
                    f"[ORACLE/fast] Попытка {last.attempt}: R²={last.r2:.2f} "
                    f"отклонил {last.rejected_by} ({last.reason[:60]}). "
                    f"Попробуй изменить только структуру, не признаки."
                )

        # Если все попытки дают низкий R² — возможно неверные признаки
        if len(self._attempts) >= 2:
            avg_r2 = sum(a.r2 for a in self._attempts) / len(self._attempts)
            if avg_r2 < 0.50:
                tops = self._inv_hints[1:3] if len(self._inv_hints) > 1 else []
                if tops:
                    return f"[ORACLE/fast] Низкий R² (<0.5). Альтернативы: {'; '.join(t[:60] for t in tops)}"

        # Летописец: если есть нарратив пути — подсказываем на 3-й попытке
        if attempt == 2 and self._chronicle_paths:
            path = self._chronicle_paths[0]
            return f"[ORACLE/chronicle] Похожий прошлый путь: {path[:150]}"

        # v10.20: история решений Navigator — показываем лучшую гипотезу из прошлого
        if attempt == 0 and self._nav_decision_history:
            best_entry = self._nav_decision_history[0]
            best_hyps  = best_entry.get("navigator_decision", {}).get("hypotheses", [])
            best_r2    = best_entry.get("r2_blind", 0)
            best_ops   = best_entry.get("navigator_decision", {}).get("selected_operators", [])
            if best_hyps:
                return (
                    f"[ORACLE/navlog] Прошлый успех R²={best_r2:.3f}: "
                    f"Navigator брал гипотезы {best_hyps[:3]} "
                    f"с операторами {best_ops[:4]}. Попробуй похожую структуру."
                )

        return None

    def _llm_hint(self, attempt: int) -> Optional[str]:
        """Стратегический совет через LLM (Gemma/SYNTHESIS_MODEL)."""
        try:
            from .navigator import ollama_chat

            # Строим картину текущей сессии
            attempts_str = "\n".join(
                f"  Попытка {a.attempt}: {a.tried[:40]} R²={a.r2:.3f}"
                + (f" отклонил {a.rejected_by}: {a.reason[:40]}" if a.rejected_by else "")
                for a in self._attempts[-4:]
            )

            inv_str = "\n".join(f"  {h[:80]}" for h in self._inv_hints[:3])
            # v10.20: добавляем историю успешных решений Navigator
            nav_hist_str = ""
            if self._nav_decision_history:
                nav_hist_lines = []
                for _ne in self._nav_decision_history[:3]:
                    _nh = _ne.get("navigator_decision", {})
                    nav_hist_lines.append(
                        f"  R²={_ne.get('r2_blind',0):.3f}: "
                        f"гипотезы={_nh.get('hypotheses',[])[:3]} "
                        f"операторы={_nh.get('selected_operators',[])[:4]} "
                        f"reasoning={_nh.get('reasoning','')[:60]}"
                    )
                nav_hist_str = "\n".join(nav_hist_lines)
            rules_str = self._meta_rules[:200] if self._meta_rules else "нет"
            # Нарративы Летописца — путь к прошлым успехам
            chronicle_str = "\n".join(f"  {p[:100]}" for p in self._chronicle_paths[:2])
            # v10.15: СТОП-паттерны — отклонённые Матрёшкой
            rejected_str = "\n".join(f"  {w[:120]}" for w in self._rejected_warnings[:5])
            # v10.15: спорные — хороший R², но LLM усомнилась (мягкое предупреждение)
            disputed_str = "\n".join(f"  {w[:120]}" for w in self._disputed_warnings[:4])

            prompt = (
                f"Ты — стратегический советник системы символьной регрессии.\n"
                f"Анализируй текущий поиск и дай ОДИН конкретный совет.\n\n"
                f"ЗАДАЧА: {self._data_meta[:200]}\n"
                f"Размерности признаков: {self._dim_codes}\n"
                f"Домен: {self._domain or 'неизвестен'}\n\n"
                f"ИСТОРИЯ ПОПЫТОК:\n{attempts_str}\n\n"
                f"ИЗВЕСТНЫЕ ПАТТЕРНЫ ДЛЯ ЭТИХ РАЗМЕРНОСТЕЙ:\n"
                f"{inv_str or '  нет данных'}\n\n"
                + (
                    f"УСПЕШНЫЕ РЕШЕНИЯ NAVIGATOR ИЗ ПРОШЛЫХ ПРОГОНОВ:\n"
                    f"{nav_hist_str}\n\n"
                    if nav_hist_str else ""
                ) +
                f"ПУТИ К УСПЕХУ (нарративы Летописца):\n"
                f"{chronicle_str or '  нет данных'}\n\n"
                f"ПРАВИЛА ИЗ ПРОШЛЫХ СЕССИЙ:\n{rules_str}\n\n"
                + (
                    f"⛔ ЗАПРЕЩЁННЫЕ ПАТТЕРНЫ (плохой R², отклонены — НЕ ПРЕДЛАГАТЬ):\n"
                    f"{rejected_str}\n\n"
                    if rejected_str else ""
                )
                + (
                    f"⚠️  СПОРНЫЕ ПАТТЕРНЫ (хороший R², но LLM усомнилась — "
                    f"можно переформулировать):\n"
                    f"{disputed_str}\n\n"
                    if disputed_str else ""
                )
                + (
                    f"🌍 ПОНИМАНИЕ МИРА (структура природы из прошлых открытий):\n"
                    f"{world_str}\n\n"
                    if world_str else ""
                )
                + f"ЗАДАНИЕ: Дай ОДНУ конкретную подсказку для следующего поиска.\n"
                f"Формат: 'Попробуй [структуру/оператор/признак] потому что [причина]'\n"
                f"Максимум 2 предложения. Будь конкретен."
            )

            start = time.time()
            response = ollama_chat(
                prompt, model=self.model, host=self.host,
                temperature=0.3, num_predict=120,
            ).strip()
            elapsed = time.time() - start

            # Фильтруем ошибки ollama — не передаём их как подсказки
            if not response or response.startswith("[OLLAMA_ERROR]") or len(response) < 10:
                log.debug("[Oracle] LLM вернул ошибку или пустой ответ: %s", response[:50])
                return None

            self._llm_calls += 1
            hint = f"[ORACLE/llm] {response}"
            self._oracle_log.append({
                "attempt": attempt,
                "hint":    response,
                "elapsed": round(elapsed, 1),
                "ts":      datetime.now().isoformat(),
            })
            log.info("[Oracle] LLM совет (%.1fs, вызов %d/%d): %s",
                     elapsed, self._llm_calls, ORACLE_MAX_LLM_CALLS, response[:60])
            print(f"  [Oracle] Совет: {response[:80]}")
            return hint

        except Exception as e:
            log.debug("[Oracle] LLM hint error: %s", e)
            return None

    # ── Финализация ───────────────────────────────────────────────

    def finalize(
        self,
        formula_final: str = "",
        r2_final:      float = 0.0,
        domain:        str  = "",
        passed:        bool = False,
    ) -> None:
        """
        Вызывается в конце сессии.
        Записывает выводы Oracle в episodic_memory для следующих сессий.
        """
        if not self._attempts:
            return

        elapsed_total = time.time() - self._session_start
        n_attempts    = len(self._attempts)
        best_r2       = max((a.r2 for a in self._attempts), default=0.0)
        top_rejectors = {}
        for a in self._attempts:
            if a.rejected_by:
                top_rejectors[a.rejected_by] = top_rejectors.get(a.rejected_by, 0) + 1

        session_summary = {
            "ts":            datetime.now().isoformat(),
            "event":         "oracle_session",
            "domain":        domain or self._domain,
            "dim_codes":     self._dim_codes,
            "n_attempts":    n_attempts,
            "best_r2":       round(best_r2, 4),
            "final_r2":      round(r2_final, 4),
            "passed":        passed,
            "formula_final": formula_final[:100],
            "llm_calls":     self._llm_calls,
            "oracle_hints":  self._oracle_log,
            "top_rejectors": top_rejectors,
            "elapsed_sec":   round(elapsed_total, 1),
        }

        # Сохраняем в episodic_memory
        try:
            from .episodic_memory import get_memory, MEMORY_DIR
            path = MEMORY_DIR / "oracle_sessions.jsonl"
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(session_summary, ensure_ascii=False) + "\n")
            log.info("[Oracle] Сессия сохранена: %d попыток R²=%.3f passed=%s",
                     n_attempts, r2_final, passed)
        except Exception as e:
            log.debug("[Oracle] finalize save: %s", e)

        # Обновляем invariant_learned если нашли что-то хорошее
        if passed and r2_final >= 0.85 and formula_final and self._oracle_log:
            try:
                from .episodic_memory import get_memory
                path_summary = " → ".join(
                    f"попытка {a.attempt}: R²={a.r2:.2f}" for a in self._attempts[-3:]
                )
                # Не дублируем — invariant_learned уже пишется в engine.py
                # Oracle только логирует что его советы помогли
                hints_used = [h["hint"][:80] for h in self._oracle_log if h.get("hint")]
                if hints_used:
                    log.info("[Oracle] Советы которые помогли: %s", hints_used)
            except Exception:
                pass

        print(f"\n  [Oracle] Сессия завершена: {n_attempts} попыток, "
              f"R²={r2_final:.3f}, LLM вызовов={self._llm_calls}")

    # ── Статистика ────────────────────────────────────────────────

    def status(self) -> str:
        """Краткий статус Oracle для отображения."""
        n = len(self._attempts)
        if n == 0:
            return f"Oracle[0 попыток, {len(self._inv_hints)} подсказок из памяти]"
        best = max(a.r2 for a in self._attempts)
        return (f"Oracle[{n} попыток, best_R²={best:.3f}, "
                f"LLM={self._llm_calls}/{ORACLE_MAX_LLM_CALLS}]")

    @staticmethod
    def load_past_sessions(limit: int = 10) -> List[Dict]:
        """Загружает историю прошлых сессий Oracle."""
        try:
            from .episodic_memory import MEMORY_DIR
            path = MEMORY_DIR / "oracle_sessions.jsonl"
            if not path.exists():
                return []
            lines = path.read_text(encoding="utf-8").strip().splitlines()
            sessions = []
            for line in lines[-limit:]:
                try:
                    sessions.append(json.loads(line))
                except Exception:
                    continue
            return sessions
        except Exception:
            return []
