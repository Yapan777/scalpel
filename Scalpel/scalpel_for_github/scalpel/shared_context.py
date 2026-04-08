"""
shared_context.py — Общая доска координации v1.0 (уровень 4)

Вместо слепого конвейера:
    Хирург → Surgery → Препаратор → Navigator

Теперь модули слышат друг друга:
    Хирург пишет намерение → Препаратор читает и сигнализирует план →
    Координатор видит конфликт → Хирург получает второй шанс →
    Препаратор переоценивает на чистых данных

Принципы:
  - SharedContext — пассивная доска, не логика
  - Логика координации — в engine.py (orchestrator)
  - Каждый модуль пишет своё состояние и читает чужое
  - История событий для отладки и обучения
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List, Optional


# ─────────────────────────────────────────────────────────────────
# СОБЫТИЯ — что происходило на доске
# ─────────────────────────────────────────────────────────────────

@dataclass
class CtxEvent:
    ts:     float
    actor:  str   # "surgeon" / "preparator" / "navigator" / "coordinator"
    action: str   # "intent" / "revision_request" / "applied" / "override"
    detail: str   # человекочитаемое описание

    def __str__(self) -> str:
        return f"[{self.actor}] {self.action}: {self.detail}"


# ─────────────────────────────────────────────────────────────────
# ГЛАВНЫЙ ОБЪЕКТ
# ─────────────────────────────────────────────────────────────────

@dataclass
class SharedContext:
    """
    Общая доска координации между модулями Scalpel.
    v10.26: все поля сгруппированы ДО методов (dataclass требование).
    """

    # ── Хирург (yi:9b) ────────────────────────────────────────────
    surgeon_intent:        str   = "pending"
    surgeon_outlier_ratio: float = 0.0
    surgeon_iqr_mult:      float = 3.0
    surgeon_cut_fraction:  float = 0.025
    surgeon_pass:          int   = 0

    # ── Surgery (алгоритм) ────────────────────────────────────────
    surgery_performed:   bool  = False
    surgery_removed_pct: float = 0.0
    y_negative_after:    float = 0.0

    # ── Препаратор ────────────────────────────────────────────────
    prep_intent:      str   = "pending"
    prep_applied:     str   = "pending"
    prep_confidence:  float = 1.0
    prep_pass:        int   = 0

    # ── Координация ───────────────────────────────────────────────
    revision_needed:  bool  = False
    revision_reason:  str   = ""
    revision_count:   int   = 0
    MAX_REVISIONS:    int   = 2

    # ── МетаПаттерны ─────────────────────────────────────────────
    active_patterns:  List  = field(default_factory=list)
    pattern_hints:    dict  = field(default_factory=dict)

    # ── Физик — право вето ───────────────────────────────────────
    veto_count:       int  = 0
    veto_active:      bool = False
    veto_reason:      str  = ""
    veto_suggestion:  str  = ""
    MAX_VETOES:       int  = 2

    # ── Navigator ─────────────────────────────────────────────────
    nav_hypotheses:   List[str] = field(default_factory=list)
    nav_operators:    List[str] = field(default_factory=list)
    nav_features:     List[str] = field(default_factory=list)
    nav_reasoning:    str       = ""
    nav_pass:         int       = 0

    # ── Матрёшка — роли ──────────────────────────────────────────
    role_verdicts:    dict = field(default_factory=dict)
    role_flags:       dict = field(default_factory=dict)

    # ── История событий (последнее поле) ─────────────────────────
    events: List[CtxEvent] = field(default_factory=list)

    # ─────────────────────────────────────────────────────────────
    # API для хирурга
    # ─────────────────────────────────────────────────────────────

    def surgeon_write(
        self,
        intent:        str,
        outlier_ratio: float,
        iqr_mult:      float,
        cut_fraction:  float,
    ) -> None:
        """Хирург записывает своё намерение после анализа данных."""
        self.surgeon_intent        = intent
        self.surgeon_outlier_ratio = outlier_ratio
        self.surgeon_iqr_mult      = iqr_mult
        self.surgeon_cut_fraction  = cut_fraction
        self.surgeon_pass         += 1
        self._log("surgeon", "intent",
                  f"intent={intent} outliers={outlier_ratio:.1%} "
                  f"iqr_k={iqr_mult} cut={cut_fraction:.3f}")

    def surgeon_escalate(self) -> dict:
        """
        Возвращает параметры для более агрессивного второго прохода.
        Каждая ревизия/вето снижает iqr_mult на 0.5 и увеличивает cut на 50%.
        BUG FIX: используем max(revision_count, veto_count, 1) — гарантируем
        что escalation ВСЕГДА агрессивнее дефолта (не равен ему при count=0).
        """
        _step   = max(self.revision_count, self.veto_count, 1)
        new_iqr  = max(1.5,  self.surgeon_iqr_mult   - 0.5 * _step)
        new_cut  = min(0.05, self.surgeon_cut_fraction * (1.5 ** _step))
        self._log("surgeon", "escalate",
                  f"iqr_k {self.surgeon_iqr_mult:.1f}→{new_iqr:.1f} "
                  f"cut {self.surgeon_cut_fraction:.3f}→{new_cut:.3f}")
        return {"iqr_multiplier": new_iqr, "cut_fraction": new_cut, "force_cut": True}

    # ─────────────────────────────────────────────────────────────
    # API для surgery (алгоритм)
    # ─────────────────────────────────────────────────────────────

    def surgery_write(self, performed: bool, removed_pct: float) -> None:
        """Surgery записывает результат."""
        self.surgery_performed   = performed
        self.surgery_removed_pct = removed_pct
        self._log("surgery", "applied",
                  f"performed={performed} removed={removed_pct:.2f}%")

    def update_negative_ratio(self, y) -> None:
        """Обновляет долю отрицательных значений после хирургии."""
        import numpy as np
        self.y_negative_after = float(np.mean(y < 0))
        self._log("surgery", "stats",
                  f"negative_after={self.y_negative_after:.1%}")

    # ─────────────────────────────────────────────────────────────
    # API для препаратора
    # ─────────────────────────────────────────────────────────────

    def prep_signal_intent(self, intent: str, confidence: float = 1.0) -> None:
        """
        Препаратор сигнализирует намерение ДО применения.
        Это ключевой момент координации — хирург может ещё вмешаться.
        """
        self.prep_intent    = intent
        self.prep_confidence = confidence
        self.prep_pass      += 1
        self._log("preparator", "intent",
                  f"planning={intent} confidence={confidence:.2f} "
                  f"(negative_in_data={self.y_negative_after:.1%})")

    def prep_write_applied(self, applied: str) -> None:
        """Препаратор записывает что реально применил."""
        self.prep_applied = applied
        self._log("preparator", "applied", f"transform={applied}")

    # ─────────────────────────────────────────────────────────────
    # API координатора (engine.py)
    # ─────────────────────────────────────────────────────────────

    def coordinator_check(self) -> bool:
        """
        Проверяет нужна ли ревизия.
        Возвращает True если нужен второй проход хирургии.

        Правила:
          1. Препаратор хочет sqrt/log но после хирургии ещё есть отрицательные
          2. Не превышен лимит ревизий
        """
        if self.revision_count >= self.MAX_REVISIONS:
            self._log("coordinator", "revision_blocked",
                      f"MAX_REVISIONS={self.MAX_REVISIONS} достигнут")
            return False

        # Правило 1: sqrt/log несовместимы с отрицательными значениями
        if self.prep_intent in ("sqrt", "log") and self.y_negative_after > 0.02:
            self.revision_needed = True
            self.revision_reason = (
                f"Препаратор хочет {self.prep_intent}, "
                f"но {self.y_negative_after:.1%} значений отрицательные. "
                f"Хирург должен их убрать."
            )
            self.revision_count += 1
            self._log("coordinator", "revision_request", self.revision_reason)
            return True

        self._log("coordinator", "check_ok",
                  f"prep_intent={self.prep_intent} "
                  f"negative={self.y_negative_after:.1%} — ревизия не нужна")
        return False

    def coordinator_reset_prep(self) -> None:
        """Сбрасывает состояние препаратора для повторного анализа."""
        self.prep_intent    = "pending"
        self.prep_applied   = "pending"
        self.revision_needed = False   # BUG FIX: сбрасываем флаг ревизии
        self._log("coordinator", "reset_prep",
                  "препаратор сброшен для повторного анализа")

    # ─────────────────────────────────────────────────────────────
    # Вспомогательное
    # ─────────────────────────────────────────────────────────────

    def _log(self, actor: str, action: str, detail: str) -> None:
        self.events.append(CtxEvent(
            ts=time.time(), actor=actor, action=action, detail=detail,
        ))



    # ─────────────────────────────────────────────────────────────
    # API для Физика (вето)
    # ─────────────────────────────────────────────────────────────

    def issue_veto(self, reason: str, suggestion: str = "") -> None:
        """Физик выставляет вето на предложение Navigator."""
        self.veto_active     = True
        self.veto_reason     = reason
        self.veto_suggestion = suggestion
        self.veto_count     += 1
        self._log("physicist", "veto",
                  f"[{self.veto_count}/{self.MAX_VETOES}] {reason}")

    def clear_veto(self) -> None:
        """Сбрасывает активное вето после пересмотра."""
        self.veto_active    = False
        self.veto_reason    = ""
        self.veto_suggestion = ""
        self._log("physicist", "veto_cleared", "вето снято — данные пересмотрены")

    def veto_for_navigator(self) -> str:
        """Строка для failure_logs Navigator — что именно было не так."""
        if not self.veto_reason:
            return ""
        return (
            f"[ВЕТО ФИЗИКА] {self.veto_reason}"
            + (f" Рекомендация: {self.veto_suggestion}" if self.veto_suggestion else "")
        )

    # ─────────────────────────────────────────────────────────────
    # API для МетаПаттернов
    # ─────────────────────────────────────────────────────────────

    def load_patterns(self, data_stats: dict) -> None:
        """Загружает применимые паттерны для текущего закона."""
        try:
            from .meta_patterns import get_pattern_engine
            engine = get_pattern_engine()
            self.active_patterns = engine.find_matching(data_stats)
            for target in ("surgeon", "preparator", "navigator"):
                hint = engine.hints_for_prompt(data_stats, target)
                if hint:
                    self.pattern_hints[target] = hint
            n = len(self.active_patterns)
            self._log("coordinator", "patterns_loaded",
                      f"{n} паттернов применимы к данным")
            if n > 0:
                print(f"  [МетаПаттерны] {n} паттернов применимы к этому закону")
        except Exception as e:
            import logging
            logging.getLogger("scalpel").debug("[MetaPatterns] load: %s", e)

    def get_pattern_hint(self, target: str) -> str:
        """Возвращает строку подсказок для конкретного модуля."""
        return self.pattern_hints.get(target, "")

    def get_pattern_action(self, target: str, action_type: str):
        """Возвращает рекомендованное значение действия (или None)."""
        for p in self.active_patterns:
            if p.target == target and p.action_type == action_type:
                return p.action_value
        return None

    # ─────────────────────────────────────────────────────────────
    # API для Navigator
    # ─────────────────────────────────────────────────────────────

    def navigator_write(
        self,
        hypotheses: List[str],
        operators:  List[str],
        features:   List[str],
        reasoning:  str = "",
    ) -> None:
        """Navigator записывает своё решение ДО запуска PySR."""
        self.nav_hypotheses = hypotheses
        self.nav_operators  = operators
        self.nav_features   = features
        self.nav_reasoning  = reasoning
        self.nav_pass      += 1
        self._log("navigator", "decision",
                  f"hyps={hypotheses[:2]} ops={operators[:4]} "
                  f"features={features}")

    # ─────────────────────────────────────────────────────────────
    # API для ролей Матрёшки
    # ─────────────────────────────────────────────────────────────

    def role_write(self, role: str, verdict: str, flags: List[str] = None) -> None:
        """Роль записывает вердикт и флаги."""
        self.role_verdicts[role] = verdict
        if flags:
            self.role_flags[role] = flags
        self._log(role.lower(), "verdict",
                  f"verdict={verdict}" + (f" flags={flags}" if flags else ""))

    def get_role_flags(self) -> List[str]:
        """Все флаги от всех ролей — Navigator читает это в следующей итерации."""
        all_flags = []
        for flags in self.role_flags.values():
            all_flags.extend(flags)
        return list(set(all_flags))

    # ─────────────────────────────────────────────────────────────
    # Сводка для промптов ролей
    # ─────────────────────────────────────────────────────────────

    def ctx_for_roles(self) -> str:
        """
        Возвращает компактную строку которую видит каждая роль Матрёшки.
        Содержит: что сделал хирург, что решил препаратор, что предложил Navigator.
        """
        lines = ["[SharedContext — история обработки данных]"]

        # Хирург
        if self.surgeon_intent and self.surgeon_intent != "pending":
            lines.append(
                f"  Хирург: intent={self.surgeon_intent}, "
                f"outliers={self.surgeon_outlier_ratio:.1%}, "
                f"проходов={self.surgeon_pass}"
            )
        if self.surgery_performed:
            lines.append(
                f"  Surgery: удалено {self.surgery_removed_pct:.1f}%, "
                f"ревизий={self.revision_count}"
            )

        # Препаратор
        if self.prep_applied and self.prep_applied != "pending":
            lines.append(f"  Препаратор: трансформация={self.prep_applied}")

        # Navigator
        if self.nav_hypotheses:
            lines.append(
                f"  Navigator: гипотезы={self.nav_hypotheses[:3]}, "
                f"операторы={self.nav_operators[:4]}"
            )
            if self.nav_reasoning:
                lines.append(f"  Navigator reasoning: {self.nav_reasoning[:120]}")

        # Флаги предыдущих ролей (если это не первая роль в Матрёшке)
        prev_flags = self.get_role_flags()
        if prev_flags:
            lines.append(f"  Флаги предыдущих ролей: {prev_flags}")

        if len(lines) == 1:
            return ""   # нечего показывать
        return "\n".join(lines)

    def ctx_for_navigator(self) -> str:
        """
        Возвращает строку которую видит Navigator в следующей HADI-итерации.
        Содержит флаги от ролей — что они нашли не так в предыдущей формуле.
        """
        flags = self.get_role_flags()
        verdicts = self.role_verdicts
        if not flags and not verdicts:
            return ""

        lines = ["[SharedContext — обратная связь от ролей]"]
        for role, verdict in verdicts.items():
            role_flags = self.role_flags.get(role, [])
            flag_str = f" [{', '.join(role_flags)}]" if role_flags else ""
            lines.append(f"  {role}: {verdict}{flag_str}")
        if flags:
            lines.append(f"  Суммарные проблемы: {flags}")
        return "\n".join(lines)

    def summary(self) -> str:
        """Краткая сводка для логов."""
        lines = [
            f"SharedContext — {len(self.events)} событий, "
            f"ревизий={self.revision_count}/{self.MAX_REVISIONS}",
            f"  Хирург:     intent={self.surgeon_intent} "
            f"outliers={self.surgeon_outlier_ratio:.1%}",
            f"  Surgery:    performed={self.surgery_performed} "
            f"negative_after={self.y_negative_after:.1%}",
            f"  Препаратор: intent={self.prep_intent} → "
            f"applied={self.prep_applied}",
        ]
        if self.nav_hypotheses:
            lines.append(f"  Navigator:  hyps={self.nav_hypotheses[:2]} "
                         f"ops={self.nav_operators[:3]}")
        if self.role_verdicts:
            vstr = " | ".join(f"{r}={v}" for r, v in self.role_verdicts.items())
            lines.append(f"  Матрёшка:   {vstr}")
        if self.veto_count > 0:
            status = "активно" if self.veto_active else "снято"
            lines.append(f"  Вето:       {self.veto_count}x ({status}) — {self.veto_reason[:80]}")
        if self.revision_count > 0:
            lines.append(f"  Ревизия:    {self.revision_reason}")
        return "\n".join(lines)

    def to_log_dict(self) -> dict:
        """Для записи в jsonl логи."""
        return {
            "surgeon_intent":        self.surgeon_intent,
            "surgeon_outlier_ratio": round(self.surgeon_outlier_ratio, 4),
            "surgery_performed":     self.surgery_performed,
            "y_negative_after":      round(self.y_negative_after, 4),
            "prep_intent":           self.prep_intent,
            "prep_applied":          self.prep_applied,
            "nav_hypotheses":        self.nav_hypotheses[:3],
            "nav_operators":         self.nav_operators[:4],
            "nav_pass":              self.nav_pass,
            "role_verdicts":         self.role_verdicts,
            "role_flags":            self.role_flags,
            "veto_count":            self.veto_count,
            "veto_reason":           self.veto_reason,
            "revision_count":        self.revision_count,
            "revision_reason":       self.revision_reason,
            "events":                [str(e) for e in self.events],
        }
