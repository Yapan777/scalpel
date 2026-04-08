"""
episodic_memory.py — Эпизодическая память ролей v9.9.2.

Каждая роль Матрёшки накапливает специализированную базу знаний:
  - Что принимала → паттерны правильных формул
  - Что отклоняла → паттерны ложных корреляций
  - Почему        → конкретные аргументы из анализа

Структура памяти (JSONL, один файл на роль):
  {
    "ts":        "2024-01-15T10:23:45",
    "formula":   "f0 / f1",
    "skeleton":  "v / v",
    "verdict":   "ПРИНЯТА",
    "analysis":  "Размерностно корректно...",
    "r2_train":  0.95,
    "r2_blind":  0.91,
    "complexity": 3,
    "domain":    "Finance",
    "tags":      ["ratio", "dimensionless"]
  }

Использование в BootstrapFewShot:
  memory.recall(role, limit=5) → List[dspy.Example]
"""
from __future__ import annotations

import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import (
    ROLE_FAILURE_DIR, ROLE_NAMES, ROLE_VAULT_TAGS,
    DSPY_FEW_SHOT_MAX,
)

log = logging.getLogger("scalpel")

# Папка эпизодической памяти (рядом с role_failures)
MEMORY_DIR = ROLE_FAILURE_DIR.parent / "episodic_memory"

# Slug-имена ролей для файлов
_ROLE_SLUG = {
    "Скептик":   "skeptic",
    "Физик":     "physicist",
    "Прагматик": "pragmatist",
    "Мистик":    "mystic",
}

# Задачи ролей (для реконструкции dspy.Example)
_ROLE_TASKS = {
    "Скептик":   "Найди слабые места. Почему формула ложна? Какие данные опровергнут?",
    "Физик":     "Проверь размерностную корректность. Соответствует известным законам?",
    "Прагматик": "Оцени практическую применимость. В каких условиях ломается?",
    "Мистик":    "Найди аналогии с известными физическими законами или структурами.",
}


# ══════════════════════════════════════════════════════════════════
# УТИЛИТА: Скелет формулы
# ══════════════════════════════════════════════════════════════════

def _formula_skeleton(formula: str) -> str:
    """f0*f1 + 3.14 → v * v + c"""
    tokens = re.findall(r"[a-zA-Z_]\w*|[\d.]+|[+\-*/^()]", formula)
    _OPS   = {"sin","cos","exp","log","abs","sqrt","tanh","pow"}
    result = []
    for t in tokens:
        if t in _OPS:                      result.append(t)
        elif re.match(r"^[a-zA-Z_]", t):  result.append("v")
        elif re.match(r"^\d",         t):  result.append("c")
        else:                              result.append(t)
    return " ".join(result)


def _formula_tags(formula: str, skeleton: str) -> List[str]:
    """Автоматически тегирует формулу по структуре."""
    tags = []
    if "/" in formula:     tags.append("ratio")
    if "sqrt" in formula:  tags.append("sqrt")
    if "log"  in formula:  tags.append("log")
    if "exp"  in formula:  tags.append("exp")
    if "*"    in formula:  tags.append("product")
    if "+"    in formula:  tags.append("additive")
    n_vars = skeleton.count("v")
    tags.append(f"vars_{n_vars}")
    return tags


# ══════════════════════════════════════════════════════════════════
# EPISODIC MEMORY
# ══════════════════════════════════════════════════════════════════

class EpisodicMemory:
    """
    Эпизодическая память всех ролей Матрёшки.

    Хранит выводы ролей в JSONL:
      episodic_memory/
        skeptic_memory.jsonl
        physicist_memory.jsonl
        pragmatist_memory.jsonl
        mystic_memory.jsonl

    Методы:
      remember()   — сохранить вывод роли
      recall()     — загрузить примеры для BootstrapFewShot
      stats()      — статистика по роли
      top_patterns() — самые частые паттерны ПРИНЯТА/ОТКЛОНЕНА
    """

    def __init__(self, memory_dir: Path = MEMORY_DIR) -> None:
        self.memory_dir = memory_dir
        self.memory_dir.mkdir(parents=True, exist_ok=True)

    # ── Сохранение ────────────────────────────────────────────────

    def remember(
        self,
        role_name:  str,
        formula:    str,
        verdict:    str,
        analysis:   str,
        r2_train:   float = 0.0,
        r2_blind:   float = 0.0,
        complexity: int   = 0,
        domain:     str   = "",
    ) -> None:
        """
        Сохраняет вывод роли в её личную память.
        Вызывается автоматически после каждого аудита.
        """
        skeleton = _formula_skeleton(formula)
        tags     = _formula_tags(formula, skeleton)

        record = {
            "ts":         datetime.now().isoformat(),
            "formula":    formula,
            "skeleton":   skeleton,
            "verdict":    verdict,
            "analysis":   analysis[:500],   # обрезаем длинные тексты
            "r2_train":   round(r2_train,  4),
            "r2_blind":   round(r2_blind,  4),
            "complexity": complexity,
            "domain":     domain,
            "tags":       tags,
        }

        path = self._path(role_name)
        try:
            with path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
            log.debug("[Memory/%s] Запомнено: %s → %s", role_name, formula, verdict)
        except Exception as e:
            log.warning("[Memory/%s] Ошибка записи: %s", role_name, e)

    def remember_scientific_cycle(
        self,
        formula:     str,
        question:    str,
        physicist:   str   = "",   # Физик: structural_critique
        skeptic:     str   = "",   # Скептик: analysis
        delphi_ops:  list  = None, # Delphi: new_operator_hint
        delphi_exp:  float = 0.0,  # Delphi: new_exponent_hint
        variable:    str   = "",   # new_variable_hint
        domain:      str   = "",
        cycle:       int   = 1,
    ) -> None:
        """
        v10.13: Сохраняет запись научного цикла как новый тип события.
        Используется DSPy BootstrapFewShot как обучающий пример для Navigator.
        """
        skeleton = _formula_skeleton(formula)
        record = {
            "ts":        datetime.now().isoformat(),
            "event":     "scientific_cycle",          # новый тип
            "formula":   formula,
            "skeleton":  skeleton,
            "question":  question[:300],
            "физик":     physicist[:200],
            "скептик":   skeptic[:200],
            "delphi":    f"операторы={delphi_ops or []}, степень={delphi_exp}",
            "variable":  variable[:100],
            "domain":    domain,
            "cycle":     cycle,
        }
        # Сохраняем в отдельный файл scientific_cycles.jsonl
        path = self.memory_dir / "scientific_cycles.jsonl"
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
            log.debug("[Memory/Scientific] Цикл %d записан: %s", cycle, formula[:50])
        except Exception as e:
            log.warning("[Memory/Scientific] Ошибка записи: %s", e)

    def recall_scientific_cycles(
        self,
        formula:  str = "",
        limit:    int = 10,
    ) -> list:
        """
        v10.13: Загружает записи научных циклов.
        Если formula задана — фильтрует по скелету.
        Возвращает список строк-выводов для previous_cycle_conclusions.
        """
        path = self.memory_dir / "scientific_cycles.jsonl"
        if not path.exists():
            return []
        skeleton_filter = _formula_skeleton(formula) if formula else ""
        records = []
        try:
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                        if skeleton_filter and rec.get("skeleton") != skeleton_filter:
                            continue
                        records.append(rec)
                    except Exception:
                        continue
        except Exception as e:
            log.warning("[Memory/Scientific] Ошибка чтения: %s", e)
            return []

        # Берём последние limit записей, преобразуем в строки
        recent = records[-limit:]
        conclusions = []
        for rec in recent:
            parts = [f"Цикл {rec.get('cycle', '?')}"]
            if rec.get("физик"):
                parts.append(f"Физик: {rec['физик'][:80]}")
            if rec.get("variable"):
                parts.append(f"Переменная: {rec['variable']}")
            if rec.get("delphi"):
                parts.append(f"Delphi: {rec['delphi']}")
            if rec.get("скептик"):
                parts.append(f"Скептик: {rec['скептик'][:60]}")
            conclusions.append(" | ".join(parts))
        return conclusions

    def remember_curriculum(
        self,
        level:         int,
        formula_true:  str,
        formula_found: str,
        r2_blind:      float,
        noise_level:   float,
        maxsize:       int,
        fast_fail_sec: int,
        domain:        str,
        passed:        bool,
        depth:         int  = 0,   # v10.14: глубина дерева вариаций (0=оригинал)
        r2_threshold:  float = 0.80,
    ) -> None:
        """
        v10.14: Сохраняет результат curriculum датасета.
        Файл: episodic_memory/curriculum_memory.jsonl

        result_type (из спецификации):
          good   — R² >= 0.90: исследуй глубже
          medium — R² 0.80-0.90: чуть подправить
          bad    — R² < 0.80: изменить структуру

        Усиление лучших (из спецификации):
          R² >= 0.95 → сохранить 5 раз
          R² >= 0.85 → сохранить 2 раза
          иначе      → 1 раз
        """
        # Определяем result_type по спецификации
        if r2_blind >= 0.90:
            result_type = "good"
        elif r2_blind >= 0.80:
            result_type = "medium"
        else:
            result_type = "bad"

        record = {
            "ts":            datetime.now().isoformat(),
            "event":         "curriculum",
            "level":         level,
            "depth":         depth,          # v10.14: глубина дерева вариаций
            "result_type":   result_type,    # v10.14: good/medium/bad
            "formula_true":  formula_true,
            "formula_found": formula_found[:200],
            "r2_blind":      round(r2_blind, 4),
            "noise_level":   noise_level,
            "maxsize":       maxsize,
            "fast_fail_sec": fast_fail_sec,
            "domain":        domain,
            "passed":        passed,
        }
        path = self.memory_dir / "curriculum_memory.jsonl"

        # Усиление лучших примеров (из спецификации)
        weight = 5 if r2_blind >= 0.95 else (2 if r2_blind >= 0.85 else 1)

        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as f:
                for _ in range(weight):          # лучшие примеры дублируются
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
            log.debug(
                "[Memory/Curriculum] L%d depth=%d %s: %s → R²=%.4f (×%d)",
                level, depth, result_type, formula_true[:30], r2_blind, weight,
            )
        except Exception as e:
            log.warning("[Memory/Curriculum] Ошибка записи: %s", e)

    def remember_invariant_learned(
        self,
        formula_shadow:  str,
        formula_real:    str,
        r2_blind:        float,
        domain:          str,
        field:           str,
        feat_names:      list,
        dim_codes:       list,
        status:          str   = "discovery",   # "discovery" | "known_law" | "similar_to_law"
        heritage_label:  str   = "",
        explanation:     str   = "",
        path_summary:    str   = "",  # краткий путь к инварианту (из chronicle)
    ) -> None:
        """
        v10.14: Сохраняет ИНВАРИАНТ который система нашла.
        Ключевое: хранит связь dim_codes → структура формулы → домен.
        Это обучает Navigator: "если dim=[2,8] → пробуй a^1.5 (Кеплер)".

        Файл: episodic_memory/invariants_learned.jsonl
        """
        skeleton = _formula_skeleton(formula_shadow)
        # Паттерн dim_codes → структура: ключ для обучения Navigator
        dim_pattern = ",".join(str(d) for d in sorted(set(dim_codes)))

        record = {
            "ts":            datetime.now().isoformat(),
            "event":         "invariant_learned",
            "formula_shadow": formula_shadow[:200],
            "formula_real":   formula_real[:200],
            "skeleton":       skeleton,
            "r2_blind":       round(r2_blind, 4),
            "domain":         domain,
            "field":          field,
            "feat_names":     feat_names[:8],
            "dim_codes":      dim_codes[:8],
            "dim_pattern":    dim_pattern,   # "2,8" → Кеплер, "3,6" → Ньютон
            "status":         status,
            "heritage_label": heritage_label,
            "explanation":    explanation[:300],
            "path_summary":   path_summary[:300],
        }
        path = self.memory_dir / "invariants_learned.jsonl"
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
            log.info("[Memory/Invariant] %s: %s R²=%.4f",
                     status, formula_shadow[:40], r2_blind)
        except Exception as e:
            log.warning("[Memory/Invariant] Ошибка записи: %s", e)

    def recall_chronicle_paths(
        self,
        domain: str = "",
        limit:  int = 5,
    ) -> list:
        """
        v10.14+: Выводы Летописца — успешные пути к формулам.
        Oracle читает это чтобы знать: "как мы шли к этому результату
        в прошлый раз" — не только что нашли, но и как.

        Возвращает список строк вида:
          "domain=Physics dim=[2] за 2 попытки: a**1.5 R²=0.987
           Путь: пробовали f0, f0**2, нашли f0**1.5"
        """
        path = self.memory_dir / "chronicle_steps.jsonl"
        if not path.exists():
            return []
        records = []
        try:
            for line in path.read_text(encoding="utf-8").strip().splitlines():
                try:
                    rec = json.loads(line)
                    if rec.get("event") != "chronicle_final":
                        continue
                    if not rec.get("passed", False):
                        continue
                    if domain and rec.get("domain","") and domain not in rec.get("domain",""):
                        continue
                    records.append(rec)
                except Exception:
                    continue
        except Exception as e:
            log.warning("[Memory/ChronicleP] %s", e)
            return []

        records.sort(key=lambda r: float(r.get("r2_blind", 0)), reverse=True)
        records = records[:limit]

        hints = []
        for rec in records:
            parts = [
                f"domain={rec.get('domain','?')} за {rec.get('total_attempts','?')} попыток",
                f"формула={rec.get('formula_final','?')[:40]} R²={rec.get('r2_blind',0):.3f}",
            ]
            narrative = rec.get("chronicle", "")
            if narrative:
                parts.append(f"нарратив: {narrative[:120]}")
            hints.append(" | ".join(parts))
        return hints

    def recall_invariants_for_domain(
        self,
        dim_codes:  list  = None,
        domain:     str   = "",
        limit:      int   = 10,
    ) -> list:
        """
        v10.14: Загружает инварианты для данного домена / dim_codes паттерна.
        Используется Navigator: "дай мне примеры что работало с dim=[2,8]".
        Возвращает список строк-подсказок.
        """
        path = self.memory_dir / "invariants_learned.jsonl"
        if not path.exists():
            return []
        target_dim = ",".join(str(d) for d in sorted(set(dim_codes))) if dim_codes else ""
        records: list = []
        try:
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                        # Фильтр по dim_pattern ИЛИ домену
                        dim_match    = target_dim and rec.get("dim_pattern") == target_dim
                        domain_match = domain     and rec.get("domain") == domain
                        field_match  = domain     and rec.get("field",  "").lower() in domain.lower()
                        if dim_match or domain_match or field_match or (not target_dim and not domain):
                            records.append(rec)
                    except Exception:
                        continue
        except Exception as e:
            log.warning("[Memory/Invariant] Ошибка чтения: %s", e)
            return []

        # Сортируем по R²
        records.sort(key=lambda r: float(r.get("r2_blind", 0)), reverse=True)
        records = records[:limit]

        hints = []
        for rec in records:
            parts = []
            if rec.get("heritage_label"):
                parts.append(f"★ {rec['heritage_label']}")
            parts.append(f"формула={rec.get('skeleton','')} R²={rec.get('r2_blind',0):.2f}")
            if rec.get("dim_pattern"):
                parts.append(f"dim=[{rec['dim_pattern']}]")
            if rec.get("explanation"):
                parts.append(f"→ {rec['explanation'][:80]}")
            if rec.get("path_summary"):
                parts.append(f"путь: {rec['path_summary'][:60]}")
            hints.append(" | ".join(parts))
        return hints

    def remember_navigator_hypotheses(
        self,
        attempt:    int,
        hypotheses: List[str],
        features:   List[str],
        operators:  List[str],
        reasoning:  str  = "",
        domain:     str  = "",
    ) -> None:
        """
        v10.16: Сохраняет гипотезы Navigator после каждой HADI итерации.
        Позволяет DSPy и Летописцу видеть что предлагал Navigator
        и к чему это привело — система умнеет на своих идеях.
        Файл: episodic_memory/chronicle_steps.jsonl (event=navigator_hypotheses)
        """
        record = {
            "ts":         datetime.now().isoformat(),
            "event":      "navigator_hypotheses",
            "attempt":    attempt,
            "hypotheses": [h[:80] for h in hypotheses[:5]],
            "features":   list(features)[:6],
            "operators":  list(operators)[:10],
            "reasoning":  reasoning[:200],
            "domain":     domain,
        }
        path = self.memory_dir / "chronicle_steps.jsonl"
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
            log.debug("[Memory/NavHyp] Попытка %d: %d гипотез записано",
                      attempt, len(hypotheses))
        except Exception as e:
            log.warning("[Memory/NavHyp] Ошибка записи: %s", e)

    def remember_chronicle_step(
        self,
        attempt:     int,
        tried:       str,
        r2:          float,
        rejected_by: str  = "",
        reason:      str  = "",
        delphi_hint: str  = "",
        led_to:      str  = "",
        domain:      str  = "",
    ) -> None:
        """
        v10.14: Сохраняет один шаг из истории поиска (Летописец).
        Файл: episodic_memory/chronicle_steps.jsonl
        """
        record = {
            "ts":          datetime.now().isoformat(),
            "event":       "chronicle_step",
            "attempt":     attempt,
            "tried":       tried[:200],
            "r2":          round(r2, 4),
            "rejected_by": rejected_by[:100],
            "reason":      reason[:200],
            "delphi_hint": delphi_hint[:100],
            "led_to":      led_to[:200],
            "domain":      domain,
        }
        path = self.memory_dir / "chronicle_steps.jsonl"
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
            log.debug("[Memory/Chronicle] Шаг %d записан: %s → R²=%.4f",
                      attempt, tried[:40], r2)
        except Exception as e:
            log.warning("[Memory/Chronicle] Ошибка записи: %s", e)

    def recall_chronicle_steps(
        self,
        domain: str = "",
        limit:  int = 20,
    ) -> List[str]:
        """
        v10.14: Загружает цепочки из истории поиска для Navigator few-shot.
        Возвращает список строк вида:
          "Попытка 2: sqrt(f0)*f1 → отклонил Физик (размерность) → Delphi: добавь / → sqrt(f0)/f1 R²=0.94"
        """
        path = self.memory_dir / "chronicle_steps.jsonl"
        if not path.exists():
            return []
        records: List[Dict] = []
        try:
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                        if rec.get("event") != "chronicle_step":
                            continue
                        if domain and rec.get("domain") != domain:
                            continue
                        records.append(rec)
                    except Exception:
                        continue
        except Exception as e:
            log.warning("[Memory/Chronicle] Ошибка чтения: %s", e)
            return []

        chains = []
        for rec in records[-limit:]:
            parts = [f"Попытка {rec.get('attempt','?')}: {rec.get('tried','')}"]
            if rec.get("rejected_by"):
                reason_short = (rec.get("reason") or "")[:60]
                parts.append(
                    f"→ отклонил {rec['rejected_by']}"
                    + (f" ({reason_short})" if reason_short else "")
                )
            if rec.get("delphi_hint"):
                parts.append(f"→ Delphi: {rec['delphi_hint']}")
            if rec.get("led_to"):
                parts.append(f"→ {rec['led_to']} R²={rec.get('r2', 0.0):.2f}")
            chains.append(" ".join(parts))
        return chains

    def remember_chronicle_final(
        self,
        level:         int,
        formula_final: str,
        r2_blind:      float,
        total_attempts: int,
        chronicle_text: str = "",
        domain:        str  = "",
        passed:        bool = True,
    ) -> None:
        """
        v10.14: Финальная запись Летописца после каждого уровня curriculum
        или после принятия формулы в LLM-фазе.
        Событие "chronicle_final" — итог поиска.
        Файл: episodic_memory/chronicle_steps.jsonl
        """
        record = {
            "ts":             datetime.now().isoformat(),
            "event":          "chronicle_final",
            "level":          level,
            "formula_final":  formula_final[:200],
            "r2_blind":       round(r2_blind, 4),
            "total_attempts": total_attempts,
            "chronicle":      chronicle_text[:500] if chronicle_text else "",
            "domain":         domain,
            "passed":         passed,
        }
        path = self.memory_dir / "chronicle_steps.jsonl"
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
            log.debug(
                "[Memory/Chronicle] Финал уровня %d: %s R²=%.4f",
                level, formula_final[:40], r2_blind,
            )
        except Exception as e:
            log.warning("[Memory/Chronicle] Ошибка финальной записи: %s", e)

    # ── Извлечение для BootstrapFewShot ───────────────────────────

    def recall(
        self,
        role_name: str,
        verdict_filter: Optional[str] = None,  # "ПРИНЯТА" | "ОТКЛОНЕНА" | None
        limit:     int  = DSPY_FEW_SHOT_MAX,
        min_r2:    float = 0.0,
    ) -> List[Any]:
        """
        Загружает записи из памяти роли как dspy.Example.
        Возвращает пустой список если DSPy недоступен.

        verdict_filter=None → все записи
        verdict_filter="ПРИНЯТА" → только подтверждённые формулы
        """
        try:
            import dspy
            _dspy_ok = True
        except ImportError:
            _dspy_ok = False

        records = self._load_records(role_name)
        if not records:
            return []

        # Фильтрация
        if verdict_filter:
            records = [r for r in records if r.get("verdict") == verdict_filter]
        if min_r2 > 0:
            records = [r for r in records
                       if float(r.get("r2_blind", r.get("r2_train", 0))) >= min_r2]

        # Сортировка: лучший R² blind первым
        records.sort(key=lambda r: float(r.get("r2_blind", 0)), reverse=True)
        records = records[:limit]

        if not _dspy_ok:
            return records  # возвращаем raw dict если dspy нет

        examples = []
        task = _ROLE_TASKS.get(role_name, "Оцени формулу.")
        for rec in records:
            try:
                ex = dspy.Example(
                    role_name       = role_name,
                    role_task       = task,
                    formula         = rec.get("formula", ""),
                    formula_metrics = (
                        f"r2_train={rec.get('r2_train',0):.3f}, "
                        f"r2_blind={rec.get('r2_blind',0):.3f}, "
                        f"complexity={rec.get('complexity',0)}, "
                        f"domain={rec.get('domain','unknown')}"
                    ),
                    verdict         = rec.get("verdict", "УСЛОВНО"),
                    analysis        = rec.get("analysis", ""),
                ).with_inputs("role_name", "role_task", "formula", "formula_metrics")
                examples.append(ex)
            except Exception:
                continue

        log.info("[Memory/%s] recall: %d примеров (filter=%s)",
                 role_name, len(examples), verdict_filter or "all")
        return examples

    # ── Статистика ────────────────────────────────────────────────

    def stats(self, role_name: str) -> Dict:
        """
        Возвращает статистику по памяти роли.

        {
          "total": 142,
          "accepted": 89,
          "rejected": 31,
          "conditional": 22,
          "acceptance_rate": 0.63,
          "top_skeletons": {"v / v": 23, "sqrt ( v ) * v": 15, ...},
          "avg_r2_accepted": 0.91,
        }
        """
        records = self._load_records(role_name)
        if not records:
            return {"total": 0, "accepted": 0, "rejected": 0,
                    "conditional": 0, "acceptance_rate": 0.0}

        accepted    = [r for r in records if r.get("verdict") == "ПРИНЯТА"]
        rejected    = [r for r in records if r.get("verdict") == "ОТКЛОНЕНА"]
        conditional = [r for r in records if r.get("verdict") == "УСЛОВНО"]

        # Топ скелеты
        from collections import Counter
        skeleton_counter = Counter(r.get("skeleton","") for r in records)
        top_skeletons = dict(skeleton_counter.most_common(5))

        # Средний R² для принятых
        r2_vals = [float(r.get("r2_blind", r.get("r2_train", 0)))
                   for r in accepted if r.get("r2_blind") or r.get("r2_train")]
        avg_r2 = sum(r2_vals) / len(r2_vals) if r2_vals else 0.0

        return {
            "total":            len(records),
            "accepted":         len(accepted),
            "rejected":         len(rejected),
            "conditional":      len(conditional),
            "acceptance_rate":  round(len(accepted) / len(records), 3),
            "top_skeletons":    top_skeletons,
            "avg_r2_accepted":  round(avg_r2, 4),
        }

    def stats_all(self) -> Dict[str, Dict]:
        """Статистика по всем 4 ролям."""
        return {role: self.stats(role) for role in ROLE_NAMES}

    # ── Паттерны ─────────────────────────────────────────────────

    def top_patterns(
        self,
        role_name: str,
        verdict:   str = "ПРИНЯТА",
        top_n:     int = 5,
    ) -> List[Dict]:
        """
        Возвращает топ-N паттернов (скелетов) по вердикту.
        Полезно для диагностики что роль считает "хорошим".

        Пример вывода:
          [{"skeleton": "v / v", "count": 23, "avg_r2": 0.91}, ...]
        """
        from collections import defaultdict
        records = self._load_records(role_name)
        filtered = [r for r in records if r.get("verdict") == verdict]

        groups: Dict[str, List[float]] = defaultdict(list)
        for r in filtered:
            sk = r.get("skeleton", "unknown")
            groups[sk].append(float(r.get("r2_blind", r.get("r2_train", 0))))

        result = []
        for sk, r2_list in sorted(groups.items(),
                                   key=lambda x: len(x[1]), reverse=True)[:top_n]:
            result.append({
                "skeleton": sk,
                "count":    len(r2_list),
                "avg_r2":   round(sum(r2_list)/len(r2_list), 4) if r2_list else 0.0,
            })
        return result

    def print_summary(self) -> None:
        """Печатает сводку памяти всех ролей в консоль."""
        print(f"\n  {'═'*58}")
        print(f"  EPISODIC MEMORY — Сводка по ролям")
        print(f"  {'═'*58}")
        for role in ROLE_NAMES:
            s = self.stats(role)
            if s["total"] == 0:
                print(f"  {role:12s}: (пусто)")
                continue
            bar_len  = 20
            acc_fill = int(bar_len * s["acceptance_rate"])
            bar      = "█" * acc_fill + "░" * (bar_len - acc_fill)
            print(
                f"  {role:12s}: [{bar}] "
                f"{s['accepted']:3d}✓ {s['rejected']:3d}✗ {s['conditional']:3d}? "
                f"| всего={s['total']:4d} | "
                f"avg_R²={s['avg_r2_accepted']:.3f}"
            )
        print(f"  {'═'*58}\n")

    # ── Внутренние методы ─────────────────────────────────────────

    def _path(self, role_name: str) -> Path:
        slug = _ROLE_SLUG.get(role_name, role_name.lower())
        return self.memory_dir / f"{slug}_memory.jsonl"

    def _load_records(self, role_name: str) -> List[Dict]:
        path = self._path(role_name)
        if not path.exists():
            return []
        records = []
        try:
            for line in path.read_text(encoding="utf-8").strip().splitlines():
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        except Exception as e:
            log.warning("[Memory/%s] Ошибка чтения: %s", role_name, e)
        return records


# ══════════════════════════════════════════════════════════════════
# ГЛОБАЛЬНЫЙ ЭКЗЕМПЛЯР
# ══════════════════════════════════════════════════════════════════

# Один экземпляр на весь процесс
_global_memory: Optional[EpisodicMemory] = None


def get_memory() -> EpisodicMemory:
    """Возвращает глобальный экземпляр EpisodicMemory."""
    global _global_memory
    if _global_memory is None:
        _global_memory = EpisodicMemory()
    return _global_memory
