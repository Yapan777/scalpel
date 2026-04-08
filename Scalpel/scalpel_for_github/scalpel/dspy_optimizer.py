"""
dspy_optimizer.py — Self-Evolving Optimizer v9.9.

Логика:
1. GoldLoader   — читает gold_formulas.json, строит dspy.Example
2. NavModule    — DSPy-модуль Штурмана (Predict + ChainOfThought)
3. AuditModule  — DSPy-модуль одной роли Матрёшки
4. HADIModule   — DSPy-модуль рефлексии при DEATH
5. compile_nav  — BootstrapFewShot компиляция Штурмана
6. save/load    — сериализация скомпилированных весов

СВЯЩЕННЫЙ ЗАКОН: этот модуль не трогает shuffle_test,
cross_blind, ShadowMapper. Метрики качества — только PySR R²_blind.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .config import (
    DSPY_COMPILED_PATH, DSPY_FAILURE_LOG, DSPY_FEW_SHOT_MAX,
    DSPY_GOLD_MIN_R2, DSPY_RECOMPILE_DAYS,
    GOLD_PATH, REJECTED_PATH, DISPUTED_PATH, OLLAMA_HOST, OLLAMA_MODEL, NAVIGATOR_MODEL,
    NAVIGATOR_FAST_MODEL, NAV_TIMEOUT_SEC,
    VAULT_DIR,
)

# v10.14: статистика роста знаний Летописца
CHRONICLE_STATS_PATH = VAULT_DIR / "chronicle_dspy_stats.jsonl"

log = logging.getLogger("scalpel")

try:
    import dspy
    from dspy.teleprompt import BootstrapFewShot
    DSPY_AVAILABLE = True
except ImportError:
    dspy = None  # type: ignore
    BootstrapFewShot = None  # type: ignore
    DSPY_AVAILABLE = False

from .dspy_signatures import (
    NavSignature, AuditSignature, HADIReflectSignature,
)


# ═══════════════════════════════════════════════════════════════
# СЕКЦИЯ 1: LM ИНИЦИАЛИЗАЦИЯ
# ═══════════════════════════════════════════════════════════════

_LM_INITIALIZED = False


def init_dspy_lm(
    model: str = OLLAMA_MODEL,
    host:  str = OLLAMA_HOST,
) -> bool:
    """
    Инициализирует DSPy LM через Ollama.
    Возвращает True если успешно, False если DSPy/Ollama недоступны.
    """
    global _LM_INITIALIZED
    if not DSPY_AVAILABLE:
        log.warning("[DSPy] dspy-ai не установлен. Работаем в Legacy-режиме.")
        return False
    # FIX БАГ 9: проверяем доступность Ollama ДО инициализации DSPy
    try:
        import urllib.request as _ur
        _ur.urlopen(f"{host.rstrip('/')}/api/tags", timeout=5).read()
    except Exception as _ping_err:
        log.warning("[DSPy] Ollama недоступна (%s) — Legacy-режим.", _ping_err)
        return False

    # FIX БАГ 11: сбрасываем флаг перед повторной инициализацией
    # run_llm_phase() вызывается после run_engine() — второй вызов должен работать
    global _LM_INITIALIZED
    _LM_INITIALIZED = False

    try:
        # FIX БАГ 13: пробуем оба провайдера — разные версии litellm/Ollama
        # поддерживают разные форматы
        lm = None
        for provider, base in [
            (f"ollama/{model}",      f"{host.rstrip('/')}"),      # litellm >= 1.0
            (f"ollama_chat/{model}", f"{host.rstrip('/')}"),      # litellm >= 1.3
            (f"ollama/{model}",      f"{host.rstrip('/')}/v1"),   # OpenAI-совместимый
        ]:
            try:
                _lm_test = dspy.LM(
                    provider,
                    api_base=base,
                    api_key="ollama",
                    temperature=0.2,
                    max_tokens=512,
                    timeout=15,
                )
                # Быстрая проверка: один короткий вызов
                _lm_test("Hi", max_tokens=5)
                lm = dspy.LM(
                    provider,
                    api_base=base,
                    api_key="ollama",
                    temperature=0.2,
                    max_tokens=2048,
                    timeout=600,
                )
                log.info("[DSPy] Рабочий провайдер: %s @ %s", provider, base)
                break
            except Exception as _probe_err:
                log.debug("[DSPy] Провайдер %s не подошёл: %s", provider, _probe_err)
                continue
        if lm is None:
            raise RuntimeError("Ни один из Ollama провайдеров не работает")
        dspy.configure(lm=lm)
        _LM_INITIALIZED = True
        log.info("[DSPy] LM инициализирован: ollama/%s", model)
        return True
    except Exception as e:
        log.warning("[DSPy] Не удалось инициализировать LM: %s", e)
        _LM_INITIALIZED = False
        return False


# ═══════════════════════════════════════════════════════════════
# СЕКЦИЯ 2: GOLD LOADER — few-shot примеры из хранилища
# ═══════════════════════════════════════════════════════════════

class GoldLoader:
    """
    Загружает успешные инварианты из gold_formulas.json
    и превращает их в dspy.Example для BootstrapFewShot.

    Каждый золотой пример — это пара (вход → выход):
      вход:  data_meta заглушка + пустой failure_logs
      выход: гипотеза которая привела к инварианту
    """

    def __init__(self, gold_path: Path = GOLD_PATH):
        self.gold_path = gold_path

    def load_examples(
        self,
        min_r2:   float = DSPY_GOLD_MIN_R2,
        max_count: int  = DSPY_FEW_SHOT_MAX,
    ) -> List[Any]:
        """
        Возвращает список dspy.Example из лучших формул.
        Сортирует по r2_blind (лучшие первые).
        """
        if not DSPY_AVAILABLE:
            return []
        records = self._read_gold()
        if not records:
            log.info("[GoldLoader] gold_formulas.json пуст или не найден.")
            return []

        # Фильтруем по качеству
        good = [
            r for r in records
            if float(r.get("r2_blind", r.get("r2_train", 0))) >= min_r2
        ]
        # Сортируем: лучший R² blind первым
        good.sort(key=lambda r: float(r.get("r2_blind", 0)), reverse=True)
        good = good[:max_count]

        examples = []
        for rec in good:
            ex = self._record_to_example(rec)
            if ex is not None:
                examples.append(ex)

        log.info("[GoldLoader] Загружено %d золотых примеров (min_r2=%.2f).",
                 len(examples), min_r2)
        return examples

    def load_failure_examples(self, max_count: int = 10) -> List[Any]:
        """
        Загружает примеры DEATH из dspy_failure_log.jsonl
        для HADIReflectModule.
        """
        if not DSPY_AVAILABLE or not DSPY_FAILURE_LOG.exists():
            return []
        examples = []
        try:
            lines = DSPY_FAILURE_LOG.read_text(encoding="utf-8").strip().splitlines()
            for line in lines[-max_count:]:
                try:
                    rec = json.loads(line)
                    ex = dspy.Example(
                        death_report   = json.dumps(rec.get("death_report", {})),
                        data_meta      = rec.get("data_meta", ""),
                        attempt_number = rec.get("attempt_number", "1 of 3"),
                        failure_type         = rec.get("failure_type", "UNKNOWN"),
                        corrected_strategy   = rec.get("corrected_strategy", ""),
                        new_hypotheses       = rec.get("new_hypotheses", ""),
                    ).with_inputs("death_report", "data_meta", "attempt_number")
                    examples.append(ex)
                except Exception:
                    continue
        except Exception as e:
            log.warning("[GoldLoader] Ошибка чтения failure log: %s", e)
        return examples

    def load_rejected_examples(self, max_count: int = 12) -> List[Any]:
        """
        v10.15: Загружает антипримеры из rejected_formulas.json.

        Navigator видит их как "СТОП-паттерны":
        структуры которые уже проверялись Матрёшкой и были отклонены —
        не предлагать снова на похожих данных.
        """
        if not DSPY_AVAILABLE or not REJECTED_PATH.exists():
            return []
        examples = []
        try:
            with REJECTED_PATH.open(encoding="utf-8") as f:
                data = json.load(f)
            records = data.get("formulas", [])
            # Самые свежие антипримеры важнее — берём последние
            records = records[-max_count:]
            for rec in records:
                formula   = rec.get("formula", "")
                skeleton  = rec.get("skeleton", "")
                rejected_by = rec.get("rejected_by", "Матрёшка")
                reason    = rec.get("reason", "")
                lesson    = rec.get("lesson", "")
                r2_tr     = rec.get("r2_train", 0.0)
                r2_bl     = rec.get("r2_blind", 0.0)
                if not formula:
                    continue
                # Формируем failure_logs как будто Navigator видит историю
                anti_note = (
                    f"[АНТИПРИМЕР] Формула '{formula}' (skeleton: {skeleton}) "
                    f"была отклонена {rejected_by}. "
                    f"R²_train={r2_tr:.3f}, R²_blind={r2_bl:.3f}. "
                    f"Причина: {reason}. "
                    f"Урок: {lesson}"
                )
                try:
                    ex = dspy.Example(
                        data_meta         = anti_note,
                        failure_logs      = f"[REJECTED by {rejected_by}: {reason[:120]}]",
                        selected_features = "f0",
                        hypothesis        = f"# НЕ ИСПОЛЬЗОВАТЬ: {formula}\n# Причина: {reason[:80]}",
                        reasoning         = lesson,
                        formula_str       = f"# ОТКЛОНЕНО: {formula}",
                    ).with_inputs("data_meta", "failure_logs")
                    examples.append(ex)
                except Exception:
                    continue
        except Exception as e:
            log.warning("[GoldLoader] Ошибка чтения rejected_formulas: %s", e)

        log.info("[GoldLoader] Загружено %d антипримеров (СТОП-паттерны для Navigator).",
                 len(examples))
        return examples

    def load_disputed_examples(self, max_count: int = 8) -> List[Any]:
        """
        v10.15: Загружает спорные формулы из disputed_formulas.json.

        Это формулы с хорошим R² (>= REJECTED_R2_SAFE_MAX) которые тем не менее
        были отклонены Матрёшкой — возможные ложные негативы.

        В отличие от rejected (жёсткий запрет), disputed даёт Navigator
        мягкий сигнал: «эта структура математически работает, но LLM
        усомнилась — стоит проверить тщательнее, не отбрасывать сразу».
        """
        if not DSPY_AVAILABLE or not DISPUTED_PATH.exists():
            return []
        examples = []
        try:
            with DISPUTED_PATH.open(encoding="utf-8") as f:
                data = json.load(f)
            records = data.get("formulas", [])[-max_count:]
            for rec in records:
                formula  = rec.get("formula", "")
                skeleton = rec.get("skeleton", "")
                reason   = rec.get("reason", "")
                r2_bl    = rec.get("r2_blind", 0.0)
                r2_tr    = rec.get("r2_train", 0.0)
                unani    = rec.get("unanimous", False)
                if not formula:
                    continue
                dispute_note = (
                    f"[СПОРНАЯ] Формула '{formula}' (skeleton: {skeleton}) "
                    f"R²_blind={r2_bl:.3f} — математически хорошая, но LLM отклонила. "
                    f"Причина: {reason}. "
                    f"{'Единогласно — высокий риск ложного негатива.' if unani else ''}"
                    f"Можно переформулировать и попробовать снова."
                )
                try:
                    ex = dspy.Example(
                        data_meta         = dispute_note,
                        failure_logs      = f"[DISPUTED R²={r2_bl:.3f}: {reason[:100]}]",
                        selected_features = "f0",
                        hypothesis        = (
                            f"# СПОРНАЯ ФОРМУЛА (R²={r2_bl:.3f} — хороший):\n"
                            f"# {formula}\n"
                            f"# Попробуй упростить или переписать иначе."
                        ),
                        reasoning         = f"R²_blind={r2_bl:.3f} > порога, но отклонена. Переформулировать.",
                        formula_str       = formula,
                    ).with_inputs("data_meta", "failure_logs")
                    examples.append(ex)
                except Exception:
                    continue
        except Exception as e:
            log.warning("[GoldLoader] Ошибка чтения disputed_formulas: %s", e)

        log.info("[GoldLoader] Загружено %d спорных формул (мягкие предупреждения).",
                 len(examples))
        return examples

    def load_chronicle_examples(
        self,
        max_count:   int   = 30,
        min_r2:      float = 0.75,
    ) -> List[Any]:
        """
        v10.14: Загружает цепочки Летописца как dspy.Example.

        Стратегия улучшения — два типа примеров:

        1. chronicle_step (каждая итерация):
           → учит Navigator ПОЧЕМУ отклонили и ЧТО помогло
           → failure_logs = [{"hypothesis": цепочка, "death_reason": причина}]

        2. chronicle_final (итог поиска):
           → усиливает успешные паттерны
           → вес пропорционален R²: R²≥0.95 → 5×, R²≥0.85 → 2×, иначе 1×

        Чем больше накоплено записей — тем лучше компиляция.
        """
        if not DSPY_AVAILABLE:
            return []
        try:
            from .episodic_memory import get_memory
            mem = get_memory()
        except Exception as e:
            log.warning("[Chronicle] Не удалось загрузить episodic_memory: %s", e)
            return []

        examples: List[Any] = []

        # ── Тип 1: chronicle_step → учим на цепочках мышления ────
        chains = mem.recall_chronicle_steps(limit=max_count)
        for chain in chains:
            ex = self._chain_to_example(chain)
            if ex is not None:
                examples.append(ex)

        # ── Тип 2: chronicle_final → усиливаем успешные паттерны ─
        finals = self._load_chronicle_finals()
        for rec in finals:
            r2     = float(rec.get("r2_blind", 0.0))
            passed = rec.get("passed", False)
            if not passed or r2 < min_r2:
                continue
            ex = self._final_to_example(rec)
            if ex is None:
                continue
            # Усиление: лучшие примеры копируются несколько раз
            weight = 5 if r2 >= 0.95 else (2 if r2 >= 0.85 else 1)
            examples.extend([ex] * weight)

        log.info(
            "[Chronicle] Загружено %d примеров из хроники (%d chains, %d finals×weight)",
            len(examples), len(chains), len(finals),
        )
        # Сохраняем статистику для мониторинга роста
        self._save_chronicle_stats(len(chains), len(finals), len(examples))
        return examples

    def _chain_to_example(self, chain: str) -> Optional[Any]:
        """Преобразует строку-цепочку в dspy.Example."""
        if not DSPY_AVAILABLE or not chain:
            return None
        try:
            # v10.14: БАГ F исправлен — split по " → " (с пробелами)
            # чтобы не ломать формулы типа "f0→f1" (без пробелов)
            parts       = chain.split(" → ")
            formula     = parts[0].split(":")[-1].strip() if parts else ""
            reason      = ""
            delphi_hint = ""
            led_to      = ""
            for part in parts[1:]:
                p = part.strip()
                if "отклонил" in p:
                    reason = p
                elif "Delphi" in p:
                    delphi_hint = p.replace("Delphi:", "").strip()
                elif "R²" in p:
                    led_to = p.split("R²")[0].strip()

            # failure_logs объясняет Navigator почему предыдущий путь провалился
            failure_entry = {
                "hypothesis":   formula[:100],
                "death_reason": reason[:120] if reason else "отклонено",
                "delphi_hint":  delphi_hint[:80],
                "source":       "chronicle_chain",
            }
            # hypotheses = следующий шаг (то что реально помогло)
            next_hyp = led_to[:60] if led_to else formula

            return dspy.Example(
                data_meta          = "n_samples=150, features=[f0,f1,f2]",
                failure_logs       = json.dumps([failure_entry]),
                selected_features  = "f0, f1",
                selected_operators = "+,-,*,/,sqrt,log,exp",
                hypotheses         = next_hyp,
                ooda_stable        = "true",
                reasoning          = chain[:300],
            ).with_inputs("data_meta", "failure_logs")
        except Exception as e:
            log.debug("[Chronicle] chain_to_example ошибка: %s", e)
            return None

    def _final_to_example(self, rec: Dict) -> Optional[Any]:
        """Преобразует chronicle_final в усиленный dspy.Example."""
        if not DSPY_AVAILABLE:
            return None
        try:
            formula   = rec.get("formula_final", "")
            r2        = float(rec.get("r2_blind", 0.0))
            attempts  = int(rec.get("total_attempts", 1))
            chronicle = rec.get("chronicle", "")[:200]
            domain    = rec.get("domain", "")
            level     = rec.get("level", 0)

            data_meta = (
                f"n_samples=150, domain={domain or 'unknown'}, "
                f"curriculum_level={level}, r2_achieved={r2:.3f}, "
                f"total_attempts={attempts}"
            )
            return dspy.Example(
                data_meta          = data_meta,
                failure_logs       = "[]",
                selected_features  = "f0, f1",
                selected_operators = "+,-,*,/,sqrt,log",
                hypotheses         = formula[:80],
                ooda_stable        = "true",
                reasoning          = (
                    f"Финальная формула R²={r2:.3f} за {attempts} попыток. "
                    + chronicle
                ),
            ).with_inputs("data_meta", "failure_logs")
        except Exception as e:
            log.debug("[Chronicle] final_to_example ошибка: %s", e)
            return None

    def load_invariant_examples(
        self,
        max_count: int = 20,
    ) -> List[Any]:
        """
        v10.14: Загружает invariants_learned.jsonl как dspy.Example.

        Это САМЫЕ ЦЕННЫЕ примеры — система сама нашла инвариант.
        Каждый пример содержит:
          - dim_pattern → структуру формулы (Navigator учится паттерну)
          - путь к инварианту (как пришли к нему)
          - объяснение физического смысла

        Усиление:
          known_law  → 5× (подтверждённый закон природы)
          discovery  → 5× (новое открытие системы)
          similar    → 2×
        """
        if not DSPY_AVAILABLE:
            return []
        try:
            from .episodic_memory import EpisodicMemory, MEMORY_DIR
            path = MEMORY_DIR / "invariants_learned.jsonl"
            if not path.exists():
                return []
            records = []
            seen_skeletons = set()
            for line in path.read_text(encoding="utf-8").strip().splitlines():
                try:
                    rec = json.loads(line)
                    # Убираем дубли по скелету
                    # v10.14 БАГ 3: (skeleton, domain) — не только skeleton
                    skel   = rec.get("skeleton", "")
                    domain_key = rec.get("domain", "")
                    if (skel, domain_key) in seen_skeletons:
                        continue
                    seen_skeletons.add((skel, domain_key))
                    records.append(rec)
                except Exception:
                    continue
        except Exception as e:
            log.warning("[Invariant] Ошибка загрузки: %s", e)
            return []

        records.sort(key=lambda r: float(r.get("r2_blind", 0)), reverse=True)
        records = records[:max_count]

        examples = []
        for rec in records:
            ex = self._invariant_to_example(rec)
            if ex is None:
                continue
            status = rec.get("status", "similar_to_law")
            weight = 5 if status in ("known_law", "discovery") else 2
            examples.extend([ex] * weight)

        log.info("[Invariant] Загружено %d invariant-примеров", len(examples))
        return examples

    @staticmethod
    def _invariant_to_example(rec: Dict) -> Optional[Any]:
        """Преобразует invariant_learned запись в dspy.Example."""
        if not DSPY_AVAILABLE:
            return None
        try:
            skeleton    = rec.get("skeleton", "")
            dim_pattern = rec.get("dim_pattern", "")
            domain      = rec.get("domain", "")
            field       = rec.get("field", "")
            r2          = float(rec.get("r2_blind", 0))
            heritage    = rec.get("heritage_label", "")
            explanation = rec.get("explanation", "")
            path_sum    = rec.get("path_summary", "")
            feat_names  = rec.get("feat_names", ["f0", "f1"])

            # Контекст для Navigator: dim_pattern → правильная структура
            reasoning_parts = []
            if heritage:
                reasoning_parts.append(f"★ {heritage}")
            if explanation:
                reasoning_parts.append(explanation[:150])
            if path_sum:
                reasoning_parts.append(f"путь: {path_sum[:100]}")
            reasoning_parts.append(f"домен={domain}({field}), dim=[{dim_pattern}], R²={r2:.3f}")

            return dspy.Example(
                data_meta          = (
                    f"n_samples=150, features=[{','.join(feat_names[:4])}], "
                    f"dim_codes=[{dim_pattern}], domain={domain}, "
                    f"field={field}, r2_achieved={r2:.3f}"
                ),
                failure_logs       = "[]",
                selected_features  = ", ".join(feat_names[:4]),
                selected_operators = "+,-,*,/,sqrt,log,exp",
                hypotheses         = rec.get("formula_shadow", "")[:80],
                ooda_stable        = "true",
                reasoning          = " | ".join(reasoning_parts)[:400],
            ).with_inputs("data_meta", "failure_logs")
        except Exception as e:
            log.debug("[Invariant] example error: %s", e)
            return None

    def load_curriculum_examples(
        self,
        max_count:   int   = 20,
        min_r2:      float = 0.70,
    ) -> List[Any]:
        """
        v10.14: Загружает curriculum_memory.jsonl как dspy.Example.

        Ключевое отличие от chronicle/gold:
        - Знает ПРАВИЛЬНЫЙ ответ (formula_true) — это supervision signal
        - Знает сложность задачи (noise_level, level, maxsize)
        - Пара (условие → правильный ответ) — идеал для few-shot

        Усиление по уровню: уровень 4 (физика) → 3×, уровень 3 → 2×, остальные 1×
        """
        if not DSPY_AVAILABLE:
            return []
        try:
            from .episodic_memory import EpisodicMemory, MEMORY_DIR
            path = MEMORY_DIR / "curriculum_memory.jsonl"
            if not path.exists():
                return []
            records = []
            for line in path.read_text(encoding="utf-8").strip().splitlines():
                try:
                    rec = json.loads(line)
                    if rec.get("event") != "curriculum":
                        continue
                    if float(rec.get("r2_blind", 0)) < min_r2:
                        continue
                    if not rec.get("passed", False):
                        continue
                    records.append(rec)
                except Exception:
                    continue
        except Exception as e:
            log.warning("[Curriculum] Ошибка загрузки curriculum_memory: %s", e)
            return []

        # Сортируем: лучшие R² первыми
        records.sort(key=lambda r: float(r.get("r2_blind", 0)), reverse=True)
        records = records[:max_count]

        examples = []
        for rec in records:
            ex = self._curriculum_record_to_example(rec)
            if ex is None:
                continue
            # Усиление: физические законы (уровень 4) самые ценные
            level  = int(rec.get("level", 1))
            weight = 3 if level == 4 else (2 if level == 3 else 1)
            examples.extend([ex] * weight)

        log.info("[Curriculum] Загружено %d curriculum примеров (%d записей × weight).",
                 len(examples), len(records))
        return examples

    @staticmethod
    def _curriculum_record_to_example(rec: Dict) -> Optional[Any]:
        """Преобразует curriculum_memory запись в dspy.Example."""
        if not DSPY_AVAILABLE:
            return None
        try:
            formula_true  = rec.get("formula_true",  "")
            formula_found = rec.get("formula_found", "")
            r2            = float(rec.get("r2_blind", 0))
            level         = int(rec.get("level", 1))
            noise         = float(rec.get("noise_level", 0.01))
            domain        = rec.get("domain", "curriculum")

            data_meta = (
                f"n_samples=150, domain={domain}, "
                f"curriculum_level={level}, noise={noise:.0%}, "
                f"r2_blind={r2:.3f}, true_formula_hint=known"
            )
            return dspy.Example(
                data_meta          = data_meta,
                failure_logs       = "[]",
                selected_features  = "f0, f1",
                selected_operators = "+,-,*,/,sqrt,log,exp,abs",
                hypotheses         = formula_found[:80] if formula_found else formula_true[:80],
                ooda_stable        = "true",
                reasoning          = (
                    f"Curriculum L{level}: истинная формула '{formula_true[:60]}', "
                    f"найдена '{formula_found[:40]}', R²={r2:.3f}."
                ),
            ).with_inputs("data_meta", "failure_logs")
        except Exception as e:
            log.debug("[Curriculum] record_to_example ошибка: %s", e)
            return None

    @staticmethod
    def _count_chronicle_records() -> int:
        """Считает общее число записей в chronicle_steps.jsonl."""
        try:
            from .episodic_memory import MEMORY_DIR
            path = MEMORY_DIR / "chronicle_steps.jsonl"
            if not path.exists():
                return 0
            return sum(1 for l in path.read_text(encoding="utf-8").strip().splitlines() if l)
        except Exception:
            return 0

    def _chronicle_grew_since_compile(self, n_now: int) -> bool:
        """True если chronicle вырос с момента последней компиляции."""
        try:
            count_path = CHRONICLE_STATS_PATH.with_name("chronicle_count_at_compile.json")
            if not count_path.exists():
                return True  # Первый запуск — компилируем
            saved = json.loads(count_path.read_text(encoding="utf-8"))
            n_saved = int(saved.get("n_chronicle", 0))
            grew = n_now > n_saved
            if grew:
                log.info("[Siege] Chronicle вырос: %d → %d записей", n_saved, n_now)
            return grew
        except Exception:
            return True  # При ошибке — перекомпилируем

    @staticmethod
    def _save_chronicle_count(n: int) -> None:
        """Сохраняет число chronicle записей на момент компиляции."""
        try:
            count_path = CHRONICLE_STATS_PATH.with_name("chronicle_count_at_compile.json")
            count_path.parent.mkdir(parents=True, exist_ok=True)
            count_path.write_text(
                json.dumps({"n_chronicle": n, "ts": datetime.now().isoformat()}),
                encoding="utf-8",
            )
        except Exception:
            pass

    def _load_chronicle_finals(self) -> List[Dict]:
        """
        Читает chronicle_final записи из chronicle_steps.jsonl.
        Уровни: -1 = обычный запуск, 0 = неизвестно, 1-4 = curriculum.
        Все уровни используются для обучения.
        """
        try:
            from .episodic_memory import EpisodicMemory, MEMORY_DIR
            path = MEMORY_DIR / "chronicle_steps.jsonl"
            if not path.exists():
                return []
            records = []
            for line in path.read_text(encoding="utf-8").strip().splitlines():
                try:
                    rec = json.loads(line)
                    if rec.get("event") == "chronicle_final":
                        records.append(rec)
                except Exception:
                    continue
            return records
        except Exception as e:
            log.warning("[Chronicle] Ошибка чтения finals: %s", e)
            return []

    @staticmethod
    def _save_chronicle_stats(n_chains: int, n_finals: int, total_examples: int) -> None:
        """
        Сохраняет статистику роста знаний Летописца.
        Позволяет видеть как накапливается обучение со временем.
        """
        try:
            CHRONICLE_STATS_PATH.parent.mkdir(parents=True, exist_ok=True)
            record = {
                "ts":             datetime.now().isoformat(),
                "n_chains":       n_chains,
                "n_finals":       n_finals,
                "total_examples": total_examples,
            }
            with CHRONICLE_STATS_PATH.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception:
            pass  # Статистика — не критична

    def _read_gold(self) -> List[Dict]:
        if not self.gold_path.exists():
            return []
        try:
            with self.gold_path.open(encoding="utf-8") as f:
                data = json.load(f)
            return data.get("formulas", [])
        except Exception as e:
            log.warning("[GoldLoader] Ошибка чтения gold: %s", e)
            return []

    @staticmethod
    def _record_to_example(rec: Dict) -> Optional[Any]:
        """Превращает запись из gold в dspy.Example."""
        if not DSPY_AVAILABLE:
            return None
        formula = rec.get("formula", "")
        if not formula:
            return None

        r2_tr  = rec.get("r2_train", 0.0)
        r2_bl  = rec.get("r2_blind", 0.0)
        compl  = rec.get("complexity", 0)
        skeleton = rec.get("skeleton", formula)

        # Точное число признаков из поля n_features или из скелета
        n_features = rec.get("n_features", None)
        if n_features is None:
            n_features = max(skeleton.count("v") if skeleton else 1, 1)
        n_vars = max(int(n_features), 1)

        vars_str    = ", ".join(f"f{i}:1" for i in range(n_vars))
        feats_str   = ", ".join(f"f{i}" for i in range(n_vars))
        n_samples_t = rec.get("n_samples", 150)
        data_meta   = (
            f"n_samples={n_samples_t}, features=[{vars_str}], "
            f"r2_train_achieved={r2_tr:.3f}, r2_blind={r2_bl:.3f}"
        )

        # ── Разнообразный reasoning — учим ПРОЦЕССУ мышления ────────
        # Каждый пример описывает разный аналитический подход
        # чтобы DSPy не копировал шаблон а думал самостоятельно
        ops_in_formula = []
        if "sqrt" in formula:  ops_in_formula.append("sqrt")
        if "**2"  in formula:  ops_in_formula.append("square")
        if "**3"  in formula:  ops_in_formula.append("cube")
        if "log"  in formula:  ops_in_formula.append("log")
        if "exp"  in formula:  ops_in_formula.append("exp")
        if "/"    in formula:  ops_in_formula.append("division")
        if "*"    in formula and "**" not in formula: ops_in_formula.append("multiplication")

        ops_hint = ", ".join(ops_in_formula) if ops_in_formula else "basic arithmetic"

        # Генерируем несколько конкурирующих гипотез вокруг формулы
        alt_hyps = [formula]
        if n_vars == 1:
            alt_hyps += [f"f0**2", f"f0**3", f"sqrt(f0)", f"log(f0)"]
        elif n_vars == 2:
            alt_hyps += [f"f0*f1", f"f0/f1", f"f0+f1", f"f0**2*f1"]
        elif n_vars == 3:
            alt_hyps += [f"f0*f1*f2", f"f0*f1/f2", f"sqrt(f0*f1)*f2"]
        elif n_vars >= 4:
            alt_hyps += [f"f0*f1/(f2+f3)", f"f0*f1*f2/f3", f"sqrt(f0*f1*f2*f3)"]
        # Перемешиваем чтобы правильная формула не всегда первая
        import random
        random.shuffle(alt_hyps)
        hyps_str = ";".join(alt_hyps[:4])

        reasoning = (
            f"Data has {n_vars} feature(s): {feats_str}. "
            f"Detected operators: {ops_hint}. "
            f"Tried multiple structures, best fit uses {formula}. "
            f"R²={r2_bl:.3f} on blind test confirms the pattern."
        )

        return dspy.Example(
            data_meta          = data_meta,
            failure_logs       = "[]",
            selected_features  = feats_str,
            selected_operators = "+,-,*,/,sqrt,log,exp",
            hypotheses         = hyps_str,
            ooda_stable        = "true",
            reasoning          = reasoning,
        ).with_inputs("data_meta", "failure_logs")


# ═══════════════════════════════════════════════════════════════
# СЕКЦИЯ 3: DSPy МОДУЛИ
# ═══════════════════════════════════════════════════════════════

if DSPY_AVAILABLE:
    class NavModule(dspy.Module):
        """
        Штурман v9.9 — DSPy-модуль выбора признаков и гипотез.
        ChainOfThought даёт явную цепочку рассуждений.
        """
        def __init__(self):
            super().__init__()
            self.cot = dspy.ChainOfThought(NavSignature)

        def forward(
            self,
            data_meta:    str,
            failure_logs: str = "[]",
        ):
            return self.cot(
                data_meta    = data_meta,
                failure_logs = failure_logs,
            )


    class AuditModule(dspy.Module):
        """
        Одна роль Матрёшки — DSPy-модуль аудита формулы.
        Predict (без CoT) — одна роль, один абзац, один вердикт.
        """
        def __init__(self):
            super().__init__()
            self.predict = dspy.Predict(AuditSignature)

        def forward(
            self,
            role_name:       str,
            role_task:       str,
            formula:         str,
            formula_metrics: str,
        ):
            return self.predict(
                role_name       = role_name,
                role_task       = role_task,
                formula         = formula,
                formula_metrics = formula_metrics,
            )


    class HADIReflectModule(dspy.Module):
        """
        HADI-рефлексия — DSPy-модуль диагностики смерти.
        ChainOfThought: нужно понять ПОЧЕМУ умерло, прежде чем предложить новое.
        """
        def __init__(self):
            super().__init__()
            self.cot = dspy.ChainOfThought(HADIReflectSignature)

        def forward(
            self,
            death_report:   str,
            data_meta:      str,
            attempt_number: str,
        ):
            return self.cot(
                death_report   = death_report,
                data_meta      = data_meta,
                attempt_number = attempt_number,
            )

else:
    # Заглушки
    class NavModule:         pass  # type: ignore
    class AuditModule:       pass  # type: ignore
    class HADIReflectModule: pass  # type: ignore


# ═══════════════════════════════════════════════════════════════
# СЕКЦИЯ 4: КОМПИЛЯТОР (BootstrapFewShot)
# ═══════════════════════════════════════════════════════════════

def _nav_metric(example: Any, prediction: Any, trace=None) -> bool:
    """
    Метрика для BootstrapFewShot Штурмана.
    Засчитываем пример если:
    - hypotheses не пусты
    - selected_features содержит 2+ признака
    - ooda_stable — валидное значение
    СВЯЩЕННЫЙ ЗАКОН: метрика не включает shuffle_test / cross_blind.
    """
    if not DSPY_AVAILABLE:
        return False
    try:
        hyps  = getattr(prediction, "hypotheses", "") or ""
        feats = getattr(prediction, "selected_features", "") or ""
        ooda  = getattr(prediction, "ooda_stable", "") or ""

        hyps_list  = [h.strip() for h in hyps.split(";") if h.strip()]
        feats_list = [f.strip() for f in feats.split(",") if f.strip()]

        return (
            len(hyps_list)  >= 1
            and len(feats_list) >= 2
            and ooda.lower() in ("true", "false")
        )
    except Exception:
        return False


def _make_seed_demos() -> List[Any]:
    """
    Хардкодные seed-примеры для первого запуска когда gold/chronicle пусты.

    Цель: NavModule знает ожидаемый формат вывода с самого начала,
    не гадает по голой сигнатуре NavSignature.

    Примеры охватывают три типичных сценария:
      1. Две переменные (кинетика / степенной закон)
      2. Три переменные (комбинации признаков)
      3. Провал предыдущей попытки (failure_logs не пуст)
    """
    if not DSPY_AVAILABLE:
        return []
    try:
        return [
            dspy.Example(
                data_meta="n_samples=100, features=[f0:mass, f1:velocity], dim_codes=[1,1]",
                failure_logs="[]",
                selected_features="f0, f1",
                selected_operators="+,-,*,/,sqrt",
                hypotheses="f0*f1**2;f0*f1;sqrt(f0)*f1;f0/f1;f0+f1",
                ooda_stable="true",
                reasoning="Два признака одной размерности — пробуем степенные комбинации",
            ).with_inputs("data_meta", "failure_logs"),
            dspy.Example(
                data_meta="n_samples=150, features=[f0:1, f1:1, f2:1], dim_codes=[1,2,1]",
                failure_logs="[]",
                selected_features="f0, f1, f2",
                selected_operators="+,-,*,/,log,exp",
                hypotheses="f0/f1;f0*f2+f1;log(f0)*f1;f0*f1/f2;f0+f1*f2",
                ooda_stable="true",
                reasoning="Три признака — пробуем отношения и произведения попарно",
            ).with_inputs("data_meta", "failure_logs"),
            dspy.Example(
                data_meta="n_samples=200, features=[f0:1, f1:1], dim_codes=[2,1]",
                failure_logs='[{"hypothesis":"f0+f1","death_reason":"R²_blind=0.12 — слишком просто"}]',
                selected_features="f0, f1",
                selected_operators="*,/,sqrt,log,exp",
                hypotheses="f0/f1;sqrt(f0/f1);log(f0)*f1;f0**2/f1;exp(f0)*f1",
                ooda_stable="false",
                reasoning="Линейная гипотеза провалилась — переходим к нелинейным структурам",
            ).with_inputs("data_meta", "failure_logs"),
        ]
    except Exception as e:
        log.debug("[SeedDemos] Ошибка создания seed-демо: %s", e)
        return []


def compile_navigator(
    gold_examples: List[Any],
    max_bootstrapped: int = 3,
) -> Optional[Any]:
    """
    Компилирует NavModule через LabeledFewShot (v10.16).
    Использует золотые примеры как ready-made few-shot демо — без генерации.

    v10.16 FIX: BootstrapFewShot заменён на LabeledFewShot.
    BootstrapFewShot вызывал LLM для каждого примера во время compile
    (6 вызовов × deepseek-r1:14b на CPU = 10+ минут зависания).
    LabeledFewShot просто вставляет готовые gold примеры как демо — мгновенно.

    Качество не страдает: gold примеры уже содержат правильные входы/выходы.
    LLM вызовы происходят только в navigate() во время реальной работы.
    """
    if not DSPY_AVAILABLE or not _LM_INITIALIZED:
        log.warning("[Compiler] DSPy LM не инициализирован. Пропускаем компиляцию.")
        return None
    if not gold_examples:
        seeds = _make_seed_demos()
        mod = NavModule()
        if seeds:
            mod.cot.demos = seeds
            log.info("[Compiler] Нет gold-примеров — NavModule запущен с %d seed-демо.", len(seeds))
        else:
            log.info("[Compiler] Нет gold-примеров и seed недоступны. Используем base NavModule.")
        return mod

    try:
        from dspy.teleprompt import LabeledFewShot
        k = min(len(gold_examples), DSPY_FEW_SHOT_MAX)
        optimizer = LabeledFewShot(k=k)
        base_module  = NavModule()
        compiled_mod = optimizer.compile(base_module, trainset=gold_examples)
        log.info("[Compiler] LabeledFewShot завершён ✓ (%d демо, 0 LLM вызовов)", k)
        print(f"  [Siege] ✓ Navigator скомпилирован мгновенно ({k} gold примеров)")
        return compiled_mod
    except Exception as e:
        log.warning("[Compiler] LabeledFewShot упал: %s. Fallback → base module с демо.", e)
        # Fallback: просто вставляем демо напрямую
        mod = NavModule()
        try:
            mod.cot.demos = gold_examples[:DSPY_FEW_SHOT_MAX]
        except Exception:
            pass
        return mod


# ═══════════════════════════════════════════════════════════════
# СЕКЦИЯ 5: СОХРАНЕНИЕ / ЗАГРУЗКА СКОМПИЛИРОВАННОЙ МОДЕЛИ
# ═══════════════════════════════════════════════════════════════

def save_compiled_model(module: Any, path: Path = DSPY_COMPILED_PATH) -> bool:
    """
    Сохраняет скомпилированные "веса" (few-shot примеры) в JSON.
    DSPy program.save() сериализует демонстрации и инструкции.
    """
    if not DSPY_AVAILABLE or module is None:
        return False
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        module.save(str(path))
        # Добавляем метаданные включая хеш gold примеров
        import hashlib as _hl
        gold_path = path.parent / "gold_formulas.json"
        gold_hash = _hl.md5(gold_path.read_bytes()).hexdigest() if gold_path.exists() else ""
        meta = {
            "compiled_at":    datetime.now().isoformat(),
            "expires_at":     (datetime.now() + timedelta(days=DSPY_RECOMPILE_DAYS)).isoformat(),
            "version":        "10.14.0",
            "n_examples":     len(getattr(module, "_compiled_examples", [])),
            "gold_hash":      gold_hash,
        }
        meta_path = path.with_suffix(".meta.json")
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        log.info("[Compiler] Скомпилированная модель сохранена: %s", path)
        return True
    except Exception as e:
        log.warning("[Compiler] Ошибка сохранения: %s", e)
        return False


def load_compiled_model(path: Path = DSPY_COMPILED_PATH) -> Optional[Any]:
    """
    Загружает скомпилированный NavModule из JSON.
    Проверяет срок годности (DSPY_RECOMPILE_DAYS).
    Возвращает модуль или None если устарел/не найден.
    """
    if not DSPY_AVAILABLE:
        return None
    if not path.exists():
        log.info("[Compiler] Скомпилированная модель не найдена: %s", path)
        return None

    # Проверяем срок
    meta_path = path.with_suffix(".meta.json")
    if meta_path.exists():
        try:
            meta     = json.loads(meta_path.read_text(encoding="utf-8"))
            expires  = datetime.fromisoformat(meta.get("expires_at", "1970-01-01"))
            if datetime.now() > expires:
                log.info("[Compiler] Скомпилированная модель устарела. Нужна перекомпиляция.")
                return None
        except Exception:
            pass

    try:
        module = NavModule()
        module.load(str(path))
        log.info("[Compiler] Загружена скомпилированная модель: %s", path)
        return module
    except Exception as e:
        log.warning("[Compiler] Ошибка загрузки: %s. Перекомпиляция.", e)
        return None


def needs_recompile(path: Path = DSPY_COMPILED_PATH) -> bool:
    """True если нужна перекомпиляция (нет файла, устарел, или gold изменился)."""
    if not path.exists():
        return True
    meta_path = path.with_suffix(".meta.json")
    if not meta_path.exists():
        return True
    try:
        meta    = json.loads(meta_path.read_text(encoding="utf-8"))
        expires = datetime.fromisoformat(meta.get("expires_at", "1970-01-01"))
        if datetime.now() > expires:
            return True
        # Проверяем что gold примеры не изменились с момента компиляции
        import hashlib as _hl
        gold_path = path.parent / "gold_formulas.json"
        if gold_path.exists():
            gold_hash = _hl.md5(gold_path.read_bytes()).hexdigest()
            if meta.get("gold_hash") != gold_hash:
                log.info("[Compiler] Gold примеры изменились — перекомпиляция.")
                return True
        return False
    except Exception:
        return True


# ═══════════════════════════════════════════════════════════════
# СЕКЦИЯ 6: FAILURE LOG (для HADI-рефлексии)
# ═══════════════════════════════════════════════════════════════

def log_failure_example(
    death_report:      Dict,
    data_meta:         str,
    attempt_number:    str,
    failure_type:      str  = "UNKNOWN",
    corrected_strategy: str = "",
    new_hypotheses:    str  = "",
) -> None:
    """
    Записывает пример DEATH в JSONL-файл для будущего обучения HADIModule.
    Каждый провал = один dspy.Example потенциально.
    """
    try:
        DSPY_FAILURE_LOG.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "timestamp":          datetime.now().isoformat(),
            "death_report":       death_report,
            "data_meta":          data_meta,
            "attempt_number":     attempt_number,
            "failure_type":       failure_type,
            "corrected_strategy": corrected_strategy,
            "new_hypotheses":     new_hypotheses,
        }
        with DSPY_FAILURE_LOG.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as e:
        log.warning("[FailureLog] Ошибка записи: %s", e)


# ═══════════════════════════════════════════════════════════════
# СЕКЦИЯ 7: ФАСАД — единая точка входа для engine.py
# ═══════════════════════════════════════════════════════════════

class DSPyOrchestrator:
    @staticmethod
    def _count_chronicle_records() -> int:
        try:
            from .episodic_memory import MEMORY_DIR
            path = MEMORY_DIR / "chronicle_steps.jsonl"
            if not path.exists():
                return 0
            return sum(1 for l in path.read_text(encoding="utf-8").strip().splitlines() if l)
        except Exception:
            return 0

    @staticmethod
    def _save_chronicle_count(n: int) -> None:
        """FIX БАГ 7: метод отсутствовал в DSPyOrchestrator — был только в GoldLoader."""
        try:
            from .config import VAULT_DIR
            import json as _json
            from datetime import datetime as _dt
            count_path = VAULT_DIR / "chronicle_count_at_compile.json"
            count_path.parent.mkdir(parents=True, exist_ok=True)
            count_path.write_text(
                _json.dumps({"n_chronicle": n, "ts": _dt.now().isoformat()}),
                encoding="utf-8",
            )
        except Exception:
            pass

    def _chronicle_grew_since_compile(self, n_now: int) -> bool:
        """FIX БАГ 7: метод отсутствовал в DSPyOrchestrator."""
        try:
            from .config import VAULT_DIR
            import json as _json
            count_path = VAULT_DIR / "chronicle_count_at_compile.json"
            if not count_path.exists():
                return True
            saved = _json.loads(count_path.read_text(encoding="utf-8"))
            n_saved = int(saved.get("n_chronicle", 0))
            return n_now > n_saved
        except Exception:
            return True

    """
    Фасад v9.9: управляет полным жизненным циклом DSPy.

    Использование в engine.py (Siege Mode 2.0):
        orch = DSPyOrchestrator()
        orch.siege_compile()         # compile BEFORE PySR
        nav_result = orch.navigate(data_meta, failure_logs)
        audit_result = orch.audit(role, task, formula, metrics)
        reflect = orch.reflect_on_death(death_report, data_meta, attempt)
        orch.promote_to_gold(formula)  # после успеха
    """

    def __init__(
        self,
        model: str = NAVIGATOR_MODEL,  # v10.14: deepseek для Navigator
        host:  str = OLLAMA_HOST,
    ):
        self.model   = model
        self.host    = host
        self.nav_mod: Optional[Any] = None
        self.audit_mod: Optional[Any] = None
        self.hadi_mod: Optional[Any] = None
        self._lm_ok  = False
        self.loader  = GoldLoader()
        # v10.16: кэш навигатора в рамках сессии.
        # Ключ = hash(data_meta + failure_logs), значение = dict результата.
        # Сбрасывается при каждом новом siege_compile().
        self._nav_cache: Dict[str, Dict] = {}


    # ── Siege Mode 2.0: компиляция ДО PySR ───────────────────────

    def siege_compile(self) -> bool:
        """
        Протокол Siege Mode 2.0:
          1. Инициализируем LM (Ollama должна быть живой)
          2. Загружаем скомпилированную модель (если свежая)
          3. Если нет / устарела → BootstrapFewShot из gold_formulas.json
          4. Сохраняем результат в dspy_compiled_model.json
          5. Возвращаем True (после этого engine.py делает ollama_stop())

        НЕ запускает PySR — только компилирует промпты.
        """
        if not DSPY_AVAILABLE:
            log.info("[Siege] DSPy недоступен. Legacy-режим.")
            return False

        # v10.16: сбрасываем кэш при каждой новой компиляции
        self._nav_cache = {}

        # Шаг 1: LM
        print(f"  [Siege] Шаг 1/4: подключаемся к Ollama ({NAVIGATOR_MODEL})…")
        self._lm_ok = init_dspy_lm(self.model, self.host)
        if not self._lm_ok:
            log.warning("[Siege] Ollama недоступна. Legacy-режим.")
            print(
                "\n  [DSPy] ⚠️  Ollama не запущена — роли работают в Legacy-режиме.\n"
                "  Для полного режима: запусти 'ollama serve' в отдельном окне,\n"
                "  затем перезапусти run_test.py.\n"
            )
            return False

        # Шаг 2: Audit и HADI — всегда base (не требуют компиляции)
        print("  [Siege] Шаг 2/4: инициализация Audit и HADI модулей…")
        self.audit_mod = AuditModule()
        self.hadi_mod  = HADIReflectModule()

        # Шаг 3: Nav — загружаем или компилируем
        print("  [Siege] Шаг 3/4: загружаем/компилируем Navigator…")
        # v10.14: Кэш инвалидируется если накопились новые chronicle записи
        # Без этого новые знания Летописца не попадают в DSPy 7 дней
        cached     = load_compiled_model()
        n_chronicle_now = self._count_chronicle_records()

        if cached is not None and not self._chronicle_grew_since_compile(n_chronicle_now):
            log.info("[Siege] Загружена кэшированная модель (%d chronicle записей).",
                     n_chronicle_now)
            self.nav_mod = cached
        else:
            if cached is not None:
                log.info("[Siege] Перекомпиляция: накопилось новых chronicle записей.")
            else:
                log.info("[Siege] Компилируем NavModule через BootstrapFewShot…")

            gold_ex      = self.loader.load_examples()
            chronicle_ex = self.loader.load_chronicle_examples()
            curr_ex      = self.loader.load_curriculum_examples()

            # v10.14: мета-примеры (5× усиление, синтез всего опыта)
            try:
                from .meta_reflection import load_meta_examples as _load_meta
                meta_ex = _load_meta()
            except Exception:
                meta_ex = []

            # v10.14: примеры-открытия (5× — самые ценные, система нашла новый закон)
            try:
                from .discovery import load_discoveries_as_examples as _load_disc
                disc_ex = _load_disc()
            except Exception:
                disc_ex = []

            # v10.14: примеры из найденных инвариантов (5×: dim_pattern → структура)
            inv_ex = self.loader.load_invariant_examples()

            # v10.15: антипримеры — СТОП-паттерны отклонённые Матрёшкой (плохой R²)
            # Ставим в начало: Navigator должен видеть их первыми
            rejected_ex = self.loader.load_rejected_examples()

            # v10.15: спорные — хороший R², но LLM усомнилась (мягкое предупреждение)
            disputed_ex = self.loader.load_disputed_examples()

            # Rejected(жёсткий стоп) → Disputed(мягкий) → Gold → Invariants → Discoveries → Meta → Chronicle → Curriculum
            all_examples = rejected_ex + disputed_ex + gold_ex + inv_ex + disc_ex + meta_ex + chronicle_ex + curr_ex
            log.info(
                "[Siege] Trainset: %d rejected(СТОП) + %d disputed(спорные) + %d gold + "
                "%d invariants + %d discoveries + %d meta + %d chronicle + %d curriculum = %d итого",
                len(rejected_ex), len(disputed_ex), len(gold_ex), len(inv_ex), len(disc_ex),
                len(meta_ex), len(chronicle_ex), len(curr_ex), len(all_examples),
            )
            self.nav_mod = compile_navigator(all_examples)
            if self.nav_mod is not None:
                self._save_chronicle_count(n_chronicle_now)
                save_compiled_model(self.nav_mod)

        # Шаг 4: HADIModule дообучаем на failure log
        print("  [Siege] Шаг 4/4: дообучение HADI на истории провалов…")
        fail_ex = self.loader.load_failure_examples()
        if fail_ex and self.hadi_mod is not None and DSPY_AVAILABLE:
            try:
                opt = BootstrapFewShot(
                    metric=lambda ex, pred, trace=None: bool(
                        getattr(pred, "failure_type", "")
                        and getattr(pred, "new_hypotheses", "")
                    ),
                    max_bootstrapped_demos=2,
                    max_labeled_demos=min(len(fail_ex), 3),
                )
                self.hadi_mod = opt.compile(HADIReflectModule(), trainset=fail_ex)
                log.info("[Siege] HADIModule дообучен на %d примерах смерти.", len(fail_ex))
            except Exception as e:
                log.warning("[Siege] HADIModule компиляция упала: %s", e)

        log.info("[Siege] ✓ Компиляция завершена. Готов к ollama_stop() → PySR.")
        print("  [Siege] ✓ Компиляция завершена!")
        return True

    # ── Navigation ────────────────────────────────────────────────

    def navigate(
        self,
        data_meta:    str,
        failure_logs: str = "[]",
    ) -> Dict:
        """
        Запрашивает стратегию у Штурмана.
        Возвращает dict с ключами как у NavSignature.
        При ошибке — fallback словарь.

        v10.16: кэш по хэшу (data_meta + failure_logs).
        Если входы не изменились — возвращает кэшированный результат
        без повторного вызова LLM.
        """
        if not self._lm_ok or self.nav_mod is None:
            return self._nav_fallback()

        # v10.16: ключ кэша только по data_meta — она стабильна в рамках сессии.
        # failure_logs намеренно исключён из ключа: он меняется каждую HADI итерацию
        # (добавляются новые провалы), поэтому хэш по нему всегда разный и кэш
        # никогда не срабатывал бы. Navigator при HIT всё равно получает свежий
        # failure_logs через обогащение в engine.py ПОСЛЕ вызова navigate().
        import hashlib as _hashlib
        _cache_key = _hashlib.md5(data_meta.encode("utf-8")).hexdigest()

        if _cache_key in self._nav_cache:
            log.debug("[NavCache] HIT key=%s — пропускаем LLM вызов", _cache_key[:8])
            # BUG FIX: возвращаем копию, не ссылку — защита от мутации кэша
            return dict(self._nav_cache[_cache_key])

        try:
            import threading as _threading
            import time as _time

            # v10.16: индикатор что Navigator думает (DSPy синхронный — лог молчит)
            _stop_indicator = _threading.Event()

            _effective_timeout = NAV_TIMEOUT_SEC
            print(f"  [Navigator] 🧠 {self.model} (таймаут {NAV_TIMEOUT_SEC}с)", flush=True)

            def _thinking_dots(stop_event):
                _elapsed = 0
                while not stop_event.is_set():
                    _time.sleep(15)
                    _elapsed += 15
                    if not stop_event.is_set():
                        left = _effective_timeout - _elapsed
                        print(f"  [Navigator] ⏳ думает… {_elapsed}с (таймаут через {max(0,left)}с)", flush=True)

            _indicator_thread = _threading.Thread(
                target=_thinking_dots, args=(_stop_indicator,), daemon=True
            )
            _indicator_thread.start()

            # Запускаем nav_mod в отдельном потоке с таймаутом
            _pred_result = [None]
            _pred_error  = [None]

            def _run_nav():
                try:
                    _pred_result[0] = self.nav_mod(
                        data_meta    = data_meta,
                        failure_logs = failure_logs,
                    )
                except Exception as _e:
                    _pred_error[0] = _e

            _nav_thread = _threading.Thread(target=_run_nav, daemon=True)
            _nav_thread.start()
            _nav_thread.join(timeout=_effective_timeout)
            _stop_indicator.set()

            if _nav_thread.is_alive():
                log.warning("[DSPy Nav] Таймаут %ds — Fallback.", _effective_timeout)
                print(f"  [Navigator] ⚠️  Таймаут {_effective_timeout}с — Fallback.", flush=True)
                return self._nav_fallback()

            if _pred_error[0] is not None:
                raise _pred_error[0]

            pred = _pred_result[0]

            # v10.16: показываем цепочку мышления deepseek-r1 (<think>...</think>)
            try:
                _lm_history = dspy.settings.lm.history
                if _lm_history:
                    _last = _lm_history[-1]
                    _raw = ""
                    # DSPy хранит ответ в разных местах в зависимости от версии
                    if isinstance(_last, dict):
                        _raw = (
                            _last.get("response", "")
                            or _last.get("outputs", [""])[0]
                            or ""
                        )
                    # Ищем <think>...</think>
                    import re as _re
                    _think_match = _re.search(r"<think>(.*?)</think>", str(_raw), _re.DOTALL)
                    if _think_match:
                        _think_text = _think_match.group(1).strip()
                        # Показываем первые 400 символов чтобы не заспамить лог
                        _think_short = _think_text[:400] + ("…" if len(_think_text) > 400 else "")
                        print(f"\n  [Navigator 🧠 think]\n{_think_short}\n")
            except Exception:
                pass  # не критично — просто не показываем

            result = {
                "selected_features":  getattr(pred, "selected_features", ""),
                "selected_operators": getattr(pred, "selected_operators", "+,-,*,/"),
                "hypotheses":         getattr(pred, "hypotheses", ""),
                "ooda_stable":        getattr(pred, "ooda_stable", "true"),
                "reasoning":          getattr(pred, "reasoning", ""),
            }

            # v10.16: итоговый вывод что решил Navigator
            print(f"\n  [Navigator ✅ решение]")
            print(f"  Признаки:  {result['selected_features']}")
            print(f"  Операторы: {result['selected_operators']}")
            print(f"  Гипотезы:  {result['hypotheses']}")
            print(f"  Reasoning: {result['reasoning'][:120]}")
            print(f"  OODA:      {result['ooda_stable']}\n")

            # v10.16: сохраняем в кэш
            self._nav_cache[_cache_key] = result
            log.debug("[NavCache] MISS key=%s — результат закэширован", _cache_key[:8])
            return result
        except Exception as e:
            log.warning("[DSPy Nav] Ошибка: %s. Fallback.", e)
            return self._nav_fallback()

    # ── Audit ─────────────────────────────────────────────────────

    def audit_role(
        self,
        role_name:       str,
        role_task:       str,
        formula:         str,
        formula_metrics: str,
    ) -> Tuple[str, str]:
        """
        Запрашивает вердикт одной роли Матрёшки.
        Возвращает (verdict, analysis).
        """
        if not self._lm_ok or self.audit_mod is None:
            return "УСЛОВНО", f"[{role_name}] DSPy недоступен."
        # v10.14: обогащаем formula_metrics мета-контекстом и контекстом открытий
        try:
            from .meta_context import get_matryoshka_context as _mat_ctx
            _extra = _mat_ctx(formula=formula, role=role_name)
            if _extra:
                formula_metrics = formula_metrics + f"\n{_extra[:200]}"
        except Exception:
            pass

        try:
            pred = self.audit_mod(
                role_name       = role_name,
                role_task       = role_task,
                formula         = formula,
                formula_metrics = formula_metrics,
            )
            verdict  = getattr(pred, "verdict",  "УСЛОВНО").strip().upper()
            analysis = getattr(pred, "analysis", "").strip()

            # Нормализация: только три допустимых значения
            if "ОТКЛОНЕНА" in verdict:
                verdict = "ОТКЛОНЕНА"
            elif "ПРИНЯТА" in verdict:
                verdict = "ПРИНЯТА"
            else:
                verdict = "УСЛОВНО"

            return verdict, analysis
        except Exception as e:
            log.warning("[DSPy Audit] %s: %s. Fallback УСЛОВНО.", role_name, e)
            return "УСЛОВНО", f"[{role_name}] DSPy ошибка: {e}"

    # ── HADI Reflection ───────────────────────────────────────────

    def reflect_on_death(
        self,
        death_report:   Dict,
        data_meta:      str,
        attempt_number: str,
    ) -> Dict:
        """
        Анализирует смерть через HADIReflectModule.
        Записывает результат в failure log для будущего обучения.
        """
        report_str = json.dumps(death_report, ensure_ascii=False)
        result = {
            "failure_type":       "UNKNOWN",
            "corrected_strategy": "",
            "new_hypotheses":     "",
        }

        # v10.14 БАГ 5: HADI видит исторические паттерны ошибок из мета-рефлексии
        try:
            from .meta_context import get_hadi_context as _hctx
            _hmeta = _hctx()
            if _hmeta:
                data_meta = data_meta + "\n" + _hmeta[:200]
        except Exception:
            pass

        if self._lm_ok and self.hadi_mod is not None:
            try:
                pred = self.hadi_mod(
                    death_report   = report_str,
                    data_meta      = data_meta,
                    attempt_number = attempt_number,
                )
                result["failure_type"]       = getattr(pred, "failure_type", "UNKNOWN")
                result["corrected_strategy"] = getattr(pred, "corrected_strategy", "")
                result["new_hypotheses"]     = getattr(pred, "new_hypotheses", "")
                log.info("[HADI Reflect] failure_type=%s", result["failure_type"])
            except Exception as e:
                log.warning("[HADI Reflect] Ошибка: %s", e)

        # Записываем в failure log в любом случае
        log_failure_example(
            death_report       = death_report,
            data_meta          = data_meta,
            attempt_number     = attempt_number,
            failure_type       = result["failure_type"],
            corrected_strategy = result["corrected_strategy"],
            new_hypotheses     = result["new_hypotheses"],
        )
        return result

    # ── Утилиты ───────────────────────────────────────────────────

    @staticmethod
    def _nav_fallback() -> Dict:
        return {
            "selected_features":  "f0,f1",
            "selected_operators": "+,-,*,/,sqrt,log",
            "hypotheses":         "f0/f1;f0*f1;sqrt(f0)*f1;f0+f1;f0*f0/f1",
            "ooda_stable":        "true",
            "reasoning":          "Fallback: DSPy недоступен.",
        }

    @property
    def is_active(self) -> bool:
        return DSPY_AVAILABLE and self._lm_ok
