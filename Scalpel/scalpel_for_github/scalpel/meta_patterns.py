"""
meta_patterns.py — Мета-паттерны v1.0 (уровень 4, пункт 3)

Система накапливает не воспоминания о конкретных законах,
а ПРАВИЛА вида:
    "когда outlier_ratio > 3% И ratio > 40 → всегда log, не sqrt"
    "когда n_features=1 И ratio < 5 → Navigator пробует степени"

Архитектура:
  PatternStore   — JSON-хранилище правил с доказательствами
  MetaPattern    — одно правило: условие + действие + статистика
  MetaPatternEngine — поиск, применение, обновление, извлечение

Интеграция:
  1. Начало закона  → find_matching(data_stats) → ctx.active_patterns
  2. Хирург         → читает паттерны → корректирует iqr/cut
  3. Препаратор     → читает паттерны → уточняет трансформацию
  4. Navigator      → читает паттерны → уточняет операторы/гипотезы
  5. Конец закона   → update_from_result(r2) → success_rate растёт
  6. После 10 законов → extract_from_logs() LLM извлекает новые паттерны
"""
from __future__ import annotations

import json
import logging
import time
import uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Any

import numpy as np

log = logging.getLogger("scalpel")

try:
    from .config import VAULT_DIR, OLLAMA_HOST, SYNTHESIS_MODEL
    _VAULT_DIR     = VAULT_DIR
    _OLLAMA_HOST   = OLLAMA_HOST
    _EXTRACT_MODEL = SYNTHESIS_MODEL   # gemma2:9b для извлечения паттернов
except ImportError:
    _VAULT_DIR     = Path("scalpel_vault")
    _OLLAMA_HOST   = "http://localhost:11434"
    _EXTRACT_MODEL = "gemma2:9b"

PATTERNS_PATH  = _VAULT_DIR / "meta_patterns.json"
MIN_EVIDENCE   = 3     # минимум наблюдений чтобы паттерн считался надёжным
MIN_SUCCESS    = 0.6   # минимум success_rate чтобы паттерн применялся
EXTRACT_EVERY  = 10    # извлекать новые паттерны каждые N законов


# ═══════════════════════════════════════════════════════════════════
# СТРУКТУРА ПАТТЕРНА
# ═══════════════════════════════════════════════════════════════════

@dataclass
class MetaPattern:
    """
    Одно обобщённое правило вида:
        ЕСЛИ [условия на данные] → ТО [рекомендация для модуля]
    """
    pattern_id:   str
    target:       str   # "surgeon" | "preparator" | "navigator"
    description:  str   # человекочитаемое описание

    # ── Условия (все должны выполняться) ─────────────────────────
    cond_outlier_ratio_min: float = 0.0     # outlier_ratio >=
    cond_outlier_ratio_max: float = 1.0     # outlier_ratio <=
    cond_ratio_min:         float = 0.0     # data ratio >=
    cond_ratio_max:         float = 1e9     # data ratio <=
    cond_negative_min:      float = 0.0     # negative_fraction >=
    cond_negative_max:      float = 1.0     # negative_fraction <=
    cond_n_features:        int   = -1      # -1=любое, N=точно N признаков
    cond_n_features_min:    int   = 0       # минимум признаков (0=без ограничения)
    cond_n_features_max:    int   = 99      # максимум признаков

    # ── Действие (с дефолтами чтобы соблюсти порядок dataclass) ─
    action_type:  str   = ""   # "transform" | "iqr_adjust" | "operator_add" | "hypothesis_template"
    action_value: str   = ""   # "log" | "2.5" | "sqrt" | "f0^2" | ...

    # ── Доказательства ───────────────────────────────────────────
    evidence_count: int   = 0
    success_count:  int   = 0
    created_at:     str   = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%S"))
    last_updated:   str   = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%S"))
    source:         str   = "manual"   # "manual" | "llm_extracted" | "auto_inferred"

    @property
    def success_rate(self) -> float:
        return self.success_count / max(self.evidence_count, 1)

    @property
    def is_reliable(self) -> bool:
        """Паттерн достаточно подтверждён чтобы применяться."""
        return (self.evidence_count >= MIN_EVIDENCE
                and self.success_rate >= MIN_SUCCESS)

    def matches(self, data_stats: dict) -> bool:
        """Проверяет применим ли паттерн к данным."""
        out_r = data_stats.get("outlier_ratio",   0.0)
        ratio = data_stats.get("ratio",            1.0)
        neg   = data_stats.get("negative_fraction", 0.0)
        nf    = data_stats.get("n_features",        1)

        return (
            self.cond_outlier_ratio_min <= out_r <= self.cond_outlier_ratio_max
            and self.cond_ratio_min     <= ratio <= self.cond_ratio_max
            and self.cond_negative_min  <= neg   <= self.cond_negative_max
            and (self.cond_n_features == -1 or self.cond_n_features == nf)
            and nf >= self.cond_n_features_min   # BUG FIX: проверяем минимум признаков
            and nf <= self.cond_n_features_max
        )

    def as_hint(self) -> str:
        """Строка-подсказка для промпта LLM."""
        conf = f"{self.success_rate:.0%}" if self.evidence_count >= MIN_EVIDENCE else "новый"
        return (
            f"[Паттерн/{self.target}] {self.description} "
            f"(успех={conf}, n={self.evidence_count})"
        )

    def to_dict(self) -> dict:
        d = asdict(self)
        d["success_rate"] = self.success_rate
        d["is_reliable"]  = self.is_reliable
        return d


# ═══════════════════════════════════════════════════════════════════
# ХРАНИЛИЩЕ
# ═══════════════════════════════════════════════════════════════════

class PatternStore:
    """JSON-хранилище паттернов с атомарной записью."""

    def __init__(self, path: Path = PATTERNS_PATH):
        self.path = path
        self._patterns: List[MetaPattern] = []
        self._loaded = False

    def load(self) -> List[MetaPattern]:
        if self._loaded:
            return self._patterns
        self._patterns = []
        if self.path.exists():
            try:
                raw = json.loads(self.path.read_text(encoding="utf-8"))
                for p in raw.get("patterns", []):
                    try:
                        # Убираем вычисляемые поля перед созданием dataclass
                        p.pop("success_rate", None)
                        p.pop("is_reliable",  None)
                        self._patterns.append(MetaPattern(**p))
                    except Exception as e:
                        log.debug("[MetaPatterns] Пропущен паттерн: %s", e)
                log.info("[MetaPatterns] Загружено %d паттернов", len(self._patterns))
            except Exception as e:
                log.warning("[MetaPatterns] Ошибка загрузки: %s", e)
        else:
            # Первый запуск — создаём базовые паттерны из экспертных знаний
            self._patterns = _build_seed_patterns()
            self.save()
        self._loaded = True
        return self._patterns

    def save(self) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "version":    "1.0",
                "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "count":      len(self._patterns),
                "patterns":   [p.to_dict() for p in self._patterns],
            }
            self.path.write_text(
                json.dumps(data, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception as e:
            log.warning("[MetaPatterns] Ошибка сохранения: %s", e)

    def add(self, pattern: MetaPattern) -> None:
        self._patterns.append(pattern)

    def get_all(self) -> List[MetaPattern]:
        return self._patterns


# ═══════════════════════════════════════════════════════════════════
# ПАТТЕРНЫ-ЗАТРАВКИ (экспертные знания)
# ═══════════════════════════════════════════════════════════════════

def _build_seed_patterns() -> List[MetaPattern]:
    """
    Начальный набор паттернов из физических принципов.
    Они начинают с evidence_count=0 и накапливают подтверждения.
    """
    seeds = [
        # ── Препаратор ────────────────────────────────────────────
        MetaPattern(
            pattern_id   = "prep_log_high_ratio",
            target       = "preparator",
            description  = "ratio > 100 И positive > 90% → log трансформация",
            cond_ratio_min      = 100.0,
            cond_negative_max   = 0.10,
            action_type  = "transform",
            action_value = "log",
            source       = "expert",
        ),
        MetaPattern(
            pattern_id   = "prep_sqrt_medium_ratio",
            target       = "preparator",
            description  = "20 < ratio <= 100 И negative < 5% → sqrt трансформация",
            cond_ratio_min      = 20.0,
            cond_ratio_max      = 100.0,
            cond_negative_max   = 0.05,
            action_type  = "transform",
            action_value = "sqrt",
            source       = "expert",
        ),
        MetaPattern(
            pattern_id   = "prep_none_with_negatives",
            target       = "preparator",
            description  = "negative 5–15% → не применять sqrt (искажает данные)",
            cond_negative_min   = 0.05,
            cond_negative_max   = 0.14,   # FIX v10.27 #9: не перекрываться с standardize (>15%)
            action_type  = "transform",
            action_value = "none",
            source       = "expert",
        ),
        MetaPattern(
            pattern_id   = "prep_standardize_many_negatives",
            target       = "preparator",
            description  = "negative > 15% → standardize",
            cond_negative_min   = 0.15,
            action_type  = "transform",
            action_value = "standardize",
            source       = "expert",
        ),

        # ── Хирург ────────────────────────────────────────────────
        MetaPattern(
            pattern_id   = "surgeon_aggressive_high_outliers",
            target       = "surgeon",
            description  = "outlier_ratio > 5% → агрессивная хирургия (iqr_k=2.0)",
            cond_outlier_ratio_min = 0.05,
            action_type  = "iqr_adjust",
            action_value = "2.0",
            source       = "expert",
        ),
        MetaPattern(
            pattern_id   = "surgeon_gentle_low_outliers",
            target       = "surgeon",
            description  = "outlier_ratio < 2% → мягкая хирургия (iqr_k=3.5)",
            cond_outlier_ratio_max = 0.02,
            action_type  = "iqr_adjust",
            action_value = "3.5",
            source       = "expert",
        ),

        # ── Navigator ─────────────────────────────────────────────
        MetaPattern(
            pattern_id   = "nav_powers_single_feature",
            target       = "navigator",
            description  = "n_features=1 И ratio < 10 → пробовать степени f0^2, f0^3",
            cond_n_features     = 1,
            cond_ratio_max      = 10.0,
            action_type  = "hypothesis_template",
            action_value = "f0^2;f0^3;f0^0.5",
            source       = "expert",
        ),
        MetaPattern(
            pattern_id        = "nav_division_multi_feature",
            target            = "navigator",
            description       = "n_features >= 3 И ratio > 20 → пробовать деление признаков",
            cond_ratio_min    = 20.0,
            cond_n_features     = -1,
            cond_n_features_min = 3,   # BUG FIX: минимум 3 признака
            cond_n_features_max = 99,
            action_type  = "hypothesis_template",
            action_value = "f0/f1;f0*f1/f2;f0/(f1*f2)",
            source       = "expert",
        ),
        MetaPattern(
            pattern_id   = "nav_log_operators_high_ratio",
            target       = "navigator",
            description  = "ratio > 100 → добавить log и exp в операторы",
            cond_ratio_min  = 100.0,
            action_type  = "operator_add",
            action_value = "log;exp",
            source       = "expert",
        ),
        MetaPattern(
            pattern_id   = "nav_sqrt_operator_medium_ratio",
            target       = "navigator",
            description  = "20 < ratio <= 100 И negative < 5% → добавить sqrt в операторы",
            cond_ratio_min      = 20.0,
            cond_ratio_max      = 100.0,
            cond_negative_max   = 0.05,
            action_type  = "operator_add",
            action_value = "sqrt",
            source       = "expert",
        ),
    ]
    log.info("[MetaPatterns] Созданы %d seed паттернов", len(seeds))
    return seeds


# ═══════════════════════════════════════════════════════════════════
# ГЛАВНЫЙ ДВИЖОК
# ═══════════════════════════════════════════════════════════════════

class MetaPatternEngine:
    """
    Движок мета-паттернов.
    Один экземпляр на весь бенчмарк (создаётся в run_feynman.py или engine.py).
    """

    def __init__(self):
        self._store     = PatternStore()
        self._patterns  = self._store.load()
        self._law_count = self._load_law_count()

    # ─────────────────────────────────────────────────────────────
    # 1. Поиск применимых паттернов
    # ─────────────────────────────────────────────────────────────

    def find_matching(
        self,
        data_stats:    dict,
        targets:       List[str] = None,
        reliable_only: bool      = False,
    ) -> List[MetaPattern]:
        """
        Находит паттерны применимые к данным.

        data_stats должен содержать:
            outlier_ratio, ratio, negative_fraction, n_features

        reliable_only=True → только паттерны с evidence >= MIN_EVIDENCE
        """
        result = []
        for p in self._patterns:
            if targets and p.target not in targets:
                continue
            if reliable_only and not p.is_reliable:
                continue
            if p.matches(data_stats):
                result.append(p)
        return result

    def hints_for_prompt(
        self,
        data_stats: dict,
        target:     str,
    ) -> str:
        """
        Возвращает строку подсказок для промпта конкретного модуля.
        Включает только надёжные паттерны (или seed если нет накопленных).
        """
        matching = self.find_matching(data_stats, targets=[target])
        if not matching:
            return ""
        lines = ["[МетаПаттерны — накопленный опыт системы]"]
        for p in matching[:4]:   # максимум 4 подсказки
            lines.append(f"  • {p.as_hint()}")
        return "\n".join(lines)

    def action_value_for(
        self,
        data_stats:  dict,
        target:      str,
        action_type: str,
    ) -> Optional[str]:
        """
        Возвращает значение действия для конкретного типа.
        Например: action_value_for(stats, "preparator", "transform") → "log"
        Приоритет: надёжные паттерны > seed паттерны > None.
        """
        matching = [
            p for p in self.find_matching(data_stats, targets=[target])
            if p.action_type == action_type
        ]
        if not matching:
            return None
        # Сортируем: сначала надёжные, потом по success_rate
        matching.sort(key=lambda p: (p.is_reliable, p.success_rate), reverse=True)
        return matching[0].action_value

    # ─────────────────────────────────────────────────────────────
    # 2. Обновление после результата
    # ─────────────────────────────────────────────────────────────

    def update_from_result(
        self,
        data_stats:      dict,
        actions_taken:   dict,   # {"transform": "log", "iqr_k": 3.5, ...}
        r2_result:       float,
        success_threshold: float = 0.5,
    ) -> None:
        """
        Обновляет success_count/evidence_count для применённых паттернов.
        Вызывается после каждого завершённого закона.
        """
        success = r2_result >= success_threshold
        updated = 0

        for p in self._patterns:
            if not p.matches(data_stats):
                continue

            # Проверяем совпало ли действие паттерна с тем что сделала система
            action_matches = False
            if p.action_type == "transform":
                action_matches = (actions_taken.get("transform") == p.action_value)
            elif p.action_type == "iqr_adjust":
                iqr_taken = actions_taken.get("iqr_k", 3.0)
                iqr_pat   = float(p.action_value)
                action_matches = abs(iqr_taken - iqr_pat) < 0.6
            elif p.action_type == "operator_add":
                ops_taken = set(actions_taken.get("operators", []))
                pat_ops   = set(p.action_value.split(";"))
                action_matches = bool(ops_taken & pat_ops)
            elif p.action_type == "hypothesis_template":
                # Считаем совпадение если хоть один template использовался
                hyps_taken = actions_taken.get("hypotheses", [])
                templates  = p.action_value.split(";")
                action_matches = any(t in str(hyps_taken) for t in templates)

            if action_matches:
                p.evidence_count += 1
                if success:
                    p.success_count += 1
                p.last_updated = time.strftime("%Y-%m-%dT%H:%M:%S")
                updated += 1

        self._law_count += 1
        self._save_law_count()

        if updated > 0:
            self._store.save()
            log.info("[MetaPatterns] Обновлено %d паттернов (R²=%.4f, success=%s)",
                     updated, r2_result, success)

        # Каждые EXTRACT_EVERY законов — LLM извлекает новые паттерны
        if self._law_count % EXTRACT_EVERY == 0:
            self._extract_from_logs()

    # ─────────────────────────────────────────────────────────────
    # 3. LLM-извлечение новых паттернов из логов
    # ─────────────────────────────────────────────────────────────

    def _extract_from_logs(self) -> None:
        """
        Читает накопленные логи и просит LLM сформулировать новые паттерны.
        Вызывается автоматически каждые EXTRACT_EVERY законов.
        """
        import urllib.request

        # Собираем данные из логов
        surgeon_data   = _read_jsonl(_VAULT_DIR / "surgeon_log.jsonl",   limit=20)
        preparator_data = _read_jsonl(_VAULT_DIR / "preparator_log.jsonl", limit=20)

        if len(surgeon_data) + len(preparator_data) < 5:
            log.info("[MetaPatterns] Мало данных для извлечения паттернов")
            return

        # Текущие паттерны для контекста
        current = [p.description for p in self._patterns if p.evidence_count > 0]

        prompt = f"""You are analyzing accumulated data from a symbolic regression system.
Extract NEW generalizable patterns that aren't already known.

SURGEON LOG (last {len(surgeon_data)} laws):
{json.dumps(surgeon_data[:10], ensure_ascii=False, indent=1)}

PREPARATOR LOG (last {len(preparator_data)} laws):
{json.dumps(preparator_data[:10], ensure_ascii=False, indent=1)}

ALREADY KNOWN PATTERNS:
{chr(10).join(f"  - {p}" for p in current[:8])}

Find up to 3 NEW patterns not already known. For each pattern provide:
- target: "surgeon" or "preparator" or "navigator"
- description: one sentence rule in Russian
- condition: which data statistics trigger this (outlier_ratio, ratio, negative_fraction, n_features)
- action_type: "transform" or "iqr_adjust" or "operator_add" or "hypothesis_template"
- action_value: the specific value (e.g. "log", "2.5", "sqrt", "f0^2;f0^3")

Respond ONLY with JSON array:
[
  {{
    "target": "preparator",
    "description": "...",
    "cond_ratio_min": 0,
    "cond_ratio_max": 1000000,
    "cond_outlier_ratio_min": 0,
    "cond_outlier_ratio_max": 1,
    "cond_negative_min": 0,
    "cond_negative_max": 1,
    "cond_n_features": -1,
    "action_type": "transform",
    "action_value": "log"
  }}
]"""

        try:
            payload = json.dumps({
                "model":  _EXTRACT_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.3, "num_predict": 600},
            }).encode()
            req = urllib.request.Request(
                f"{_OLLAMA_HOST}/api/generate",
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=120) as resp:
                text = json.loads(resp.read()).get("response", "").strip()

            # Парсим JSON
            start = text.find("[")
            end   = text.rfind("]") + 1
            if start < 0 or end <= start:
                return
            new_patterns_raw = json.loads(text[start:end])

            added = 0
            existing_descs = {p.description for p in self._patterns}

            for raw in new_patterns_raw:
                desc = raw.get("description", "")
                if not desc or desc in existing_descs:
                    continue
                if raw.get("action_type") not in ("transform", "iqr_adjust",
                                                    "operator_add", "hypothesis_template"):
                    continue
                try:
                    new_p = MetaPattern(
                        pattern_id   = f"llm_{uuid.uuid4().hex[:8]}",
                        target       = raw.get("target", "preparator"),
                        description  = desc,
                        cond_outlier_ratio_min = float(raw.get("cond_outlier_ratio_min", 0)),
                        cond_outlier_ratio_max = float(raw.get("cond_outlier_ratio_max", 1)),
                        cond_ratio_min         = float(raw.get("cond_ratio_min", 0)),
                        cond_ratio_max         = float(raw.get("cond_ratio_max", 1e9)),
                        cond_negative_min      = float(raw.get("cond_negative_min", 0)),
                        cond_negative_max      = float(raw.get("cond_negative_max", 1)),
                        cond_n_features        = int(raw.get("cond_n_features", -1)),
                        action_type  = raw.get("action_type", "transform"),
                        action_value = str(raw.get("action_value", "none")),
                        source       = "llm_extracted",
                    )
                    self._store.add(new_p)
                    self._patterns.append(new_p)
                    existing_descs.add(desc)
                    added += 1
                    log.info("[MetaPatterns] Новый паттерн: %s", desc)
                except Exception as pe:
                    log.debug("[MetaPatterns] Паттерн пропущен: %s", pe)

            if added > 0:
                self._store.save()
                print(f"  [МетаПаттерны] ✨ Извлечено {added} новых паттернов из логов")

        except Exception as e:
            log.warning("[MetaPatterns] LLM извлечение: %s", e)

    # ─────────────────────────────────────────────────────────────
    # Вспомогательное
    # ─────────────────────────────────────────────────────────────

    def _load_law_count(self) -> int:
        counter_path = _VAULT_DIR / "meta_patterns_counter.json"
        try:
            if counter_path.exists():
                return json.loads(counter_path.read_text())["count"]
        except Exception:
            pass
        return 0

    def _save_law_count(self) -> None:
        counter_path = _VAULT_DIR / "meta_patterns_counter.json"
        try:
            counter_path.parent.mkdir(parents=True, exist_ok=True)
            counter_path.write_text(json.dumps({"count": self._law_count}))
        except Exception:
            pass

    def summary(self) -> str:
        total    = len(self._patterns)
        reliable = sum(1 for p in self._patterns if p.is_reliable)
        return (f"МетаПаттерны: {total} правил ({reliable} надёжных), "
                f"законов обработано: {self._law_count}")


# ═══════════════════════════════════════════════════════════════════
# УТИЛИТЫ
# ═══════════════════════════════════════════════════════════════════

def _read_jsonl(path: Path, limit: int = 20) -> List[dict]:
    if not path.exists():
        return []
    try:
        lines = path.read_text(encoding="utf-8").strip().splitlines()
        result = []
        for line in lines[-limit:]:
            try:
                result.append(json.loads(line))
            except Exception:
                pass
        return result
    except Exception:
        return []


# Глобальный экземпляр — создаётся один раз при импорте модуля
_engine: Optional[MetaPatternEngine] = None

def get_pattern_engine() -> MetaPatternEngine:
    """Возвращает глобальный движок паттернов (ленивая инициализация)."""
    global _engine
    if _engine is None:
        _engine = MetaPatternEngine()
        log.info("[MetaPatterns] %s", _engine.summary())
    return _engine
