"""
ram_queue.py — Строгая очередь RAM v9.9.2-dataaware.

ПРИНЦИП: каждая роль Матрёшки — отдельный слот.
Протокол одного слота:

  ┌─────────────────────────────────────────────────────┐
  │  1. RAM CHECK   — достаточно памяти?                │
  │  2. LOAD        — инициализируем модуль роли        │
  │  3. RUN         — один вызов LLM                    │
  │  4. UNLOAD      — del module                        │
  │  5. ollama_stop — освобождаем VRAM                  │
  │  6. gc.collect  — возвращаем страницы ОС            │
  │  7. SETTLE      — sleep(cooldown)                   │
  └─────────────────────────────────────────────────────┘

Между ролями никогда не держим два модуля в RAM одновременно.

v9.9.2-dataaware:
  НОВОЕ: LLM теперь видит реальные данные, а не только формулу.
  _build_data_context() — формирует блок контекста:
    - реальные имена признаков (через shadow→real маппинг)
    - 8 строк реальных данных (X_samples + y_samples)
    - предсказания формулы на этих строках (y_pred_samples)
    - остатки (residuals) — где именно формула ошибается
    - статистика остатков: max_error, bias, worst_rows
  Этот блок инжектируется в промпт каждой роли (Legacy и DSPy).
  engine.py должен передавать: data_samples, y_pred_samples, real_names.
"""
from __future__ import annotations

import gc
import json
import logging
import os
import platform
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple

import numpy as np

from .config import (
    OLLAMA_MODEL, OLLAMA_HOST, ROLE_MODELS,
    RAM_ROLE_MIN_GB, RAM_ROLE_COOLDOWN_SEC,
    RAM_GC_SETTLE_SEC, RAM_ROLE_TIMEOUT_SEC,
    ROLE_COMPILED_DIR, ROLE_FAILURE_DIR,
    ROLE_NAMES, ROLE_VAULT_TAGS,
    VAULT_DIR,
)

log = logging.getLogger("scalpel")

# Сколько строк данных показывать LLM (баланс: информативность vs длина промпта)
DATA_CONTEXT_ROWS      = 8
# Сколько «худших» строк (по |residual|) показывать отдельно
DATA_CONTEXT_WORST     = 3


# ══════════════════════════════════════════════════════════════════
# УТИЛИТЫ ПАМЯТИ
# ══════════════════════════════════════════════════════════════════

def _avail_ram_gb() -> float:
    try:
        import psutil
        return psutil.virtual_memory().available / 1024**3
    except ImportError:
        pass
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / 1024**2
    except Exception:
        pass
    return 3.0


def _ollama_stop_silent(model: str = OLLAMA_MODEL) -> None:
    """Останавливает Ollama без шума в консоли."""
    # v10.39: сначала пробуем API keep_alive=0 — надёжнее чем CLI
    try:
        import urllib.request as _ur, json as _js
        _host = OLLAMA_HOST
        _payload = _js.dumps({"model": model, "keep_alive": 0}).encode()
        _req = _ur.Request(
            f"{_host}/api/generate",
            data=_payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with _ur.urlopen(_req, timeout=5) as _r:
            _r.read()
    except Exception:
        pass
    # Дополнительно CLI
    _os = platform.system()
    cmd = f"ollama stop {model} >nul 2>&1" if _os == "Windows" \
          else f"ollama stop {model} 2>/dev/null"
    os.system(cmd)


def _gc_and_settle(settle_sec: float = RAM_GC_SETTLE_SEC) -> int:
    """gc.collect + пауза для возврата страниц ОС."""
    collected = gc.collect()
    time.sleep(settle_sec)
    return collected


# ══════════════════════════════════════════════════════════════════
# DATA CONTEXT BUILDER  ← НОВЫЙ БЛОК v9.9.2-dataaware
# ══════════════════════════════════════════════════════════════════

def _build_data_context(
    X_samples:      Optional[np.ndarray],   # shape (N, F) — строки данных
    y_samples:      Optional[np.ndarray],   # shape (N,)   — реальные значения
    y_pred_samples: Optional[np.ndarray],   # shape (N,)   — предсказания формулы
    shadow_names:   List[str],              # ["x0","x1",...] — теневые имена
    real_names:     Optional[List[str]],    # ["temperature","pressure",...] или None
    n_rows:         int = DATA_CONTEXT_ROWS,
    n_worst:        int = DATA_CONTEXT_WORST,
) -> str:
    """
    Строит текстовый блок контекста данных для инжекции в промпт LLM.

    Что включает:
      1. Маппинг shadow→real (если real_names переданы)
      2. Таблица: N строк реальных данных + y_real + y_pred + residual
      3. Статистика остатков: max, mean_abs, bias, std
      4. Топ-N «худших» строк (где |residual| максимален)

    Если данные не переданы — возвращает пустую строку (graceful degradation).
    Это позволяет вызывать без изменения старого кода (None → "").
    """
    # ── Проверка входных данных ───────────────────────────────────
    if X_samples is None or y_samples is None or y_pred_samples is None:
        return ""

    try:
        X  = np.asarray(X_samples,      dtype=float)
        y  = np.asarray(y_samples,      dtype=float)
        yp = np.asarray(y_pred_samples, dtype=float)
    except Exception as e:
        log.debug("[DataContext] Не удалось привести к float: %s", e)
        return ""

    if X.ndim != 2 or y.ndim != 1 or yp.ndim != 1:
        return ""
    if len(y) == 0 or X.shape[0] != len(y) or len(y) != len(yp):
        return ""

    # ── Имена признаков ───────────────────────────────────────────
    feat_names = real_names if real_names and len(real_names) == X.shape[1] \
                 else shadow_names[:X.shape[1]]

    lines: List[str] = []
    lines.append("─" * 60)
    lines.append("  РЕАЛЬНЫЕ ДАННЫЕ (выборка для проверки формулы)")
    lines.append("─" * 60)

    # ── Маппинг shadow → real ─────────────────────────────────────
    if real_names and len(real_names) == len(shadow_names):
        mapping_pairs = [
            f"{s}={r}" for s, r in zip(shadow_names, real_names)
            if s != r
        ]
        if mapping_pairs:
            lines.append("  Маппинг признаков: " + ", ".join(mapping_pairs))

    # ── Выбираем строки для показа ────────────────────────────────
    # Берём равномерную выборку из всего диапазона данных
    N        = len(y)
    n_show   = min(n_rows, N)
    indices  = np.round(np.linspace(0, N - 1, n_show)).astype(int)

    residuals     = y - yp
    abs_residuals = np.abs(residuals)

    # ── Заголовок таблицы ─────────────────────────────────────────
    header_parts = [f"{name[:10]:>10}" for name in feat_names[:6]]
    header_parts += ["     y_real", "     y_pred", "  residual"]
    lines.append("  " + "  ".join(header_parts))
    lines.append("  " + "─" * (len("  ".join(header_parts))))

    # ── Строки таблицы ────────────────────────────────────────────
    for i in indices:
        row_parts = []
        for j in range(min(6, X.shape[1])):
            val = X[i, j]
            # Умный форматтер: целые числа без лишних нулей
            if abs(val) >= 1000 or (abs(val) < 0.01 and val != 0):
                row_parts.append(f"{val:>10.3e}")
            else:
                row_parts.append(f"{val:>10.4g}")
        row_parts.append(f"{y[i]:>10.4g}")
        row_parts.append(f"{yp[i]:>10.4g}")
        resid = residuals[i]
        sign  = "+" if resid >= 0 else ""
        row_parts.append(f"  {sign}{resid:.4g}")
        lines.append("  " + "  ".join(row_parts))

    # ── Статистика остатков ───────────────────────────────────────
    lines.append("")
    lines.append("  СТАТИСТИКА ОСТАТКОВ (residuals = y_real − y_pred):")
    lines.append(f"    max |residual| = {abs_residuals.max():.4g}")
    lines.append(f"    mean |residual| = {abs_residuals.mean():.4g}")
    lines.append(f"    std residual  = {residuals.std():.4g}")
    bias = residuals.mean()
    lines.append(f"    bias (mean)   = {bias:+.4g}  "
                 f"{'⚠ систематическое смещение' if abs(bias) > abs_residuals.mean() * 0.3 else 'ОК'}")

    # Относительная ошибка (MAPE) если y не содержит нулей
    nonzero_mask = np.abs(y) > 1e-12
    if nonzero_mask.sum() > 0:
        mape = np.mean(np.abs(residuals[nonzero_mask] / y[nonzero_mask])) * 100
        lines.append(f"    MAPE          = {mape:.2f}%")

    # ── Топ-N худших строк ────────────────────────────────────────
    if N > n_worst:
        worst_idx = np.argsort(abs_residuals)[-n_worst:][::-1]
        lines.append("")
        lines.append(f"  ТОП-{n_worst} ХУДШИХ ТОЧЕК (где формула ошибается больше всего):")
        for rank, wi in enumerate(worst_idx, 1):
            feat_str = ", ".join(
                f"{feat_names[j][:8]}={X[wi, j]:.4g}"
                for j in range(min(4, X.shape[1]))
            )
            lines.append(
                f"    #{rank}: [{feat_str}]  "
                f"y_real={y[wi]:.4g}  y_pred={yp[wi]:.4g}  "
                f"residual={residuals[wi]:+.4g}"
            )

    # ── Диапазоны данных (помогает Физику с размерностью) ─────────
    lines.append("")
    lines.append("  ДИАПАЗОНЫ ПРИЗНАКОВ:")
    for j, name in enumerate(feat_names[:6]):
        col = X[:, j]
        lines.append(
            f"    {name[:12]:12s}: [{col.min():.4g} … {col.max():.4g}]  "
            f"mean={col.mean():.4g}"
        )
    lines.append(f"    {'y (target)':12s}: [{y.min():.4g} … {y.max():.4g}]  "
                 f"mean={y.mean():.4g}")
    lines.append("─" * 60)

    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════
# РЕЗУЛЬТАТ ОДНОЙ РОЛИ
# ══════════════════════════════════════════════════════════════════

@dataclass
class RoleResult:
    role_name:             str
    verdict:               str          # ПРИНЯТА | ОТКЛОНЕНА | УСЛОВНО | ВОЗДЕРЖАЛАСЬ
    analysis:              str
    ram_before:            float = 0.0
    ram_after:             float = 0.0
    elapsed_sec:           float = 0.0
    used_dspy:             bool  = False
    error:                 str   = ""
    # v10.8: обратная связь для Navigator
    structural_critique:   str   = ""   # что структурно не так
    improvement_suggestion: str  = ""   # конкретный совет для следующего PySR
    # v10.18: полный сырой ответ модели
    full_response:         str   = ""   # raw LLM output


# ══════════════════════════════════════════════════════════════════
# RAM СЛОТ — контекстный менеджер одной роли
# ══════════════════════════════════════════════════════════════════

@contextmanager
def role_ram_slot(
    role_name:  str,
    model:      str = OLLAMA_MODEL,
    min_ram_gb: float = RAM_ROLE_MIN_GB,
    cooldown:   float = RAM_ROLE_COOLDOWN_SEC,
) -> Generator[Dict, None, None]:
    """
    Контекстный менеджер одного RAM-слота.

    Использование:
        with role_ram_slot("Скептик") as slot:
            slot["ready"]  # True если RAM достаточно
            # ... вызов LLM ...
        # После выхода: ollama_stop + gc + sleep

    Гарантирует очистку даже при исключении.
    """
    ram_before = _avail_ram_gb()
    slot: Dict = {
        "role_name":   role_name,
        "ready":       ram_before >= min_ram_gb,
        "ram_before":  ram_before,
        "ram_after":   0.0,
        "start_time":  time.time(),
    }

    if not slot["ready"]:
        log.warning(
            "[RamSlot/%s] МАЛО RAM: %.2f ГБ < %.2f ГБ. Роль пропущена.",
            role_name, ram_before, min_ram_gb,
        )

    log.info("[RamSlot/%s] ▶ Старт. RAM=%.2f ГБ", role_name, ram_before)
    print(f"  [RAM Queue] ▶ {role_name}: {ram_before:.2f} ГБ свободно")

    try:
        yield slot
    finally:
        # ── ВЫГРУЗКА ─────────────────────────────────────────────
        _ollama_stop_silent(model)
        collected = _gc_and_settle(cooldown)
        ram_after = _avail_ram_gb()
        slot["ram_after"] = ram_after
        elapsed = time.time() - slot["start_time"]

        log.info(
            "[RamSlot/%s] ◀ Завершено. RAM: %.2f→%.2f ГБ (+%.2f) gc=%d t=%.1fs",
            role_name, ram_before, ram_after,
            ram_after - ram_before, collected, elapsed,
        )
        print(
            f"  [RAM Queue] ◀ {role_name}: "
            f"{ram_before:.2f}→{ram_after:.2f} ГБ  t={elapsed:.1f}s"
        )


# ══════════════════════════════════════════════════════════════════
# PER-ROLE DSPy MODULE LOADER
# ══════════════════════════════════════════════════════════════════

class RoleModuleLoader:
    """
    Загружает/компилирует DSPy-модуль конкретной роли.

    Каждая роль хранит свою скомпилированную модель отдельно:
        scalpel_vault/role_models/skeptic_compiled.json
        scalpel_vault/role_models/physicist_compiled.json
        ...

    Failure логи тоже раздельные:
        scalpel_vault/role_failures/skeptic.jsonl
        ...
    """

    _ROLE_SLUG = {
        "Скептик":   "skeptic",
        "Физик":     "physicist",
        "Прагматик": "pragmatist",
        "Мистик":    "mystic",
    }

    def __init__(self) -> None:
        ROLE_COMPILED_DIR.mkdir(parents=True, exist_ok=True)
        ROLE_FAILURE_DIR.mkdir(parents=True, exist_ok=True)

    def compiled_path(self, role_name: str) -> Path:
        slug = self._ROLE_SLUG.get(role_name, role_name.lower())
        return ROLE_COMPILED_DIR / f"{slug}_compiled.json"

    def failure_log_path(self, role_name: str) -> Path:
        slug = self._ROLE_SLUG.get(role_name, role_name.lower())
        return ROLE_FAILURE_DIR / f"{slug}.jsonl"

    def load_role_examples(
        self,
        role_name: str,
        gold_path: Optional[Path] = None,
        max_count: int = 5,
    ) -> List[Any]:
        """
        Загружает few-shot примеры для роли.
        v9.9.2: сначала EpisodicMemory, потом gold_formulas.json как fallback.
        """
        # ── Путь 1: EpisodicMemory (главный источник) ─────────────
        try:
            from .episodic_memory import get_memory
            mem_examples = get_memory().recall(
                role_name      = role_name,
                verdict_filter = "ПРИНЯТА",
                limit          = max_count,
                min_r2         = 0.80,
            )
            if mem_examples:
                log.info("[RoleLoader/%s] EpisodicMemory: %d примеров.",
                         role_name, len(mem_examples))
                return mem_examples
        except Exception as e:
            log.debug("[RoleLoader/%s] EpisodicMemory недоступна: %s", role_name, e)

        # ── Путь 2: gold_formulas.json (fallback) ─────────────────
        try:
            from .dspy_optimizer import DSPY_AVAILABLE
            if not DSPY_AVAILABLE:
                return []
            import dspy
        except ImportError:
            return []

        if gold_path is None:
            from .config import GOLD_PATH
            gold_path = GOLD_PATH

        if not gold_path.exists():
            return []

        tag = ROLE_VAULT_TAGS.get(role_name, "")
        try:
            import json as _json
            data    = _json.loads(gold_path.read_text(encoding="utf-8"))
            records = data.get("formulas", [])
        except Exception:
            return []

        # Берём записи с тегом этой роли (или все хорошие если тега нет)
        if tag:
            good = [r for r in records if tag in r.get("tags", [])]
            # FIX v10.16: если тег не нашёлся — fallback на все хорошие
            # (у seed-формул нет тегов ролей, иначе module=None → Legacy)
            if not good:
                log.debug("[RoleLoader/%s] Тег '%s' не найден — fallback на все R²≥0.85", role_name, tag)
                good = [r for r in records if float(r.get("r2_blind", 0)) >= 0.85]
        else:
            good = [r for r in records
                    if float(r.get("r2_blind", 0)) >= 0.85]

        good.sort(key=lambda r: float(r.get("r2_blind", 0)), reverse=True)
        good = good[:max_count]

        examples = []
        for rec in good:
            try:
                formula  = rec.get("formula", "")
                r2_tr    = rec.get("r2_train", 0.0)
                r2_bl    = rec.get("r2_blind", 0.0)
                compl    = rec.get("complexity", 0)
                metrics  = (
                    f"r2_train={r2_tr:.3f}, r2_blind={r2_bl:.3f}, "
                    f"complexity={compl}"
                )
                ex = dspy.Example(
                    role_name       = role_name,
                    role_task       = self._role_task(role_name),
                    formula         = formula,
                    formula_metrics = metrics,
                    verdict         = "ПРИНЯТА",
                    analysis        = (
                        f"Formula validated: R²_blind={r2_bl:.3f}, "
                        f"complexity={compl}."
                    ),
                ).with_inputs("role_name", "role_task", "formula", "formula_metrics")
                examples.append(ex)
            except Exception:
                continue

        log.info("[RoleLoader/%s] Загружено %d примеров.", role_name, len(examples))
        return examples

    def load_failure_examples(
        self,
        role_name: str,
        max_count: int = 8,
    ) -> List[Any]:
        """Загружает примеры ОТКЛОНЕНА для роли из failure log."""
        try:
            from .dspy_optimizer import DSPY_AVAILABLE
            if not DSPY_AVAILABLE:
                return []
            import dspy
        except ImportError:
            return []

        path = self.failure_log_path(role_name)
        if not path.exists():
            return []

        examples = []
        try:
            lines = path.read_text(encoding="utf-8").strip().splitlines()
            for line in lines[-max_count:]:
                try:
                    rec = json.loads(line)
                    ex = dspy.Example(
                        role_name       = role_name,
                        role_task       = self._role_task(role_name),
                        formula         = rec.get("formula", ""),
                        formula_metrics = rec.get("metrics", ""),
                        verdict         = rec.get("verdict", "ОТКЛОНЕНА"),
                        analysis        = rec.get("analysis", ""),
                    ).with_inputs("role_name", "role_task", "formula", "formula_metrics")
                    examples.append(ex)
                except Exception:
                    continue
        except Exception as e:
            log.warning("[RoleLoader/%s] Ошибка failure log: %s", role_name, e)
        return examples

    def compile_role_module(
        self,
        role_name: str,
        examples:  List[Any],
    ) -> Optional[Any]:
        """
        Компилирует AuditModule для конкретной роли.
        Возвращает скомпилированный модуль или None.
        """
        try:
            from .dspy_optimizer import DSPY_AVAILABLE, AuditModule
            from dspy.teleprompt import BootstrapFewShot
            if not DSPY_AVAILABLE or not examples:
                return None
        except ImportError:
            return None

        def _role_metric(ex: Any, pred: Any, trace=None) -> bool:
            got  = getattr(pred, "verdict", "").strip().upper()
            want = getattr(ex,   "verdict", "").strip().upper()
            return got == want and bool(getattr(pred, "analysis", "").strip())

        try:
            opt = BootstrapFewShot(
                metric=_role_metric,
                max_bootstrapped_demos=2,
                max_labeled_demos=min(len(examples), 4),
            )
            compiled = opt.compile(AuditModule(), trainset=examples)
            log.info("[RoleLoader/%s] BootstrapFewShot ✓", role_name)
            return compiled
        except Exception as e:
            log.warning("[RoleLoader/%s] Компиляция упала: %s", role_name, e)
            return None

    def save_role_module(self, role_name: str, module: Any) -> bool:
        if module is None:
            return False
        try:
            path = self.compiled_path(role_name)
            module.save(str(path))
            log.info("[RoleLoader/%s] Сохранено: %s", role_name, path)
            return True
        except Exception as e:
            log.warning("[RoleLoader/%s] Ошибка сохранения: %s", role_name, e)
            return False

    def load_role_module(self, role_name: str) -> Optional[Any]:
        try:
            from .dspy_optimizer import DSPY_AVAILABLE, AuditModule
            if not DSPY_AVAILABLE:
                return None
        except ImportError:
            return None

        path = self.compiled_path(role_name)
        if not path.exists():
            return None
        try:
            mod = AuditModule()
            mod.load(str(path))
            log.info("[RoleLoader/%s] Загружена скомпилированная модель.", role_name)
            return mod
        except Exception as e:
            log.warning("[RoleLoader/%s] Ошибка загрузки: %s", role_name, e)
            return None

    def log_role_result(
        self,
        role_name:  str,
        formula:    str,
        metrics:    str,
        verdict:    str,
        analysis:   str,
    ) -> None:
        """
        Записывает результат роли в failure log.
        ПРИНЯТА → enriches examples, ОТКЛОНЕНА → teaches what to reject.
        """
        path = self.failure_log_path(role_name)
        try:
            record = {
                "formula":  formula,
                "metrics":  metrics,
                "verdict":  verdict,
                "analysis": analysis,
            }
            with path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception as e:
            log.warning("[RoleLoader/%s] Ошибка записи: %s", role_name, e)

    @staticmethod
    def _role_task(role_name: str) -> str:
        tasks = {
            "Скептик":   "Найди слабые места. Почему формула ложна? Какие данные опровергнут?",
            "Физик":     "Проверь размерностную корректность. Соответствует известным законам?",
            "Прагматик": "Оцени практическую применимость. В каких условиях ломается?",
            "Мистик":    "Найди аналогии с известными физическими законами или структурами.",
        }
        return tasks.get(role_name, "Оцени формулу.")


# ══════════════════════════════════════════════════════════════════
# MATRYOSHKA QUEUE — главный оркестратор
# ══════════════════════════════════════════════════════════════════

def _extract_role_flags(analysis: str, role_name: str) -> list:
    """
    v10.24: Извлекает семантические флаги из текста анализа роли.
    Navigator читает эти флаги в следующей HADI-итерации.
    """
    flags = []
    text = analysis.lower()
    # Физик — размерностные проблемы
    if role_name == "Физик":
        if any(w in text for w in ("размерност", "dimension", "единиц", "units")):
            if any(w in text for w in ("неверн", "wrong", "несовместим", "противоречи")):
                flags.append("wrong_dimensions")
    # Скептик — переобучение
    if role_name == "Скептик":
        if any(w in text for w in ("переобуч", "overfit", "случайн", "артефакт")):
            flags.append("overfitting")
        if any(w in text for w in ("сложн", "complex", "избыточн")):
            flags.append("too_complex")
    # Физик или Прагматик — пропущенная переменная
    if any(w in text for w in ("пропущен", "missing", "нет переменн", "не хватает")):
        flags.append("missing_variable")
    return flags


class MatryoshkaQueue:
    """
    Строгая RAM-очередь для 4 ролей Матрёшки.

    Каждая роль:
      1. Получает свой RAM-слот
      2. Загружает свой скомпилированный DSPy-модуль (или legacy)
      3. Делает один вызов LLM
      4. Освобождает RAM перед следующей ролью
      5. Сохраняет результат в свой failure log

    В RAM НИКОГДА не находятся два модуля одновременно.

    v9.9.2-dataaware:
      run() принимает дополнительные параметры:
        X_samples      — numpy array (N, F): строки реальных данных
        y_samples      — numpy array (N,):   реальные значения таргета
        y_pred_samples — numpy array (N,):   предсказания формулы на этих строках
        real_names     — List[str]:          реальные имена признаков (de-shadow)
      Все параметры опциональны — при None поведение идентично v9.9.1.
    """

    def __init__(
        self,
        model:       str  = OLLAMA_MODEL,
        host:        str  = OLLAMA_HOST,
        dspy_active: bool = False,
        chat_fn=None,
        role_models: dict = None,   # v10.9: {role_name: model_name}
    ) -> None:
        self.model       = model
        self.host        = host
        self.dspy_active = dspy_active
        self.loader      = RoleModuleLoader()
        self._chat_fn    = chat_fn
        # Маппинг роль→модель. Если не задан — используем ROLE_MODELS из config
        # Если роль не найдена в маппинге — fallback на self.model
        self.role_models = role_models if role_models is not None else ROLE_MODELS

    def run(
        self,
        formula_shadow:  str,
        shadow_names:    List[str],
        r2_train:        float,
        complexity:      int,
        domain_type:     str = "",
        r2_blind:        float = 0.0,   # FIX: был всегда = r2_train (баг)
        # ── v9.9.2: реальные данные ───────────────────────────────
        X_samples:       Optional[np.ndarray] = None,
        y_samples:       Optional[np.ndarray] = None,
        y_pred_samples:  Optional[np.ndarray] = None,
        real_names:      Optional[List[str]]  = None,
        ctx:             Any                  = None,   # v10.24: SharedContext
    ) -> Tuple[str, str, List["RoleResult"]]:
        """
        Запускает 4 роли строго по одной.

        Возвращает:
            consensus   — ПРИНЯТА | ОТКЛОНЕНА | СПОРНО
            report      — полный текст отчёта
            results     — список RoleResult по каждой роли
        """
        ROLE_ORDER = ROLE_NAMES

        metrics_str = (
            f"r2_train={r2_train:.4f}, complexity={complexity}, "
            f"features={shadow_names[:8]}, domain={domain_type or 'unknown'}"
        )

        # ── Строим блок контекста данных ОДИН РАЗ для всех ролей ──
        # Это экономит CPU — не пересчитываем для каждой роли отдельно.
        data_context = _build_data_context(
            X_samples      = X_samples,
            y_samples      = y_samples,
            y_pred_samples = y_pred_samples,
            shadow_names   = shadow_names,
            real_names     = real_names,
        )

        if data_context:
            print("  [DataContext] ✓ Реальные данные переданы в роли LLM "
                  f"({X_samples.shape[0]} строк, {X_samples.shape[1]} признаков)")
        else:
            print("  [DataContext] ⚠ Данные не переданы — роли работают только с метриками")

        all_results: List[RoleResult] = []

        # v10.24: SharedContext — строим сводку один раз для всех ролей
        _ctx_summary = ""
        if ctx is not None:
            try:
                _ctx_summary = ctx.ctx_for_roles()
                if _ctx_summary:
                    log.info("[SharedContext→Матрёшка] Сводка передана ролям (%d символов)",
                             len(_ctx_summary))
            except Exception as _ctx_err:
                log.debug("[SharedContext] ctx_for_roles ошибка: %s", _ctx_err)

        print(f"\n  {'═'*58}")
        print(f"  МАТРЁШКА RAM-ОЧЕРЕДЬ — 4 роли по одной")
        print(f"  {'═'*58}")

        for role_name in ROLE_ORDER:
            role_task = self.loader._role_task(role_name)

            # v10.22: передаём что уже сказали предыдущие роли
            # Каждая следующая роль СЛЫШИТ предыдущих — настоящий совет, не опрос
            _prev_verdicts = ""
            if all_results:
                _prev_lines = []
                for _pr in all_results:
                    _short = _pr.analysis[:200].replace("\n", " ").strip()
                    _prev_lines.append(
                        f"  {_pr.role_name} [{_pr.verdict}]: {_short}"
                    )
                _prev_verdicts = (
                    "\n\nМНЕНИЯ ПРЕДЫДУЩИХ ЭКСПЕРТОВ (учти их перед своим ответом):\n"
                    + "\n".join(_prev_lines)
                    + "\n"
                )

            result    = self._run_one_role(
                role_name       = role_name,
                role_task       = role_task,
                formula         = formula_shadow,
                metrics_str     = metrics_str,
                shadow_names    = shadow_names,
                r2_train        = r2_train,
                complexity      = complexity,
                domain_type     = domain_type,
                data_context    = data_context,
                prev_verdicts   = _prev_verdicts,  # v10.22: голоса предыдущих
                ctx_summary     = _ctx_summary,    # v10.24: история обработки
            )
            all_results.append(result)

            # v10.24: роль записывает вердикт в SharedContext
            if ctx is not None:
                try:
                    # Извлекаем флаги из текста анализа роли
                    _role_flags = _extract_role_flags(result.analysis, role_name)
                    ctx.role_write(role_name, result.verdict, _role_flags)
                    # Обновляем ctx_summary для следующих ролей
                    _ctx_summary = ctx.ctx_for_roles()
                except Exception as _ctx_wr:
                    log.debug("[SharedContext] role_write ошибка: %s", _ctx_wr)

            self.loader.log_role_result(
                role_name = role_name,
                formula   = formula_shadow,
                metrics   = metrics_str,
                verdict   = result.verdict,
                analysis  = result.analysis,
            )

            # ── EPISODIC MEMORY ───────────────────────────────────
            try:
                from .episodic_memory import get_memory
                get_memory().remember(
                    role_name  = role_name,
                    formula    = formula_shadow,
                    verdict    = result.verdict,
                    analysis   = result.analysis,
                    r2_train   = r2_train,
                    r2_blind   = r2_blind,   # FIX: был r2_train — теперь правильное значение
                    complexity = complexity,
                    domain     = domain_type,
                )
            except Exception as _mem_err:
                log.debug("[Memory] Ошибка записи: %s", _mem_err)

        # ── Консенсус ─────────────────────────────────────────────
        verdicts = [r.verdict for r in all_results]
        # v10.14: ВОЗДЕРЖАЛАСЬ = роль не голосует (не за и не против)
        # Считаем только тех кто высказался
        active = [v for v in verdicts if v != "ВОЗДЕРЖАЛАСЬ"]
        abstained = len(verdicts) - len(active)
        accepted  = active.count("ПРИНЯТА")
        rejected  = active.count("ОТКЛОНЕНА")

        if abstained > 0:
            abstain_names = [r.role_name for r in all_results if r.verdict == "ВОЗДЕРЖАЛАСЬ"]
            log.info("[Consensus] Воздержались: %s (всего %d)", abstain_names, abstained)

        # Для консенсуса нужно большинство от АКТИВНЫХ голосов
        # При 2 активных: нужно оба (2/2) → иначе СПОРНО
        # При 3 активных: нужно 2/3 (простое большинство)
        # При 4 активных: нужно 3/4 для принятия, 2/4 для отклонения
        n_active = max(len(active), 1)
        accept_threshold = (n_active // 2) + 1        # строгое большинство
        reject_threshold = (n_active // 2) + 1        # столько же для отклонения

        if not active:
            consensus = "СПОРНО"   # все воздержались — нет данных
        elif accepted >= accept_threshold:
            consensus = "ПРИНЯТА"
        elif rejected >= reject_threshold:
            consensus = "ОТКЛОНЕНА"
        else:
            consensus = "СПОРНО"

        report = self._build_report(
            formula_shadow, r2_train, complexity,
            consensus, accepted, all_results,
        )

        if consensus == "ПРИНЯТА":
            self._tag_accepted_roles(all_results, formula_shadow)

        return consensus, report, all_results

    # ── Одна роль ─────────────────────────────────────────────────

    def _run_one_role(
        self,
        role_name:    str,
        role_task:    str,
        formula:      str,
        metrics_str:  str,
        shadow_names: List[str],
        r2_train:     float,
        complexity:   int,
        domain_type:  str,
        data_context: str = "",   # ← NEW
        prev_verdicts: str = "",  # v10.22: голоса предыдущих ролей
        ctx_summary:   str = "",  # v10.24: SharedContext — история обработки
    ) -> RoleResult:
        """
        Запускает одну роль в изолированном RAM-слоте.
        """
        # v10.9: выбираем модель для этой роли
        role_model = self.role_models.get(role_name, self.model)
        if role_model != self.model:
            print(f"  [v10.9] {role_name} → {role_model}")
        with role_ram_slot(role_name, role_model) as slot:
            if not slot["ready"]:
                return RoleResult(
                    role_name  = role_name,
                    verdict    = "УСЛОВНО",
                    analysis   = f"[{role_name}] Пропущена: нехватка RAM.",
                    ram_before = slot["ram_before"],
                    error      = "LOW_RAM",
                )

            t_start = time.time()
            verdict, analysis, used_dspy, critique, suggestion, _full_r = self._call_role(
                role_name    = role_name,
                role_task    = role_task,
                formula      = formula,
                metrics_str  = metrics_str,
                shadow_names = shadow_names,
                r2_train     = r2_train,
                complexity   = complexity,
                domain_type  = domain_type,
                data_context = data_context + prev_verdicts + (
                    ("\n\n" + ctx_summary) if ctx_summary else ""
                ),  # v10.22+v10.24: предыдущие роли + история обработки
                model_override = role_model,
            )
            elapsed = time.time() - t_start

        # v10.21: если DSPy ответил менее чем за 10 сек — это заглушка без Ollama.
        # Принудительно перезапускаем в Legacy-режиме.
        if used_dspy and elapsed < 10.0:
            log.warning("[Queue/%s] DSPy заглушка (%.1fs < 10s) → Legacy.", role_name, elapsed)
            print(f"  [Queue] ⚡ {role_name}: DSPy заглушка ({elapsed:.1f}с) → реальный Ollama вызов", flush=True)
            t_leg = time.time()
            verdict, analysis, _, critique, suggestion, full_resp = self._call_role(
                role_name=role_name, role_task=role_task, formula=formula,
                metrics_str=metrics_str, shadow_names=shadow_names,
                r2_train=r2_train, complexity=complexity, domain_type=domain_type,
                data_context=data_context + prev_verdicts + (
                    ("\n\n" + ctx_summary) if ctx_summary else ""
                ), model_override=role_model,
                force_legacy=True,
            )
            elapsed  = time.time() - t_leg
            used_dspy = False
        else:
            full_resp = analysis

        return RoleResult(
            role_name              = role_name,
            verdict                = verdict,
            analysis               = analysis,
            ram_before             = slot["ram_before"],
            ram_after              = slot.get("ram_after", 0.0),
            elapsed_sec            = elapsed,
            used_dspy              = used_dspy,
            structural_critique    = critique,
            improvement_suggestion = suggestion,
            full_response          = full_resp,
        )

    def _call_role(
        self,
        role_name:    str,
        role_task:    str,
        formula:      str,
        metrics_str:  str,
        shadow_names: List[str],
        r2_train:     float,
        complexity:   int,
        domain_type:  str,
        force_legacy: bool = False,   # v10.21: пропустить DSPy, сразу Legacy
        data_context: str = "",
        model_override: str = "",   # v10.9: модель для этой конкретной роли
    ) -> Tuple[str, str, bool, str, str]:
        """
        Вызывает LLM для роли.
        Возвращает (verdict, analysis, used_dspy, structural_critique, improvement_suggestion).

        v9.9.2: data_context инжектируется и в DSPy-путь, и в Legacy-путь.
        v10.8: возвращает structural_critique и improvement_suggestion для Navigator.
        """
        # ── Путь 1: DSPy per-role module ─────────────────────────
        if self.dspy_active and not force_legacy:
            module = self.loader.load_role_module(role_name)

            if module is None:
                examples = (
                    self.loader.load_role_examples(role_name)
                    + self.loader.load_failure_examples(role_name)
                )
                if examples:
                    module = self.loader.compile_role_module(role_name, examples)
                    if module:
                        self.loader.save_role_module(role_name, module)

            if module is not None:
                try:
                    # v9.9.2: добавляем data_context в formula_metrics
                    # чтобы не менять сигнатуру AuditModule (обратная совместимость).
                    # Лимит 2000 символов — защита от переполнения контекста малых
                    # моделей (deepseek-r1:8b имеет context window ~4096 токенов).
                    enriched_metrics = metrics_str
                    if data_context:
                        _ctx_trimmed = data_context[:2000] + (
                            "\n  [обрезано — полный контекст в legacy-пути]"
                            if len(data_context) > 2000 else ""
                        )
                        enriched_metrics = metrics_str + "\n\n" + _ctx_trimmed

                    pred     = module(
                        role_name       = role_name,
                        role_task       = role_task,
                        formula         = formula,
                        formula_metrics = enriched_metrics,
                    )
                    # v10.9: role_model записываем в лог (DSPy путь)
                    if model_override:
                        log.debug("[v10.9/%s] DSPy через %s", role_name, model_override)
                    verdict  = getattr(pred, "verdict",  "УСЛОВНО").strip().upper()
                    analysis = getattr(pred, "analysis", "").strip()

                    # v10.21: DSPy без примеров возвращает пустой ответ за 3 сек.
                    # Это заглушка — принудительно падаем в Legacy (реальный Ollama).
                    if not analysis or len(analysis) < 15:
                        log.warning("[Queue/%s] DSPy пустой ответ (%d симв) → Legacy.",
                                    role_name, len(analysis))
                        raise ValueError("DSPy empty — Legacy")

                    if "ВОЗДЕРЖАЛАСЬ" in verdict: verdict = "ВОЗДЕРЖАЛАСЬ"
                    elif "ОТКЛОНЕНА" in verdict:  verdict = "ОТКЛОНЕНА"
                    elif "ПРИНЯТА" in verdict:    verdict = "ПРИНЯТА"
                    else:                         verdict = "УСЛОВНО"

                    critique    = getattr(pred, "structural_critique",   "").strip()
                    suggestion  = getattr(pred, "improvement_suggestion","").strip()
                    return verdict, analysis, True, critique, suggestion, analysis  # 6й = full_response
                except Exception as e:
                    log.warning("[Queue/%s] DSPy упал: %s. Legacy.", role_name, e)
                finally:
                    del module
                    gc.collect()
                # DSPy упал — падаем в legacy, critique будет из него

        # ── Путь 2: Legacy ollama_chat ────────────────────────────
        if self._chat_fn is not None:
            _ollama_chat = self._chat_fn
        else:
            from . import navigator as _nav_mod
            _ollama_chat = _nav_mod.ollama_chat

        domain_line = f"Домен: {domain_type}\n" if domain_type else ""

        # v9.9.2: data_context вставляется между метриками и заданием роли.
        # Структура промпта:
        #   [Роль + формула + метрики]
        #   [РЕАЛЬНЫЕ ДАННЫЕ — таблица + остатки]  ← NEW
        #   [Задание роли]
        #   [Инструкция по формату ответа]
        data_block = ""
        if data_context:
            data_block = f"\n{data_context}\n"

        # Мистику — разрешаем нестандартные аналогии явно
        mystik_addition = ""
        if role_name == "Мистик":
            mystik_addition = (
                "Твоя особая задача: найти НЕОЧЕВИДНЫЕ аналогии. "
                "Может ли эта формула быть связана с другим научным доменом? "
                "Есть ли похожие паттерны в природе, музыке, экономике? "
                "Нестандартная идея лучше банального повтора. "
            )

        msg = (
            f"Роль: {role_name}\n{domain_line}"
            f"Формула: {formula}\n"
            f"R²_train={r2_train:.4f}, complexity={complexity}\n"
            f"Признаки: {', '.join(shadow_names[:8])}\n"
            f"{data_block}"
            f"{mystik_addition}"
            f"Задание: {role_task}\n\n"
            f"ВАЖНО: если у тебя НЕТ достаточно данных или уверенности для оценки — "
            f"не придумывай. Напиши ВОЗДЕРЖАЛАСЬ и объясни ПОЧЕМУ не можешь судить.\n"
            f"Примеры: 'нет данных о размерностях', 'формула вне моей компетенции', "
            f"'недостаточно точек для статистики'.\n"
            f"Ответ в одном абзаце. В конце ОБЯЗАТЕЛЬНО одно слово: "
            f"ПРИНЯТА, ОТКЛОНЕНА, УСЛОВНО или ВОЗДЕРЖАЛАСЬ.\n"
            f"ЯЗЫК: сначала можешь обдумать на любом языке, "
            f"но ВЕСЬ финальный текст ответа — СТРОГО на русском языке. "
            f"Никакого китайского, английского, японского в ответе."
        )

        # v10.14: температура = уровень творчества
        # Высокая температура = LLM берёт менее вероятные токены = нестандартные идеи
        # НО: только там где нестандартность полезна (Мистик, Navigator)
        # Скептик и Физик ДОЛЖНЫ быть холодными — иначе случайно примут плохую формулу
        role_temperatures = {
            "Скептик":   0.1,   # строгий детектив — никакой фантазии
            "Физик":     0.15,  # точная наука — минимальное творчество
            "Прагматик": 0.45,  # гибкий практик
            "Мистик":    0.92,  # ищет неожиданные аналогии — высокое творчество
        }
        temperature = role_temperatures.get(role_name, 0.3)
        active_model = model_override if model_override else self.model

        raw = _ollama_chat(
            msg, model=active_model, host=self.host,
            timeout=RAM_ROLE_TIMEOUT_SEC, temperature=temperature, num_predict=400,
        )

        if raw.startswith("[OLLAMA_ERROR]"):
            return "УСЛОВНО", f"[{role_name}] Ollama недоступна.", False, "", "", ""

        last = raw.upper().split()[-3:]
        if "ВОЗДЕРЖАЛАСЬ" in last:  verdict = "ВОЗДЕРЖАЛАСЬ"
        elif "ОТКЛОНЕНА" in last:   verdict = "ОТКЛОНЕНА"
        elif "ПРИНЯТА" in last:     verdict = "ПРИНЯТА"
        else:                     verdict = "УСЛОВНО"

        # v10.8: парсим critique и suggestion из legacy текста
        # Ищем паттерны "missing", "add", "try", "include", "should be"
        critique   = ""
        suggestion = ""
        raw_lower  = raw.lower()
        # Простой эвристический парсинг — ищем предложения с ключевыми словами
        for sent in raw.replace(".", ".\n").split("\n"):
            s = sent.strip()
            if not s:
                continue
            sl = s.lower()
            if not critique and any(w in sl for w in ["missing", "absent", "lacks", "without",
                                                       "не учтён", "отсутствует", "не хватает",
                                                       "неполн", "пропущен"]):
                critique = s[:200]
            if not suggestion and any(w in sl for w in ["add", "try", "include", "consider",
                                                          "should", "need", "добавить",
                                                          "попробовать", "включить", "нужно"]):
                suggestion = s[:200]
        if not critique:
            critique = "structure is complete" if verdict == "ПРИНЯТА" else "see analysis"
        if not suggestion:
            suggestion = "no improvement needed" if verdict == "ПРИНЯТА" else "see analysis"

        return verdict, raw, False, critique, suggestion, raw  # 6й = full_response

    # ── Вспомогательные методы ────────────────────────────────────

    @staticmethod
    def _build_report(
        formula:   str,
        r2_train:  float,
        complexity: int,
        consensus: str,
        accepted:  int,
        results:   List[RoleResult],
    ) -> str:
        lines = [
            "═" * 62,
            "  CONSENSUS REPORT — Матрёшка v9.9.2-dataaware",
            f"  Формула:   {formula}",
            f"  R²_train:  {r2_train:.4f}   Сложность: {complexity}",
            f"  Консенсус: {consensus} ({accepted}/4 ПРИНЯТА)",
            "═" * 62,
            "",
        ]
        all_llm_unavail = all(
            "недоступна" in (r.analysis or "").lower() for r in results
        )
        if all_llm_unavail:
            lines.insert(5, "  ⚠ LLM недоступна — все вердикты УСЛОВНО (без Ollama)")
        for r in results:
            dspy_badge = "[DSPy]" if r.used_dspy else "[Legacy]"
            ram_delta  = r.ram_after - r.ram_before
            lines += [
                f"══ {r.role_name} {dspy_badge} ══",
                f"   Вердикт: {r.verdict}  "
                f"RAM: {r.ram_before:.2f}→{r.ram_after:.2f} "
                f"(Δ{ram_delta:+.2f} ГБ)  t={r.elapsed_sec:.1f}s",
                r.analysis,
                f"   [Вердикт: {r.verdict}]",
                "",
            ]
        # ── v10.8: КОНСИЛИУМ — что улучшить ─────────────────────────
        has_critiques = any(
            r.structural_critique and r.structural_critique != "structure is complete"
            for r in results
        )
        if has_critiques or consensus != "ПРИНЯТА":
            lines += [
                "",
                "═" * 62,
                "  КОНСИЛИУМ — РЕКОМЕНДАЦИИ ДЛЯ СЛЕДУЮЩЕГО PySR",
                "═" * 62,
            ]
            for r in results:
                critique   = r.structural_critique   or "—"
                suggestion = r.improvement_suggestion or "—"
                lines += [
                    f"  {r.role_name}:",
                    f"    Проблема:  {critique[:120]}",
                    f"    Совет PySR: {suggestion[:120]}",
                    "",
                ]
            # Общий итог — самые частые советы
            all_suggestions = [
                r.improvement_suggestion for r in results
                if r.improvement_suggestion
                and r.improvement_suggestion != "no improvement needed"
            ]
            if all_suggestions:
                lines += [
                    "  ── Итог консилиума ──",
                    f"  Приоритет: {all_suggestions[0][:120]}",
                    "═" * 62,
                ]

        return "\n".join(lines)

    def _tag_accepted_roles(
        self,
        results: List[RoleResult],
        formula: str,
    ) -> None:
        """После консенсуса ПРИНЯТА — добавляем теги ролей в vault."""
        try:
            from .config import GOLD_PATH
            import json as _json

            if not GOLD_PATH.exists():
                return
            data     = _json.loads(GOLD_PATH.read_text(encoding="utf-8"))
            formulas = data.get("formulas", [])

            for rec in formulas:
                if rec.get("formula") == formula:
                    tags = rec.get("tags", [])
                    for r in results:
                        if r.verdict == "ПРИНЯТА":
                            tag = ROLE_VAULT_TAGS.get(r.role_name, "")
                            if tag and tag not in tags:
                                tags.append(tag)
                    rec["tags"] = tags
                    break

            data["last_updated"] = __import__("datetime").datetime.now().isoformat()
            tmp = GOLD_PATH.with_suffix(".tmp")
            tmp.write_text(
                _json.dumps(data, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            tmp.replace(GOLD_PATH)
            log.info("[Queue] Теги ролей добавлены в vault: %s", formula)
        except Exception as e:
            log.warning("[Queue] Не удалось добавить теги: %s", e)


# ══════════════════════════════════════════════════════════════════
# RAM СТАТУС (для диагностики)
# ══════════════════════════════════════════════════════════════════

def ram_status_report(results: List[RoleResult]) -> str:
    """Краткий отчёт по использованию RAM по всем ролям."""
    lines = ["  RAM REPORT по ролям:"]
    total_freed = 0.0
    for r in results:
        delta = r.ram_after - r.ram_before
        total_freed += max(0, delta)
        lines.append(
            f"    {r.role_name:12s}  "
            f"{r.ram_before:.2f}→{r.ram_after:.2f} ГБ  "
            f"Δ{delta:+.2f}  {'DSPy' if r.used_dspy else 'Legacy':6s}  "
            f"{r.verdict}"
        )
    lines.append(f"    {'─'*52}")
    lines.append(f"    Итого освобождено: +{total_freed:.2f} ГБ")
    return "\n".join(lines)
