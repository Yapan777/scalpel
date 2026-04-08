"""
topological_surgery.py — Topological Surgery v10.3.9 (Метод Перельмана).

Реализует топологическую хирургию данных перед подачей в PySR:
  1. SingularityDetector  — обнаружение точек разрыва (сингулярностей)
                            через вторую производную (локальную кривизну).
  2. perform_surgery()    — «хирургическое вырезание» 2-3% аномальных точек
                            + сшивание гладких участков (numpy, до Julia).
  3. ricci_flow_smooth()  — предварительный Ricci Flow: фильтр Савицкого-Голея
                            для удаления микро-шума («выпускание воздуха»).
  4. format_surgery_report() — Вердикт Перельмана для Матрёшки.

RAM Guard: вся хирургия живёт исключительно в numpy.
           Julia (PySR) получает уже «чистый скелет» данных.

Связь с монолитом:
  engine.py   → вызывает ricci_flow_smooth() + perform_surgery() до build_pysr()
  EngineResult → хранит surgery_report, surgery_pct, poincare_invariant
  audit.py    → format_surgery_report() вшивается в финальный отчёт Матрёшки
"""
from __future__ import annotations

import gc
import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
from scipy.signal import savgol_filter  # type: ignore

log = logging.getLogger("scalpel")

# ─────────────────────────────────────────────────────────────────────────────
# КОНСТАНТЫ МОДУЛЯ
# ─────────────────────────────────────────────────────────────────────────────

# Кривизна: точка — сингулярность, если кривизна > SINGULARITY_CURVATURE_RATIO × median
SINGULARITY_CURVATURE_RATIO: float = 10.0

# Порог хирургии: % самых аномальных точек для удаления (2-3 %)
SURGERY_CUT_FRACTION: float = 0.025   # 2.5 % по умолчанию

# Savitzky-Golay: окно и степень полинома для Ricci Flow
RICCI_SG_WINDOW: int = 7   # нечётное, минимум 5
RICCI_SG_POLYORDER: int = 3

# Минимальная длина данных для применения фильтра
RICCI_MIN_SAMPLES: int = 20

# После хирургии — если R² >= этого порога, признаём Пуанкаре-инвариант
POINCARE_R2_THRESHOLD: float = 0.82

# ─────────────────────────────────────────────────────────────────────────────
# DATACLASS РЕЗУЛЬТАТА
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SurgeryResult:
    """Результат топологической хирургии одного прогона."""

    # Исходные размеры
    n_original:       int   = 0
    n_after:          int   = 0

    # Сингулярности
    singularity_mask: np.ndarray = field(default_factory=lambda: np.array([], dtype=bool))
    n_singularities:  int         = 0

    # Кривизна
    curvature_values: np.ndarray = field(default_factory=lambda: np.array([]))
    curvature_mean:   float       = 0.0
    curvature_max:    float       = 0.0

    # Хирургия
    surgery_performed: bool  = False
    surgery_pct:       float = 0.0   # % удалённых точек
    cut_indices:       np.ndarray = field(default_factory=lambda: np.array([], dtype=int))

    # Ricci Flow
    ricci_applied:    bool = False

    # Финальный вердикт
    poincare_invariant: bool = False
    report_lines:       List[str] = field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────────
# 1. IQR OUTLIER DETECTOR — Детектор выбросов (метод Тьюки)
# ─────────────────────────────────────────────────────────────────────────────

def detect_outliers_iqr(
    y: np.ndarray,
    iqr_multiplier: float = 3.0,
) -> Tuple[np.ndarray, float, float, float]:
    """
    Обнаруживает выбросы через межквартильный размах (IQR, метод Тьюки).

    Выброс: точка вне [Q1 - k*IQR, Q3 + k*IQR] где k=iqr_multiplier.
    k=1.5 → стандартный критерий (умеренные выбросы)
    k=3.0 → экстремальные выбросы (консервативный, меньше ложных срабатываний)

    Преимущество над кривизной: не зависит от порядка точек и не реагирует
    на гладкие нелинейности формулы — только на статистические аномалии.

    Возвращает:
        (outlier_mask, Q1, Q3, IQR)
    """
    n = len(y)
    if n < 4:
        return np.zeros(n, dtype=bool), 0.0, 0.0, 0.0

    Q1  = float(np.percentile(y, 25))
    Q3  = float(np.percentile(y, 75))
    IQR = Q3 - Q1

    if IQR < 1e-12:
        # Почти константные данные — выбросов нет
        return np.zeros(n, dtype=bool), Q1, Q3, IQR

    lower = Q1 - iqr_multiplier * IQR
    upper = Q3 + iqr_multiplier * IQR

    outlier_mask = (y < lower) | (y > upper)
    n_out = int(outlier_mask.sum())

    log.info(
        "[IQR Detector] Q1=%.4f Q3=%.4f IQR=%.4f  "
        "bounds=[%.4f, %.4f]  outliers=%d/%d (%.1f%%)",
        Q1, Q3, IQR, lower, upper, n_out, n, 100*n_out/max(n,1),
    )
    return outlier_mask, Q1, Q3, IQR


# ─────────────────────────────────────────────────────────────────────────────
# 2. FRACTAL CUTTING — Хирургия
# ─────────────────────────────────────────────────────────────────────────────

def perform_surgery(
    X: np.ndarray,
    y: np.ndarray,
    threshold: float = 0.95,
    cut_fraction: float = SURGERY_CUT_FRACTION,
    curvature_ratio: float = SINGULARITY_CURVATURE_RATIO,  # оставлен для совместимости
    iqr_multiplier: float = 3.0,
    force_cut: bool = False,  # FIX v10.23: хирург явно сказал "режь" — обходим threshold
) -> Tuple[np.ndarray, np.ndarray, SurgeryResult]:
    """
    Топологическая хирургия данных (v10.4 — IQR-based).

    Алгоритм:
      1. IQR детектор: находим выбросы вне [Q1 - 3*IQR, Q3 + 3*IQR].
         Адаптивный порог — не зависит от масштаба данных.
      2. Если выбросов > (1 - threshold)*n — применяем хирургию.
      3. Удаляем найденные выбросы (не больше cut_fraction от n).
      4. Возвращаем чистые данные для Ricci Flow и PySR.

    Порядок вызова в engine.py:
      perform_surgery()  →  ricci_flow_smooth()  →  PySR
      (сначала убрать выбросы, потом сгладить оставшееся)
    """
    n = len(y)
    result = SurgeryResult(n_original=n)

    # ── IQR детектор выбросов ─────────────────────────────────────
    outlier_mask, Q1, Q3, IQR = detect_outliers_iqr(y.ravel(), iqr_multiplier)

    result.singularity_mask = outlier_mask
    result.n_singularities  = int(outlier_mask.sum())
    # Храним IQR-статистику в полях кривизны для отчёта
    result.curvature_mean   = float(IQR)
    result.curvature_max    = float(Q3 - Q1)

    # Доля выбросов
    outlier_fraction = result.n_singularities / max(n, 1)

    # ── Решение о хирургии ───────────────────────────────────────
    needs_surgery = outlier_fraction > (1.0 - threshold)

    # FIX v10.23: если хирург явно сказал резать — уважаем его решение
    # даже если выбросов "мало" по threshold. Найден хоть один → режем.
    if force_cut and result.n_singularities > 0:
        needs_surgery = True

    if not needs_surgery:
        log.info(
            "[Surgery] Не требуется. outlier_fraction=%.3f <= %.3f",
            outlier_fraction, 1.0 - threshold,
        )
        result.n_after = n
        result.report_lines = _build_report_lines(result, surgery_skipped=True)
        return X, y, result

    # ── Fractal Cutting: удаляем выбросы (не больше cut_fraction) ─
    n_cut     = min(result.n_singularities, max(1, int(np.ceil(n * cut_fraction))))
    keep_mask = np.ones(n, dtype=bool)

    if n_cut < result.n_singularities:
        # Выбросов больше чем разрешено резать — берём самые экстремальные
        y_flat    = y.ravel()
        distances = np.abs(y_flat - np.median(y_flat))
        cut_idx   = np.where(outlier_mask)[0]
        cut_idx   = cut_idx[np.argsort(distances[cut_idx])[::-1]][:n_cut]
        keep_mask[cut_idx] = False
    else:
        keep_mask[outlier_mask] = False

    X_clean = X[keep_mask]
    y_clean = y[keep_mask]

    gc.collect()

    surgery_pct = 100.0 * n_cut / n
    log.info(
        "[Surgery] Выполнена: n_cut=%d (%.2f%%)  n_keep=%d",
        n_cut, surgery_pct, len(y_clean),
    )

    result.surgery_performed = True
    result.surgery_pct       = surgery_pct
    result.cut_indices       = np.array([], dtype=int)
    result.n_after           = len(y_clean)
    result.report_lines      = _build_report_lines(result, surgery_skipped=False)

    return X_clean, y_clean, result


# ─────────────────────────────────────────────────────────────────────────────
# 3. RICCI FLOW — Сглаживание (фильтр Савицкого-Голея)
# ─────────────────────────────────────────────────────────────────────────────

def ricci_flow_smooth(
    X: np.ndarray,
    y: np.ndarray,
    window: int = RICCI_SG_WINDOW,
    polyorder: int = RICCI_SG_POLYORDER,
) -> Tuple[np.ndarray, np.ndarray, bool]:
    """
    Ricci Flow — предварительное «выпускание воздуха» из формы данных.

    Применяет фильтр Савицкого-Голея к y для сглаживания микро-шума,
    оставляя «скелет» формы — кривую без дрожания.

    Аналогия: нормализованный поток Риччи Перельмана — равномерное
    «сдутие» неоднородностей метрики до сферически-симметричного состояния.

    RAM Guard: scipy.signal.savgol_filter работает в numpy,
               результат → numpy-массив до вызова Julia.

    Аргументы:
        X         — матрица признаков (не сглаживается, только y)
        y         — целевой вектор
        window    — ширина окна SG-фильтра (нечётное число)
        polyorder — порядок полинома SG

    Возвращает:
        (X_unchanged, y_smoothed, ricci_applied: bool)
    """
    n = len(y)

    # Нет смысла сглаживать совсем короткие ряды
    if n < RICCI_MIN_SAMPLES:
        log.info("[RicciFlow] Пропуск: n=%d < %d", n, RICCI_MIN_SAMPLES)
        return X, y, False

    # v10.3.9 RICCI FIX: автоподстройка окна под размер данных.
    # Шаг 1: clamp окна к n (нечётному)
    actual_window = min(window, n if n % 2 == 1 else n - 1)
    # Шаг 2: окно должно быть > polyorder + 1 (требование scipy)
    actual_window = max(actual_window, polyorder + 2)
    # Шаг 3: окно обязано быть нечётным
    if actual_window % 2 == 0:
        actual_window += 1
    # Шаг 4 (КРИТИЧНЫЙ): после max() окно могло стать больше n.
    # При малых датасетах (n < polyorder+2) это приведёт к ValueError.
    # Финальный clamp — окно никогда не превышает n.
    if actual_window > n:
        # Не можем гарантировать валидные параметры SG — пропускаем фильтр
        log.warning(
            "[RicciFlow] Пропуск: actual_window=%d > n=%d (слишком мало точек для SG).",
            actual_window, n,
        )
        return X, y, False

    try:
        y_smooth = savgol_filter(
            y.ravel(),
            window_length=actual_window,
            polyorder=min(polyorder, actual_window - 1),
            mode="nearest",   # граничный режим без артефактов
        ).astype(y.dtype)

        noise_removed = float(np.std(y.ravel() - y_smooth))
        log.info(
            "[RicciFlow] SG window=%d polyorder=%d  σ_noise=%.4f",
            actual_window, polyorder, noise_removed,
        )
        return X, y_smooth, True

    except Exception as exc:
        log.warning("[RicciFlow] SG-фильтр не применён: %s", exc)
        return X, y, False


# ─────────────────────────────────────────────────────────────────────────────
# 4. REPORTING — Вердикт Перельмана
# ─────────────────────────────────────────────────────────────────────────────

def _build_report_lines(result: SurgeryResult, surgery_skipped: bool) -> List[str]:
    """Внутренний построитель строк отчёта."""
    lines = []
    if surgery_skipped:
        lines.append(
            f"[SURGERY PERFORMED: 0.00% удалено] — "
            f"Хирургия не потребовалась "
            f"(сингулярностей: {result.n_singularities}/{result.n_original})"
        )
    else:
        lines.append(
            f"[SURGERY PERFORMED: {result.surgery_pct:.2f}% удалено] — "
            f"Вырезано {result.n_original - result.n_after} из {result.n_original} точек"
        )
    return lines


def mark_poincare_invariant(result: SurgeryResult, r2_after: Optional[float]) -> SurgeryResult:
    """
    Финальная пометка Пуанкаре-инварианта.

    Вызывается из engine.py после того, как PySR нашёл формулу
    на «прооперированных» данных.

    Если после хирургии R² >= POINCARE_R2_THRESHOLD — хаос
    «стянулся в сферу»: топологический инвариант найден.
    """
    if r2_after is not None and r2_after >= POINCARE_R2_THRESHOLD:
        result.poincare_invariant = True
        result.report_lines.append(
            "[POINCARE INVARIANT DETECTED: СТРУКТУРА СГЛАЖЕНА]"
        )
        log.info(
            "[Surgery] POINCARE INVARIANT: R²=%.4f >= %.4f",
            r2_after, POINCARE_R2_THRESHOLD,
        )
    else:
        r2_str = f"{r2_after:.4f}" if r2_after is not None else "N/A"
        result.report_lines.append(
            f"[POINCARE INVARIANT: НЕ ОБНАРУЖЕН] — R²={r2_str} < {POINCARE_R2_THRESHOLD}"
        )
    return result


def format_surgery_report(result: SurgeryResult) -> str:
    """
    Форматирует полный раздел «Вердикт Перельмана» для финального отчёта
    Матрёшки и FINAL_REPORT_v10.txt.
    """
    sep = "─" * 62
    lines = [
        sep,
        "  ✂  TOPOLOGICAL SURGERY v10.3.9 (Метод Перельмана)",
        sep,
        f"  Исходных точек:      {result.n_original}",
        f"  Сингулярностей:      {result.n_singularities}"
        f" ({100*result.n_singularities/max(result.n_original,1):.1f}%)",
        f"  κ_mean:              {result.curvature_mean:.4f}",
        f"  κ_max:               {result.curvature_max:.4f}",
        f"  Ricci Flow applied:  {'✓' if result.ricci_applied else '○'}",
        f"  Хирургия:            {'✓ ВЫПОЛНЕНА' if result.surgery_performed else '○ не потребовалась'}",
    ]

    if result.surgery_performed:
        lines.append(f"  Удалено точек:       {result.n_original - result.n_after}"
                     f" ({result.surgery_pct:.2f}%)")
        lines.append(f"  Осталось точек:      {result.n_after}")

    lines.append(sep)

    # Строки вердикта (SURGERY PERFORMED + POINCARE INVARIANT)
    for verdict_line in result.report_lines:
        lines.append(f"  {verdict_line}")

    lines.append(sep)
    return "\n".join(lines)
