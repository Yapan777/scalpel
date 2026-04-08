"""
test_topological_surgery.py — Тесты модуля Topological Surgery v10.3.8.

Покрытие:
  - detect_singularities: корректная кривизна, пометка зон
  - ricci_flow_smooth:    SG-фильтр, RAM Guard (только numpy), edge cases
  - perform_surgery:      порог активации, процент удаления, сшивание
  - mark_poincare_invariant: оба исхода
  - format_surgery_report: структура строки отчёта
  - RAM Guard: убеждаемся, что X остаётся numpy.ndarray на всех этапах
"""
import numpy as np
import pytest

from scalpel.topological_surgery import (
    SINGULARITY_CURVATURE_RATIO,
    SURGERY_CUT_FRACTION,
    POINCARE_R2_THRESHOLD,
    SurgeryResult,
    detect_singularities,
    format_surgery_report,
    mark_poincare_invariant,
    perform_surgery,
    ricci_flow_smooth,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _smooth_signal(n: int = 100) -> np.ndarray:
    """Гладкий синус без аномалий."""
    return np.sin(np.linspace(0, 4 * np.pi, n)).astype(np.float64)


def _signal_with_spike(n: int = 100, spike_pos: int = 50,
                       spike_amplitude: float = 100.0) -> np.ndarray:
    """Гладкий сигнал с одним резким выбросом (сингулярностью)."""
    y = _smooth_signal(n)
    y[spike_pos] += spike_amplitude
    return y


def _make_X(n: int = 100, p: int = 2) -> np.ndarray:
    rng = np.random.default_rng(42)
    return rng.random((n, p)).astype(np.float64)


# ─────────────────────────────────────────────────────────────────────────────
# 1. detect_singularities
# ─────────────────────────────────────────────────────────────────────────────

class TestDetectSingularities:

    def test_smooth_signal_no_singularities(self):
        """Гладкий сигнал не должен давать сингулярностей."""
        y = _smooth_signal(200)
        mask, curv, mean_k, max_k = detect_singularities(y)
        # Допускаем до 2% пометок из-за граничных эффектов
        assert mask.sum() / len(y) < 0.02

    def test_spike_detected_as_singularity(self):
        """Резкий выброс должен быть помечен как сингулярность."""
        y = _signal_with_spike(100, spike_pos=50, spike_amplitude=200.0)
        mask, curv, mean_k, max_k = detect_singularities(y)
        # Точка спайка и её соседи должны быть помечены
        assert mask[50] or mask[49] or mask[51], \
            "Spike region not flagged as singularity"

    def test_curvature_shape_matches_input(self):
        y = _smooth_signal(80)
        mask, curv, _, _ = detect_singularities(y)
        assert curv.shape == (80,)
        assert mask.shape == (80,)

    def test_short_signal_no_crash(self):
        """Сигналы длиной < 3 не должны падать."""
        for n in (0, 1, 2):
            y = np.zeros(n)
            mask, curv, mean_k, max_k = detect_singularities(y)
            assert len(mask) == n

    def test_constant_signal_no_singularities(self):
        """Константный сигнал: нулевая кривизна, нет сингулярностей."""
        y = np.ones(50) * 7.5
        mask, curv, mean_k, max_k = detect_singularities(y)
        assert mask.sum() == 0
        assert np.allclose(curv, 0.0)

    def test_custom_ratio(self):
        """Более низкий порог кривизны даёт больше сингулярностей."""
        y = _signal_with_spike(100, spike_amplitude=10.0)
        mask_strict, _, _, _ = detect_singularities(y, curvature_ratio=20.0)
        mask_loose,  _, _, _ = detect_singularities(y, curvature_ratio=3.0)
        assert mask_loose.sum() >= mask_strict.sum()


# ─────────────────────────────────────────────────────────────────────────────
# 2. ricci_flow_smooth
# ─────────────────────────────────────────────────────────────────────────────

class TestRicciFlowSmooth:

    def test_output_shapes_unchanged(self):
        X = _make_X(100)
        y = _smooth_signal(100)
        X_out, y_out, applied = ricci_flow_smooth(X, y)
        assert X_out.shape == X.shape
        assert y_out.shape == y.shape

    def test_X_is_unchanged(self):
        """Ricci Flow не должен трогать X — только y."""
        X = _make_X(100)
        y = _smooth_signal(100)
        X_out, _, _ = ricci_flow_smooth(X, y)
        np.testing.assert_array_equal(X, X_out)

    def test_applied_flag_true_for_long_signal(self):
        X = _make_X(100)
        y = _smooth_signal(100)
        _, _, applied = ricci_flow_smooth(X, y)
        assert applied is True

    def test_not_applied_for_short_signal(self):
        """Для коротких сигналов фильтр должен быть пропущен."""
        X = _make_X(10)
        y = _smooth_signal(10)
        _, _, applied = ricci_flow_smooth(X, y)
        assert applied is False

    def test_noise_reduction(self):
        """После фильтра дисперсия шума должна уменьшиться."""
        rng = np.random.default_rng(0)
        y_clean = _smooth_signal(200)
        noise   = rng.normal(0, 0.5, 200)
        y_noisy = y_clean + noise
        X = _make_X(200)
        _, y_out, applied = ricci_flow_smooth(X, y_noisy)
        assert applied
        # Расстояние от сглаженного сигнала до чистого < расстояние шумного
        assert np.std(y_out - y_clean) < np.std(y_noisy - y_clean)

    def test_ram_guard_output_is_numpy(self):
        """RAM Guard: на выходе должен быть np.ndarray, не pandas, не список."""
        X = _make_X(50)
        y = _smooth_signal(50)
        X_out, y_out, _ = ricci_flow_smooth(X, y)
        assert isinstance(X_out, np.ndarray)
        assert isinstance(y_out, np.ndarray)


# ─────────────────────────────────────────────────────────────────────────────
# 3. perform_surgery
# ─────────────────────────────────────────────────────────────────────────────

class TestPerformSurgery:

    def test_smooth_data_no_surgery(self):
        """Гладкие данные не требуют хирургии."""
        X = _make_X(200)
        y = _smooth_signal(200)
        X_out, y_out, res = perform_surgery(X, y, threshold=0.95)
        assert not res.surgery_performed
        assert res.n_after == 200

    def test_spike_data_triggers_surgery(self):
        """
        Данные с большим числом резких выбросов должны активировать хирургию.
        Создаём сигнал, где >5% точек — сингулярности.
        """
        rng = np.random.default_rng(1)
        n = 200
        y = _smooth_signal(n)
        # Добавляем 15 резких спайков (7.5% точек)
        spike_idx = rng.choice(n, size=15, replace=False)
        y[spike_idx] += rng.uniform(500, 1000, size=15)
        X = _make_X(n)
        X_out, y_out, res = perform_surgery(
            X, y, threshold=0.95, cut_fraction=0.025
        )
        assert res.surgery_performed
        assert res.surgery_pct > 0.0
        assert res.n_after < n

    def test_cut_fraction_respected(self):
        """Процент удаления не должен превышать cut_fraction + 1 точка (округление ceil)."""
        rng = np.random.default_rng(2)
        n = 300
        y = _smooth_signal(n)
        y[rng.choice(n, size=30, replace=False)] += 1000.0
        X = _make_X(n)
        cut_frac = 0.025
        _, _, res = perform_surgery(X, y, threshold=0.95, cut_fraction=cut_frac)
        if res.surgery_performed:
            max_allowed_pct = (cut_frac + 1 / n) * 100
            assert res.surgery_pct <= max_allowed_pct + 0.1  # допуск на ceil

    def test_output_shapes_consistent(self):
        """X_out и y_out должны иметь совпадающее число строк."""
        X = _make_X(100)
        y = _smooth_signal(100)
        y[50] += 9999.0
        X_out, y_out, res = perform_surgery(X, y)
        assert X_out.shape[0] == y_out.shape[0]
        assert X_out.shape[0] == res.n_after

    def test_ram_guard_numpy_only(self):
        """RAM Guard: X и y на выходе — numpy.ndarray."""
        X = _make_X(100)
        y = _smooth_signal(100)
        X_out, y_out, _ = perform_surgery(X, y)
        assert isinstance(X_out, np.ndarray)
        assert isinstance(y_out, np.ndarray)

    def test_ordering_preserved_after_stitch(self):
        """После сшивания y_out соответствует y[kept_indices] в правильном порядке."""
        rng = np.random.default_rng(42)
        n = 100
        X = _make_X(n)
        y = _smooth_signal(n)
        y[30] += 5000.0
        y[60] += 5000.0
        _, y_out, res = perform_surgery(X, y, threshold=0.95)
        if res.surgery_performed:
            # FIX: setdiff1d всегда сортирован — проверяем y_out а не индексы
            kept = np.setdiff1d(np.arange(n), res.cut_indices)
            # y_out должен точно совпадать с y[kept] (с допуском на float32)
            assert len(y_out) == len(kept), \
                f"Длина y_out={len(y_out)} != len(kept)={len(kept)}"
            np.testing.assert_allclose(
                y_out, y[kept], rtol=1e-5,
                err_msg="y_out не соответствует y[kept_indices] — порядок нарушен"
            )

    def test_min_samples_edge_case(self):
        """Хирургия на 3 точках не должна падать."""
        X = np.array([[1.0], [2.0], [3.0]])
        y = np.array([1.0, 100.0, 1.0])
        X_out, y_out, res = perform_surgery(X, y)
        assert X_out.shape[0] == y_out.shape[0]


# ─────────────────────────────────────────────────────────────────────────────
# 4. mark_poincare_invariant
# ─────────────────────────────────────────────────────────────────────────────

class TestMarkPoincareInvariant:

    def test_high_r2_marks_invariant(self):
        res = SurgeryResult(surgery_performed=True, surgery_pct=2.5)
        res = mark_poincare_invariant(res, r2_after=0.90)
        assert res.poincare_invariant is True
        assert any("POINCARE INVARIANT DETECTED" in l for l in res.report_lines)

    def test_low_r2_no_invariant(self):
        res = SurgeryResult(surgery_performed=True, surgery_pct=2.5)
        res = mark_poincare_invariant(res, r2_after=0.50)
        assert res.poincare_invariant is False
        assert any("НЕ ОБНАРУЖЕН" in l for l in res.report_lines)

    def test_none_r2_no_invariant(self):
        res = SurgeryResult()
        res = mark_poincare_invariant(res, r2_after=None)
        assert res.poincare_invariant is False

    def test_boundary_r2_exactly_threshold(self):
        """R² ровно на пороге — инвариант засчитывается."""
        res = SurgeryResult()
        res = mark_poincare_invariant(res, r2_after=POINCARE_R2_THRESHOLD)
        assert res.poincare_invariant is True

    def test_report_lines_populated(self):
        res = SurgeryResult()
        res = mark_poincare_invariant(res, r2_after=0.95)
        assert len(res.report_lines) >= 1


# ─────────────────────────────────────────────────────────────────────────────
# 5. format_surgery_report
# ─────────────────────────────────────────────────────────────────────────────

class TestFormatSurgeryReport:

    def _make_result(self, performed: bool = True, poincare: bool = True) -> SurgeryResult:
        res = SurgeryResult(
            n_original=100,
            n_after=97 if performed else 100,
            n_singularities=8,
            curvature_mean=0.012,
            curvature_max=1.450,
            surgery_performed=performed,
            surgery_pct=3.0 if performed else 0.0,
            ricci_applied=True,
            poincare_invariant=poincare,
            report_lines=[
                "[SURGERY PERFORMED: 3.00% удалено]",
                "[POINCARE INVARIANT DETECTED: СТРУКТУРА СГЛАЖЕНА]" if poincare
                else "[POINCARE INVARIANT: НЕ ОБНАРУЖЕН]",
            ],
        )
        return res

    def test_report_contains_surgery_header(self):
        report = format_surgery_report(self._make_result())
        assert "TOPOLOGICAL SURGERY" in report

    def test_report_contains_surgery_pct_line(self):
        report = format_surgery_report(self._make_result(performed=True))
        assert "SURGERY PERFORMED" in report

    def test_report_contains_poincare_line_when_detected(self):
        report = format_surgery_report(self._make_result(poincare=True))
        assert "POINCARE INVARIANT DETECTED" in report

    def test_report_contains_not_detected_when_absent(self):
        report = format_surgery_report(self._make_result(poincare=False))
        assert "НЕ ОБНАРУЖЕН" in report

    def test_report_is_string(self):
        report = format_surgery_report(self._make_result())
        assert isinstance(report, str)
        assert len(report) > 50

    def test_no_surgery_report(self):
        res = SurgeryResult(n_original=100, n_after=100, surgery_performed=False)
        res.report_lines = ["[SURGERY PERFORMED: 0.00% удалено] — Хирургия не потребовалась"]
        report = format_surgery_report(res)
        assert "не потребовалась" in report or "SURGERY PERFORMED" in report


# ─────────────────────────────────────────────────────────────────────────────
# 6. Интеграционный тест: полный пайплайн хирургии
# ─────────────────────────────────────────────────────────────────────────────

class TestSurgeryPipeline:

    def test_full_pipeline_clean_data(self):
        """Гладкие данные: Ricci применён, хирургия не нужна, инвариант при высоком R²."""
        n = 150
        rng = np.random.default_rng(99)
        X = rng.random((n, 3))
        y = _smooth_signal(n)

        # Шаг 1: Ricci Flow
        X, y, ricci_ok = ricci_flow_smooth(X, y)
        assert ricci_ok

        # Шаг 2: Хирургия
        X, y, res = perform_surgery(X, y, threshold=0.95)
        res.ricci_applied = ricci_ok
        assert not res.surgery_performed  # чистые данные

        # Шаг 3: Симулируем высокий R² от PySR
        res = mark_poincare_invariant(res, r2_after=0.93)
        assert res.poincare_invariant

        # Шаг 4: Отчёт
        report = format_surgery_report(res)
        assert "POINCARE INVARIANT DETECTED" in report

    def test_full_pipeline_noisy_data(self):
        """Зашумлённые данные: хирургия активируется, R² низкий → нет инварианта."""
        n = 200
        rng = np.random.default_rng(7)
        y = _smooth_signal(n)
        # Создаём много сингулярностей (>5%)
        spike_idx = rng.choice(n, size=20, replace=False)
        y[spike_idx] += rng.uniform(800, 1500, size=20)
        X = _make_X(n)

        X, y, ricci_ok = ricci_flow_smooth(X, y)
        X, y, res = perform_surgery(X, y, threshold=0.95)
        res.ricci_applied = ricci_ok

        if res.surgery_performed:
            assert res.surgery_pct > 0

        res = mark_poincare_invariant(res, r2_after=0.45)
        assert not res.poincare_invariant

        report = format_surgery_report(res)
        assert "SURGERY" in report
