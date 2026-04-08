"""
test_dspy_v99.py — Тесты v9.9 DSPy-компонентов.
Все тесты работают БЕЗ реального DSPy и без Ollama.
Запуск: pytest tests/test_dspy_v99.py -v
"""
import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock


# ═══════════════════════════════════════════════════════════
# СЕКЦИЯ 1: GoldLoader
# ═══════════════════════════════════════════════════════════

class TestGoldLoader:

    def _write_gold(self, tmp_path: Path, formulas: list) -> Path:
        p = tmp_path / "gold_formulas.json"
        p.write_text(json.dumps({"formulas": formulas, "count": len(formulas)}),
                     encoding="utf-8")
        return p

    def test_load_empty_vault(self, tmp_path):
        from scalpel.dspy_optimizer import GoldLoader
        loader = GoldLoader(gold_path=tmp_path / "nonexistent.json")
        result = loader.load_examples()
        assert result == []

    def test_load_filters_by_min_r2(self, tmp_path):
        from scalpel.dspy_optimizer import GoldLoader
        gold_p = self._write_gold(tmp_path, [
            {"formula": "f0/f1", "r2_train": 0.95, "r2_blind": 0.92,
             "complexity": 3, "skeleton": "v / v"},
            {"formula": "f0+f1", "r2_train": 0.60, "r2_blind": 0.55,
             "complexity": 2, "skeleton": "v + v"},
        ])
        loader = GoldLoader(gold_path=gold_p)
        # min_r2=0.85 — только первая запись проходит
        with patch("scalpel.dspy_optimizer.DSPY_AVAILABLE", True), \
             patch("scalpel.dspy_optimizer.dspy") as mock_dspy:
            mock_dspy.Example = _MockExample
            results = loader.load_examples(min_r2=0.85)
        assert len(results) == 1

    def test_load_sorted_by_r2_blind(self, tmp_path):
        from scalpel.dspy_optimizer import GoldLoader
        gold_p = self._write_gold(tmp_path, [
            {"formula": "f0/f1", "r2_train": 0.90, "r2_blind": 0.88,
             "complexity": 3, "skeleton": "v / v"},
            {"formula": "f0*f1", "r2_train": 0.95, "r2_blind": 0.93,
             "complexity": 2, "skeleton": "v * v"},
        ])
        loader = GoldLoader(gold_path=gold_p)
        with patch("scalpel.dspy_optimizer.DSPY_AVAILABLE", True), \
             patch("scalpel.dspy_optimizer.dspy") as mock_dspy:
            mock_dspy.Example = _MockExample
            results = loader.load_examples(min_r2=0.80)
        # Лучший R²_blind идёт первым
        assert results[0].formula_hint == "f0*f1"

    def test_load_respects_max_count(self, tmp_path):
        from scalpel.dspy_optimizer import GoldLoader
        formulas = [
            {"formula": f"f{i}", "r2_train": 0.95, "r2_blind": 0.90,
             "complexity": i+1, "skeleton": "v"}
            for i in range(10)
        ]
        gold_p = self._write_gold(tmp_path, formulas)
        loader = GoldLoader(gold_path=gold_p)
        with patch("scalpel.dspy_optimizer.DSPY_AVAILABLE", True), \
             patch("scalpel.dspy_optimizer.dspy") as mock_dspy:
            mock_dspy.Example = _MockExample
            results = loader.load_examples(min_r2=0.85, max_count=3)
        assert len(results) <= 3

    def test_load_returns_empty_without_dspy(self, tmp_path):
        from scalpel.dspy_optimizer import GoldLoader
        gold_p = self._write_gold(tmp_path, [
            {"formula": "f0/f1", "r2_train": 0.95, "r2_blind": 0.92,
             "complexity": 3, "skeleton": "v / v"},
        ])
        loader = GoldLoader(gold_path=gold_p)
        with patch("scalpel.dspy_optimizer.DSPY_AVAILABLE", False):
            results = loader.load_examples()
        assert results == []


# ═══════════════════════════════════════════════════════════
# СЕКЦИЯ 2: needs_recompile / save / load
# ═══════════════════════════════════════════════════════════

class TestCompiledModelLifecycle:

    def test_needs_recompile_no_file(self, tmp_path):
        from scalpel.dspy_optimizer import needs_recompile
        assert needs_recompile(tmp_path / "nonexistent.json") is True

    def test_needs_recompile_fresh_file(self, tmp_path):
        from scalpel.dspy_optimizer import needs_recompile
        from datetime import datetime, timedelta
        path     = tmp_path / "compiled.json"
        meta     = tmp_path / "compiled.meta.json"
        path.write_text("{}", encoding="utf-8")
        expires  = (datetime.now() + timedelta(days=7)).isoformat()
        meta.write_text(json.dumps({"expires_at": expires}), encoding="utf-8")
        assert needs_recompile(path) is False

    def test_needs_recompile_expired(self, tmp_path):
        from scalpel.dspy_optimizer import needs_recompile
        from datetime import datetime, timedelta
        path     = tmp_path / "compiled.json"
        meta     = tmp_path / "compiled.meta.json"
        path.write_text("{}", encoding="utf-8")
        expires  = (datetime.now() - timedelta(days=1)).isoformat()  # вчера
        meta.write_text(json.dumps({"expires_at": expires}), encoding="utf-8")
        assert needs_recompile(path) is True


# ═══════════════════════════════════════════════════════════
# СЕКЦИЯ 3: DSPyOrchestrator fallback (без LLM)
# ═══════════════════════════════════════════════════════════

class TestOrchestratorFallback:

    def test_navigate_returns_fallback_without_dspy(self):
        from scalpel.dspy_optimizer import DSPyOrchestrator
        orch = DSPyOrchestrator()
        # _lm_ok = False по умолчанию
        result = orch.navigate("n_samples=100, features=[f0:10, f1:5]")
        assert "hypotheses" in result
        assert "selected_features" in result
        # Fallback должен давать хоть что-то
        assert len(result["hypotheses"]) > 0

    def test_audit_role_returns_conditional_without_dspy(self):
        from scalpel.dspy_optimizer import DSPyOrchestrator
        orch = DSPyOrchestrator()
        verdict, analysis = orch.audit_role(
            role_name="Скептик",
            role_task="Найди слабые места",
            formula="f0 / f1",
            formula_metrics="r2_train=0.95",
        )
        assert verdict == "УСЛОВНО"
        assert "DSPy" in analysis

    def test_is_active_false_without_lm(self):
        from scalpel.dspy_optimizer import DSPyOrchestrator
        orch = DSPyOrchestrator()
        assert orch.is_active is False


# ═══════════════════════════════════════════════════════════
# СЕКЦИЯ 4: log_failure_example
# ═══════════════════════════════════════════════════════════

class TestFailureLog:

    def test_failure_log_created(self, tmp_path):
        from scalpel.dspy_optimizer import log_failure_example
        with patch("scalpel.dspy_optimizer.DSPY_FAILURE_LOG",
                   tmp_path / "failures.jsonl"):
            log_failure_example(
                death_report={"r2_achieved": 0.5, "death_reasons": ["r2<threshold"]},
                data_meta="n_samples=100",
                attempt_number="1 of 3",
                failure_type="OVERFIT",
            )
        assert (tmp_path / "failures.jsonl").exists()

    def test_failure_log_appends(self, tmp_path):
        from scalpel.dspy_optimizer import log_failure_example
        log_path = tmp_path / "failures.jsonl"
        with patch("scalpel.dspy_optimizer.DSPY_FAILURE_LOG", log_path):
            log_failure_example({"death_reasons": ["a"]}, "meta", "1 of 3")
            log_failure_example({"death_reasons": ["b"]}, "meta", "2 of 3")
        lines = log_path.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 2

    def test_failure_log_valid_json_lines(self, tmp_path):
        from scalpel.dspy_optimizer import log_failure_example
        log_path = tmp_path / "failures.jsonl"
        with patch("scalpel.dspy_optimizer.DSPY_FAILURE_LOG", log_path):
            log_failure_example(
                {"r2_achieved": 0.3, "death_reasons": ["shuffle_failed"]},
                "n_samples=50, features=[f0:10, f1:5]",
                "2 of 3",
                failure_type="WRONG_STRUCTURE",
                corrected_strategy="try ratios instead",
                new_hypotheses="f0/f1;sqrt(f0)",
            )
        record = json.loads(log_path.read_text(encoding="utf-8").strip())
        assert record["failure_type"] == "WRONG_STRUCTURE"
        assert record["new_hypotheses"] == "f0/f1;sqrt(f0)"


# ═══════════════════════════════════════════════════════════
# СЕКЦИЯ 5: nav_decision_from_dspy (конвертер)
# ═══════════════════════════════════════════════════════════

class TestNavDecisionFromDspy:

    def test_basic_conversion(self):
        from scalpel.navigator import nav_decision_from_dspy
        dspy_result = {
            "selected_features":  "f0,f2",
            "selected_operators": "+,-,*,/,sqrt",
            "hypotheses":         "f0/f2;sqrt(f0)*f2;f0*f2",
            "ooda_stable":        "true",
            "reasoning":          "f0/f2 dimensionally consistent",
        }
        nav = nav_decision_from_dspy(dspy_result, ["f0", "f1", "f2", "f3"])
        assert nav.selected_features == ["f0", "f2"]
        assert "f0/f2" in nav.hypotheses
        assert nav.ooda_stable is True

    def test_ooda_false(self):
        from scalpel.navigator import nav_decision_from_dspy
        dspy_result = {
            "selected_features":  "f0,f1",
            "selected_operators": "/",
            "hypotheses":         "f0/f1",
            "ooda_stable":        "false",
            "reasoning":          "",
        }
        nav = nav_decision_from_dspy(dspy_result, ["f0", "f1"])
        assert nav.ooda_stable is False

    def test_hypotheses_split_by_semicolon(self):
        from scalpel.navigator import nav_decision_from_dspy
        dspy_result = {
            "selected_features":  "f0,f1",
            "selected_operators": "+,*",
            "hypotheses":         "f0+f1;f0*f1;sqrt(f0)",
            "ooda_stable":        "true",
            "reasoning":          "",
        }
        nav = nav_decision_from_dspy(dspy_result, ["f0", "f1"])
        assert len(nav.hypotheses) == 3

    def test_hypotheses_limited_to_5(self):
        from scalpel.navigator import nav_decision_from_dspy
        dspy_result = {
            "selected_features":  "f0,f1",
            "selected_operators": "+",
            "hypotheses":         ";".join([f"f0+f{i}" for i in range(10)]),
            "ooda_stable":        "true",
            "reasoning":          "",
        }
        nav = nav_decision_from_dspy(dspy_result, ["f0", "f1"])
        assert len(nav.hypotheses) <= 5

    def test_unknown_features_filtered(self):
        from scalpel.navigator import nav_decision_from_dspy
        dspy_result = {
            "selected_features":  "f0,f99,f1",  # f99 не существует
            "selected_operators": "+",
            "hypotheses":         "f0+f1",
            "ooda_stable":        "true",
            "reasoning":          "",
        }
        nav = nav_decision_from_dspy(dspy_result, ["f0", "f1"])
        assert "f99" not in nav.selected_features


# ═══════════════════════════════════════════════════════════
# СЕКЦИЯ 6: Священные метрики — проверяем что они НЕ DSPy
# ═══════════════════════════════════════════════════════════

class TestSacredMetrics:

    def test_shuffle_test_is_pure_python(self):
        """shuffle_test не должен импортировать dspy."""
        import inspect
        from scalpel.engine import shuffle_test
        src = inspect.getsource(shuffle_test)
        assert "dspy" not in src
        assert "SACRED" in src  # самопроверка присутствует

    def test_cross_blind_is_pure_python(self):
        """cross_blind не должен импортировать dspy."""
        import inspect
        from scalpel.engine import cross_blind
        src = inspect.getsource(cross_blind)
        assert "dspy" not in src
        assert "SACRED" in src

    def test_shuffle_test_result(self):
        """shuffle_test работает корректно."""
        import numpy as np
        from scalpel.engine import shuffle_test
        np.random.seed(42)
        X = np.random.rand(100, 2)
        y = X[:, 0] * 2 + X[:, 1]
        # Идеальный предиктор
        def perfect(Xp): return Xp[:, 0] * 2 + Xp[:, 1]
        passed, p = shuffle_test(perfect, X, y, n_perm=200, p_threshold=0.05)
        assert passed is True
        assert 0.0 <= p <= 1.0

    def test_cross_blind_result(self):
        """cross_blind работает корректно."""
        import numpy as np
        from scalpel.engine import cross_blind
        np.random.seed(42)
        X = np.random.rand(200, 2)
        y = X[:, 0] * 3 - X[:, 1]
        def pred(Xp): return Xp[:, 0] * 3 - Xp[:, 1]
        passed, r2 = cross_blind(pred, X, y, n_folds=5)
        assert passed is True
        assert r2 > 0.95

    def test_shadow_mapper_sacred(self):
        """ShadowMapper использует только Python-стандарт."""
        import inspect
        from scalpel.shadow import ShadowMapper
        src = inspect.getsource(ShadowMapper)
        assert "dspy" not in src


# ═══════════════════════════════════════════════════════════
# Вспомогательный класс для mock dspy.Example
# ═══════════════════════════════════════════════════════════

class _MockExample:
    """Имитирует dspy.Example для тестов без реального dspy."""
    def __init__(self, **kwargs):
        self._data = kwargs
        # Для проверки сортировки — храним формулу
        self.formula_hint = kwargs.get("hypotheses", "")

    def with_inputs(self, *args):
        return self

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        return self._data.get(name, "")
