"""
test_episodic_memory.py — Тесты эпизодической памяти v9.9.2.
Все тесты без DSPy и без Ollama.
Запуск: pytest tests/test_episodic_memory.py -v
"""
import json
import pytest
from pathlib import Path
from unittest.mock import patch


# ═══════════════════════════════════════════════════════════
# СЕКЦИЯ 1: remember() — запись в память
# ═══════════════════════════════════════════════════════════

class TestRemember:

    def test_creates_file_on_first_remember(self, tmp_path):
        from scalpel.episodic_memory import EpisodicMemory
        mem = EpisodicMemory(tmp_path)
        mem.remember("Скептик", "f0/f1", "ПРИНЯТА", "Хорошая формула")
        assert (tmp_path / "skeptic_memory.jsonl").exists()

    def test_separate_files_per_role(self, tmp_path):
        from scalpel.episodic_memory import EpisodicMemory
        mem = EpisodicMemory(tmp_path)
        mem.remember("Скептик",   "f0/f1", "ПРИНЯТА",   "OK")
        mem.remember("Физик",     "f0*f1", "ОТКЛОНЕНА", "NO")
        mem.remember("Прагматик", "f0+f1", "УСЛОВНО",   "?")
        mem.remember("Мистик",    "f0-f1", "ПРИНЯТА",   "OK")
        assert (tmp_path / "skeptic_memory.jsonl").exists()
        assert (tmp_path / "physicist_memory.jsonl").exists()
        assert (tmp_path / "pragmatist_memory.jsonl").exists()
        assert (tmp_path / "mystic_memory.jsonl").exists()

    def test_record_has_all_fields(self, tmp_path):
        from scalpel.episodic_memory import EpisodicMemory
        mem = EpisodicMemory(tmp_path)
        mem.remember(
            role_name  = "Физик",
            formula    = "sqrt(f0) * f1",
            verdict    = "ПРИНЯТА",
            analysis   = "Размерностно корректно.",
            r2_train   = 0.95,
            r2_blind   = 0.91,
            complexity = 4,
            domain     = "Physics",
        )
        record = json.loads(
            (tmp_path / "physicist_memory.jsonl").read_text(encoding="utf-8").strip()
        )
        assert record["formula"]    == "sqrt(f0) * f1"
        assert record["verdict"]    == "ПРИНЯТА"
        assert record["r2_train"]   == 0.95
        assert record["r2_blind"]   == 0.91
        assert record["complexity"] == 4
        assert record["domain"]     == "Physics"
        assert "ts"                 in record
        assert "skeleton"           in record
        assert "tags"               in record

    def test_appends_multiple_records(self, tmp_path):
        from scalpel.episodic_memory import EpisodicMemory
        mem = EpisodicMemory(tmp_path)
        for i in range(5):
            mem.remember("Скептик", f"f{i}/f0", "ПРИНЯТА", f"Анализ {i}")
        lines = (tmp_path / "skeptic_memory.jsonl").read_text(
            encoding="utf-8").strip().splitlines()
        assert len(lines) == 5

    def test_analysis_truncated_to_500(self, tmp_path):
        from scalpel.episodic_memory import EpisodicMemory
        mem = EpisodicMemory(tmp_path)
        long_analysis = "x" * 1000
        mem.remember("Мистик", "f0*f1", "УСЛОВНО", long_analysis)
        record = json.loads(
            (tmp_path / "mystic_memory.jsonl").read_text(encoding="utf-8").strip()
        )
        assert len(record["analysis"]) <= 500

    def test_skeleton_generated(self, tmp_path):
        from scalpel.episodic_memory import EpisodicMemory
        mem = EpisodicMemory(tmp_path)
        mem.remember("Физик", "f0 / f1 + 3.14", "ПРИНЯТА", "OK")
        record = json.loads(
            (tmp_path / "physicist_memory.jsonl").read_text(encoding="utf-8").strip()
        )
        assert "v" in record["skeleton"]  # переменные
        assert "c" in record["skeleton"]  # константа


# ═══════════════════════════════════════════════════════════
# СЕКЦИЯ 2: recall() — загрузка из памяти
# ═══════════════════════════════════════════════════════════

class TestRecall:

    def _populate(self, mem, role, n_accepted=3, n_rejected=2):
        for i in range(n_accepted):
            mem.remember(role, f"f{i}/f0", "ПРИНЯТА",
                        f"OK {i}", r2_blind=0.90+i*0.01)
        for i in range(n_rejected):
            mem.remember(role, f"f{i}+f0", "ОТКЛОНЕНА",
                        f"NO {i}", r2_blind=0.50)

    def test_recall_all(self, tmp_path):
        from scalpel.episodic_memory import EpisodicMemory
        mem = EpisodicMemory(tmp_path)
        self._populate(mem, "Скептик")
        records = mem.recall("Скептик")
        # Без dspy возвращает raw dict
        assert len(records) <= 5

    def test_recall_filter_accepted(self, tmp_path):
        from scalpel.episodic_memory import EpisodicMemory
        mem = EpisodicMemory(tmp_path)
        self._populate(mem, "Физик", n_accepted=3, n_rejected=4)
        records = mem.recall("Физик", verdict_filter="ПРИНЯТА")
        assert all(r.get("verdict") == "ПРИНЯТА" for r in records)

    def test_recall_filter_rejected(self, tmp_path):
        from scalpel.episodic_memory import EpisodicMemory
        mem = EpisodicMemory(tmp_path)
        self._populate(mem, "Прагматик", n_accepted=2, n_rejected=3)
        records = mem.recall("Прагматик", verdict_filter="ОТКЛОНЕНА")
        assert all(r.get("verdict") == "ОТКЛОНЕНА" for r in records)

    def test_recall_limit(self, tmp_path):
        from scalpel.episodic_memory import EpisodicMemory
        mem = EpisodicMemory(tmp_path)
        for i in range(10):
            mem.remember("Мистик", f"f{i}", "ПРИНЯТА", "OK", r2_blind=0.9)
        records = mem.recall("Мистик", limit=3)
        assert len(records) <= 3

    def test_recall_sorted_by_r2_blind(self, tmp_path):
        from scalpel.episodic_memory import EpisodicMemory
        mem = EpisodicMemory(tmp_path)
        mem.remember("Скептик", "f0/f1", "ПРИНЯТА", "OK", r2_blind=0.85)
        mem.remember("Скептик", "f0*f1", "ПРИНЯТА", "OK", r2_blind=0.95)
        mem.remember("Скептик", "f0+f1", "ПРИНЯТА", "OK", r2_blind=0.75)
        records = mem.recall("Скептик", verdict_filter="ПРИНЯТА")
        r2s = [float(r.get("r2_blind", 0)) for r in records]
        assert r2s == sorted(r2s, reverse=True)

    def test_recall_empty_when_no_memory(self, tmp_path):
        from scalpel.episodic_memory import EpisodicMemory
        mem = EpisodicMemory(tmp_path)
        records = mem.recall("Физик")
        assert records == []

    def test_recall_min_r2_filter(self, tmp_path):
        from scalpel.episodic_memory import EpisodicMemory
        mem = EpisodicMemory(tmp_path)
        mem.remember("Физик", "f0/f1", "ПРИНЯТА", "OK", r2_blind=0.95)
        mem.remember("Физик", "f0+f1", "ПРИНЯТА", "OK", r2_blind=0.60)
        records = mem.recall("Физик", min_r2=0.80)
        r2s = [float(r.get("r2_blind", 0)) for r in records]
        assert all(r2 >= 0.80 for r2 in r2s)


# ═══════════════════════════════════════════════════════════
# СЕКЦИЯ 3: stats() — статистика
# ═══════════════════════════════════════════════════════════

class TestStats:

    def test_stats_empty(self, tmp_path):
        from scalpel.episodic_memory import EpisodicMemory
        mem   = EpisodicMemory(tmp_path)
        stats = mem.stats("Скептик")
        assert stats["total"]    == 0
        assert stats["accepted"] == 0

    def test_stats_counts(self, tmp_path):
        from scalpel.episodic_memory import EpisodicMemory
        mem = EpisodicMemory(tmp_path)
        mem.remember("Физик", "f0/f1", "ПРИНЯТА",   "OK", r2_blind=0.95)
        mem.remember("Физик", "f0+f1", "ПРИНЯТА",   "OK", r2_blind=0.90)
        mem.remember("Физик", "f0*f1", "ОТКЛОНЕНА", "NO")
        mem.remember("Физик", "f0-f1", "УСЛОВНО",   "?")
        stats = mem.stats("Физик")
        assert stats["total"]       == 4
        assert stats["accepted"]    == 2
        assert stats["rejected"]    == 1
        assert stats["conditional"] == 1

    def test_acceptance_rate(self, tmp_path):
        from scalpel.episodic_memory import EpisodicMemory
        mem = EpisodicMemory(tmp_path)
        mem.remember("Прагматик", "f0/f1", "ПРИНЯТА",   "OK")
        mem.remember("Прагматик", "f0+f1", "ПРИНЯТА",   "OK")
        mem.remember("Прагматик", "f0*f1", "ОТКЛОНЕНА", "NO")
        mem.remember("Прагматик", "f0-f1", "ОТКЛОНЕНА", "NO")
        stats = mem.stats("Прагматик")
        assert abs(stats["acceptance_rate"] - 0.5) < 0.01

    def test_avg_r2_accepted(self, tmp_path):
        from scalpel.episodic_memory import EpisodicMemory
        mem = EpisodicMemory(tmp_path)
        mem.remember("Мистик", "f0/f1", "ПРИНЯТА", "OK", r2_blind=0.90)
        mem.remember("Мистик", "f0*f1", "ПРИНЯТА", "OK", r2_blind=0.80)
        stats = mem.stats("Мистик")
        assert abs(stats["avg_r2_accepted"] - 0.85) < 0.01

    def test_stats_all_returns_4_roles(self, tmp_path):
        from scalpel.episodic_memory import EpisodicMemory
        mem   = EpisodicMemory(tmp_path)
        stats = mem.stats_all()
        assert len(stats) == 4
        assert "Скептик"   in stats
        assert "Физик"     in stats
        assert "Прагматик" in stats
        assert "Мистик"    in stats


# ═══════════════════════════════════════════════════════════
# СЕКЦИЯ 4: top_patterns()
# ═══════════════════════════════════════════════════════════

class TestTopPatterns:

    def test_top_patterns_accepted(self, tmp_path):
        from scalpel.episodic_memory import EpisodicMemory
        mem = EpisodicMemory(tmp_path)
        # f0/f1 принята 3 раза
        for i in range(3):
            mem.remember("Скептик", "f0 / f1", "ПРИНЯТА", "OK", r2_blind=0.9)
        # f0*f1 принята 1 раз
        mem.remember("Скептик", "f0 * f1", "ПРИНЯТА", "OK", r2_blind=0.85)
        patterns = mem.top_patterns("Скептик", verdict="ПРИНЯТА")
        assert patterns[0]["count"] == 3  # f0/f1 первый

    def test_top_patterns_empty_role(self, tmp_path):
        from scalpel.episodic_memory import EpisodicMemory
        mem = EpisodicMemory(tmp_path)
        patterns = mem.top_patterns("Физик", verdict="ПРИНЯТА")
        assert patterns == []


# ═══════════════════════════════════════════════════════════
# СЕКЦИЯ 5: _formula_skeleton и _formula_tags
# ═══════════════════════════════════════════════════════════

class TestFormulaUtils:

    def test_skeleton_ratio(self):
        from scalpel.episodic_memory import _formula_skeleton
        assert _formula_skeleton("f0 / f1") == "v / v"

    def test_skeleton_with_constant(self):
        from scalpel.episodic_memory import _formula_skeleton
        s = _formula_skeleton("f0 + 3.14")
        assert "v" in s and "c" in s

    def test_skeleton_with_sqrt(self):
        from scalpel.episodic_memory import _formula_skeleton
        s = _formula_skeleton("sqrt(f0) * f1")
        assert "sqrt" in s

    def test_tags_ratio(self):
        from scalpel.episodic_memory import _formula_tags
        tags = _formula_tags("f0 / f1", "v / v")
        assert "ratio" in tags

    def test_tags_sqrt(self):
        from scalpel.episodic_memory import _formula_tags
        tags = _formula_tags("sqrt(f0) * f1", "sqrt ( v ) * v")
        assert "sqrt" in tags

    def test_tags_var_count(self):
        from scalpel.episodic_memory import _formula_tags
        tags = _formula_tags("f0 * f1 * f2", "v * v * v")
        assert "vars_3" in tags


# ═══════════════════════════════════════════════════════════
# СЕКЦИЯ 6: get_memory() singleton
# ═══════════════════════════════════════════════════════════

class TestGetMemory:

    def test_returns_same_instance(self):
        from scalpel.episodic_memory import get_memory, _global_memory
        import scalpel.episodic_memory as em_mod
        # Сбрасываем синглтон
        em_mod._global_memory = None
        m1 = get_memory()
        m2 = get_memory()
        assert m1 is m2

    def test_returns_episodic_memory(self):
        from scalpel.episodic_memory import get_memory, EpisodicMemory
        assert isinstance(get_memory(), EpisodicMemory)
