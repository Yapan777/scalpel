"""
test_ram_queue.py — Тесты строгой RAM-очереди v9.9.1.
Все тесты работают без реального Ollama и без DSPy.
Запуск: pytest tests/test_ram_queue.py -v
"""
import gc
import json
import time
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock, call

# Импорты пакета на уровне модуля (требуются тестам класса TestMatryoshkaQueue)
from scalpel.ram_queue import MatryoshkaQueue, RoleResult, ram_status_report, role_ram_slot


# ═══════════════════════════════════════════════════════════
# СЕКЦИЯ 1: role_ram_slot (контекстный менеджер)
# ═══════════════════════════════════════════════════════════

class TestRoleRamSlot:

    def test_slot_yields_ready_when_enough_ram(self):
        from scalpel.ram_queue import role_ram_slot
        with patch("scalpel.ram_queue._avail_ram_gb", return_value=4.0), \
             patch("scalpel.ram_queue._ollama_stop_silent"), \
             patch("scalpel.ram_queue._gc_and_settle"):
            with role_ram_slot("Скептик", min_ram_gb=1.2) as slot:
                assert slot["ready"] is True
                assert slot["ram_before"] == 4.0

    def test_slot_not_ready_when_low_ram(self):
        from scalpel.ram_queue import role_ram_slot
        with patch("scalpel.ram_queue._avail_ram_gb", return_value=0.8), \
             patch("scalpel.ram_queue._ollama_stop_silent"), \
             patch("scalpel.ram_queue._gc_and_settle"):
            with role_ram_slot("Скептик", min_ram_gb=1.2) as slot:
                assert slot["ready"] is False

    def test_slot_always_calls_ollama_stop_on_exit(self):
        """ollama_stop вызывается даже при исключении внутри слота."""
        from scalpel.ram_queue import role_ram_slot
        mock_stop = MagicMock()
        with patch("scalpel.ram_queue._avail_ram_gb", return_value=4.0), \
             patch("scalpel.ram_queue._ollama_stop_silent", mock_stop), \
             patch("scalpel.ram_queue._gc_and_settle"):
            try:
                with role_ram_slot("Физик") as slot:
                    raise RuntimeError("Симулируем ошибку")
            except RuntimeError:
                pass
        mock_stop.assert_called_once()

    def test_slot_always_calls_gc_on_exit(self):
        """gc.collect вызывается всегда при выходе из слота."""
        from scalpel.ram_queue import role_ram_slot
        mock_gc = MagicMock(return_value=10)
        with patch("scalpel.ram_queue._avail_ram_gb", return_value=4.0), \
             patch("scalpel.ram_queue._ollama_stop_silent"), \
             patch("scalpel.ram_queue._gc_and_settle", mock_gc):
            with role_ram_slot("Прагматик") as slot:
                pass
        mock_gc.assert_called_once()

    def test_slot_records_ram_after(self):
        """Слот записывает RAM после очистки."""
        from scalpel.ram_queue import role_ram_slot
        ram_values = [3.0, 3.5]  # до и после gc
        ram_iter   = iter(ram_values)

        with patch("scalpel.ram_queue._avail_ram_gb", side_effect=ram_iter), \
             patch("scalpel.ram_queue._ollama_stop_silent"), \
             patch("scalpel.ram_queue._gc_and_settle"):
            with role_ram_slot("Мистик") as slot:
                pass
        assert slot["ram_after"] == 3.5

    def test_slots_run_sequentially(self):
        """Тест что слоты не пересекаются во времени."""
        from scalpel.ram_queue import role_ram_slot
        events = []
        with patch("scalpel.ram_queue._avail_ram_gb", return_value=4.0), \
             patch("scalpel.ram_queue._ollama_stop_silent"), \
             patch("scalpel.ram_queue._gc_and_settle"):
            for role in ["Скептик", "Физик", "Прагматик"]:
                with role_ram_slot(role) as slot:
                    events.append(f"enter_{role}")
                events.append(f"exit_{role}")

        # Проверяем строгую очерёдность: enter1, exit1, enter2, exit2...
        assert events == [
            "enter_Скептик",  "exit_Скептик",
            "enter_Физик",    "exit_Физик",
            "enter_Прагматик","exit_Прагматик",
        ]


# ═══════════════════════════════════════════════════════════
# СЕКЦИЯ 2: RoleModuleLoader
# ═══════════════════════════════════════════════════════════

class TestRoleModuleLoader:

    def test_compiled_path_per_role(self, tmp_path):
        from scalpel.ram_queue import RoleModuleLoader
        with patch("scalpel.ram_queue.ROLE_COMPILED_DIR", tmp_path), \
             patch("scalpel.ram_queue.ROLE_FAILURE_DIR", tmp_path):
            loader = RoleModuleLoader()
        paths = {role: loader.compiled_path(role) for role in
                 ["Скептик","Физик","Прагматик","Мистик"]}
        # Все пути разные
        assert len(set(str(p) for p in paths.values())) == 4
        # Правильные slug-имена
        assert "skeptic" in str(paths["Скептик"])
        assert "physicist" in str(paths["Физик"])
        assert "pragmatist" in str(paths["Прагматик"])
        assert "mystic" in str(paths["Мистик"])

    def test_failure_log_path_per_role(self, tmp_path):
        from scalpel.ram_queue import RoleModuleLoader
        with patch("scalpel.ram_queue.ROLE_COMPILED_DIR", tmp_path), \
             patch("scalpel.ram_queue.ROLE_FAILURE_DIR", tmp_path):
            loader = RoleModuleLoader()
        paths = {role: loader.failure_log_path(role)
                 for role in ["Скептик","Физик"]}
        assert paths["Скептик"] != paths["Физик"]
        assert paths["Скептик"].suffix == ".jsonl"

    def test_log_role_result_creates_file(self, tmp_path):
        from scalpel.ram_queue import RoleModuleLoader
        with patch("scalpel.ram_queue.ROLE_COMPILED_DIR", tmp_path), \
             patch("scalpel.ram_queue.ROLE_FAILURE_DIR", tmp_path):
            loader = RoleModuleLoader()
            loader.log_role_result(
                role_name = "Скептик",
                formula   = "f0/f1",
                metrics   = "r2_train=0.95",
                verdict   = "ПРИНЯТА",
                analysis  = "Формула корректна.",
            )
        log_file = tmp_path / "skeptic.jsonl"
        assert log_file.exists()

    def test_log_role_result_appends(self, tmp_path):
        from scalpel.ram_queue import RoleModuleLoader
        with patch("scalpel.ram_queue.ROLE_COMPILED_DIR", tmp_path), \
             patch("scalpel.ram_queue.ROLE_FAILURE_DIR", tmp_path):
            loader = RoleModuleLoader()
            loader.log_role_result("Физик","f0/f1","r2=0.95","ПРИНЯТА","OK")
            loader.log_role_result("Физик","f0*f1","r2=0.70","ОТКЛОНЕНА","Плохо")
        log_file = tmp_path / "physicist.jsonl"
        lines = log_file.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 2

    def test_log_role_result_valid_json(self, tmp_path):
        from scalpel.ram_queue import RoleModuleLoader
        with patch("scalpel.ram_queue.ROLE_COMPILED_DIR", tmp_path), \
             patch("scalpel.ram_queue.ROLE_FAILURE_DIR", tmp_path):
            loader = RoleModuleLoader()
            loader.log_role_result(
                "Мистик","sqrt(f0)*f1","r2=0.92","ПРИНЯТА",
                "Напоминает закон Кеплера."
            )
        record = json.loads(
            (tmp_path / "mystic.jsonl").read_text(encoding="utf-8").strip()
        )
        assert record["verdict"]  == "ПРИНЯТА"
        assert record["formula"]  == "sqrt(f0)*f1"

    def test_role_task_all_roles(self):
        from scalpel.ram_queue import RoleModuleLoader
        for role in ["Скептик","Физик","Прагматик","Мистик"]:
            task = RoleModuleLoader._role_task(role)
            assert isinstance(task, str) and len(task) > 10

    def test_load_role_examples_empty_without_dspy(self, tmp_path):
        from scalpel.ram_queue import RoleModuleLoader
        with patch("scalpel.ram_queue.ROLE_COMPILED_DIR", tmp_path), \
             patch("scalpel.ram_queue.ROLE_FAILURE_DIR", tmp_path):
            loader = RoleModuleLoader()
        with patch("scalpel.ram_queue.DSPY_AVAILABLE", False, create=True):
            result = loader.load_role_examples("Скептик")
        assert result == []


# ═══════════════════════════════════════════════════════════
# СЕКЦИЯ 3: MatryoshkaQueue — полный прогон (mock LLM)
# ═══════════════════════════════════════════════════════════

class TestMatryoshkaQueue:

    def _make_queue(self, tmp_path):
        from scalpel.ram_queue import MatryoshkaQueue
        with patch("scalpel.ram_queue.ROLE_COMPILED_DIR", tmp_path / "models"), \
             patch("scalpel.ram_queue.ROLE_FAILURE_DIR", tmp_path / "failures"):
            q = MatryoshkaQueue(dspy_active=False)
        return q

    def test_runs_exactly_4_roles(self, tmp_path):
        from scalpel.ram_queue import MatryoshkaQueue
        call_count = []

        def mock_legacy(*args, **kwargs):
            call_count.append(1)
            return "Анализ завершён. ПРИНЯТА"

        with patch("scalpel.ram_queue._avail_ram_gb", return_value=4.0), \
             patch("scalpel.ram_queue._ollama_stop_silent"), \
             patch("scalpel.ram_queue._gc_and_settle"), \
             patch("scalpel.ram_queue.ROLE_COMPILED_DIR", tmp_path), \
             patch("scalpel.ram_queue.ROLE_FAILURE_DIR", tmp_path), \
             patch("scalpel.navigator.ollama_chat", mock_legacy):
            q = MatryoshkaQueue(dspy_active=False)
            consensus, report, results = q.run(
                formula_shadow = "f0/f1",
                shadow_names   = ["f0","f1"],
                r2_train       = 0.95,
                complexity     = 3,
            )
        assert len(results) == 4

    def test_consensus_accepted_3_of_4(self, tmp_path):
        responses = [
            "Всё верно. ПРИНЯТА",
            "Размерности OK. ПРИНЯТА",
            "Применимо. ПРИНЯТА",
            "Интересно. УСЛОВНО",
        ]
        idx = [0]
        def mock_chat(*args, **kwargs):
            r = responses[idx[0] % len(responses)]
            idx[0] += 1
            return r

        with patch("scalpel.ram_queue._avail_ram_gb", return_value=4.0), \
             patch("scalpel.ram_queue._ollama_stop_silent"), \
             patch("scalpel.ram_queue._gc_and_settle"), \
             patch("scalpel.ram_queue.ROLE_COMPILED_DIR", tmp_path), \
             patch("scalpel.ram_queue.ROLE_FAILURE_DIR", tmp_path), \
             patch("scalpel.navigator.ollama_chat", mock_chat):
            q = MatryoshkaQueue(dspy_active=False)
            consensus, _, _ = q.run("f0/f1",["f0","f1"],0.95,3)
        assert consensus == "ПРИНЯТА"

    def test_consensus_rejected(self, tmp_path):
        def mock_chat(*args, **kwargs):
            return "Формула неверна. ОТКЛОНЕНА"

        with patch("scalpel.ram_queue._avail_ram_gb", return_value=4.0), \
             patch("scalpel.ram_queue._ollama_stop_silent"), \
             patch("scalpel.ram_queue._gc_and_settle"), \
             patch("scalpel.ram_queue.ROLE_COMPILED_DIR", tmp_path), \
             patch("scalpel.ram_queue.ROLE_FAILURE_DIR", tmp_path), \
             patch("scalpel.navigator.ollama_chat", mock_chat):
            q = MatryoshkaQueue(dspy_active=False)
            consensus, _, _ = q.run("f0+f1",["f0","f1"],0.60,2)
        assert consensus == "ОТКЛОНЕНА"

    def test_consensus_disputed(self, tmp_path):
        responses = ["ПРИНЯТА","ПРИНЯТА","ОТКЛОНЕНА","УСЛОВНО"]
        idx = [0]
        def mock_chat(*args, **kwargs):
            r = responses[idx[0] % 4]
            idx[0] += 1
            return r

        with patch("scalpel.ram_queue._avail_ram_gb", return_value=4.0), \
             patch("scalpel.ram_queue._ollama_stop_silent"), \
             patch("scalpel.ram_queue._gc_and_settle"), \
             patch("scalpel.ram_queue.ROLE_COMPILED_DIR", tmp_path), \
             patch("scalpel.ram_queue.ROLE_FAILURE_DIR", tmp_path), \
             patch("scalpel.navigator.ollama_chat", mock_chat):
            q = MatryoshkaQueue(dspy_active=False)
            consensus, _, _ = q.run("f0*f1",["f0","f1"],0.80,2)
        assert consensus == "СПОРНО"

    def test_role_results_have_ram_stats(self, tmp_path):
        ram_vals = [4.0, 4.2, 4.1, 4.3, 4.2, 4.4, 4.3, 4.5]
        idx = [0]
        def ram_fn():
            v = ram_vals[idx[0] % len(ram_vals)]
            idx[0] += 1
            return v

        def mock_chat(*args, **kwargs):
            return "OK. ПРИНЯТА"

        with patch("scalpel.ram_queue._avail_ram_gb", side_effect=ram_fn), \
             patch("scalpel.ram_queue._ollama_stop_silent"), \
             patch("scalpel.ram_queue._gc_and_settle"), \
             patch("scalpel.ram_queue.ROLE_COMPILED_DIR", tmp_path), \
             patch("scalpel.ram_queue.ROLE_FAILURE_DIR", tmp_path), \
             patch("scalpel.navigator.ollama_chat", mock_chat):
            q = MatryoshkaQueue(dspy_active=False)
            _, _, results = q.run("f0/f1",["f0","f1"],0.95,3)

        for r in results:
            assert r.ram_before > 0
            assert r.elapsed_sec >= 0

    def test_low_ram_role_returns_conditional(self, tmp_path):
        """Роль с нехваткой RAM возвращает УСЛОВНО без вызова LLM."""
        call_count = [0]
        def mock_chat(*args, **kwargs):
            call_count[0] += 1
            return "ПРИНЯТА"

        with patch("scalpel.ram_queue._avail_ram_gb", return_value=0.5), \
             patch("scalpel.ram_queue._ollama_stop_silent"), \
             patch("scalpel.ram_queue._gc_and_settle"), \
             patch("scalpel.ram_queue.ROLE_COMPILED_DIR", tmp_path), \
             patch("scalpel.ram_queue.ROLE_FAILURE_DIR", tmp_path), \
             patch("scalpel.navigator.ollama_chat", mock_chat):
            q = MatryoshkaQueue(dspy_active=False)
            consensus, _, results = q.run("f0/f1",["f0","f1"],0.95,3)

        assert call_count[0] == 0          # LLM не вызывался
        assert all(r.verdict == "УСЛОВНО" for r in results)
        assert all(r.error == "LOW_RAM"   for r in results)

    def test_failure_logs_written_per_role(self, tmp_path):
        """После прогона у каждой роли есть свой failure log."""
        failures_dir = tmp_path / "failures"
        failures_dir.mkdir()

        def mock_chat(*args, **kwargs):
            return "Анализ. ПРИНЯТА"

        with patch("scalpel.ram_queue._avail_ram_gb", return_value=4.0), \
             patch("scalpel.ram_queue._ollama_stop_silent"), \
             patch("scalpel.ram_queue._gc_and_settle"), \
             patch("scalpel.ram_queue.ROLE_COMPILED_DIR", tmp_path / "models"), \
             patch("scalpel.ram_queue.ROLE_FAILURE_DIR", failures_dir), \
             patch("scalpel.navigator.ollama_chat", mock_chat):
            q = MatryoshkaQueue(dspy_active=False)
            q.run("f0/f1",["f0","f1"],0.95,3)

        log_files = list(failures_dir.glob("*.jsonl"))
        assert len(log_files) == 4           # один файл на роль

    def test_ollama_stop_called_after_each_role(self, tmp_path):
        """ollama_stop вызывается 4 раза — после каждой роли."""
        stop_calls = [0]
        def mock_stop(*args, **kwargs):
            stop_calls[0] += 1

        def mock_chat(*args, **kwargs):
            return "OK. ПРИНЯТА"

        with patch("scalpel.ram_queue._avail_ram_gb", return_value=4.0), \
             patch("scalpel.ram_queue._ollama_stop_silent", mock_stop), \
             patch("scalpel.ram_queue._gc_and_settle"), \
             patch("scalpel.ram_queue.ROLE_COMPILED_DIR", tmp_path), \
             patch("scalpel.ram_queue.ROLE_FAILURE_DIR", tmp_path), \
             patch("scalpel.navigator.ollama_chat", mock_chat):
            q = MatryoshkaQueue(dspy_active=False)
            q.run("f0/f1",["f0","f1"],0.95,3)

        assert stop_calls[0] == 4


# ═══════════════════════════════════════════════════════════
# СЕКЦИЯ 4: ram_status_report
# ═══════════════════════════════════════════════════════════

class TestRamStatusReport:

    def test_report_contains_all_roles(self):
        from scalpel.ram_queue import ram_status_report, RoleResult
        results = [
            RoleResult("Скептик",   "ПРИНЯТА",   "OK", 3.0, 3.5, 1.2),
            RoleResult("Физик",     "ОТКЛОНЕНА", "NO", 3.5, 3.8, 0.9),
            RoleResult("Прагматик", "УСЛОВНО",   "?",  3.8, 4.0, 1.1),
            RoleResult("Мистик",    "ПРИНЯТА",   "OK", 4.0, 4.2, 0.8),
        ]
        report = ram_status_report(results)
        for role in ["Скептик","Физик","Прагматик","Мистик"]:
            assert role in report

    def test_report_shows_verdicts(self):
        from scalpel.ram_queue import ram_status_report, RoleResult
        results = [
            RoleResult("Скептик", "ПРИНЯТА",   "ok", 3.0, 3.5, 1.0),
            RoleResult("Физик",   "ОТКЛОНЕНА", "no", 3.5, 3.8, 0.9),
        ]
        report = ram_status_report(results)
        assert "ПРИНЯТА"   in report
        assert "ОТКЛОНЕНА" in report
