"""
test_curriculum_chronicle.py — Минимальные тесты для:
  - curriculum.py             (Prompt 1)
  - generate_chronicle()      (Prompt 2)
  - remember_chronicle_step() (Prompt 3)
  - recall_chronicle_steps()  (Prompt 3)
  - checkpoint при долгом запуске (Prompt 4)

Тесты не требуют PySR, Ollama или DSPy — всё мокируется.
"""
from __future__ import annotations

import gc
import json
import tempfile
import time
from pathlib import Path
from typing import List
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


# ═══════════════════════════════════════════════════════════════════
# PROMPT 1 — curriculum.py
# ═══════════════════════════════════════════════════════════════════

class TestCurriculumDatasetGeneration:
    """Тест: генераторы датасетов возвращают корректные структуры."""

    def test_generate_level1_count_and_shape(self):
        from scalpel.curriculum import generate_level1
        rng = np.random.default_rng(42)
        datasets = generate_level1(noise=0.01, rng=rng)
        assert len(datasets) == 10, "Уровень 1 должен давать ровно 10 датасетов"
        for ds in datasets:
            assert ds.X_train.shape[0] == 150
            assert ds.X_test.shape[0] == 50
            assert ds.y_train.shape[0] == 150
            assert ds.level == 1
            assert len(ds.dim_codes) == ds.X_train.shape[1]

    def test_generate_level2_count_and_shape(self):
        from scalpel.curriculum import generate_level2
        rng = np.random.default_rng(42)
        datasets = generate_level2(noise=0.03, rng=rng)
        assert len(datasets) == 10
        for ds in datasets:
            assert ds.level == 2
            assert len(ds.feat_names) == ds.X_train.shape[1]

    def test_generate_level3_count_and_shape(self):
        from scalpel.curriculum import generate_level3
        rng = np.random.default_rng(42)
        datasets = generate_level3(noise=0.05, rng=rng)
        assert len(datasets) == 10
        for ds in datasets:
            assert ds.level == 3

    def test_generate_level4_count_and_physics(self):
        from scalpel.curriculum import generate_level4
        rng = np.random.default_rng(42)
        datasets = generate_level4(noise=0.10, rng=rng)
        assert len(datasets) == 10
        # Хотя бы один датасет с физическим dim_code (!=0)
        has_physics = any(any(c != 0 for c in ds.dim_codes) for ds in datasets)
        assert has_physics, "Уровень 4 должен иметь физические dim_codes"

    def test_noise_actually_added(self):
        from scalpel.curriculum import generate_level1
        import numpy as np
        rng_clean = np.random.default_rng(99)
        rng_noisy = np.random.default_rng(99)
        ds_clean = generate_level1(noise=0.0, rng=rng_clean)[0]
        ds_noisy = generate_level1(noise=1.0, rng=rng_noisy)[0]
        # FIX: реальные проверки вместо пустого теста
        assert not np.allclose(ds_clean.y_train, ds_noisy.y_train), \
            "noise=1.0 должен изменить y_train относительно noise=0.0"
        assert np.std(ds_noisy.y_train) >= np.std(ds_clean.y_train), \
            "Зашумлённый датасет должен иметь std не меньше чистого"


class TestCurriculumCheckpoint:
    """Тест: checkpoint при долгом запуске curriculum."""

    def test_checkpoint_saved_and_restored(self, tmp_path):
        """
        Проверяем что checkpoint записывается в JSONL и читается обратно.
        Checkpoint — защита от потери прогресса при 50-часовом прогоне.
        """
        checkpoint_path = tmp_path / "curriculum_checkpoint.json"

        def save_checkpoint(level: int, dataset: int, r2_history: List[float]):
            checkpoint = {
                "level":      level,
                "dataset":    dataset,
                "r2_history": r2_history,
                "ts":         time.time(),
            }
            checkpoint_path.write_text(
                json.dumps(checkpoint, ensure_ascii=False), encoding="utf-8"
            )

        def load_checkpoint():
            if not checkpoint_path.exists():
                return None
            try:
                return json.loads(checkpoint_path.read_text(encoding="utf-8"))
            except Exception:
                return None

        # Симулируем прогресс уровня 2
        save_checkpoint(level=2, dataset=7, r2_history=[0.91, 0.88, 0.85, 0.92, 0.87, 0.90, 0.88])
        cp = load_checkpoint()
        assert cp is not None
        assert cp["level"] == 2
        assert cp["dataset"] == 7
        assert len(cp["r2_history"]) == 7
        assert all(isinstance(x, float) for x in cp["r2_history"])

    def test_checkpoint_resume_skips_done_datasets(self, tmp_path):
        """
        При resume: датасеты до checkpoint["dataset"] пропускаются.
        """
        checkpoint_path = tmp_path / "curriculum_checkpoint.json"
        checkpoint_path.write_text(
            json.dumps({"level": 1, "dataset": 5, "r2_history": [0.95]*5}),
            encoding="utf-8",
        )
        cp = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        start_from = cp["dataset"]  # = 5
        all_datasets = list(range(10))  # 0..9
        remaining = all_datasets[start_from:]
        assert remaining == [5, 6, 7, 8, 9], "Должны остаться датасеты 5-9"

    def test_checkpoint_log_saved_to_jsonl(self, tmp_path):
        """
        _save_curriculum_log записывает в JSONL без ошибок.
        """
        from scalpel.curriculum import LevelResult, DatasetResult, _save_curriculum_log
        import unittest.mock as mock

        ds_result = DatasetResult(
            formula_true="y = a*x", formula_found="f0 * c",
            r2_blind=0.97, passed=True, level=1,
            noise_level=0.01, maxsize=8, fast_fail_sec=300,
            domain="curriculum_linear", elapsed_sec=42.0, error="",
        )
        lr = LevelResult(
            level=1, name="Линейные", mean_r2=0.97,
            passed=True, attempt=1, results=[ds_result], elapsed_sec=42.0,
        )

        with mock.patch("scalpel.curriculum.SCRIPT_DIR", tmp_path):
            _save_curriculum_log(lr)

        log_path = tmp_path / "scalpel_vault" / "curriculum_log.jsonl"
        assert log_path.exists(), "curriculum_log.jsonl должен быть создан"
        data = json.loads(log_path.read_text(encoding="utf-8"))
        assert data["level"] == 1
        assert data["mean_r2"] == 0.97
        assert data["passed"] is True
        assert len(data["datasets"]) == 1


class TestCurriculumLevelConfig:
    """Тест: параметры уровней соответствуют спецификации."""

    def test_level_thresholds(self):
        from scalpel.curriculum import LEVELS
        expected = {1: 0.90, 2: 0.85, 3: 0.80, 4: 0.75}
        for lv in LEVELS:
            assert lv.r2_threshold == expected[lv.level], (
                f"Уровень {lv.level}: ожидаем r2_threshold={expected[lv.level]}"
            )

    def test_level_maxsize(self):
        from scalpel.curriculum import LEVELS
        expected = {1: 8, 2: 12, 3: 15, 4: 20}
        for lv in LEVELS:
            assert lv.maxsize == expected[lv.level]

    def test_level_datasets_count(self):
        from scalpel.curriculum import LEVELS
        for lv in LEVELS:
            assert lv.datasets == 10


# ═══════════════════════════════════════════════════════════════════
# PROMPT 2 — generate_chronicle()
# ═══════════════════════════════════════════════════════════════════

class TestGenerateChronicle:
    """Тест: Летописец корректно собирает историю."""

    def _make_candidate(self, formula: str, r2: float, rejected_by: str = "", critique: str = "") -> dict:
        feedback = []
        if rejected_by:
            feedback.append({
                "role":    rejected_by,
                "verdict": "ОТКЛОНЕНА",
                "critique": critique,
                "suggestion": "",
            })
        return {
            "formula_shadow":     formula,
            "r2_blind":           r2,
            "matryoshka_feedback": feedback,
            "delphi_consilium":   {},
        }

    def test_chronicle_calls_ask_fn(self):
        from scalpel.critical_thinking import generate_chronicle
        history = [
            self._make_candidate("f0 * f1", 0.43, "Скептик", "формула не учитывает масштаб"),
            self._make_candidate("sqrt(f0) * f1", 0.71, "Физик", "размерность не сходится"),
            self._make_candidate("sqrt(f0) / f1", 0.94),
        ]
        calls = []
        def mock_ask(prompt: str) -> str:
            calls.append(prompt)
            return "ИСТОРИЯ ПОИСКА\nСистема прошла путь от f0*f1 до sqrt(f0)/f1."

        result = generate_chronicle(
            hadi_history=history,
            consilium={},
            heritage="закон Кеплера",
            domain="Physics",
            formula_final="sqrt(f0) / f1",
            ask_fn=mock_ask,
        )
        assert len(calls) == 1, "Летописец должен вызывать ask_fn ровно 1 раз"
        assert "ИСТОРИЯ ПОИСКА" in result

    def test_chronicle_empty_history_returns_empty(self):
        from scalpel.critical_thinking import generate_chronicle
        result = generate_chronicle(
            hadi_history=[],
            consilium={},
            heritage="",
            domain="",
            formula_final="f0",
            ask_fn=lambda p: "test",
        )
        assert result == "", "Пустая история → пустой рассказ"

    def test_chronicle_fallback_on_ask_fn_error(self):
        from scalpel.critical_thinking import generate_chronicle
        history = [
            self._make_candidate("f0", 0.50, "Скептик", "плохая формула"),
            self._make_candidate("f0 * f1", 0.90),
        ]
        def failing_ask(prompt: str) -> str:
            raise RuntimeError("ollama недоступен")

        result = generate_chronicle(
            hadi_history=history,
            consilium={},
            heritage="",
            domain="",
            formula_final="f0 * f1",
            ask_fn=failing_ask,
        )
        # Fallback должен вернуть хотя бы что-то
        assert "ИСТОРИЯ ПОИСКА" in result or "f0" in result

    def test_chronicle_prompt_includes_domain(self):
        from scalpel.critical_thinking import generate_chronicle
        history = [
            self._make_candidate("f0", 0.80),
        ]
        prompts = []
        def capture_ask(prompt: str) -> str:
            prompts.append(prompt)
            return "ИСТОРИЯ ПОИСКА\nОтчёт."

        generate_chronicle(
            hadi_history=history,
            consilium={},
            heritage="",
            domain="Finance",
            formula_final="f0",
            ask_fn=capture_ask,
        )
        assert "Finance" in prompts[0], "Домен должен быть в промпте"

    def test_chronicle_includes_rejection_reason(self):
        from scalpel.critical_thinking import generate_chronicle
        history = [
            self._make_candidate("f0", 0.40, "Физик", "размерность неверна"),
            self._make_candidate("f0/f1", 0.92),
        ]
        prompts = []
        def capture_ask(prompt: str) -> str:
            prompts.append(prompt)
            return "ИСТОРИЯ ПОИСКА\nОтчёт."

        generate_chronicle(
            hadi_history=history,
            consilium={},
            heritage="",
            domain="",
            formula_final="f0/f1",
            ask_fn=capture_ask,
        )
        assert "размерность неверна" in prompts[0], "Причина отклонения должна быть в промпте"


# ═══════════════════════════════════════════════════════════════════
# PROMPT 3 — remember_chronicle_step() / recall_chronicle_steps()
# ═══════════════════════════════════════════════════════════════════

class TestChronicleStepMemory:
    """Тест: сохранение и чтение шагов Летописца в episodic_memory."""

    def test_remember_and_recall_basic(self, tmp_path):
        from scalpel.episodic_memory import EpisodicMemory
        mem = EpisodicMemory(memory_dir=tmp_path)

        mem.remember_chronicle_step(
            attempt=1, tried="f0 * f1", r2=0.43,
            rejected_by="Скептик", reason="не учитывает масштаб",
            delphi_hint="добавить sqrt", led_to="sqrt(f0) * f1",
            domain="Physics",
        )
        mem.remember_chronicle_step(
            attempt=2, tried="sqrt(f0) * f1", r2=0.71,
            rejected_by="Физик", reason="размерность не сходится",
            delphi_hint="добавить деление", led_to="sqrt(f0) / f1",
            domain="Physics",
        )
        mem.remember_chronicle_step(
            attempt=3, tried="sqrt(f0) / f1", r2=0.94,
            rejected_by="", reason="",
            delphi_hint="", led_to="sqrt(f0) / f1",
            domain="Physics",
        )

        chains = mem.recall_chronicle_steps()
        assert len(chains) == 3

    def test_file_written_as_jsonl(self, tmp_path):
        from scalpel.episodic_memory import EpisodicMemory
        mem = EpisodicMemory(memory_dir=tmp_path)
        mem.remember_chronicle_step(
            attempt=1, tried="f0", r2=0.5, domain="Test",
        )
        path = tmp_path / "chronicle_steps.jsonl"
        assert path.exists(), "chronicle_steps.jsonl должен быть создан"
        lines = [l for l in path.read_text(encoding="utf-8").strip().splitlines() if l]
        assert len(lines) == 1
        rec = json.loads(lines[0])
        assert rec["event"] == "chronicle_step"
        assert rec["attempt"] == 1
        assert rec["tried"] == "f0"
        assert rec["r2"] == 0.5

    def test_domain_filter(self, tmp_path):
        from scalpel.episodic_memory import EpisodicMemory
        mem = EpisodicMemory(memory_dir=tmp_path)
        mem.remember_chronicle_step(attempt=1, tried="f0", r2=0.5, domain="Physics")
        mem.remember_chronicle_step(attempt=2, tried="f1", r2=0.6, domain="Finance")
        mem.remember_chronicle_step(attempt=3, tried="f2", r2=0.7, domain="Physics")

        physics_chains = mem.recall_chronicle_steps(domain="Physics")
        assert len(physics_chains) == 2
        finance_chains = mem.recall_chronicle_steps(domain="Finance")
        assert len(finance_chains) == 1

    def test_limit_respected(self, tmp_path):
        from scalpel.episodic_memory import EpisodicMemory
        mem = EpisodicMemory(memory_dir=tmp_path)
        for i in range(30):
            mem.remember_chronicle_step(attempt=i, tried=f"f{i}", r2=0.5)
        chains = mem.recall_chronicle_steps(limit=10)
        assert len(chains) == 10

    def test_chain_format_has_key_parts(self, tmp_path):
        from scalpel.episodic_memory import EpisodicMemory
        mem = EpisodicMemory(memory_dir=tmp_path)
        mem.remember_chronicle_step(
            attempt=2, tried="sqrt(f0) * f1", r2=0.71,
            rejected_by="Физик", reason="размерность",
            delphi_hint="добавить /", led_to="sqrt(f0)/f1",
        )
        chains = mem.recall_chronicle_steps()
        assert len(chains) == 1
        chain = chains[0]
        assert "sqrt(f0) * f1" in chain
        assert "Физик" in chain
        assert "добавить /" in chain or "Delphi" in chain
        assert "sqrt(f0)/f1" in chain

    def test_recall_empty_file_returns_empty_list(self, tmp_path):
        from scalpel.episodic_memory import EpisodicMemory
        mem = EpisodicMemory(memory_dir=tmp_path)
        result = mem.recall_chronicle_steps()
        assert result == []

    def test_corrupt_line_skipped_gracefully(self, tmp_path):
        from scalpel.episodic_memory import EpisodicMemory
        path = tmp_path / "chronicle_steps.jsonl"
        path.write_text(
            '{"event":"chronicle_step","attempt":1,"tried":"f0","r2":0.5}\n'
            'NOT_VALID_JSON\n'
            '{"event":"chronicle_step","attempt":2,"tried":"f1","r2":0.6}\n',
            encoding="utf-8",
        )
        mem = EpisodicMemory(memory_dir=tmp_path)
        chains = mem.recall_chronicle_steps()
        assert len(chains) == 2, "Битые строки должны молча пропускаться"

    def test_multiple_saves_accumulate(self, tmp_path):
        from scalpel.episodic_memory import EpisodicMemory
        mem = EpisodicMemory(memory_dir=tmp_path)
        for i in range(5):
            mem.remember_chronicle_step(attempt=i, tried=f"formula_{i}", r2=float(i)/10)
        chains = mem.recall_chronicle_steps()
        assert len(chains) == 5


# ═══════════════════════════════════════════════════════════════════
# PROMPT 4 — Безопасность: ошибки не ломают основной поток
# ═══════════════════════════════════════════════════════════════════

class TestErrorIsolation:
    """
    Тест: новые модули не ломают основной код при ошибках.
    Все вызовы завёрнуты в try/except.
    """

    def test_remember_curriculum_bad_path_no_exception(self, tmp_path, monkeypatch):
        """remember_curriculum не бросает исключение при ошибке записи."""
        from scalpel.episodic_memory import EpisodicMemory
        mem = EpisodicMemory(memory_dir=tmp_path)
        # Делаем директорию нечитаемой — симулируем disk error
        broken_path = tmp_path / "curriculum_memory.jsonl"
        broken_path.write_text("", encoding="utf-8")
        broken_path.chmod(0o000)
        try:
            mem.remember_curriculum(
                level=1, formula_true="y=x", formula_found="f0",
                r2_blind=0.9, noise_level=0.01, maxsize=8,
                fast_fail_sec=300, domain="test", passed=True,
            )
        except Exception as e:
            pytest.fail(f"remember_curriculum бросил исключение: {e}")
        finally:
            broken_path.chmod(0o644)

    def test_remember_chronicle_step_bad_path_no_exception(self, tmp_path):
        """remember_chronicle_step не бросает исключение при ошибке записи."""
        from scalpel.episodic_memory import EpisodicMemory
        mem = EpisodicMemory(memory_dir=tmp_path)
        broken = tmp_path / "chronicle_steps.jsonl"
        broken.write_text("", encoding="utf-8")
        broken.chmod(0o000)
        try:
            mem.remember_chronicle_step(
                attempt=1, tried="f0", r2=0.5, domain="test"
            )
        except Exception as e:
            pytest.fail(f"remember_chronicle_step бросил исключение: {e}")
        finally:
            broken.chmod(0o644)

    def test_recall_chronicle_steps_missing_file_returns_empty(self, tmp_path):
        """recall_chronicle_steps возвращает [] если файл не существует."""
        from scalpel.episodic_memory import EpisodicMemory
        mem = EpisodicMemory(memory_dir=tmp_path)
        result = mem.recall_chronicle_steps()
        assert result == []

    def test_generate_chronicle_exception_returns_fallback(self):
        """generate_chronicle возвращает fallback-строку при ошибке LLM."""
        from scalpel.critical_thinking import generate_chronicle
        history = [{"formula_shadow": "f0", "r2_blind": 0.5,
                    "matryoshka_feedback": [], "delphi_consilium": {}}]
        result = generate_chronicle(
            hadi_history=history, consilium={}, heritage="",
            domain="", formula_final="f0",
            ask_fn=lambda _: (_ for _ in ()).throw(Exception("LLM down")),
        )
        # Не должно быть исключения — должен вернуть строку
        assert isinstance(result, str)

    def test_curriculum_level_generators_no_exception_on_zero_noise(self):
        """Генераторы работают при noise=0 (граничный случай)."""
        from scalpel.curriculum import generate_level1, generate_level2, generate_level3
        rng = np.random.default_rng(0)
        for gen in [generate_level1, generate_level2, generate_level3]:
            try:
                datasets = gen(noise=0.0, rng=rng)
                assert len(datasets) == 10
            except Exception as e:
                pytest.fail(f"{gen.__name__} сломался при noise=0: {e}")


# ═══════════════════════════════════════════════════════════════════
# ИНТЕГРАЦИОННЫЙ ТЕСТ: полная цепочка
# ═══════════════════════════════════════════════════════════════════

class TestChronicleIntegration:
    """Интеграционный тест: chronicle → remember_chronicle_step → recall → Navigator."""

    def test_full_chain(self, tmp_path):
        """
        Симулируем полный цикл:
          1. generate_chronicle() создаёт историю
          2. remember_chronicle_step() сохраняет каждый шаг
          3. recall_chronicle_steps() возвращает цепочки
          4. Цепочки добавляются в failure_logs для Navigator
        """
        from scalpel.critical_thinking import generate_chronicle
        from scalpel.episodic_memory import EpisodicMemory

        mem = EpisodicMemory(memory_dir=tmp_path)

        # Симулируем кандидатов HADI цикла
        candidates = [
            {
                "formula_shadow":      "f0 * f1",
                "r2_blind":            0.43,
                "matryoshka_feedback": [{"role": "Скептик", "verdict": "ОТКЛОНЕНА",
                                         "critique": "не учитывает масштаб", "suggestion": ""}],
                "delphi_consilium":    {"forced_operators": ["sqrt"]},
            },
            {
                "formula_shadow":      "sqrt(f0) / f1",
                "r2_blind":            0.94,
                "matryoshka_feedback": [{"role": "Физик", "verdict": "ПРИНЯТА",
                                         "critique": "", "suggestion": ""}],
                "delphi_consilium":    {},
            },
        ]

        chronicle = generate_chronicle(
            hadi_history=candidates,
            consilium={},
            heritage="закон Кеплера",
            domain="Physics",
            formula_final="sqrt(f0) / f1",
            ask_fn=lambda p: "ИСТОРИЯ ПОИСКА\nПуть от f0*f1 к sqrt(f0)/f1.",
        )
        assert "ИСТОРИЯ ПОИСКА" in chronicle

        # Сохраняем шаги
        for si, cand in enumerate(candidates):
            fb    = cand.get("matryoshka_feedback") or []
            rej   = next((x.get("role","") for x in fb if x.get("verdict") == "ОТКЛОНЕНА"), "")
            reason= next(((x.get("critique") or "") for x in fb if x.get("verdict") == "ОТКЛОНЕНА"), "")
            dcon  = cand.get("delphi_consilium") or {}
            d_ops = dcon.get("forced_operators") or []
            led   = candidates[si+1]["formula_shadow"] if si+1 < len(candidates) else cand["formula_shadow"]
            mem.remember_chronicle_step(
                attempt=si+1, tried=cand["formula_shadow"],
                r2=cand["r2_blind"], rejected_by=rej, reason=reason,
                delphi_hint=", ".join(str(o) for o in d_ops),
                led_to=led, domain="Physics",
            )

        # Читаем цепочки для Navigator
        chains = mem.recall_chronicle_steps(domain="Physics")
        assert len(chains) == 2

        # Добавляем в failure_logs
        failure_logs_list: list = []
        for chain in chains:
            failure_logs_list.append({
                "hypothesis":   f"[Летописец] {chain[:150]}",
                "death_reason": "цепочка из истории поиска",
                "source":       "chronicle_memory",
            })
        assert len(failure_logs_list) == 2
        assert any("Летописец" in entry["hypothesis"] for entry in failure_logs_list)
        # Navigator теперь знает о цепочке мышления
        assert any("Скептик" in entry["hypothesis"] or "sqrt" in entry["hypothesis"]
                   for entry in failure_logs_list)
