"""
test_vault.py — Тесты для GoldVault._skeleton() и _atomic_append().
Тестируем чистую логику без реальных файлов (tmp_path фикстура pytest).
Запуск: pytest tests/test_vault.py -v
"""
import json
import pytest
from pathlib import Path
from scalpel.vault import GoldVault


# ── _skeleton() ───────────────────────────────────────────────────────────────

def test_skeleton_variables_become_v():
    result = GoldVault._skeleton("f0 / f1")
    assert result == "v / v"


def test_skeleton_constants_become_c():
    result = GoldVault._skeleton("f0 + 3.14")
    assert result == "v + c"


def test_skeleton_operators_preserved():
    result = GoldVault._skeleton("sqrt(f0) * f1")
    assert "sqrt" in result
    assert "*" in result


def test_skeleton_complex_formula():
    result = GoldVault._skeleton("f0 * f0 / f1 + 2.5")
    assert "v" in result
    assert "c" in result


def test_skeleton_empty_formula():
    result = GoldVault._skeleton("")
    assert result == ""


# ── _atomic_append() + _atomic_write() ────────────────────────────────────────

def test_atomic_append_creates_file(tmp_path):
    path = tmp_path / "test_gold.json"
    record = {"id": "abc123", "formula": "f0 / f1", "r2_train": 0.95}
    GoldVault._atomic_append(path, record)
    assert path.exists()


def test_atomic_append_correct_structure(tmp_path):
    path = tmp_path / "test_gold.json"
    record = {"id": "abc123", "formula": "f0 / f1"}
    GoldVault._atomic_append(path, record)
    data = json.loads(path.read_text(encoding="utf-8"))
    assert "formulas" in data
    assert data["count"] == 1
    assert data["formulas"][0]["formula"] == "f0 / f1"


def test_atomic_append_accumulates(tmp_path):
    """Два вызова → два record в списке."""
    path = tmp_path / "test_gold.json"
    GoldVault._atomic_append(path, {"id": "1", "formula": "f0"})
    GoldVault._atomic_append(path, {"id": "2", "formula": "f1"})
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["count"] == 2
    assert len(data["formulas"]) == 2


def test_atomic_append_no_data_loss_on_corrupt(tmp_path):
    """Если файл повреждён — начинаем заново, не падаем."""
    path = tmp_path / "test_gold.json"
    path.write_text("CORRUPTED DATA {{{", encoding="utf-8")
    record = {"id": "abc", "formula": "f0"}
    GoldVault._atomic_append(path, record)  # не должно бросать исключение
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["count"] == 1


def test_atomic_write_uses_tmp_then_replace(tmp_path):
    """После записи .tmp файл не должен оставаться."""
    path = tmp_path / "gold.json"
    data = {"formulas": [], "count": 0}
    GoldVault._atomic_write(path, data)
    tmp = path.with_suffix(".tmp")
    assert not tmp.exists()
    assert path.exists()
