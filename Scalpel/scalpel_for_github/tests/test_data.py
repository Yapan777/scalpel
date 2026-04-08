"""
test_data.py — Тесты для загрузки данных.
Запуск: pytest tests/test_data.py -v
"""
import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from scalpel.data import load_csv


# ── Фикстуры ─────────────────────────────────────────────────────────────────

@pytest.fixture
def single_csv(tmp_path) -> Path:
    """Один CSV с 100 строками."""
    df = pd.DataFrame({
        "price":  np.random.rand(100) * 100,
        "volume": np.random.rand(100) * 1000,
        "target": np.random.rand(100),
    })
    p = tmp_path / "data.csv"
    df.to_csv(p, index=False)
    return p


@pytest.fixture
def train_test_csvs(tmp_path):
    """Два CSV: train + test."""
    df_train = pd.DataFrame({
        "price":  np.random.rand(80) * 100,
        "volume": np.random.rand(80) * 1000,
        "target": np.random.rand(80),
    })
    df_test = pd.DataFrame({
        "price":  np.random.rand(20) * 100,
        "volume": np.random.rand(20) * 1000,
        "target": np.random.rand(20),
    })
    train_p = tmp_path / "train.csv"
    test_p  = tmp_path / "test.csv"
    df_train.to_csv(train_p, index=False)
    df_test.to_csv(test_p, index=False)
    return train_p, test_p


# ── Один CSV ──────────────────────────────────────────────────────────────────

def test_single_csv_loads(single_csv):
    X_tr, y_tr, X_te, y_te, feat_names, target_col = load_csv(
        train_path=str(single_csv),
        test_path="nonexistent.csv",
    )
    assert X_tr.shape[1] == 2      # price + volume
    assert target_col == "target"


def test_single_csv_split_ratio(single_csv):
    X_tr, y_tr, X_te, y_te, feat_names, target_col = load_csv(
        train_path=str(single_csv),
        test_path="nonexistent.csv",
        test_size=0.2,
    )
    total = len(y_tr) + len(y_te)
    assert total == 100
    assert len(y_te) == 20   # 20% от 100


def test_single_csv_feature_names(single_csv):
    _, _, _, _, feat_names, _ = load_csv(
        train_path=str(single_csv),
        test_path="nonexistent.csv",
    )
    assert "price"  in feat_names
    assert "volume" in feat_names
    assert "target" not in feat_names


def test_single_csv_explicit_target(single_csv):
    _, _, _, _, feat_names, target_col = load_csv(
        train_path=str(single_csv),
        test_path="nonexistent.csv",
        target_col="price",
    )
    assert target_col == "price"
    assert "price" not in feat_names
    assert "volume" in feat_names


def test_single_csv_auto_target_is_last_column(single_csv):
    """Если target не указан — берём последний столбец."""
    _, _, _, _, _, target_col = load_csv(
        train_path=str(single_csv),
        test_path="nonexistent.csv",
    )
    assert target_col == "target"


# ── Train + Test CSV ──────────────────────────────────────────────────────────

def test_two_csvs_correct_sizes(train_test_csvs):
    train_p, test_p = train_test_csvs
    X_tr, y_tr, X_te, y_te, _, _ = load_csv(
        train_path=str(train_p),
        test_path=str(test_p),
    )
    assert len(y_tr) == 80
    assert len(y_te) == 20


def test_two_csvs_float64(train_test_csvs):
    train_p, test_p = train_test_csvs
    X_tr, y_tr, X_te, y_te, _, _ = load_csv(
        train_path=str(train_p),
        test_path=str(test_p),
    )
    assert X_tr.dtype == np.float64
    assert y_tr.dtype == np.float64


# ── Ошибки ────────────────────────────────────────────────────────────────────

def test_missing_files_raises():
    with pytest.raises(FileNotFoundError):
        load_csv(
            train_path="nonexistent_train.csv",
            test_path="nonexistent_test.csv",
        )
