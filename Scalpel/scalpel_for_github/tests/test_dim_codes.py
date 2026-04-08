"""
test_dim_codes.py — Тесты для определения размерности признаков.
Запуск: pytest tests/test_dim_codes.py -v
"""
import pytest
from scalpel.dim_codes import dim_code


# ── Финансовые признаки → 10 ──────────────────────────────────────────────────

@pytest.mark.parametrize("name", ["price", "cost", "usd_value", "eur_rate", "revenue_total"])
def test_financial_features(name):
    assert dim_code(name) == 10


# ── Объём/количество → 5 ─────────────────────────────────────────────────────

@pytest.mark.parametrize("name", ["volume", "qty", "count_users", "num_orders", "amount"])
def test_volume_features(name):
    assert dim_code(name) == 5


# ── Безразмерные → 0 ─────────────────────────────────────────────────────────

@pytest.mark.parametrize("name", ["ratio", "rate", "pct_change", "percent", "share", "rel_val"])
def test_dimensionless_features(name):
    assert dim_code(name) == 0


# ── Время → 8 ────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("name", ["time", "date_col", "day_of_week", "hour", "month", "year", "age"])
def test_time_features(name):
    assert dim_code(name) == 8


# ── Масса → 3 ────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("name", ["mass", "weight", "kg_total", "gram_value"])
def test_mass_features(name):
    assert dim_code(name) == 3


# ── Длина → 2 ────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("name", ["length", "dist", "meter_val", "km_traveled", "radius", "area"])
def test_length_features(name):
    assert dim_code(name) == 2


# ── Температура → 4 ──────────────────────────────────────────────────────────

@pytest.mark.parametrize("name", ["temp", "kelvin", "celsius", "heat_val"])
def test_temperature_features(name):
    assert dim_code(name) == 4


# ── Сила/энергия → 6 ─────────────────────────────────────────────────────────

@pytest.mark.parametrize("name", ["force", "newton", "pressure", "energy", "joule_val"])
def test_force_features(name):
    assert dim_code(name) == 6


# ── Неизвестное → 1 ──────────────────────────────────────────────────────────

@pytest.mark.parametrize("name", ["xyz_feature", "col_a", "magic_col", "feature_99"])
def test_unknown_features(name):
    assert dim_code(name) == 1


# ── Регистронезависимость ─────────────────────────────────────────────────────

def test_case_insensitive():
    assert dim_code("PRICE")       == dim_code("price")
    assert dim_code("Temperature") == dim_code("temperature")
    assert dim_code("VOLUME")      == dim_code("volume")
