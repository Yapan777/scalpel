"""
test_shadow.py — Тесты для ShadowMapper.
Запуск: pytest tests/test_shadow.py -v
"""
import pytest
from scalpel.shadow import ShadowMapper


# ── build() ───────────────────────────────────────────────────────────────────

def test_build_returns_shadow_names():
    """build() должен вернуть [f0, f1, f2, ...]"""
    mapper = ShadowMapper()
    result = mapper.build(["price", "volume", "ratio"])
    assert result == ["f0", "f1", "f2"]


def test_build_single_feature():
    mapper = ShadowMapper()
    result = mapper.build(["temperature"])
    assert result == ["f0"]


def test_build_sets_active():
    mapper = ShadowMapper()
    assert mapper.active is False
    mapper.build(["price", "volume"])
    assert mapper.active is True


def test_build_creates_fingerprint():
    mapper = ShadowMapper()
    mapper.build(["price", "volume"])
    assert len(mapper.fingerprint) == 16  # sha256[:16]


def test_build_different_inputs_different_fingerprints():
    m1, m2 = ShadowMapper(), ShadowMapper()
    m1.build(["price", "volume"])
    m2.build(["mass", "length"])
    assert m1.fingerprint != m2.fingerprint


# ── anonymize() ───────────────────────────────────────────────────────────────

def test_anonymize_known_names():
    mapper = ShadowMapper()
    mapper.build(["price", "volume", "ratio"])
    assert mapper.anonymize(["price", "ratio"]) == ["f0", "f2"]


def test_anonymize_unknown_name_passthrough():
    """Неизвестное имя не трогаем — возвращаем как есть."""
    mapper = ShadowMapper()
    mapper.build(["price"])
    assert mapper.anonymize(["unknown_feature"]) == ["unknown_feature"]


# ── restore() ─────────────────────────────────────────────────────────────────

def test_restore_simple_formula():
    mapper = ShadowMapper()
    mapper.build(["price", "volume"])
    restored = mapper.restore("f0 / f1")
    assert restored == "price / volume"


def test_restore_complex_formula():
    mapper = ShadowMapper()
    mapper.build(["price", "volume", "ratio"])
    restored = mapper.restore("sqrt(f0) * f2 + f1")
    assert restored == "sqrt(price) * ratio + volume"


def test_restore_no_collision_f1_vs_f10():
    """
    Критический тест: f1 не должен заменяться внутри f10.
    FIX-V5 сортирует по длине токена (длинные первыми).
    """
    features = [f"feat_{i}" for i in range(12)]  # feat_0 .. feat_11
    mapper = ShadowMapper()
    mapper.build(features)
    formula = "f10 + f1"
    restored = mapper.restore(formula)
    assert "feat_10" in restored
    assert "feat_1 " in restored or restored.endswith("feat_1")
    assert "feat_10" in restored  # f10 не стал feat_1_0


def test_restore_inactive_returns_original():
    """Если build() не вызывался — formule не трогаем."""
    mapper = ShadowMapper()
    formula = "f0 + f1"
    assert mapper.restore(formula) == formula


# ── reverse_mapping ───────────────────────────────────────────────────────────

def test_reverse_mapping_correct():
    mapper = ShadowMapper()
    mapper.build(["price", "volume"])
    rm = mapper.reverse_mapping
    assert rm == {"f0": "price", "f1": "volume"}


def test_reverse_mapping_is_copy():
    """Изменение reverse_mapping не должно ломать внутренний словарь."""
    mapper = ShadowMapper()
    mapper.build(["price", "volume"])
    rm = mapper.reverse_mapping
    rm["f0"] = "hacked"
    assert mapper.restore("f0") == "price"  # внутри не изменилось
