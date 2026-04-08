"""
test_navigator.py — Тесты для парсера ответа Navigator (без Ollama).
Тестируем только _parse_nav() — логику, а не сеть.
Запуск: pytest tests/test_navigator.py -v
"""
import json
import pytest
from scalpel.navigator import _parse_nav, _FALLBACK_HYPOTHESES, _DEFAULT_OPS


SHADOW_NAMES = ["f0", "f1", "f2", "f3"]


# ── Нормальный JSON ───────────────────────────────────────────────────────────

def test_parse_valid_json():
    raw = json.dumps({
        "selected_features":  ["f0", "f2"],
        "selected_operators": ["+", "-", "*", "/"],
        "hypotheses":         ["f0 / f2", "sqrt(f0)"],
        "ooda_stable":        True,
        "reasoning":          "тест",
    })
    dec = _parse_nav(raw, SHADOW_NAMES)
    assert dec.selected_features  == ["f0", "f2"]
    assert dec.selected_operators == ["+", "-", "*", "/"]
    assert "f0 / f2" in dec.hypotheses
    assert dec.ooda_stable is True


def test_parse_reasoning_truncated():
    """reasoning обрезается до 120 символов."""
    long_reason = "x" * 200
    raw = json.dumps({
        "selected_features":  ["f0", "f1"],
        "selected_operators": ["+"],
        "hypotheses":         ["f0 + f1"],
        "reasoning":          long_reason,
    })
    dec = _parse_nav(raw, SHADOW_NAMES)
    assert len(dec.reasoning) <= 120


# ── Fallback на ошибках ───────────────────────────────────────────────────────

def test_parse_empty_string_fallback():
    dec = _parse_nav("", SHADOW_NAMES)
    assert dec.hypotheses == _FALLBACK_HYPOTHESES
    assert dec.selected_operators == _DEFAULT_OPS


def test_parse_ollama_error_fallback():
    dec = _parse_nav("[OLLAMA_ERROR] URLError: Connection refused", SHADOW_NAMES)
    assert dec.hypotheses == _FALLBACK_HYPOTHESES


def test_parse_no_json_fallback():
    dec = _parse_nav("Вот мой ответ без JSON!", SHADOW_NAMES)
    assert dec.hypotheses == _FALLBACK_HYPOTHESES


def test_parse_broken_json_fallback():
    dec = _parse_nav('{"selected_features": ["f0"', SHADOW_NAMES)
    assert dec.hypotheses == _FALLBACK_HYPOTHESES


# ── JSON в markdown-обёртке ───────────────────────────────────────────────────

def test_parse_json_in_markdown():
    """Ollama иногда оборачивает ответ в ```json ... ```"""
    inner = json.dumps({
        "selected_features":  ["f0", "f1"],
        "selected_operators": ["*", "/"],
        "hypotheses":         ["f0 * f1"],
        "ooda_stable":        False,
    })
    raw = f"```json\n{inner}\n```"
    dec = _parse_nav(raw, SHADOW_NAMES)
    assert "f0 * f1" in dec.hypotheses
    assert dec.ooda_stable is False


# ── Валидация признаков ───────────────────────────────────────────────────────

def test_parse_filters_unknown_features():
    """Признаки не из shadow_names отфильтровываются."""
    raw = json.dumps({
        "selected_features":  ["f0", "f99"],  # f99 не существует
        "selected_operators": ["+"],
        "hypotheses":         ["f0 + f1"],
    })
    dec = _parse_nav(raw, SHADOW_NAMES)
    assert "f99" not in dec.selected_features


def test_parse_too_few_features_uses_all():
    """Если отобрано < 2 признаков — используем все."""
    raw = json.dumps({
        "selected_features":  ["f0"],  # только один
        "selected_operators": ["+"],
        "hypotheses":         ["f0 + f1"],
    })
    dec = _parse_nav(raw, SHADOW_NAMES)
    assert dec.selected_features == list(SHADOW_NAMES)


# ── Валидация операторов ──────────────────────────────────────────────────────

def test_parse_filters_unknown_operators():
    """Незнакомые операторы отфильтровываются."""
    raw = json.dumps({
        "selected_features":  ["f0", "f1"],
        "selected_operators": ["+", "unknown_op", "xor"],
        "hypotheses":         ["f0 + f1"],
    })
    dec = _parse_nav(raw, SHADOW_NAMES)
    assert "unknown_op" not in dec.selected_operators
    assert "xor" not in dec.selected_operators
    assert "+" in dec.selected_operators


def test_parse_empty_operators_uses_default():
    raw = json.dumps({
        "selected_features":  ["f0", "f1"],
        "selected_operators": [],
        "hypotheses":         ["f0 + f1"],
    })
    dec = _parse_nav(raw, SHADOW_NAMES)
    assert dec.selected_operators == _DEFAULT_OPS


# ── Лимит гипотез ─────────────────────────────────────────────────────────────

def test_parse_hypotheses_limited_to_5():
    """Даже если Ollama вернула 10 гипотез — берём только топ-5."""
    raw = json.dumps({
        "selected_features":  ["f0", "f1"],
        "selected_operators": ["+"],
        "hypotheses":         [f"f0 + f{i}" for i in range(10)],
    })
    dec = _parse_nav(raw, SHADOW_NAMES)
    assert len(dec.hypotheses) <= 5
