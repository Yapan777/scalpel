"""
test_audit.py — Тесты для парсинга вердиктов Матрёшки.
Тестируем только логику консенсуса — без реальной Ollama.
Запуск: pytest tests/test_audit.py -v

FIX v10.14:
  - matryoshka_audit возвращает 4 значения: (consensus, report, role_results, consilium)
  - Все тесты распаковывают 4 значения вместо 2
  - ROLE_COMPILED_DIR / ROLE_FAILURE_DIR изолированы через tmp_path
  - ollama_chat мокируется на всех трёх путях:
      scalpel.audit.ollama_chat      — основная функция (RAM Queue)
      scalpel.audit._ask_syn         — SYNTHESIS_MODEL (Dialectic, Delphi)
      scalpel.audit._ask_chr         — CHRONICLE_MODEL (Sinquain)
    Проще мокировать через navigator.ollama_chat — единый источник
"""
import pytest
from unittest.mock import patch, MagicMock
from scalpel.audit import matryoshka_audit, MATRYOSHKA_ROLES


def _audit_patches(tmp_path, mock_chat_fn):
    """
    Полная изоляция: мокируем ollama_chat везде, изолируем все пути.
    Патчим navigator.ollama_chat — единая точка для всех вызовов LLM в проекте.
    """
    return [
        # Основной chat — для RAM Queue ролей
        patch("scalpel.audit.ollama_chat",               side_effect=mock_chat_fn),
        # navigator.ollama_chat — для _ask_syn/_ask_chr (Dialectic, Delphi, Sinquain)
        patch("scalpel.navigator.ollama_chat",           side_effect=mock_chat_fn),
        # Пути
        patch("scalpel.audit.SCRIPT_DIR",                tmp_path),
        patch("scalpel.ram_queue.ROLE_COMPILED_DIR",     tmp_path / "models"),
        patch("scalpel.ram_queue.ROLE_FAILURE_DIR",      tmp_path / "failures"),
        # Остановка Ollama — не нужна в тестах
        patch("scalpel.audit.ollama_stop",               MagicMock(), create=True),
        patch("scalpel.engine.ollama_stop",              MagicMock()),
    ]


def test_consensus_accepted_when_3_plus_accepted(tmp_path):
    """3+ ПРИНЯТА → консенсус ПРИНЯТА."""
    responses = [
        "Формула корректна. ПРИНЯТА",
        "Размерности верны. ПРИНЯТА",
        "Применима на практике. ПРИНЯТА",
        "Напоминает закон Кеплера. УСЛОВНО",
    ]
    idx = 0
    def mock_chat(*args, **kwargs):
        nonlocal idx
        r = responses[idx % len(responses)]
        idx += 1
        return r

    p = _audit_patches(tmp_path, mock_chat)
    with p[0], p[1], p[2], p[3], p[4], p[5], p[6]:
        consensus, report, role_results, consilium = matryoshka_audit(
            formula_shadow="f0 / f1",
            shadow_names=["f0", "f1"],
            r2_train=0.95,
            r2_blind=0.92,
            complexity=3,
        )
    assert consensus == "ПРИНЯТА"
    assert "ПРИНЯТА" in report


def test_consensus_rejected_when_3_plus_rejected(tmp_path):
    """3+ ОТКЛОНЕНА → консенсус ОТКЛОНЕНА."""
    responses = [
        "Формула неверна. ОТКЛОНЕНА",
        "Нарушение размерностей. ОТКЛОНЕНА",
        "Не применима. ОТКЛОНЕНА",
        "Аналогий нет. УСЛОВНО",
    ]
    idx = 0
    def mock_chat(*args, **kwargs):
        nonlocal idx
        r = responses[idx % len(responses)]
        idx += 1
        return r

    p = _audit_patches(tmp_path, mock_chat)
    with p[0], p[1], p[2], p[3], p[4], p[5], p[6]:
        consensus, _, _, _ = matryoshka_audit(
            formula_shadow="f0 + f1",
            shadow_names=["f0", "f1"],
            r2_train=0.60,
            r2_blind=0.55,
            complexity=2,
        )
    assert consensus == "ОТКЛОНЕНА"


def test_consensus_disputed_when_mixed(tmp_path):
    """2 ПРИНЯТА + 1 ОТКЛОНЕНА + 1 УСЛОВНО → СПОРНО."""
    responses = [
        "Хорошо. ПРИНЯТА",
        "Размерности OK. ПРИНЯТА",
        "Проблемы есть. ОТКЛОНЕНА",
        "Структура интересная. УСЛОВНО",
    ]
    idx = 0
    def mock_chat(*args, **kwargs):
        nonlocal idx
        r = responses[idx % len(responses)]
        idx += 1
        return r

    p = _audit_patches(tmp_path, mock_chat)
    with p[0], p[1], p[2], p[3], p[4], p[5], p[6]:
        consensus, _, _, _ = matryoshka_audit(
            formula_shadow="f0 * f1",
            shadow_names=["f0", "f1"],
            r2_train=0.80,
            r2_blind=0.77,
            complexity=2,
        )
    assert consensus == "СПОРНО"


def test_ollama_error_treated_as_conditional(tmp_path):
    """При ошибке Ollama → все УСЛОВНО → консенсус СПОРНО (не падаем)."""
    def mock_chat(*args, **kwargs):
        return "[OLLAMA_ERROR] Connection refused"

    p = _audit_patches(tmp_path, mock_chat)
    with p[0], p[1], p[2], p[3], p[4], p[5], p[6]:
        consensus, report, _, _ = matryoshka_audit(
            formula_shadow="f0",
            shadow_names=["f0"],
            r2_train=0.50,
            r2_blind=0.45,
            complexity=1,
        )
    assert consensus == "СПОРНО"
    assert "LLM недоступна" in report


def test_report_saved_to_file(tmp_path):
    """Отчёт должен сохраняться в CONSENSUS_REPORT.txt."""
    def mock_chat(*args, **kwargs):
        return "Всё OK. ПРИНЯТА"

    p = _audit_patches(tmp_path, mock_chat)
    with p[0], p[1], p[2], p[3], p[4], p[5], p[6]:
        matryoshka_audit(
            formula_shadow="f0 / f1",
            shadow_names=["f0", "f1"],
            r2_train=0.95,
            r2_blind=0.92,
            complexity=3,
        )
    report_file = tmp_path / "scalpel_vault" / "CONSENSUS_REPORT.txt"
    assert report_file.exists()
    content = report_file.read_text(encoding="utf-8")
    assert "f0 / f1" in content


def test_returns_exactly_4_values(tmp_path):
    """matryoshka_audit обязана возвращать ровно 4 значения."""
    def mock_chat(*args, **kwargs):
        return "ПРИНЯТА"

    p = _audit_patches(tmp_path, mock_chat)
    with p[0], p[1], p[2], p[3], p[4], p[5], p[6]:
        result = matryoshka_audit(
            formula_shadow="f0 / f1",
            shadow_names=["f0", "f1"],
            r2_train=0.95,
            r2_blind=0.92,
            complexity=3,
        )
    assert len(result) == 4, \
        f"Ожидалось 4 значения, получено {len(result)}: {[type(x).__name__ for x in result]}"
    consensus, report, role_results, consilium = result
    assert isinstance(consilium, dict)
    assert isinstance(role_results, list)


def test_r2_blind_passed_correctly(tmp_path):
    """r2_blind должен передаваться отдельно от r2_train (был баг r2_blind=r2_train)."""
    captured = {}
    def mock_chat(*args, **kwargs):
        return "ПРИНЯТА"

    # Патчим episodic_memory чтобы поймать что туда записывается
    mock_memory = MagicMock()
    p = _audit_patches(tmp_path, mock_chat)
    with p[0], p[1], p[2], p[3], p[4], p[5], p[6], \
         patch("scalpel.ram_queue.get_memory", return_value=mock_memory):
        matryoshka_audit(
            formula_shadow="f0 / f1",
            shadow_names=["f0", "f1"],
            r2_train=0.95,
            r2_blind=0.88,   # специально другое значение
            complexity=3,
        )
    # Проверяем что remember() вызывался с r2_blind=0.88, не с r2_train=0.95
    if mock_memory.remember.called:
        call_kwargs = mock_memory.remember.call_args_list[0][1]
        assert call_kwargs.get("r2_blind") != call_kwargs.get("r2_train"), \
            "r2_blind должен отличаться от r2_train — был баг когда они всегда равны"
        assert call_kwargs.get("r2_blind") == pytest.approx(0.88, abs=0.01), \
            f"r2_blind должен быть 0.88, получено {call_kwargs.get('r2_blind')}"


# ── Структурные тесты ──────────────────────────────────────────────

def test_exactly_4_roles():
    assert len(MATRYOSHKA_ROLES) == 4


def test_role_names():
    names = [r[0] for r in MATRYOSHKA_ROLES]
    assert "Скептик"   in names
    assert "Физик"     in names
    assert "Прагматик" in names
    assert "Мистик"    in names
