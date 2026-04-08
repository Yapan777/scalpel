"""
run_test.py — Честный тест системы

Генерирует зашумленные данные (формула скрыта).
Система видит только числа — никаких подсказок.
Если найдёт правильную формулу — система работает.

Запуск: python run_test.py
"""
import sys, os, time
sys.path.insert(0, os.getcwd())

import numpy as np
import pandas as pd

# ── Проверка Ollama ПЕРЕД запуском ───────────────────────────────────────────
def _check_ollama(host: str = "http://localhost:11434", wait: bool = True) -> bool:
    """Проверяет доступность Ollama. Если недоступна — объясняет что делать."""
    import urllib.request
    for attempt in range(1, 4):
        try:
            urllib.request.urlopen(f"{host}/api/tags", timeout=3).read()
            if attempt > 1:
                print("  ✓ Ollama доступна!")
            return True
        except Exception:
            if attempt == 1:
                print("\n" + "=" * 60)
                print("  ⚠️  OLLAMA НЕ ЗАПУЩЕНА")
                print("=" * 60)
                print("  DSPy и LLM-роли работают через Ollama.")
                print("  Без неё система перейдёт в Legacy-режим")
                print("  (роли отвечают УСЛОВНО, DSPy не оптимизирован).\n")
                print("  Чтобы включить полный режим:")
                print("  1. Открой НОВОЕ окно терминала")
                print("  2. Запусти: ollama serve")
                print("  3. Дождись 'Listening on 127.0.0.1:11434'\n")
                if not wait:
                    print("  Продолжаем в Legacy-режиме...\n")
                    return False
                print("  Ожидаю Ollama (30 сек)... нажми Ctrl+C чтобы продолжить без неё")
            if wait and attempt <= 3:
                time.sleep(10)
    print("  Ollama не запустилась — продолжаем в Legacy-режиме.\n")
    return False

_ollama_ok = _check_ollama(wait=True)
if _ollama_ok:
    print("  ✓ Ollama доступна — DSPy активен, LLM работают в полном режиме\n")

# Генерируем данные (формула намеренно скрыта в комментарии)
rng = np.random.default_rng(42)
x_train = np.linspace(0.5, 4.0, 150)
x_test  = np.linspace(0.5, 4.0, 50)

# Целевые значения с 10% шумом
y_clean_tr = x_train ** 1.5
y_clean_te = x_test  ** 1.5
y_train = y_clean_tr + rng.normal(0, 0.10 * np.std(y_clean_tr), len(x_train))
y_test  = y_clean_te + rng.normal(0, 0.10 * np.std(y_clean_te), len(x_test))

# Сохраняем — нейтральные имена, только числа
pd.DataFrame({"x": x_train, "y": y_train}).to_csv("train.csv", index=False)
pd.DataFrame({"x": x_test,  "y": y_test}).to_csv("test.csv",   index=False)

print("Данные сгенерированы: train.csv, test.csv")
print(f"Шум: 10% | {len(x_train)} train, {len(x_test)} test строк")
print("Запускаем Scalpel...\n")

from scalpel.engine import run_engine, run_llm_phase

X_train = x_train.reshape(-1,1).astype("float64")   # FIX БАГ 3: float64 — единый тип с data.py и engine.py
X_test  = x_test.reshape(-1,1).astype("float64")

print("=" * 60)
result = run_engine(
    X_train     = X_train,
    y_train     = y_train.astype("float64"),   # FIX БАГ 3: float64
    X_test      = X_test,
    y_test      = y_test.astype("float64"),    # FIX БАГ 3: float64
    feat_names  = ["f0"],
    target_col  = "y",
    domain_type = "Physics",
    phase       = "pysr",
    dim_codes   = [0],
    noise_hint  = 0.10,
    skip_heritage = False,
)

print("=" * 60)
print(f"PySR результат: {result.formula_real}")
print(f"R²_blind: {result.r2_blind:.4f}")
print("\nЗапускаем LLM верификацию (Матрёшка)...")
print("=" * 60)

result_llm = run_llm_phase()
if result_llm and result_llm.formula_real:
    print(f"Финальная формула: {result_llm.formula_real}")
    print(f"Consensus: {result_llm.consensus}")
    r2 = result_llm.r2_blind
    if r2 > 0.90:
        print("\n✅ ОТЛИЧНО! Система работает правильно")
    elif r2 > 0.75:
        print("\n✅ ХОРОШО! Формула найдена")
    else:
        print("\n⚠️ Формула слабая — попробуйте увеличить PYSR_FAST_FAIL_SEC в config.py")

# v10.15: показываем Layer 2 если был найден
try:
    import json
    from pathlib import Path
    from scalpel.config import RESIDUAL_RESULT_PATH
    if RESIDUAL_RESULT_PATH.exists():
        rd = json.loads(RESIDUAL_RESULT_PATH.read_text(encoding="utf-8"))
        layers = rd.get("layers", [])
        if layers:
            last = layers[-1]
            print("\n" + "=" * 60)
            if last.get("layer2_formula") and last.get("layer2_consensus") == "ПРИНЯТА":
                print("🔬 НАЙДЕН ВТОРОЙ ФИЗИЧЕСКИЙ СЛОЙ!")
                print(f"   Layer 1: {last['layer1_formula']}  R²={last['layer1_r2']:.4f}")
                print(f"   Layer 2: {last['layer2_formula']}  R²={last['layer2_r2_blind']:.4f}")
                print(f"   Итоговая: y = {last['combined_formula']}")
                print(f"   R²_combined = {last['combined_r2']:.4f}")
            elif last.get("ran"):
                print(f"🔬 ResidualScan: остатки — шум "
                      f"(R²_linear={last.get('residual_r2',0):.3f}). "
                      f"Данные объясняются одним законом.")
except Exception:
    pass

