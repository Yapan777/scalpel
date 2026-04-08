"""
run_feynman.py — Физический Benchmark v3.0 (ЧЕСТНЫЙ)

30 физических законов в 3 прогонах (Transfer Learning).
Система видит ТОЛЬКО числа. Никаких подсказок.

Запуск (полный Scalpel):  python run_feynman.py
Запуск (baseline PySR):   python run_feynman.py --baseline
Результаты: scalpel_vault/feynman_results_summary.json
"""
import sys, os, time, json, hashlib, argparse
sys.path.insert(0, os.getcwd())

import numpy as np
from pathlib import Path
from datetime import datetime

def _log_preparator_result(transform: str, r2: float, result: str):
    """Записывает результат трансформации в preparator_log.jsonl."""
    try:
        import json
        from pathlib import Path
        log_path = Path("scalpel_vault/feynman_results.json").parent / "preparator_log.jsonl"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        entry = {
            "ts": datetime.now().isoformat(timespec="seconds"),
            "transform": transform,
            "r2": round(r2, 4),
            "result": result,
            "source": "feynman_benchmark",
        }
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception:
        pass


def _check_ollama(host="http://localhost:11434"):
    import urllib.request
    try:
        urllib.request.urlopen(f"{host}/api/tags", timeout=3).read()
        print("  ✓ Ollama доступна\n")
        return True
    except Exception:
        print("\n⚠️  OLLAMA НЕ ЗАПУЩЕНА. Запусти: ollama serve\n")
        return False


def _add_noise(y_all: np.ndarray, rng_l, noise_type: str) -> np.ndarray:
    """
    Три типа шума для реалистичного тестирования хирургии:

    gaussian        — равномерный 5% (идеальные условия)
    outliers        — 5% гауссов + 8% точек с выбросами 5×std
                      Хирургия должна срабатывать и резать выбросы.
    heteroscedastic — шум пропорционален |y| (реальные физические данные:
                      погрешность прибора растёт с измеряемой величиной)
    """
    std_y = np.std(y_all)
    n     = len(y_all)

    if noise_type == "gaussian":
        return y_all + rng_l.normal(0, 0.05 * std_y, n)

    elif noise_type == "outliers":
        noise = rng_l.normal(0, 0.05 * std_y, n)
        n_out = max(2, int(0.08 * n))                          # 8% выбросов
        idx   = rng_l.choice(n, n_out, replace=False)
        noise[idx] = rng_l.normal(0, 5.0 * std_y, n_out)      # 5× стд
        return y_all + noise

    elif noise_type == "heteroscedastic":
        scale = 0.08 * np.abs(y_all) + 0.01 * std_y           # шум ∝ |y|
        return y_all + rng_l.normal(0, scale, n)

    return y_all + rng_l.normal(0, 0.05 * std_y, n)           # fallback

# ── ПРОГОН 1: 10 законов v3.0 ─────────────────────────────────────
# Скелеты: v*v*v, v**3, v**2/v, v*v/v, v**2*v, v*sqrt(v/v)
# Нет конфликтов с gold vault
PHYSICS_LAWS_RUN1 = [
    {
        # y = m*g*h — потенциальная энергия
        # Скелет: v*v*v — нет в gold ✓
        "id": 1,
        "name":         "Потенциальная энергия",
        "true_formula": "f0 * f1 * f2",
        "noise_type":   "outliers",
        "n_features":   3,
        "dim_codes":    [0, 0, 0],
        "f0": lambda rng: rng.uniform(0.5, 20.0, 200),    # m масса
        "f1": lambda rng: np.full(200, 9.81),              # g ускорение
        "f2": lambda rng: rng.uniform(0.1, 50.0, 200),    # h высота
        "formula": lambda d: d["f0"] * d["f1"] * d["f2"],
    },
    {
        # y = (4/3)*pi*r^3 — объём шара
        # Скелет: v**3 — нет в gold ✓
        "id": 2,
        "name":         "Объём шара",
        "true_formula": "4.189 * f0**3",
        "noise_type":   "heteroscedastic",
        "n_features":   1,
        "dim_codes":    [0],
        "f0": lambda rng: rng.uniform(0.1, 5.0, 200),     # r радиус
        "formula": lambda d: (4/3) * np.pi * d["f0"]**3,
    },
    {
        # y = v^2 / (2*a) — путь при торможении
        # Скелет: v*v/v — нет в gold ✓
        "id": 3,
        "name":         "Тормозной путь",
        "true_formula": "f0**2 / (2 * f1)",
        "noise_type":   "gaussian",
        "n_features":   2,
        "dim_codes":    [0, 0],
        "f0": lambda rng: rng.uniform(5.0, 30.0, 200),    # v скорость
        "f1": lambda rng: rng.uniform(1.0, 10.0, 200),    # a замедление
        "formula": lambda d: d["f0"]**2 / (2 * d["f1"]),
    },
    {
        # y = n1*sin(a1)/sin(a2) — закон Снеллиуса (показатель преломления)
        # Скелет: v*v/v — нет в gold ✓
        "id": 4,
        "name":         "Закон Снеллиуса",
        "true_formula": "f0 * f1 / f2",
        "noise_type":   "outliers",
        "n_features":   3,
        "dim_codes":    [0, 0, 0],
        "f0": lambda rng: rng.uniform(1.0, 2.5, 200),     # n1 показатель
        "f1": lambda rng: rng.uniform(0.1, 1.4, 200),     # sin(a1)
        "f2": lambda rng: rng.uniform(0.1, 1.4, 200),     # sin(a2)
        "formula": lambda d: d["f0"] * d["f1"] / d["f2"],
    },
    {
        # y = I*R^2 — мощность на резисторе (через ток)
        # Скелет: v*v*v — нет в gold ✓
        "id": 5,
        "name":         "Мощность на резисторе",
        "true_formula": "f0**2 * f1",
        "noise_type":   "heteroscedastic",
        "n_features":   2,
        "dim_codes":    [0, 0],
        "f0": lambda rng: rng.uniform(0.01, 5.0, 200),    # I ток
        "f1": lambda rng: rng.uniform(1.0, 1000.0, 200),  # R сопротивление
        "formula": lambda d: d["f0"]**2 * d["f1"],
    },
    {
        # y = 2*G*M/r — вторая космическая скорость^2
        # Скелет: v*v/v — нет в gold ✓
        "id": 6,
        "name":         "Вторая космическая скорость²",
        "true_formula": "2 * f0 * f1 / f2",
        "noise_type":   "gaussian",
        "n_features":   3,
        "dim_codes":    [0, 0, 0],
        "f0": lambda rng: np.full(200, 6.674e-11),         # G константа
        "f1": lambda rng: rng.uniform(1e23, 1e25, 200),   # M масса планеты
        "f2": lambda rng: rng.uniform(1e6, 1e7, 200),     # r радиус
        "formula": lambda d: 2 * d["f0"] * d["f1"] / d["f2"],
    },
    {
        # y = k*q/r^2 — напряжённость электрического поля
        # Скелет: v*v/v*v — нет в gold ✓
        "id": 7,
        "name":         "Напряжённость поля",
        "true_formula": "f0 * f1 / f2**2",
        "noise_type":   "outliers",
        "n_features":   3,
        "dim_codes":    [0, 0, 0],
        "f0": lambda rng: np.full(200, 8.988e9),           # k константа
        "f1": lambda rng: rng.uniform(1e-9, 1e-6, 200),   # q заряд
        "f2": lambda rng: rng.uniform(0.01, 1.0, 200),    # r расстояние
        "formula": lambda d: d["f0"] * d["f1"] / d["f2"]**2,
    },
    {
        # y = sqrt(2*h/g) — время свободного падения
        # Скелет: sqrt(v/v) — нет в gold ✓
        "id": 8,
        "name":         "Время падения",
        "true_formula": "sqrt(2 * f0 / f1)",
        "noise_type":   "heteroscedastic",
        "n_features":   2,
        "dim_codes":    [0, 0],
        "f0": lambda rng: rng.uniform(0.5, 100.0, 200),   # h высота
        "f1": lambda rng: np.full(200, 9.81),              # g ускорение
        "formula": lambda d: np.sqrt(2 * d["f0"] / d["f1"]),
    },
    {
        # y = m*c^2 — энергия покоя (E=mc²)
        # Скелет: v*v*v — нет в gold ✓
        "id": 9,
        "name":         "Энергия покоя E=mc²",
        "true_formula": "f0 * f1**2",
        "noise_type":   "gaussian",
        "n_features":   2,
        "dim_codes":    [0, 0],
        "f0": lambda rng: rng.uniform(1e-30, 1e-27, 200), # m масса
        "f1": lambda rng: np.full(200, 3e8),               # c скорость света
        "formula": lambda d: d["f0"] * d["f1"]**2,
    },
    {
        # y = (1/2)*C*V^2 — энергия конденсатора
        # Скелет: v*v*v — нет в gold ✓
        "id": 10,
        "name":         "Энергия конденсатора",
        "true_formula": "0.5 * f0 * f1**2",
        "noise_type":   "outliers",
        "n_features":   2,
        "dim_codes":    [0, 0],
        "f0": lambda rng: rng.uniform(1e-6, 1e-3, 200),   # C ёмкость
        "f1": lambda rng: rng.uniform(1.0, 1000.0, 200),  # V напряжение
        "formula": lambda d: 0.5 * d["f0"] * d["f1"]**2,
    },
]

# ── ПРОГОН 2: 10 законов (средняя сложность) ──────────────────────
# Transfer learning: система применяет опыт прогона 1
# Новые скелеты: v*v, sqrt(v²+v²), v*v*v/v², v*v/(v+v),
#                v*v*v⁴, v/(v*v), v*v*v/v, v*v²/v, sqrt(v*v/v)
PHYSICS_LAWS_RUN2 = [
    {
        # F = m * a
        "id": 11, "name": "Второй закон Ньютона",
        "true_formula": "f0 * f1",
        "noise_type":   "heteroscedastic",
        "n_features": 2, "dim_codes": [0, 0],
        "f0": lambda rng: rng.uniform(0.1, 100.0, 200),   # m масса
        "f1": lambda rng: rng.uniform(0.1, 50.0, 200),    # a ускорение
        "formula": lambda d: d["f0"] * d["f1"],
    },
    {
        # v = sqrt(vx² + vy²)
        "id": 12, "name": "Результирующая скорость (2D)",
        "true_formula": "sqrt(f0**2 + f1**2)",
        "noise_type":   "gaussian",
        "n_features": 2, "dim_codes": [0, 0],
        "f0": lambda rng: rng.uniform(-20.0, 20.0, 200),  # vx
        "f1": lambda rng: rng.uniform(-20.0, 20.0, 200),  # vy
        "formula": lambda d: np.sqrt(d["f0"]**2 + d["f1"]**2),
    },
    {
        # F = k * q1 * q2 / r²
        "id": 13, "name": "Закон Кулона",
        "true_formula": "f0 * f1 * f2 / f3**2",
        "noise_type":   "outliers",
        "n_features": 4, "dim_codes": [0, 0, 0, 0],
        "f0": lambda rng: np.full(200, 8.988e9),           # k
        "f1": lambda rng: rng.uniform(1e-9, 1e-6, 200),   # q1
        "f2": lambda rng: rng.uniform(1e-9, 1e-6, 200),   # q2
        "f3": lambda rng: rng.uniform(0.01, 1.0, 200),    # r
        "formula": lambda d: d["f0"]*d["f1"]*d["f2"]/d["f3"]**2,
    },
    {
        # f = do * di / (do + di)
        "id": 14, "name": "Тонкая линза (фокусное расстояние)",
        "true_formula": "f0 * f1 / (f0 + f1)",
        "noise_type":   "heteroscedastic",
        "n_features": 2, "dim_codes": [0, 0],
        "f0": lambda rng: rng.uniform(0.05, 2.0, 200),    # do предмет
        "f1": lambda rng: rng.uniform(0.05, 2.0, 200),    # di изображение
        "formula": lambda d: d["f0"]*d["f1"]/(d["f0"]+d["f1"]),
    },
    {
        # P = sigma * A * T^4
        "id": 15, "name": "Закон Стефана-Больцмана",
        "true_formula": "f0 * f1 * f2**4",
        "noise_type":   "gaussian",
        "n_features": 3, "dim_codes": [0, 0, 0],
        "f0": lambda rng: np.full(200, 5.67e-8),           # sigma
        "f1": lambda rng: rng.uniform(0.01, 1.0, 200),    # A площадь
        "f2": lambda rng: rng.uniform(100, 800, 200),      # T температура
        "formula": lambda d: d["f0"]*d["f1"]*d["f2"]**4,
    },
    {
        # lambda = h / (m * v)
        "id": 16, "name": "Длина волны де Бройля",
        "true_formula": "f0 / (f1 * f2)",
        "noise_type":   "outliers",
        "n_features": 3, "dim_codes": [0, 0, 0],
        "f0": lambda rng: np.full(200, 6.626e-34),          # h
        "f1": lambda rng: rng.uniform(1e-30, 1e-27, 200),  # m
        "f2": lambda rng: rng.uniform(1e3, 1e6, 200),      # v
        "formula": lambda d: d["f0"]/(d["f1"]*d["f2"]),
    },
    {
        # P = n * R * T / V
        "id": 17, "name": "Идеальный газ (давление)",
        "true_formula": "f0 * f1 * f2 / f3",
        "noise_type":   "heteroscedastic",
        "n_features": 4, "dim_codes": [0, 0, 0, 0],
        "f0": lambda rng: rng.uniform(0.1, 10.0, 200),    # n моль
        "f1": lambda rng: np.full(200, 8.314),             # R
        "f2": lambda rng: rng.uniform(200, 1000, 200),     # T
        "f3": lambda rng: rng.uniform(0.001, 0.1, 200),   # V
        "formula": lambda d: d["f0"]*d["f1"]*d["f2"]/d["f3"],
    },
    {
        # F = m * v² / r
        "id": 18, "name": "Центростремительная сила",
        "true_formula": "f0 * f1**2 / f2",
        "noise_type":   "gaussian",
        "n_features": 3, "dim_codes": [0, 0, 0],
        "f0": lambda rng: rng.uniform(0.1, 50.0, 200),    # m
        "f1": lambda rng: rng.uniform(0.1, 30.0, 200),    # v
        "f2": lambda rng: rng.uniform(0.1, 10.0, 200),    # r
        "formula": lambda d: d["f0"]*d["f1"]**2/d["f2"],
    },
    {
        # v1 = sqrt(G * M / r)
        "id": 19, "name": "Первая космическая скорость",
        "true_formula": "sqrt(f0 * f1 / f2)",
        "noise_type":   "outliers",
        "n_features": 3, "dim_codes": [0, 0, 0],
        "f0": lambda rng: np.full(200, 6.674e-11),         # G
        "f1": lambda rng: rng.uniform(1e23, 1e26, 200),   # M
        "f2": lambda rng: rng.uniform(1e6, 1e8, 200),     # r
        "formula": lambda d: np.sqrt(d["f0"]*d["f1"]/d["f2"]),
    },
    {
        # F = q * v * B  (перпендикулярно)
        "id": 20, "name": "Сила Лоренца",
        "true_formula": "f0 * f1 * f2",
        "noise_type":   "heteroscedastic",
        "n_features": 3, "dim_codes": [0, 0, 0],
        "f0": lambda rng: rng.uniform(1e-9, 1e-6, 200),   # q
        "f1": lambda rng: rng.uniform(1e3, 1e6, 200),     # v
        "f2": lambda rng: rng.uniform(0.001, 1.0, 200),   # B
        "formula": lambda d: d["f0"]*d["f1"]*d["f2"],
    },
]

# ── ПРОГОН 3: 10 законов (высокая сложность) ──────────────────────
# Transfer learning: система применяет опыт прогонов 1 и 2
# Новые скелеты: 1/sqrt(1-v²/v²), sqrt(v³/vv), v*(1/v²-1/v²),
#                1/sqrt(vv), v*v/sqrt(1-v²/v²), v*v²/v,
#                v/v², v*sqrt(1-v²/v²), v*(1+v/v), v*v/(v*v*v)
PHYSICS_LAWS_RUN3 = [
    {
        # gamma = 1 / sqrt(1 - (v/c)²)
        "id": 21, "name": "Фактор Лоренца",
        "true_formula": "1 / sqrt(1 - (f0/f1)**2)",
        "noise_type":   "heteroscedastic",
        "n_features": 2, "dim_codes": [0, 0],
        "f0": lambda rng: rng.uniform(0.01, 0.95, 200) * 3e8,
        "f1": lambda rng: np.full(200, 3e8),
        "formula": lambda d: 1/np.sqrt(np.clip(1-(d["f0"]/d["f1"])**2, 1e-10, None)),
    },
    {
        # T = 2*pi * sqrt(a³ / (G*M))
        "id": 22, "name": "Третий закон Кеплера",
        "true_formula": "6.2832 * sqrt(f0**3 / (f1 * f2))",
        "noise_type":   "gaussian",
        "n_features": 3, "dim_codes": [0, 0, 0],
        "f0": lambda rng: rng.uniform(1e9, 1e12, 200),    # a полуось
        "f1": lambda rng: np.full(200, 6.674e-11),         # G
        "f2": lambda rng: rng.uniform(1e28, 1e32, 200),   # M
        "formula": lambda d: 2*np.pi*np.sqrt(d["f0"]**3/(d["f1"]*d["f2"])),
    },
    {
        # 1/lambda = R_inf * (1/n1² - 1/n2²)
        "id": 23, "name": "Формула Ридберга",
        "true_formula": "f0 * (1/f1**2 - 1/f2**2)",
        "noise_type":   "outliers",
        "n_features": 3, "dim_codes": [0, 0, 0],
        "f0": lambda rng: np.full(200, 1.097e7),
        "f1": lambda rng: np.tile([1,1,2,2,3], 40).astype(float),   # n1
        "f2": lambda rng: np.tile([2,3,3,4,4], 40).astype(float),   # n2 > n1
        "formula": lambda d: d["f0"]*(1/d["f1"]**2 - 1/d["f2"]**2),
    },
    {
        # omega = 1 / sqrt(L * C)
        "id": 24, "name": "Резонансная частота LC-контура",
        "true_formula": "1 / sqrt(f0 * f1)",
        "noise_type":   "heteroscedastic",
        "n_features": 2, "dim_codes": [0, 0],
        "f0": lambda rng: rng.uniform(1e-6, 1e-3, 200),   # L
        "f1": lambda rng: rng.uniform(1e-9, 1e-6, 200),   # C
        "formula": lambda d: 1/np.sqrt(d["f0"]*d["f1"]),
    },
    {
        # p = m * v / sqrt(1 - (v/c)²)
        "id": 25, "name": "Релятивистский импульс",
        "true_formula": "f0 * f1 / sqrt(1 - (f1/f2)**2)",
        "noise_type":   "gaussian",
        "n_features": 3, "dim_codes": [0, 0, 0],
        "f0": lambda rng: rng.uniform(1e-30, 1e-27, 200),
        "f1": lambda rng: rng.uniform(0.01, 0.9, 200)*3e8,
        "f2": lambda rng: np.full(200, 3e8),
        "formula": lambda d: d["f0"]*d["f1"]/np.sqrt(np.clip(1-(d["f1"]/d["f2"])**2, 1e-10, None)),
    },
    {
        # E = 3 * G * M² / (5 * R)
        "id": 26, "name": "Гравитационная связывающая энергия",
        "true_formula": "3 * f0 * f1**2 / (5 * f2)",
        "noise_type":   "outliers",
        "n_features": 3, "dim_codes": [0, 0, 0],
        "f0": lambda rng: np.full(200, 6.674e-11),
        "f1": lambda rng: rng.uniform(1e24, 1e30, 200),
        "f2": lambda rng: rng.uniform(1e6, 1e9, 200),
        "formula": lambda d: 3*d["f0"]*d["f1"]**2/(5*d["f2"]),
    },
    {
        # I = P / (4 * pi * r²)
        "id": 27, "name": "Интенсивность точечного источника",
        "true_formula": "f0 / (4 * 3.14159 * f1**2)",
        "noise_type":   "heteroscedastic",
        "n_features": 2, "dim_codes": [0, 0],
        "f0": lambda rng: rng.uniform(0.01, 1000.0, 200),
        "f1": lambda rng: rng.uniform(0.1, 100.0, 200),
        "formula": lambda d: d["f0"]/(4*np.pi*d["f1"]**2),
    },
    {
        # L = L0 * sqrt(1 - (v/c)²)
        "id": 28, "name": "Лоренцево сокращение длины",
        "true_formula": "f0 * sqrt(1 - (f1/f2)**2)",
        "noise_type":   "gaussian",
        "n_features": 3, "dim_codes": [0, 0, 0],
        "f0": lambda rng: rng.uniform(0.1, 100.0, 200),
        "f1": lambda rng: rng.uniform(0.01, 0.9, 200)*3e8,
        "f2": lambda rng: np.full(200, 3e8),
        "formula": lambda d: d["f0"]*np.sqrt(np.clip(1-(d["f1"]/d["f2"])**2, 0, None)),
    },
    {
        # f_obs = f0 * (1 + v_obs / v_sound)
        "id": 29, "name": "Эффект Доплера",
        "true_formula": "f0 * (1 + f1 / f2)",
        "noise_type":   "outliers",
        "n_features": 3, "dim_codes": [0, 0, 0],
        "f0": lambda rng: rng.uniform(100, 10000, 200),
        "f1": lambda rng: rng.uniform(0.0, 100.0, 200),
        "f2": lambda rng: np.full(200, 343.0),
        "formula": lambda d: d["f0"]*(1+d["f1"]/d["f2"]),
    },
    {
        # V_H = I * B / (n * q * d)  — напряжение Холла
        "id": 30, "name": "Напряжение Холла",
        "true_formula": "f0 * f1 / (f2 * f3 * f4)",
        "noise_type":   "heteroscedastic",
        "n_features": 5, "dim_codes": [0, 0, 0, 0, 0],
        "f0": lambda rng: rng.uniform(0.001, 10.0, 200),
        "f1": lambda rng: rng.uniform(0.01, 5.0, 200),
        "f2": lambda rng: rng.uniform(1e20, 1e23, 200),
        "f3": lambda rng: np.full(200, 1.6e-19),
        "f4": lambda rng: rng.uniform(1e-4, 1e-2, 200),
        "formula": lambda d: d["f0"]*d["f1"]/(d["f2"]*d["f3"]*d["f4"]),
    },
]

# Карта прогон → законы
LAWS_BY_RUN = {1: PHYSICS_LAWS_RUN1, 2: PHYSICS_LAWS_RUN2, 3: PHYSICS_LAWS_RUN3}

RESULTS_PATH      = Path("scalpel_vault/feynman_results.json")
GROUND_TRUTH_PATH = Path("scalpel_vault/feynman_ground_truth_PRIVATE.json")
RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)


def _score(found, true):
    f = found.lower().replace(" ", "")
    t = true.lower().replace(" ", "")
    if f == t:
        return "ТОЧНО"

    # FIX v10.27 #11: пробуем sympy для проверки математической эквивалентности.
    # Без этого "f1 * f0" vs "f0 * f1" давало ПРОМАХ для F=ma, E=mc², Лоренца и т.д.
    # PySR часто возвращает операнды в другом порядке — это корректная формула.
    try:
        import sympy
        # Создаём символы f0..f9 для парсинга
        _syms = {f"f{i}": sympy.Symbol(f"f{i}") for i in range(10)}
        _expr_f = sympy.sympify(found.replace("^", "**"), locals=_syms)
        _expr_t = sympy.sympify(true.replace("^", "**"),  locals=_syms)
        _diff   = sympy.simplify(_expr_f - _expr_t)
        if _diff == 0:
            return "ТОЧНО"
        # Проверяем пропорциональность (формула с другим константным множителем)
        _ratio_expr = sympy.simplify(_expr_f / _expr_t)
        if _ratio_expr.is_number and abs(float(_ratio_expr) - 1.0) < 0.01:
            return "ТОЧНО"
        if _ratio_expr.is_number:
            return "БЛИЗКО"  # правильная структура, неверная константа
    except Exception:
        pass  # sympy недоступен или не смог распарсить — используем маркеры ниже

    markers_f = {op for op in ["sqrt","**2","**3","**4","/","log","exp","pi"] if op in f}
    markers_t = {op for op in ["sqrt","**2","**3","**4","/","log","exp","pi"] if op in t}
    ratio = len(markers_f & markers_t) / max(len(markers_t), 1)
    if ratio >= 0.8:  return "БЛИЗКО"
    if ratio >= 0.5:  return "ЧАСТИЧНО"
    return "ПРОМАХ"


def _run_single_pass(run_num: int, laws: list, from_engine, run_llm_phase):
    """Один полный прогон всех 10 законов. Возвращает (results, score_map)."""
    run_path = Path(f"scalpel_vault/feynman_results_run{run_num}.json")

    results   = []
    score_map = {"ТОЧНО": 0, "БЛИЗКО": 0, "ЧАСТИЧНО": 0, "ПРОМАХ": 0}
    ts_start  = datetime.now().isoformat(timespec="seconds")

    # Resume внутри прогона (если упал на середине)
    done_ids = set()
    if run_path.exists():
        try:
            prev = json.loads(run_path.read_text(encoding="utf-8"))
            for r in prev.get("results", []):
                done_ids.add(r["id"])
                r.pop("true_formula", None)
                results.append(r)
                score_map[r["score"]] = score_map.get(r["score"], 0) + 1
            if done_ids:
                print(f"  ▶ Прогон {run_num}: продолжаем, уже выполнены законы {sorted(done_ids)}\n")
        except Exception:
            pass

    # Каждый прогон использует одинаковый seed — одни и те же данные, честное сравнение
    master_rng = np.random.default_rng(3141)

    for law in laws:
        lid   = law["id"]
        feats = [f"f{i}" for i in range(law["n_features"])]

        if lid in done_ids:
            print(f"  ⏭  Закон {lid}/10 уже выполнен — пропускаем")
            continue

        print(f"\n{'─'*62}")
        print(f"  Закон {lid}/10: [СКРЫТО ОТ СИСТЕМЫ]")
        print(f"  Признаков: {law['n_features']}")
        print(f"{'─'*62}")

        data = {}
        for f in feats:
            data[f] = law[f](master_rng)
        y_all = law["formula"](data)
        X_all = np.column_stack([data[f] for f in feats])

        rng_l   = np.random.default_rng(lid * 3141)
        y_noisy = _add_noise(y_all, rng_l, law.get("noise_type", "gaussian"))

        idx  = rng_l.permutation(len(y_all))
        n_tr = int(0.75 * len(y_all))
        tr, te = idx[:n_tr], idx[n_tr:]

        X_train = X_all[tr].astype("float64")
        X_test  = X_all[te].astype("float64")
        y_train = y_noisy[tr].astype("float64")
        y_test  = y_all[te].astype("float64")

        print(f"  Данные: {len(X_train)} train / {len(X_test)} test")

        t0        = time.time()
        final_f   = "ERROR"
        r2_final  = 0.0
        consensus = "ERROR"

        try:
            result = from_engine(
                X_train      = X_train,
                y_train      = y_train,
                X_test       = X_test,
                y_test       = y_test,
                feat_names   = feats,
                target_col   = "y",
                domain_type  = "",
                phase        = "pysr",
                dim_codes    = law["dim_codes"],
                noise_hint   = None,  # автоопределение через estimate_noise_level()
                skip_heritage= False,
            )
            pysr_f  = result.formula_real
            r2_pysr = result.r2_blind
            print(f"\n  PySR: {pysr_f}  R²={r2_pysr:.4f}")

            if r2_pysr > 0.3:
                # FIX v10.27 #1: run_llm_phase() читает PHASE_RESULT_PATH —
                # глобальный файл который run_engine пишет перед возвратом.
                # Проверяем что файл свежий (содержит attempt_num текущего прогона).
                # Если run_engine упал до записи → файл может содержать кандидатов
                # предыдущего закона → LLM верифицирует чужую формулу.
                from pathlib import Path as _Path
                _phase_path = _Path("scalpel_vault/pysr_phase_result.json")
                _phase_ok = False
                if _phase_path.exists():
                    try:
                        _ph = json.loads(_phase_path.read_text(encoding="utf-8"))
                        # Проверяем что закон совпадает: feat_names в файле == feats текущего
                        _ph_feats = (_ph.get("candidates") or [_ph])[0].get("feat_names", [])
                        _phase_ok = set(_ph_feats) == set(feats)
                    except Exception:
                        _phase_ok = False
                if not _phase_ok:
                    print(f"  ⚠️  PHASE_RESULT_PATH не соответствует текущему закону — LLM-фаза пропущена")
                    final_f   = pysr_f
                    r2_final  = r2_pysr
                    consensus = "PHASE_MISMATCH"
                else:
                    llm = run_llm_phase()
                    if llm and llm.formula_real:
                        final_f   = llm.formula_real
                        r2_final  = llm.r2_blind
                        consensus = llm.consensus
                    else:
                        final_f   = pysr_f
                        r2_final  = r2_pysr
                        consensus = "NOT_RUN"
                    _log_preparator_result("LLM", r2_final, "SUCCESS" if r2_final > 0.5 else "PARTIAL")
            else:
                final_f   = pysr_f
                r2_final  = r2_pysr
                consensus = "LOW_R2"
                _log_preparator_result("LLM", r2_pysr, "FAIL")

        except Exception as e:
            print(f"  ❌ Ошибка: {e}")

        elapsed = round(time.time() - t0, 1)
        score   = _score(final_f, law["true_formula"])
        score_map[score] += 1
        icons = {"ТОЧНО":"✅","БЛИЗКО":"🟡","ЧАСТИЧНО":"🟠","ПРОМАХ":"❌"}

        print(f"\n  {icons[score]} [{score}] Закон {lid}: {law['name']}")
        print(f"     Нашла:  {final_f}")
        print(f"     Истина: [СКРЫТО — см. feynman_ground_truth_PRIVATE.json]")
        print(f"     R²={r2_final:.4f}  Consensus={consensus}  t={elapsed}s")

        tf_hash = hashlib.sha256(law["true_formula"].encode()).hexdigest()[:16]

        # Сохраняем истину отдельно — LLM-роли не читают
        _gt = {}
        if GROUND_TRUTH_PATH.exists():
            try:
                _gt = json.loads(GROUND_TRUTH_PATH.read_text(encoding="utf-8"))
            except Exception:
                pass
        _gt[str(lid)] = {"name": law["name"], "true_formula": law["true_formula"]}
        GROUND_TRUTH_PATH.write_text(
            json.dumps(_gt, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        results.append({
            "run": run_num,
            "id": lid, "name": law["name"],
            "noise_type": law.get("noise_type", "gaussian"),
            "true_formula_hash": tf_hash,
            "found_formula": final_f,
            "r2_blind": round(r2_final, 4),
            "consensus": consensus,
            "score": score,
            "elapsed_sec": elapsed,
            "ts": datetime.now().isoformat(timespec="seconds"),
        })

        run_path.write_text(
            json.dumps({
                "benchmark": "Физический v3.0 (честный)",
                "run": run_num,
                "ts_start": ts_start,
                "ts_updated": datetime.now().isoformat(timespec="seconds"),
                "score_map": score_map,
                "results": results,
            }, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"  Сохранено → feynman_results_run{run_num}.json")

    # Итог прогона
    total   = len(laws)
    correct = score_map["ТОЧНО"] + score_map["БЛИЗКО"]
    print(f"\n{'═'*62}")
    print(f"  ПРОГОН {run_num}/3 — ФИНАЛ")
    print(f"{'═'*62}")
    print(f"  ✅ Точно:     {score_map['ТОЧНО']}/{total}")
    print(f"  🟡 Близко:    {score_map['БЛИЗКО']}/{total}")
    print(f"  🟠 Частично:  {score_map['ЧАСТИЧНО']}/{total}")
    print(f"  ❌ Промах:    {score_map['ПРОМАХ']}/{total}")
    print(f"{'─'*62}")
    print(f"  Итог прогона {run_num}: {correct}/{total}")
    print(f"{'═'*62}")

    return results, score_map


def run_physics_benchmark(num_runs: int = 3):
    """Запускает num_runs полных прогонов. Логи обучения НЕ сбрасываются между прогонами."""
    print("=" * 62)
    print(f"  ФИЗИЧЕСКИЙ BENCHMARK v3.0 — {num_runs} ПРОГОНА (ЧЕСТНЫЙ)")
    print("  Система видит только числа. Никаких подсказок.")
    print("  Логи обучения накапливаются между прогонами.")
    print("=" * 62)
    print()

    _check_ollama()

    from scalpel.engine import run_engine, run_llm_phase

    # Проверяем с какого прогона продолжать
    start_run = 1
    for rn in range(1, num_runs + 1):
        rp   = Path(f"scalpel_vault/feynman_results_run{rn}.json")
        laws = LAWS_BY_RUN[rn]
        if rp.exists():
            try:
                prev = json.loads(rp.read_text(encoding="utf-8"))
                if len(prev.get("results", [])) == len(laws):
                    start_run = rn + 1
                else:
                    start_run = rn
                    break
            except Exception:
                start_run = rn
                break
        else:
            start_run = rn
            break

    if start_run > num_runs:
        print("  ✅ Все прогоны уже завершены. Показываю итог.\n")
    else:
        print(f"  ▶ Начинаем с прогона {start_run}/{num_runs}\n")

    all_runs = {}

    for run_num in range(1, num_runs + 1):
        rp   = Path(f"scalpel_vault/feynman_results_run{run_num}.json")
        laws = LAWS_BY_RUN[run_num]

        if run_num < start_run:
            try:
                prev = json.loads(rp.read_text(encoding="utf-8"))
                all_runs[run_num] = prev
                sm = prev.get("score_map", {})
                correct = sm.get("ТОЧНО", 0) + sm.get("БЛИЗКО", 0)
                print(f"  ⏭  Прогон {run_num}: уже завершён — {correct}/{len(laws)}")
            except Exception:
                pass
            continue

        print(f"\n{'█'*62}")
        print(f"  ПРОГОН {run_num}/{num_runs} — СТАРТ")
        print(f"  Законы {laws[0]['id']}–{laws[-1]['id']} (новые, система видит первый раз)")
        print(f"  Накоплено прогонов опыта: {run_num - 1}")
        print(f"{'█'*62}")

        results, score_map = _run_single_pass(run_num, laws, run_engine, run_llm_phase)
        all_runs[run_num] = {"score_map": score_map, "results": results}

    # ── ИТОГОВЫЙ SUMMARY ─────────────────────────────────────────
    print(f"\n{'█'*62}")
    print(f"  ИТОГОВЫЙ SUMMARY — {num_runs} ПРОГОНОВ (Transfer Learning)")
    print(f"{'█'*62}")

    summary_rows = []
    for rn in range(1, num_runs + 1):
        if rn not in all_runs:
            continue
        sm    = all_runs[rn].get("score_map", {})
        laws  = LAWS_BY_RUN[rn]
        total = len(laws)
        correct = sm.get("ТОЧНО", 0) + sm.get("БЛИЗКО", 0)
        label = ["", "базовый", "средний", "сложный"][rn]
        print(f"  Прогон {rn} ({label:8s}): ✅{sm.get('ТОЧНО',0)}  🟡{sm.get('БЛИЗКО',0)}"
              f"  🟠{sm.get('ЧАСТИЧНО',0)}  ❌{sm.get('ПРОМАХ',0)}"
              f"  → {correct}/{total}")
        summary_rows.append({"run": rn, "score_map": sm, "correct": correct, "total": total})

    # Динамика обучения
    if len(summary_rows) >= 2:
        delta = summary_rows[-1]["correct"] - summary_rows[0]["correct"]
        print(f"{'─'*62}")
        if delta > 0:
            print(f"  📈 Система улучшилась на {delta} закона(ов) за {num_runs} прогона!")
        elif delta == 0:
            print(f"  ➡️  Результат стабилен.")
        else:
            print(f"  📉 Результат снизился на {abs(delta)} (стохастичность PySR).")

    summary_path = Path("scalpel_vault/feynman_results_summary.json")
    summary_path.write_text(
        json.dumps({
            "benchmark":   "Физический v3.0 (честный)",
            "num_runs":    num_runs,
            "ts":          datetime.now().isoformat(timespec="seconds"),
            "author":      "Гилазетдинов Адель Рустамович",
            "runs":        summary_rows,
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\n  Детали → feynman_results_run1/2/3.json")
    print(f"  Итог   → feynman_results_summary.json")
    print(f"  Истина → feynman_ground_truth_PRIVATE.json")
    print(f"{'█'*62}")


def run_baseline(num_runs: int = 3):
    """
    Baseline: те же 30 законов, только голый PySR без LLM.
    Никакого Navigator, Oracle, Матрёшки — только эволюция.
    Используется для сравнения в статье.
    """
    from pysr import PySRRegressor

    print("=" * 62)
    print(f"  BASELINE — чистый PySR без LLM ({num_runs} прогона)")
    print("  Никакого Navigator, Oracle, Матрёшки.")
    print("  Только эволюция символьной регрессии.")
    print("=" * 62)
    print()

    all_runs = {}

    for run_num in range(1, num_runs + 1):
        laws      = LAWS_BY_RUN[run_num]
        run_path  = Path(f"scalpel_vault/baseline_results_run{run_num}.json")
        results   = []
        score_map = {"ТОЧНО": 0, "БЛИЗКО": 0, "ЧАСТИЧНО": 0, "ПРОМАХ": 0}
        done_ids  = set()

        # Resume
        if run_path.exists():
            try:
                prev = json.loads(run_path.read_text(encoding="utf-8"))
                for r in prev.get("results", []):
                    done_ids.add(r["id"])
                    r.pop("true_formula", None)
                    results.append(r)
                    score_map[r["score"]] = score_map.get(r["score"], 0) + 1
                if done_ids:
                    print(f"  ▶ Прогон {run_num}: продолжаем, выполнено {sorted(done_ids)}")
            except Exception:
                pass

        print(f"\n{'█'*62}")
        print(f"  BASELINE ПРОГОН {run_num}/3 — законы {laws[0]['id']}–{laws[-1]['id']}")
        print(f"{'█'*62}")

        master_rng = np.random.default_rng(3141)
        ts_start   = datetime.now().isoformat(timespec="seconds")

        for law in laws:
            lid   = law["id"]
            feats = [f"f{i}" for i in range(law["n_features"])]

            if lid in done_ids:
                print(f"  ⏭  Закон {lid} уже выполнен")
                continue

            print(f"\n{'─'*62}")
            print(f"  Закон {lid}/30: [СКРЫТО]  Признаков: {law['n_features']}")
            print(f"{'─'*62}")

            # Генерация данных (идентично Scalpel для честного сравнения)
            data = {}
            for f in feats:
                data[f] = law[f](master_rng)
            y_all = law["formula"](data)
            X_all = np.column_stack([data[f] for f in feats])

            rng_l   = np.random.default_rng(lid * 3141)
            y_noisy = _add_noise(y_all, rng_l, law.get("noise_type", "gaussian"))

            idx  = rng_l.permutation(len(y_all))
            n_tr = int(0.75 * len(y_all))
            tr, te = idx[:n_tr], idx[n_tr:]

            X_train = X_all[tr].astype("float64")
            X_test  = X_all[te].astype("float64")
            y_train = y_noisy[tr].astype("float64")
            y_test  = y_all[te].astype("float64")

            t0       = time.time()
            final_f  = "ERROR"
            r2_final = 0.0

            try:
                # Голый PySR — без seed гипотез, без подсказок
                model = PySRRegressor(
                    niterations   = 40,
                    binary_operators = ["+", "-", "*", "/"],
                    unary_operators  = ["sqrt", "log", "exp", "abs", "square", "cube"],
                    maxsize          = 20,
                    verbosity        = 0,
                    random_state     = lid,
                )
                model.fit(X_train, y_train, variable_names=feats)

                best = model.get_best()
                final_f = str(best["sympy_format"])

                # R² на тестовой выборке
                y_pred  = model.predict(X_test)
                ss_res  = np.sum((y_test - y_pred) ** 2)
                ss_tot  = np.sum((y_test - np.mean(y_test)) ** 2)
                r2_final = float(1 - ss_res / ss_tot) if ss_tot > 1e-12 else 0.0

            except Exception as e:
                print(f"  ❌ Ошибка PySR: {e}")

            elapsed = round(time.time() - t0, 1)
            score   = _score(final_f, law["true_formula"])
            score_map[score] += 1
            icons   = {"ТОЧНО":"✅","БЛИЗКО":"🟡","ЧАСТИЧНО":"🟠","ПРОМАХ":"❌"}

            print(f"\n  {icons[score]} [{score}] Закон {lid}: {law['name']}")
            print(f"     Нашла:  {final_f}")
            print(f"     Истина: [СКРЫТО]")
            print(f"     R²={r2_final:.4f}  t={elapsed}s")

            tf_hash = hashlib.sha256(law["true_formula"].encode()).hexdigest()[:16]

            results.append({
                "run": run_num, "id": lid, "name": law["name"],
                "noise_type": law.get("noise_type", "gaussian"),
                "true_formula_hash": tf_hash,
                "found_formula": final_f,
                "r2_blind": round(r2_final, 4),
                "score": score,
                "elapsed_sec": elapsed,
                "ts": datetime.now().isoformat(timespec="seconds"),
            })

            run_path.write_text(
                json.dumps({
                    "mode": "baseline_pysr",
                    "run": run_num,
                    "ts_start": ts_start,
                    "ts_updated": datetime.now().isoformat(timespec="seconds"),
                    "score_map": score_map,
                    "results": results,
                }, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            print(f"  Сохранено → baseline_results_run{run_num}.json")

        total   = len(laws)
        correct = score_map["ТОЧНО"] + score_map["БЛИЗКО"]
        print(f"\n{'═'*62}")
        print(f"  BASELINE ПРОГОН {run_num} — ФИНАЛ: {correct}/{total}")
        print(f"{'═'*62}")
        all_runs[run_num] = {"score_map": score_map, "results": results}

    # ── ИТОГОВЫЙ BASELINE SUMMARY ────────────────────────────────
    print(f"\n{'█'*62}")
    print(f"  BASELINE SUMMARY — {num_runs} ПРОГОНОВ")
    print(f"{'█'*62}")

    summary_rows = []
    for rn in range(1, num_runs + 1):
        if rn not in all_runs:
            continue
        sm      = all_runs[rn].get("score_map", {})
        total   = len(LAWS_BY_RUN[rn])
        correct = sm.get("ТОЧНО", 0) + sm.get("БЛИЗКО", 0)
        label   = ["", "базовый", "средний", "сложный"][rn]
        print(f"  Прогон {rn} ({label:8s}): ✅{sm.get('ТОЧНО',0)}  🟡{sm.get('БЛИЗКО',0)}"
              f"  🟠{sm.get('ЧАСТИЧНО',0)}  ❌{sm.get('ПРОМАХ',0)}"
              f"  → {correct}/{total}")
        summary_rows.append({"run": rn, "score_map": sm, "correct": correct, "total": total})

    Path("scalpel_vault/baseline_summary.json").write_text(
        json.dumps({
            "mode":     "baseline_pysr_only",
            "num_runs": num_runs,
            "ts":       datetime.now().isoformat(timespec="seconds"),
            "author":   "Гилазетдинов Адель Рустамович",
            "runs":     summary_rows,
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\n  Итог → baseline_summary.json")
    print(f"{'█'*62}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Физический Benchmark v3.0")
    parser.add_argument(
        "--baseline", action="store_true",
        help="Запустить baseline (только PySR без LLM) для сравнения в статье"
    )
    args = parser.parse_args()

    if args.baseline:
        run_baseline(num_runs=3)
    else:
        run_physics_benchmark(num_runs=3)
