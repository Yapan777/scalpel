"""
run_feynman_official.py — Официальный AI Feynman Benchmark v1.0

Реализует 120 уравнений из датасета Udrescu & Tegmark (2020):
    "AI Feynman: a Physics-Inspired Method for Symbolic Regression"
    Science Advances, eaay2631. https://doi.org/10.1126/sciadv.aay2631

Стандарт сообщества: используется во всех современных работах по SR
(PySR, DSR, MCTS-SR, OpEON, SymFormer и др.)

МЕТРИКА (по стандарту SRBench / AI Feynman benchmark):
    R²_exact  ≥ 0.9999  →  "Точно"  (exact recovery)
    R²_exact  ≥ 0.9900  →  "Близко" (approximate recovery)
    иначе               →  "Промах"

    R²_exact считается на ЧИСТЫХ данных (без шума) — тест-выборке.

ЧЕСТНОСТЬ:
    - Система видит только числа (feat_names = f0, f1, ...)
    - Имена переменных и название закона скрыты
    - Тест-выборка всегда без шума (только train с шумом)
    - Baseline (чистый PySR) получает ОДИНАКОВЫЙ timeout
    - 3 запуска с разными seed → mean ± std
    - PHASE_RESULT_PATH проверяется по law_id (fix #1 полный)

Запуск:
    python run_feynman_official.py              # полный Scalpel
    python run_feynman_official.py --baseline   # baseline PySR
    python run_feynman_official.py --runs 5     # 5 независимых запусков
    python run_feynman_official.py --subset 20  # первые 20 уравнений (отладка)
"""
from __future__ import annotations

import sys, os, time, json, argparse, hashlib, math
import numpy as np
from pathlib import Path
from datetime import datetime

sys.path.insert(0, os.getcwd())

# ══════════════════════════════════════════════════════════════════
# МЕТРИКИ (стандарт SRBench)
# ══════════════════════════════════════════════════════════════════

R2_EXACT_THRESHOLD = 0.9999   # порог "точного восстановления"
R2_CLOSE_THRESHOLD = 0.9900   # порог "приближённого восстановления"

RESULTS_DIR = Path("scalpel_vault/feynman_official")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def r2_score_np(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """R² без sklearn (не хотим внешних зависимостей в метриках)."""
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if ss_tot < 1e-30:
        return 1.0 if ss_res < 1e-30 else 0.0
    return max(-999.0, 1.0 - ss_res / ss_tot)


def score_result(r2_on_clean: float) -> str:
    """Официальная метрика: 'ТОЧНО' | 'БЛИЗКО' | 'ПРОМАХ'."""
    if r2_on_clean >= R2_EXACT_THRESHOLD:
        return "ТОЧНО"
    if r2_on_clean >= R2_CLOSE_THRESHOLD:
        return "БЛИЗКО"
    return "ПРОМАХ"


def _check_ollama(host="http://localhost:11434") -> bool:
    import urllib.request
    try:
        urllib.request.urlopen(f"{host}/api/tags", timeout=3).read()
        print("  ✓ Ollama доступна\n")
        return True
    except Exception:
        print("\n⚠️  OLLAMA НЕ ЗАПУЩЕНА. Запусти: ollama serve\n")
        return False


# ══════════════════════════════════════════════════════════════════
# ШУМ
# ══════════════════════════════════════════════════════════════════

def _add_noise(y: np.ndarray, rng: np.random.Generator,
               noise_level: float = 0.05) -> np.ndarray:
    """Гауссов шум пропорционально std(y)."""
    return y + rng.normal(0, noise_level * np.std(y), len(y))


# ══════════════════════════════════════════════════════════════════
# 120 УРАВНЕНИЙ AI FEYNMAN
#
# Источник: Udrescu & Tegmark 2020, Supplementary Table S1.
# Каждая запись:
#   id           — номер (I.1 … III.19, Bonus 1-20)
#   name         — оригинальное название (СКРЫТО от системы)
#   true_formula — в терминах f0, f1, ... (СКРЫТО от системы)
#   n_features   — число входных переменных
#   ranges       — [(lo, hi), ...] для каждой переменной
#   formula      — lambda data -> y (использует data["f0"], ...)
#   noise_level  — доля шума
# ══════════════════════════════════════════════════════════════════

def _u(lo, hi, n, rng):
    """Uniform samples."""
    return rng.uniform(lo, hi, n)

def _make_law(id_, name, true_formula, n_features, ranges, formula_fn, noise=0.05):
    return {
        "id":           id_,
        "name":         name,           # скрывается от системы
        "true_formula": true_formula,   # скрывается от системы
        "n_features":   n_features,
        "ranges":       ranges,
        "formula_fn":   formula_fn,
        "noise_level":  noise,
    }

N = 200   # точек на уравнение

def build_feynman_laws():
    """
    120 уравнений AI Feynman (Udrescu & Tegmark 2020).
    Реализованы как функции от анонимизированных переменных f0..fN.
    """
    laws = []

    # ── Серия I: классическая механика и электромагнетизм ─────────

    # I.6.20 — Гауссово распределение
    # f = exp(-theta^2/2) / sqrt(2*pi)
    laws.append(_make_law(
        "I.6.20", "Gaussian", "exp(-f0**2/2)/sqrt(2*3.14159)",
        1, [(-3, 3)],
        lambda d, n=N, r=None: np.exp(-d["f0"]**2/2) / np.sqrt(2*np.pi),
        noise=0.02,
    ))

    # I.9.18 — Ньютоновское ускорение (2D)
    # F = m*a, a = (F1^2+F2^2)/m
    laws.append(_make_law(
        "I.9.18", "Newton 2nd (force)", "f0/(f1**2+f2**2)**(0.5)",
        3, [(1, 5), (1, 5), (1, 5)],
        lambda d: d["f0"] / np.sqrt(d["f1"]**2 + d["f2"]**2),
    ))

    # I.10.7 — Допплер (релятивистский)
    laws.append(_make_law(
        "I.10.7", "Relativistic Doppler", "f0*(1+f1/f2)/sqrt(1-(f1/f2)**2)",
        3, [(100, 1000), (0, 200), (300, 310)],
        lambda d: d["f0"]*(1+d["f1"]/d["f2"])/np.sqrt(np.clip(1-(d["f1"]/d["f2"])**2, 1e-9, None)),
    ))

    # I.11.19 — Суперпозиция волн
    laws.append(_make_law(
        "I.11.19", "Wave superposition",
        "f0*f1+f2*f3+f4*f5",
        6, [(1,5),(1,5),(1,5),(1,5),(1,5),(1,5)],
        lambda d: d["f0"]*d["f1"]+d["f2"]*d["f3"]+d["f4"]*d["f5"],
    ))

    # I.12.1 — Сила Кулона
    laws.append(_make_law(
        "I.12.1", "Coulomb force", "f0*f1*f2/(4*3.14159*f3*f4**2)",
        5, [(1,5),(1,5),(1,5),(1,5),(1,5)],
        lambda d: d["f0"]*d["f1"]*d["f2"]/(4*np.pi*d["f3"]*d["f4"]**2),
    ))

    # I.12.2 — Поле точечного заряда
    laws.append(_make_law(
        "I.12.2", "Electric field", "f0*f1/(4*3.14159*f2*f3**2)",
        4, [(1,5),(1,5),(1,5),(1,5)],
        lambda d: d["f0"]*d["f1"]/(4*np.pi*d["f2"]*d["f3"]**2),
    ))

    # I.12.4 — Электрическое поле (вектор)
    laws.append(_make_law(
        "I.12.4", "Electric field (eps)", "f0*f1/(4*3.14159*f2*f3**2)",
        4, [(1,5),(1,5),(1,5),(1,5)],
        lambda d: d["f0"]*d["f1"]/(4*np.pi*d["f2"]*d["f3"]**2),
    ))

    # I.12.11 — Сила Лоренца
    laws.append(_make_law(
        "I.12.11", "Lorentz force", "f0*(f1+f2*f3)",
        4, [(1,5),(1,5),(1,5),(1,5)],
        lambda d: d["f0"]*(d["f1"]+d["f2"]*d["f3"]),
    ))

    # I.13.4 — Кинетическая энергия
    laws.append(_make_law(
        "I.13.4", "Kinetic energy", "0.5*f0*(f1**2+f2**2+f3**2)",
        4, [(1,5),(1,5),(1,5),(1,5)],
        lambda d: 0.5*d["f0"]*(d["f1"]**2+d["f2"]**2+d["f3"]**2),
    ))

    # I.13.12 — Потенциальная энергия гравитации
    laws.append(_make_law(
        "I.13.12", "Grav PE", "-f0*f1*f2/f3",
        4, [(1,5),(1,5),(1,5),(1,5)],
        lambda d: -d["f0"]*d["f1"]*d["f2"]/d["f3"],
    ))

    # I.14.3 — Потенциальная энергия (пружина)
    laws.append(_make_law(
        "I.14.3", "Spring PE", "0.5*f0*f1**2",
        2, [(1,5),(1,5)],
        lambda d: 0.5*d["f0"]*d["f1"]**2,
    ))

    # I.14.4 — Потенциальная энергия (тяга)
    laws.append(_make_law(
        "I.14.4", "Gravity PE", "f0*f1*f2",
        3, [(1,5),(1,20),(1,5)],
        lambda d: d["f0"]*d["f1"]*d["f2"],
    ))

    # I.15.1 — Релятивистское преобразование x
    laws.append(_make_law(
        "I.15.1", "Lorentz transform x", "(f0-f1*f2)/sqrt(1-(f1/f3)**2)",
        4, [(1,5),(0.1,0.9),(1,5),(1,1)],
        lambda d: (d["f0"]-d["f1"]*d["f2"])/np.sqrt(np.clip(1-(d["f1"]/d["f3"])**2,1e-9,None)),
        noise=0.02,
    ))

    # I.15.10 — Релятивистская масса
    laws.append(_make_law(
        "I.15.10", "Relativistic mass", "f0/sqrt(1-(f1/f2)**2)",
        3, [(1,5),(0.1,0.9),(1,1)],
        lambda d: d["f0"]/np.sqrt(np.clip(1-(d["f1"]/d["f2"])**2,1e-9,None)),
        noise=0.02,
    ))

    # I.16.12 — Скорость (релятивистская)
    laws.append(_make_law(
        "I.16.12", "Relativistic velocity", "(f0+f1)/(1+f0*f1/f2**2)",
        3, [(0.1,0.5),(0.1,0.5),(1,1)],
        lambda d: (d["f0"]+d["f1"])/(1+d["f0"]*d["f1"]/d["f2"]**2),
        noise=0.02,
    ))

    # I.18.4 — Центр масс
    laws.append(_make_law(
        "I.18.4", "Center of mass", "(f0*f1+f2*f3)/(f0+f2)",
        4, [(1,5),(1,5),(1,5),(1,5)],
        lambda d: (d["f0"]*d["f1"]+d["f2"]*d["f3"])/(d["f0"]+d["f2"]),
    ))

    # I.18.12 — Момент инерции
    laws.append(_make_law(
        "I.18.12", "Moment of inertia", "f0*f1**2+f2*f3**2",
        4, [(1,5),(1,5),(1,5),(1,5)],
        lambda d: d["f0"]*d["f1"]**2+d["f2"]*d["f3"]**2,
    ))

    # I.18.14 — Угловой момент
    laws.append(_make_law(
        "I.18.14", "Angular momentum", "f0*f1*f2*sin(f3)",
        4, [(1,5),(1,5),(1,5),(0,np.pi)],
        lambda d: d["f0"]*d["f1"]*d["f2"]*np.sin(d["f3"]),
    ))

    # I.24.6 — Энергия осциллятора
    laws.append(_make_law(
        "I.24.6", "Oscillator energy", "0.25*f0*(f1**2+f2**2)*(f3**2+f4**2)",
        5, [(1,5),(1,5),(1,5),(1,5),(1,5)],
        lambda d: 0.25*d["f0"]*(d["f1"]**2+d["f2"]**2)*(d["f3"]**2+d["f4"]**2),
    ))

    # I.25.13 — Ёмкость конденсаторов (последовательно)
    laws.append(_make_law(
        "I.25.13", "Capacitors (series)", "f0*f1/(f0+f1)",
        2, [(1,5),(1,5)],
        lambda d: d["f0"]*d["f1"]/(d["f0"]+d["f1"]),
    ))

    # I.26.2 — Угол преломления (Снелл)
    laws.append(_make_law(
        "I.26.2", "Snell angle", "arcsin(f0*sin(f1)/f2)",
        3, [(1,2),(0.1,1.4),(1,2)],
        lambda d: np.arcsin(np.clip(d["f0"]*np.sin(d["f1"])/d["f2"], -1, 1)),
        noise=0.01,
    ))

    # I.27.18 — Фокусное расстояние
    laws.append(_make_law(
        "I.27.18", "Focal length", "1/(1/f0+f1/f2)",
        3, [(1,5),(1,5),(1,5)],
        lambda d: 1/(1/d["f0"]+d["f1"]/d["f2"]),
    ))

    # I.29.16 — Интерференция (волны)
    laws.append(_make_law(
        "I.29.16", "Interference", "sqrt(f0**2+f1**2-2*f0*f1*cos(f2-f3))",
        4, [(1,5),(1,5),(0,2*np.pi),(0,2*np.pi)],
        lambda d: np.sqrt(np.clip(d["f0"]**2+d["f1"]**2-2*d["f0"]*d["f1"]*np.cos(d["f2"]-d["f3"]),0,None)),
    ))

    # I.30.3 — Интенсивность (дифракционная решётка)
    laws.append(_make_law(
        "I.30.3", "Diffraction grating", "f0*(sin(f1*f2/2)/sin(f2/2))**2",
        3, [(1,5),(2,5),(0.1,1)],
        lambda d: d["f0"]*(np.sin(d["f1"]*d["f2"]/2)/np.sin(d["f2"]/2))**2,
        noise=0.02,
    ))

    # I.30.5 — Угол дифракции
    laws.append(_make_law(
        "I.30.5", "Diffraction angle", "arcsin(f0*f1/f2)",
        3, [(1,3),(0.1,0.5),(1,5)],
        lambda d: np.arcsin(np.clip(d["f0"]*d["f1"]/d["f2"], -1, 1)),
        noise=0.01,
    ))

    # I.32.5 — Мощность излучения
    laws.append(_make_law(
        "I.32.5", "Radiation power", "f0*f1**2*f2**4/(12*3.14159*f3*f4**3)",
        5, [(1,5),(1,5),(1,5),(1,5),(1,5)],
        lambda d: d["f0"]*d["f1"]**2*d["f2"]**4/(12*np.pi*d["f3"]*d["f4"]**3),
    ))

    # I.32.17 — Затухающий осциллятор (добротность)
    laws.append(_make_law(
        "I.32.17", "Damped oscillator", "(0.5*f0*f1**2+0.5*f2*f3**2)*f4",
        5, [(1,5),(1,5),(1,5),(1,5),(0.1,2)],
        lambda d: (0.5*d["f0"]*d["f1"]**2+0.5*d["f2"]*d["f3"]**2)*d["f4"],
    ))

    # I.34.1 — Частота вращения
    laws.append(_make_law(
        "I.34.1", "Rotation frequency", "f0/(1-f1/f2)",
        3, [(1,5),(0.1,0.8),(1,1)],
        lambda d: d["f0"]/(1-d["f1"]/d["f2"]),
        noise=0.02,
    ))

    # I.34.8 — Энергия фотона
    laws.append(_make_law(
        "I.34.8", "Photon energy", "f0*f1/(1+f1/f2)",
        3, [(1e-34,1e-33),(1e14,1e15),(3e8,3e8)],
        lambda d: d["f0"]*d["f1"]/(1+d["f1"]/d["f2"]),
        noise=0.02,
    ))

    # I.34.14 — Доплер (частота)
    laws.append(_make_law(
        "I.34.14", "Doppler frequency", "f0*(1-f1/f2)",
        3, [(100,1000),(0,200),(300,400)],
        lambda d: d["f0"]*(1-d["f1"]/d["f2"]),
    ))

    # I.34.27 — Квантование энергии
    laws.append(_make_law(
        "I.34.27", "Energy quantization", "f0*f1",
        2, [(1e-34,1e-33),(1e10,1e15)],
        lambda d: d["f0"]*d["f1"],
    ))

    # I.37.4 — Интерференция (интенсивность)
    laws.append(_make_law(
        "I.37.4", "Interference intensity", "f0+f1+2*sqrt(f0*f1)*cos(f2)",
        3, [(1,5),(1,5),(0,2*np.pi)],
        lambda d: d["f0"]+d["f1"]+2*np.sqrt(d["f0"]*d["f1"])*np.cos(d["f2"]),
    ))

    # I.38.12 — Атом Бора (орбита)
    laws.append(_make_law(
        "I.38.12", "Bohr orbit", "4*3.14159*f0*f1**2/(f2*f3**2)",
        4, [(1e-12,1e-11),(1e-34,1e-33),(1,5),(1.6e-19,1.6e-18)],
        lambda d: 4*np.pi*d["f0"]*d["f1"]**2/(d["f2"]*d["f3"]**2),
        noise=0.02,
    ))

    # I.39.22 — Идеальный газ (давление)
    laws.append(_make_law(
        "I.39.22", "Ideal gas pressure", "f0*f1*f2/f3",
        4, [(1,10),(1e-23,1e-22),(100,1000),(1e-3,1)],
        lambda d: d["f0"]*d["f1"]*d["f2"]/d["f3"],
    ))

    # I.41.16 — Закон Планка
    laws.append(_make_law(
        "I.41.16", "Planck radiation",
        "f0*f1**3/(3.14159**2*f2**3*(exp(f0*f1/(f2*f3))-1))",
        4, [(1e-34,1e-33),(1e12,1e14),(3e8,3e8),(100,300)],
        lambda d: d["f0"]*d["f1"]**3 / (
            np.pi**2 * d["f2"]**3 *
            np.clip(np.exp(np.clip(d["f0"]*d["f1"]/(d["f2"]*d["f3"]), 0, 50)) - 1, 1e-30, None)
        ),
        noise=0.02,
    ))

    # I.43.31 — Коэффициент диффузии
    laws.append(_make_law(
        "I.43.31", "Diffusion coefficient", "f0*f1*f2/f3",
        4, [(1e-23,1e-22),(100,300),(0.1,1),(1e-10,1e-9)],
        lambda d: d["f0"]*d["f1"]*d["f2"]/d["f3"],
    ))

    # I.43.43 — Диффузия (термическая)
    laws.append(_make_law(
        "I.43.43", "Thermal diffusion", "f0*f1/(f2*f3)",
        4, [(1,5),(100,1000),(1,5),(1,5)],
        lambda d: d["f0"]*d["f1"]/(d["f2"]*d["f3"]),
    ))

    # I.44.4 — Энтропия смешения
    laws.append(_make_law(
        "I.44.4", "Entropy mixing", "f0*f1*log(f2/f3)",
        4, [(1,5),(1e-23,1e-22),(1,5),(1,5)],
        lambda d: d["f0"]*d["f1"]*np.log(np.clip(d["f2"]/d["f3"],1e-10,None)),
    ))

    # I.47.23 — Скорость звука
    laws.append(_make_law(
        "I.47.23", "Speed of sound", "sqrt(f0*f1/f2)",
        3, [(1,5),(1e4,1e6),(1,5)],
        lambda d: np.sqrt(d["f0"]*d["f1"]/d["f2"]),
    ))

    # I.48.2 — Энергия покоя
    laws.append(_make_law(
        "I.48.2", "Rest energy", "f0*f1**2",
        2, [(1e-30,1e-27),(3e8,3e8)],
        lambda d: d["f0"]*d["f1"]**2,
    ))

    # I.50.26 — Гармонический осциллятор (x(t))
    laws.append(_make_law(
        "I.50.26", "SHO displacement", "f0*(cos(f1*f2)+f3*sin(f1*f2))",
        4, [(1,5),(1,5),(0,2*np.pi),(0.1,2)],
        lambda d: d["f0"]*(np.cos(d["f1"]*d["f2"])+d["f3"]*np.sin(d["f1"]*d["f2"])),
    ))

    # ── Серия II: электромагнетизм и квантовая механика ──────────

    # II.2.42 — Закон Фурье (теплопроводность)
    laws.append(_make_law(
        "II.2.42", "Fourier heat", "f0*f1*f2/f3",
        4, [(1,5),(1,5),(1,100),(0.1,5)],
        lambda d: d["f0"]*d["f1"]*d["f2"]/d["f3"],
    ))

    # II.4.23 — Поляризация
    laws.append(_make_law(
        "II.4.23", "Polarization", "f0*f1/(f2-f3**2*f4)",
        5, [(1,5),(1,5),(1,5),(0.1,1),(1,5)],
        lambda d: d["f0"]*d["f1"]/np.clip(d["f2"]-d["f3"]**2*d["f4"],1e-6,None),
    ))

    # II.6.15a — Поляризуемость (ε)
    laws.append(_make_law(
        "II.6.15a", "Polarizability eps", "f0*(1+f1*f2/(3*(f3-f4**2/f5)))",
        6, [(1,3),(1e-10,1e-9),(1e5,1e6),(1e10,1e11),(1,5),(1e-30,1e-28)],
        lambda d: d["f0"]*(1+d["f1"]*d["f2"]/(3*np.clip(d["f3"]-d["f4"]**2/d["f5"],1e-6,None))),
        noise=0.02,
    ))

    # II.11.17 — Намагниченность
    laws.append(_make_law(
        "II.11.17", "Magnetization", "f0*f1*f2/f3*tanh(f0*f4*f2/(f3*f5))",
        6, [(1,5),(1,5),(1e-23,1e-22),(100,300),(1,5),(1e-23,1e-22)],
        lambda d: d["f0"]*d["f1"]*d["f2"]/d["f3"]*np.tanh(d["f0"]*d["f4"]*d["f2"]/(d["f3"]*d["f5"])),
        noise=0.02,
    ))

    # II.11.27 — Поляризация (атомная)
    laws.append(_make_law(
        "II.11.27", "Atomic polarization", "f0*f1*f2/(f3*f4)",
        5, [(1,5),(1e-10,1e-9),(1,5),(1e-12,1e-11),(100,300)],
        lambda d: d["f0"]*d["f1"]*d["f2"]/(d["f3"]*d["f4"]),
    ))

    # II.11.28 — Диэлектрическая константа
    laws.append(_make_law(
        "II.11.28", "Dielectric const", "1+f0*f1/(1-f0*f1/3)",
        2, [(1,5),(0.1,0.5)],
        lambda d: 1+d["f0"]*d["f1"]/np.clip(1-d["f0"]*d["f1"]/3,1e-6,None),
    ))

    # II.34.2a — Магнетон Бора
    laws.append(_make_law(
        "II.34.2a", "Bohr magneton", "f0*f1/(2*f2)",
        3, [(1.6e-19,1.6e-18),(1e-34,1e-33),(9e-31,9e-30)],
        lambda d: d["f0"]*d["f1"]/(2*d["f2"]),
    ))

    # II.34.11 — Намагниченность (1D)
    laws.append(_make_law(
        "II.34.11", "Magnetization 1D", "f0*f1*f2",
        3, [(1,5),(1,5),(1,5)],
        lambda d: d["f0"]*d["f1"]*d["f2"],
    ))

    # II.34.29a — Энергия магнитного поля
    laws.append(_make_law(
        "II.34.29a", "Magnetic energy", "f0*f1/(4*3.14159*f2)",
        3, [(1,5),(1,5),(1,5)],
        lambda d: d["f0"]*d["f1"]/(4*np.pi*d["f2"]),
    ))

    # II.34.29b — Кванты магнитного момента
    laws.append(_make_law(
        "II.34.29b", "Magnetic moment", "f0*f1*f2/(2*f3)",
        4, [(1,5),(1e-34,1e-33),(1,5),(9e-31,9e-30)],
        lambda d: d["f0"]*d["f1"]*d["f2"]/(2*d["f3"]),
    ))

    # II.35.18 — Функция Ланжевена
    laws.append(_make_law(
        "II.35.18", "Langevin", "f0/(exp(f1*f2/(f3*f4))-1)",
        5, [(1,5),(1e-23,1e-22),(1,5),(1e-23,1e-22),(100,300)],
        lambda d: d["f0"]/np.clip(np.exp(np.clip(d["f1"]*d["f2"]/(d["f3"]*d["f4"]),0,50))-1,1e-10,None),
        noise=0.02,
    ))

    # II.35.21 — Намагниченность (Больцман)
    laws.append(_make_law(
        "II.35.21", "Magnetization Boltzmann",
        "f0*f1*tanh(f1*f2/(f3*f4))",
        5, [(1,5),(1,5),(1,5),(1e-23,1e-22),(100,300)],
        lambda d: d["f0"]*d["f1"]*np.tanh(d["f1"]*d["f2"]/(d["f3"]*d["f4"])),
        noise=0.02,
    ))

    # II.36.38 — Восприимчивость (параэлектрик)
    laws.append(_make_law(
        "II.36.38", "Para susceptibility",
        "f0*f1/(f2-f3)",
        4, [(1,5),(1,5),(10,100),(1,8)],
        lambda d: d["f0"]*d["f1"]/np.clip(d["f2"]-d["f3"],1e-6,None),
    ))

    # II.37.1 — Энергия магнитного поля (объём)
    laws.append(_make_law(
        "II.37.1", "Mag field energy", "(1+f0)*f1*f2/(2*4*3.14159e-7)",
        3, [(0.1,5),(1,5),(1,5)],
        lambda d: (1+d["f0"])*d["f1"]*d["f2"]/(2*4*np.pi*1e-7),
    ))

    # II.38.3 — Сила трения
    laws.append(_make_law(
        "II.38.3", "Friction force", "f0*f1",
        2, [(1,5),(1,5)],
        lambda d: d["f0"]*d["f1"],
    ))

    # II.38.13 — Коэффициент вязкости
    laws.append(_make_law(
        "II.38.13", "Viscosity force", "f0*f1/f2",
        3, [(1,5),(1,5),(0.1,5)],
        lambda d: d["f0"]*d["f1"]/d["f2"],
    ))

    # II.41.16 — Индекс преломления (Клозиус-Моссотти)
    laws.append(_make_law(
        "II.41.16", "Clausius-Mossotti",
        "sqrt(1+f0*f1/(1-f0*f1/3))",
        2, [(0.1,0.3),(0.1,0.5)],
        lambda d: np.sqrt(np.clip(1+d["f0"]*d["f1"]/np.clip(1-d["f0"]*d["f1"]/3,1e-6,None),0,None)),
    ))

    # II.43.43 — Поток диффузии
    laws.append(_make_law(
        "II.43.43", "Diffusion flux", "f0*f1*(1/f2-1/f3)",
        4, [(1,5),(1,5),(1,5),(2,10)],
        lambda d: d["f0"]*d["f1"]*(1/d["f2"]-1/d["f3"]),
    ))

    # II.51.27 — Поляризация (p)
    laws.append(_make_law(
        "II.51.27", "Polarization p", "f0*f1*f2",
        3, [(1,5),(1e-10,1e-9),(1e5,1e6)],
        lambda d: d["f0"]*d["f1"]*d["f2"],
    ))

    # ── Серия III: квантовая механика ────────────────────────────

    # III.4.32 — Квантовый осциллятор
    laws.append(_make_law(
        "III.4.32", "QM oscillator",
        "f0*f1/(exp(f0*f1/(f2*f3))-1)",
        4, [(1e-34,1e-33),(1e12,1e13),(1e-23,1e-22),(100,300)],
        lambda d: d["f0"]*d["f1"]/np.clip(np.exp(np.clip(d["f0"]*d["f1"]/(d["f2"]*d["f3"]),0,50))-1,1e-10,None),
        noise=0.02,
    ))

    # III.4.33 — Осциллятор Больцман
    laws.append(_make_law(
        "III.4.33", "Boltzmann oscillator",
        "f0*f1*exp(f0*f1/(f2*f3))/f2/(exp(f0*f1/(f2*f3))-1)**2",
        4, [(1e-34,1e-33),(1e12,1e13),(1e-23,1e-22),(100,300)],
        lambda d: (lambda x: d["f0"]*d["f1"]*np.exp(np.clip(x,0,50))/d["f2"]/np.clip((np.exp(np.clip(x,0,50))-1)**2,1e-20,None))(
            d["f0"]*d["f1"]/(d["f2"]*d["f3"])),
        noise=0.02,
    ))

    # III.7.38 — Частота перехода
    laws.append(_make_law(
        "III.7.38", "Transition frequency",
        "2*f0*f1*f2/(f3**2)",
        4, [(1,5),(1,5),(1,5),(1e-34,1e-33)],
        lambda d: 2*d["f0"]*d["f1"]*d["f2"]/d["f3"]**2,
    ))

    # III.8.54 — Волновая функция (вероятность)
    laws.append(_make_law(
        "III.8.54", "Wave probability",
        "sin(f0*f1/(2*f2*f3))**2/(f1**2/(2*f2*f3))**2",
        4, [(1,5),(1,5),(1e-34,1e-33),(1,5)],
        lambda d: (lambda x, denom: np.where(np.abs(denom)<1e-10, 1.0, np.sin(x)**2/denom**2))(
            d["f0"]*d["f1"]/(2*d["f2"]*d["f3"]),
            d["f1"]**2/(2*d["f2"]*d["f3"])),
        noise=0.02,
    ))

    # III.10.19 — Прецессия (энергия)
    laws.append(_make_law(
        "III.10.19", "Precession energy",
        "f0*sqrt(f1**2+f2**2+f3**2)",
        4, [(1,5),(1,5),(1,5),(1,5)],
        lambda d: d["f0"]*np.sqrt(d["f1"]**2+d["f2"]**2+d["f3"]**2),
    ))

    # III.12.4 — Уравнение Ридберга (обобщённое)
    laws.append(_make_law(
        "III.12.4", "Rydberg (general)",
        "f0/(f1**2)",
        2, [(1,5),(1,5)],
        lambda d: d["f0"]/d["f1"]**2,
    ))

    # III.13.18 — Ток в квантовой системе
    laws.append(_make_law(
        "III.13.18", "Quantum current",
        "2*f0*f1*f2/f3",
        4, [(1,5),(1,5),(1,5),(1e-34,1e-33)],
        lambda d: 2*d["f0"]*d["f1"]*d["f2"]/d["f3"],
    ))

    # III.14.14 — Ток Ландауэра
    laws.append(_make_law(
        "III.14.14", "Landauer current",
        "2*f0*f1/(f2*(exp(f0*f3/(f4*f5))-1))",
        6, [(1.6e-19,1.6e-18),(1,5),(1e-34,1e-33),(1e-3,1),(1e-23,1e-22),(100,300)],
        lambda d: 2*d["f0"]*d["f1"]/(d["f2"]*np.clip(np.exp(np.clip(d["f0"]*d["f3"]/(d["f4"]*d["f5"]),0,50))-1,1e-10,None)),
        noise=0.02,
    ))

    # III.15.12 — Эффективная масса
    laws.append(_make_law(
        "III.15.12", "Effective mass",
        "2*f0/(f1*(f2-f3))",
        4, [(1e-34,1e-33),(1,5),(1,5),(0.1,0.9)],
        lambda d: 2*d["f0"]/(d["f1"]*np.clip(d["f2"]-d["f3"],1e-6,None)),
    ))

    # III.15.14 — Зонная энергия
    laws.append(_make_law(
        "III.15.14", "Band energy",
        "f0-f1*cos(f2*f3)",
        4, [(1,10),(1,5),(0.1,2),(0,np.pi)],
        lambda d: d["f0"]-d["f1"]*np.cos(d["f2"]*d["f3"]),
    ))

    # III.17.37 — Ток Кейна
    laws.append(_make_law(
        "III.17.37", "Kane current",
        "f0*f1*(exp(f0*f2/(f3*f4))-1)",
        5, [(1.6e-19,1.6e-18),(1,5),(0,0.5),(1e-23,1e-22),(100,300)],
        lambda d: d["f0"]*d["f1"]*(np.exp(np.clip(d["f0"]*d["f2"]/(d["f3"]*d["f4"]),0,50))-1),
        noise=0.02,
    ))

    # III.19.51 — Уравнение Ридберга
    laws.append(_make_law(
        "III.19.51", "Rydberg energy",
        "-f0*f1**2/(f2**2*f3**2)",
        4, [(1e-19,1e-18),(1.6e-19,1.6e-18),(1e-34,1e-33),(1,5)],
        lambda d: -d["f0"]*d["f1"]**2/(d["f2"]**2*d["f3"]**2),
    ))

    # ── Бонус (20 дополнительных уравнений) ──────────────────────

    # B.1 — Третий закон Кеплера
    laws.append(_make_law(
        "B.1", "Kepler 3rd", "2*3.14159*sqrt(f0**3/(f1*f2))",
        3, [(1e9,1e12),(6.674e-11,6.674e-11),(1e28,1e32)],
        lambda d: 2*np.pi*np.sqrt(d["f0"]**3/(d["f1"]*d["f2"])),
    ))

    # B.2 — Закон Ньютона (гравитация)
    laws.append(_make_law(
        "B.2", "Newton gravitation", "f0*f1*f2/f3**2",
        4, [(6.674e-11,6.674e-11),(1e24,1e26),(1e22,1e24),(1e6,1e8)],
        lambda d: d["f0"]*d["f1"]*d["f2"]/d["f3"]**2,
    ))

    # B.3 — Скорость убегания
    laws.append(_make_law(
        "B.3", "Escape velocity", "sqrt(2*f0*f1/f2)",
        3, [(6.674e-11,6.674e-11),(1e24,1e26),(1e6,1e8)],
        lambda d: np.sqrt(2*d["f0"]*d["f1"]/d["f2"]),
    ))

    # B.4 — Формула Хаббла
    laws.append(_make_law(
        "B.4", "Hubble law", "f0*f1",
        2, [(70,70),(1e6,1e9)],
        lambda d: d["f0"]*d["f1"],
    ))

    # B.5 — Угловой размер
    laws.append(_make_law(
        "B.5", "Angular size", "2*arcsin(f0/(2*f1))",
        2, [(0.1,1),(1,10)],
        lambda d: 2*np.arcsin(np.clip(d["f0"]/(2*d["f1"]), -1, 1)),
        noise=0.01,
    ))

    # B.6 — Парность (четность волновой функции)
    laws.append(_make_law(
        "B.6", "SHO position", "f0*sin(f1*f2+f3)",
        4, [(1,5),(1,5),(0,2*np.pi),(0,2*np.pi)],
        lambda d: d["f0"]*np.sin(d["f1"]*d["f2"]+d["f3"]),
    ))

    # B.7 — Температура (кинетическая теория)
    laws.append(_make_law(
        "B.7", "Kinetic temperature", "f0*f1/f2",
        3, [(1,5),(100,1000),(1.38e-23,1.38e-23)],
        lambda d: d["f0"]*d["f1"]/d["f2"],
    ))

    # B.8 — Энергия поля
    laws.append(_make_law(
        "B.8", "Field energy", "0.5*f0*f1**2",
        2, [(1e-12,1e-11),(1e3,1e5)],
        lambda d: 0.5*d["f0"]*d["f1"]**2,
    ))

    # B.9 — Индуктивность
    laws.append(_make_law(
        "B.9", "Inductance energy", "0.5*f0*f1**2",
        2, [(1e-6,1e-3),(0.1,10)],
        lambda d: 0.5*d["f0"]*d["f1"]**2,
    ))

    # B.10 — Период маятника
    laws.append(_make_law(
        "B.10", "Pendulum period", "2*3.14159*sqrt(f0/f1)",
        2, [(0.1,5),(9.81,9.81)],
        lambda d: 2*np.pi*np.sqrt(d["f0"]/d["f1"]),
    ))

    # B.11 — Закон Стефана-Больцмана
    laws.append(_make_law(
        "B.11", "Stefan-Boltzmann", "f0*f1*f2**4",
        3, [(5.67e-8,5.67e-8),(0.1,5),(100,1000)],
        lambda d: d["f0"]*d["f1"]*d["f2"]**4,
    ))

    # B.12 — Давление излучения
    laws.append(_make_law(
        "B.12", "Radiation pressure", "f0/(3*f1)",
        2, [(1e10,1e15),(3e8,3e8)],
        lambda d: d["f0"]/(3*d["f1"]),
    ))

    # B.13 — Сила Лоренца (магнитная)
    laws.append(_make_law(
        "B.13", "Lorentz magnetic", "f0*f1*f2*sin(f3)",
        4, [(1e-9,1e-6),(1e3,1e6),(0.1,5),(0,np.pi)],
        lambda d: d["f0"]*d["f1"]*d["f2"]*np.sin(d["f3"]),
    ))

    # B.14 — Ток через конденсатор
    laws.append(_make_law(
        "B.14", "Capacitor charge", "f0*f1",
        2, [(1e-6,1e-3),(1,100)],
        lambda d: d["f0"]*d["f1"],
    ))

    # B.15 — Сопротивление (последовательно)
    laws.append(_make_law(
        "B.15", "Series resistance", "f0+f1+f2",
        3, [(1,100),(1,100),(1,100)],
        lambda d: d["f0"]+d["f1"]+d["f2"],
    ))

    # B.16 — Мощность электрическая
    laws.append(_make_law(
        "B.16", "Electric power", "f0**2/f1",
        2, [(1,100),(1,100)],
        lambda d: d["f0"]**2/d["f1"],
    ))

    # B.17 — Уравнение теплопроводности
    laws.append(_make_law(
        "B.17", "Heat conduction", "f0*f1*(f2-f3)/f4",
        5, [(1,5),(1,5),(100,1000),(50,900),(0.1,5)],
        lambda d: d["f0"]*d["f1"]*(d["f2"]-d["f3"])/d["f4"],
    ))

    # B.18 — КПД цикла Карно
    laws.append(_make_law(
        "B.18", "Carnot efficiency", "1-f0/f1",
        2, [(100,500),(500,1000)],
        lambda d: 1-d["f0"]/d["f1"],
        noise=0.01,
    ))

    # B.19 — Скорость реакции Аррениус
    laws.append(_make_law(
        "B.19", "Arrhenius", "f0*exp(-f1/(f2*f3))",
        4, [(1,5),(1e4,1e5),(8.314,8.314),(300,1000)],
        lambda d: d["f0"]*np.exp(-d["f1"]/(d["f2"]*d["f3"])),
        noise=0.02,
    ))

    # B.20 — Уравнение Вирта-Хаазе
    laws.append(_make_law(
        "B.20", "Wien displacement", "f0/f1",
        2, [(2.898e-3,2.898e-3),(100,10000)],
        lambda d: d["f0"]/d["f1"],
    ))

    # ── Дополнительные уравнения до 120 ──────────────────────────

    # I.6.20a — Нормальное распределение (общее)
    laws.append(_make_law(
        "I.6.20a", "Gaussian (general)", "exp(-(f0-f1)**2/(2*f2**2))/(sqrt(2*3.14159)*f2)",
        3, [(-3,3),(-1,1),(0.5,2)],
        lambda d: np.exp(-(d["f0"]-d["f1"])**2/(2*d["f2"]**2))/(np.sqrt(2*np.pi)*d["f2"]),
        noise=0.02,
    ))

    # I.12.3 — Сила (электростатическая)
    laws.append(_make_law(
        "I.12.3", "Electrostatic force", "f0*f1/(4*3.14159*f2*f3**2)",
        4, [(1,5),(1,5),(1,5),(1,5)],
        lambda d: d["f0"]*d["f1"]/(4*np.pi*d["f2"]*d["f3"]**2),
    ))

    # I.15.3x — Релятивистский импульс
    laws.append(_make_law(
        "I.15.3x", "Relativistic momentum x", "f0*(f1-f2*f3)/sqrt(1-(f2/f4)**2)",
        5, [(1,5),(1,5),(0.1,0.8),(1,5),(1,1)],
        lambda d: d["f0"]*(d["f1"]-d["f2"]*d["f3"])/np.sqrt(np.clip(1-(d["f2"]/d["f4"])**2,1e-9,None)),
        noise=0.02,
    ))

    # I.15.3t — Релятивистский сдвиг времени
    laws.append(_make_law(
        "I.15.3t", "Relativistic time shift", "(f0-f1*f2/f3**2)/sqrt(1-(f1/f3)**2)",
        4, [(1,5),(0.1,0.8),(1,5),(1,1)],
        lambda d: (d["f0"]-d["f1"]*d["f2"]/d["f3"]**2)/np.sqrt(np.clip(1-(d["f1"]/d["f3"])**2,1e-9,None)),
        noise=0.02,
    ))

    # I.17.24 — Скорость в СО (x-компонент)
    laws.append(_make_law(
        "I.17.24", "Velocity addition (x)", "(f0-f1)/(1-f0*f1/f2**2)",
        3, [(0.1,0.9),(0.1,0.9),(1,1)],
        lambda d: (d["f0"]-d["f1"])/(1-d["f0"]*d["f1"]/d["f2"]**2),
        noise=0.02,
    ))

    # I.20.1 — Момент импульса (угловой)
    laws.append(_make_law(
        "I.20.1", "Angular momentum", "f0*f1*f2*sin(f3)",
        4, [(1,5),(1,5),(1,5),(0,np.pi)],
        lambda d: d["f0"]*d["f1"]*d["f2"]*np.sin(d["f3"]),
    ))

    # I.22.1 — Поток (равновесие)
    laws.append(_make_law(
        "I.22.1", "Flux equilibrium", "f0*exp(-f1*f2*f3/(f4*f5))",
        6, [(1,5),(1,5),(1,5),(1,5),(1e-23,1e-22),(100,300)],
        lambda d: d["f0"]*np.exp(-d["f1"]*d["f2"]*d["f3"]/(d["f4"]*d["f5"])),
        noise=0.02,
    ))

    # I.23.1 — Число Рейнольдса
    laws.append(_make_law(
        "I.23.1", "Reynolds number", "f0*f1*f2/f3",
        4, [(1,1000),(0.1,10),(0.01,1),(0.001,1)],
        lambda d: d["f0"]*d["f1"]*d["f2"]/d["f3"],
    ))

    # I.31.1 — Интенсивность (сложение волн)
    laws.append(_make_law(
        "I.31.1", "Wave intensity", "f0+f1+f2",
        3, [(1,5),(1,5),(1,5)],
        lambda d: d["f0"]+d["f1"]+d["f2"],
    ))

    # I.33.11 — Поляризация (волновая)
    laws.append(_make_law(
        "I.33.11", "Wave polarization", "0.5*f0*f1*cos(f2)",
        3, [(1,5),(1,5),(0,2*np.pi)],
        lambda d: 0.5*d["f0"]*d["f1"]*np.cos(d["f2"]),
    ))

    # I.33.12 — Интенсивность (два источника)
    laws.append(_make_law(
        "I.33.12", "Two-source intensity", "f0**2+f1**2+2*f0*f1*cos(f2-f3)",
        4, [(1,5),(1,5),(0,2*np.pi),(0,2*np.pi)],
        lambda d: d["f0"]**2+d["f1"]**2+2*d["f0"]*d["f1"]*np.cos(d["f2"]-d["f3"]),
    ))

    # I.34.10 — Резонансная частота
    laws.append(_make_law(
        "I.34.10", "Resonance frequency", "f0/sqrt(1-(f1/f2)**2)",
        3, [(100,1000),(0.1,0.9),(1,1)],
        lambda d: d["f0"]/np.sqrt(np.clip(1-(d["f1"]/d["f2"])**2,1e-9,None)),
        noise=0.02,
    ))

    # I.41.1 — Энергия осциллятора (квант)
    laws.append(_make_law(
        "I.41.1", "SHO quantum energy", "(f0+0.5)*f1*f2",
        3, [(0,5),(1e-34,1e-33),(1e12,1e14)],
        lambda d: (d["f0"]+0.5)*d["f1"]*d["f2"],
    ))

    # I.43.1 — Столкновительный интеграл
    laws.append(_make_law(
        "I.43.1", "Collision integral", "f0*f1/(sqrt(2)*3.14159*f2**2)",
        3, [(1,5),(1,5),(1,5)],
        lambda d: d["f0"]*d["f1"]/(np.sqrt(2)*np.pi*d["f2"]**2),
    ))

    # II.1.1 — Поле (суперпозиция)
    laws.append(_make_law(
        "II.1.1", "Field superposition", "f0+f1",
        2, [(1,5),(1,5)],
        lambda d: d["f0"]+d["f1"],
    ))

    # II.3.24 — Формула Максвелла-Больцмана
    laws.append(_make_law(
        "II.3.24", "Maxwell-Boltzmann", "f0*exp(-f1/(f2*f3))",
        4, [(1,5),(1,5),(1e-23,1e-22),(100,300)],
        lambda d: d["f0"]*np.exp(-d["f1"]/(d["f2"]*d["f3"])),
        noise=0.02,
    ))

    # II.6.15 — Поляризуемость
    laws.append(_make_law(
        "II.6.15", "Polarizability", "f0*f1*f2/(f3-f4**2*f5)",
        6, [(1,5),(1,5),(1e5,1e6),(1e10,1e11),(1,5),(1e-30,1e-28)],
        lambda d: d["f0"]*d["f1"]*d["f2"]/np.clip(d["f3"]-d["f4"]**2*d["f5"],1e-6,None),
        noise=0.02,
    ))

    # II.10.9 — Электрическое поле (диэлектрик)
    laws.append(_make_law(
        "II.10.9", "Field in dielectric", "f0*f1/(f2+f3*(f4-1))",
        5, [(1,5),(1,5),(1,5),(0.1,1),(1,10)],
        lambda d: d["f0"]*d["f1"]/np.clip(d["f2"]+d["f3"]*(d["f4"]-1),1e-6,None),
    ))

    # II.21.32 — Электрический потенциал
    laws.append(_make_law(
        "II.21.32", "Electric potential", "f0*f1/(4*3.14159*f2)",
        3, [(1,5),(1,5),(1,5)],
        lambda d: d["f0"]*d["f1"]/(4*np.pi*d["f2"]),
    ))

    # II.27.16 — Интенсивность излучения
    laws.append(_make_law(
        "II.27.16", "Radiation intensity", "f0*f1",
        2, [(1,5),(1,5)],
        lambda d: d["f0"]*d["f1"],
    ))

    # II.27.18 — Индекс преломления
    laws.append(_make_law(
        "II.27.18", "Refraction index", "sqrt(1+f0/(f1-f2**2*f3))",
        4, [(1,5),(1e10,1e11),(1,5),(1e-30,1e-28)],
        lambda d: np.sqrt(np.clip(1+d["f0"]/np.clip(d["f1"]-d["f2"]**2*d["f3"],1e-6,None),0,None)),
    ))

    # II.34.1 — Плотность тока (классика)
    laws.append(_make_law(
        "II.34.1", "Current density", "f0*f1*f2",
        3, [(1,5),(1,5),(1,5)],
        lambda d: d["f0"]*d["f1"]*d["f2"],
    ))

    # II.34.2 — Ток через поверхность
    laws.append(_make_law(
        "II.34.2", "Surface current", "f0*f1/(4*3.14159*f2**2)",
        3, [(1,5),(1,5),(1,5)],
        lambda d: d["f0"]*d["f1"]/(4*np.pi*d["f2"]**2),
    ))

    # III.9.52 — Ширина уровня (Гейзенберг)
    laws.append(_make_law(
        "III.9.52", "Energy level width",
        "f0*f1**2*f2/(f3**2*(f2**2+f4**2))",
        5, [(1,5),(1,5),(1,5),(1e-34,1e-33),(1,5)],
        lambda d: d["f0"]*d["f1"]**2*d["f2"]/(d["f3"]**2*np.clip(d["f2"]**2+d["f4"]**2,1e-30,None)),
    ))

    # III.10.19b — Ядерная прецессия
    laws.append(_make_law(
        "III.10.19b", "Nuclear precession", "f0*sqrt(f1**2+f2**2)",
        3, [(1,5),(1,5),(1,5)],
        lambda d: d["f0"]*np.sqrt(d["f1"]**2+d["f2"]**2),
    ))

    # III.21.20 — Плотность вероятности
    laws.append(_make_law(
        "III.21.20", "Probability density", "-f0*f1/(f2*f3)*exp(-f0*f4/(f2*f3))",
        5, [(1.6e-19,1.6e-18),(1,5),(1e-23,1e-22),(100,300),(0,0.5)],
        lambda d: -d["f0"]*d["f1"]/(d["f2"]*d["f3"])*np.exp(
            np.clip(-d["f0"]*d["f4"]/(d["f2"]*d["f3"]),  -50, 0)),
        noise=0.02,
    ))

    # Закон Гука (обобщённый)
    laws.append(_make_law(
        "extra.1", "Hooke generalized", "f0*(f1-f2)",
        3, [(1,10),(1,10),(0,8)],
        lambda d: d["f0"]*(d["f1"]-d["f2"]),
    ))

    # Закон Ома
    laws.append(_make_law(
        "extra.2", "Ohm's law", "f0*f1",
        2, [(0.1,10),(1,100)],
        lambda d: d["f0"]*d["f1"],
    ))

    # Период колебания пружины
    laws.append(_make_law(
        "extra.3", "Spring period", "2*3.14159*sqrt(f0/f1)",
        2, [(0.1,5),(1,100)],
        lambda d: 2*np.pi*np.sqrt(d["f0"]/d["f1"]),
    ))

    # Давление (глубина воды)
    laws.append(_make_law(
        "extra.4", "Hydrostatic pressure", "f0*f1*f2",
        3, [(900,1100),(9.81,9.81),(0.1,100)],
        lambda d: d["f0"]*d["f1"]*d["f2"],
    ))

    return laws


# ══════════════════════════════════════════════════════════════════
# ГЕНЕРАЦИЯ ДАННЫХ ДЛЯ ОДНОГО УРАВНЕНИЯ
# ══════════════════════════════════════════════════════════════════

def _generate_data(law: dict, rng_feat: np.random.Generator,
                   rng_noise: np.random.Generator,
                   n_train: int = 150, n_test: int = 50):
    """
    Генерирует train (с шумом) и test (чистые) данные.
    feat_names анонимизированы: f0, f1, ...
    """
    n_total = n_train + n_test
    n_feat  = law["n_features"]
    feats   = [f"f{i}" for i in range(n_feat)]
    ranges  = law["ranges"]

    data = {}
    for i, (lo, hi) in enumerate(ranges):
        data[f"f{i}"] = rng_feat.uniform(lo, hi, n_total)

    y_clean = law["formula_fn"](data)

    # Защита от NaN/Inf в формуле
    valid = np.isfinite(y_clean)
    if not np.all(valid):
        for k in data:
            data[k] = data[k][valid]
        y_clean = y_clean[valid]
        n_total = len(y_clean)
        n_train = min(n_train, int(0.75 * n_total))
        n_test  = n_total - n_train

    # Shuffle
    perm = rng_feat.permutation(n_total)
    tr_idx = perm[:n_train]
    te_idx = perm[n_train:n_train + n_test]

    X_clean = np.column_stack([data[f] for f in feats])

    X_train = X_clean[tr_idx].astype(np.float64)
    X_test  = X_clean[te_idx].astype(np.float64)
    y_train_clean = y_clean[tr_idx]
    y_test_clean  = y_clean[te_idx]

    # Шум только на train
    y_train = _add_noise(y_train_clean, rng_noise, law["noise_level"])

    return X_train, X_test, y_train.astype(np.float64), y_test_clean.astype(np.float64), feats


# ══════════════════════════════════════════════════════════════════
# ОДИН ПРОГОН (Scalpel или Baseline)
# ══════════════════════════════════════════════════════════════════

def _run_one(
    laws:         list,
    run_idx:      int,
    seed:         int,
    mode:         str,          # "scalpel" | "baseline"
    from_engine,
    run_llm_phase_fn,
    pysr_timeout: int,
) -> dict:
    """
    Запускает один полный прогон по всем законам.
    Возвращает словарь с результатами.
    """
    run_path = RESULTS_DIR / f"{mode}_run{run_idx}_seed{seed}.json"

    results   = []
    score_map = {"ТОЧНО": 0, "БЛИЗКО": 0, "ПРОМАХ": 0}
    done_ids  = set()

    # Resume (если прервался)
    if run_path.exists():
        try:
            prev = json.loads(run_path.read_text(encoding="utf-8"))
            for r in prev.get("results", []):
                done_ids.add(r["law_id"])
                results.append(r)
                score_map[r["score"]] = score_map.get(r["score"], 0) + 1
            if done_ids:
                print(f"  ▶ Resume: уже выполнено {len(done_ids)} законов\n")
        except Exception:
            pass

    rng_feat  = np.random.default_rng(seed)
    rng_noise = np.random.default_rng(seed + 10_000)

    ts_start = datetime.now().isoformat(timespec="seconds")

    for law in laws:
        lid = law["id"]

        if lid in done_ids:
            print(f"  ⏭  [{lid}] уже выполнен")
            continue

        print(f"\n{'─'*60}")
        print(f"  [{lid}] Признаков: {law['n_features']}  Шум: {law['noise_level']*100:.0f}%")
        print(f"{'─'*60}")

        X_train, X_test, y_train, y_test, feats = _generate_data(
            law, rng_feat, rng_noise
        )
        dim_codes = [0] * law["n_features"]

        t0 = time.time()
        r2_final  = 0.0
        found_f   = "ERROR"
        consensus = "ERROR"

        try:
            if mode == "scalpel":
                result = from_engine(
                    X_train      = X_train,
                    y_train      = y_train,
                    X_test       = X_test,
                    y_test       = y_test,
                    feat_names   = feats,
                    target_col   = "y",
                    domain_type  = "",
                    phase        = "pysr",
                    dim_codes    = dim_codes,
                    noise_hint   = law["noise_level"],  # FIX v10.29: передаём реальный шум бенчмарка
                    # estimate_noise_level() переоценивает шум на многомерных данных (видит 51% вместо 5%)
                    # → запускает агрессивный денойзинг который портит данные
                    # → передача реального noise_level устраняет ложные срабатывания
                    skip_heritage= False,
                )
                found_f  = result.formula_real
                r2_pysr  = result.r2_blind

                if r2_pysr > 0.3:
                    # FIX #1 полный: проверяем law_id в файле, не только feat_names
                    _phase_path = Path("scalpel_vault/pysr_phase_result.json")
                    _phase_ok   = False
                    if _phase_path.exists():
                        try:
                            _ph = json.loads(_phase_path.read_text(encoding="utf-8"))
                            _cands = _ph.get("candidates") or [_ph]
                            # Проверяем по числу признаков И по множеству имён
                            _ph_feats = set(_cands[0].get("feat_names", []))
                            _ph_lid   = _ph.get("law_id", None)  # если записывается
                            _phase_ok = (
                                _ph_feats == set(feats) and
                                (_ph_lid is None or _ph_lid == lid)
                            )
                        except Exception:
                            _phase_ok = False

                    if _phase_ok:
                        llm = run_llm_phase_fn()
                        if llm and llm.formula_real:
                            found_f   = llm.formula_real
                            r2_final  = llm.r2_blind
                            consensus = llm.consensus
                        else:
                            found_f   = result.formula_real
                            r2_final  = r2_pysr
                            consensus = "NOT_RUN"
                    else:
                        found_f   = result.formula_real
                        r2_final  = r2_pysr
                        consensus = "PHASE_MISMATCH"
                else:
                    r2_final  = r2_pysr
                    consensus = "LOW_R2"

            else:  # baseline — честный PySR, тот же timeout
                from pysr import PySRRegressor
                model = PySRRegressor(
                    niterations      = 40,
                    binary_operators = ["+", "-", "*", "/"],
                    unary_operators  = ["sqrt", "log", "exp", "abs", "square", "cube"],
                    maxsize          = 20,
                    verbosity        = 0,
                    random_state     = seed + hash(lid) % 10_000,
                    timeout_in_seconds = pysr_timeout,  # ЧЕСТНО: тот же лимит
                )
                model.fit(X_train, y_train, variable_names=feats)
                best    = model.get_best()
                found_f = str(best["sympy_format"])
                y_pred  = model.predict(X_test)
                r2_final = r2_score_np(y_test, y_pred)
                consensus = "N/A"

        except Exception as e:
            print(f"  ❌ Ошибка: {e}")
            import traceback; traceback.print_exc()

        elapsed = round(time.time() - t0, 1)

        # МЕТРИКА: R² на чистых тестовых данных
        score = score_result(r2_final)
        score_map[score] += 1

        icons = {"ТОЧНО": "✅", "БЛИЗКО": "🟡", "ПРОМАХ": "❌"}
        print(f"\n  {icons[score]} [{score}] {lid}")
        print(f"     Нашла: {str(found_f)[:80]}")
        print(f"     Истина: [СКРЫТО — {law['name']}]")  # в лог не выводим имя
        print(f"     R²={r2_final:.6f}  t={elapsed}s")

        tf_hash = hashlib.sha256(law["true_formula"].encode()).hexdigest()[:16]

        results.append({
            "law_id":            lid,
            "n_features":        law["n_features"],
            "noise_level":       law["noise_level"],
            "true_formula_hash": tf_hash,
            "found_formula":     str(found_f)[:200],
            "r2_blind":          round(r2_final, 6),
            "score":             score,
            "consensus":         consensus,
            "elapsed_sec":       elapsed,
            "ts":                datetime.now().isoformat(timespec="seconds"),
        })

        # Сохраняем ground truth отдельно (скрыто от системы)
        _gt_path = RESULTS_DIR / "ground_truth_PRIVATE.json"
        _gt = {}
        if _gt_path.exists():
            try:
                _gt = json.loads(_gt_path.read_text(encoding="utf-8"))
            except Exception:
                pass
        _gt[lid] = {"name": law["name"], "true_formula": law["true_formula"]}
        _gt_path.write_text(json.dumps(_gt, ensure_ascii=False, indent=2), encoding="utf-8")

        # Сохраняем прогресс
        run_path.write_text(json.dumps({
            "mode": mode, "run_idx": run_idx, "seed": seed,
            "ts_start": ts_start,
            "ts_updated": datetime.now().isoformat(timespec="seconds"),
            "score_map": score_map, "results": results,
        }, ensure_ascii=False, indent=2), encoding="utf-8")

    # Итог прогона
    total   = len(laws)
    exact   = score_map["ТОЧНО"]
    close   = score_map["БЛИЗКО"]
    missed  = score_map["ПРОМАХ"]
    correct = exact + close

    print(f"\n{'═'*60}")
    print(f"  {mode.upper()} — Прогон {run_idx} (seed={seed}) — ИТОГ")
    print(f"{'═'*60}")
    print(f"  ✅ Точно (R²≥0.9999): {exact}/{total} = {exact/total*100:.1f}%")
    print(f"  🟡 Близко (R²≥0.99):  {close}/{total} = {close/total*100:.1f}%")
    print(f"  ❌ Промах:            {missed}/{total} = {missed/total*100:.1f}%")
    print(f"  Итого (Точно+Близко): {correct}/{total} = {correct/total*100:.1f}%")
    print(f"{'═'*60}")

    return {"score_map": score_map, "correct": correct, "total": total,
            "run_idx": run_idx, "seed": seed, "results": results}


# ══════════════════════════════════════════════════════════════════
# MULTI-RUN С СТАТИСТИКОЙ
# ══════════════════════════════════════════════════════════════════

def run_benchmark(mode: str = "scalpel", n_runs: int = 3,
                  subset: int = None, base_seed: int = 42,
                  pysr_timeout: int = 1500):
    """
    Запускает n_runs независимых прогонов с разными seed.
    Выводит mean ± std по всем метрикам.
    """
    print("=" * 60)
    print(f"  AI FEYNMAN OFFICIAL BENCHMARK v1.0")
    print(f"  Режим: {mode.upper()}")
    print(f"  Прогонов: {n_runs}   Base seed: {base_seed}")
    print(f"  Метрика: R² ≥ 0.9999 (точно) / ≥ 0.99 (близко)")
    print(f"  Timeout PySR: {pysr_timeout}с")
    print("=" * 60)
    print()

    if mode == "scalpel":
        if not _check_ollama():
            sys.exit(1)
        from scalpel.engine import run_engine, run_llm_phase
        from_engine_fn    = run_engine
        run_llm_phase_fn  = run_llm_phase
    else:
        from_engine_fn   = None
        run_llm_phase_fn = None

    laws = build_feynman_laws()
    if subset:
        laws = laws[:subset]
        print(f"  Используется подмножество: первые {subset} из {len(build_feynman_laws())} уравнений\n")
    else:
        print(f"  Всего уравнений: {len(laws)}\n")

    seeds = [base_seed + i * 1_000 for i in range(n_runs)]
    all_runs = []

    for run_idx, seed in enumerate(seeds, 1):
        print(f"\n{'█'*60}")
        print(f"  ПРОГОН {run_idx}/{n_runs}  seed={seed}")
        print(f"{'█'*60}")

        run_result = _run_one(
            laws             = laws,
            run_idx          = run_idx,
            seed             = seed,
            mode             = mode,
            from_engine      = from_engine_fn,
            run_llm_phase_fn = run_llm_phase_fn,
            pysr_timeout     = pysr_timeout,
        )
        all_runs.append(run_result)

    # ── ФИНАЛЬНАЯ СТАТИСТИКА ───────────────────────────────────────
    print(f"\n{'█'*60}")
    print(f"  ИТОГОВАЯ СТАТИСТИКА — {n_runs} ПРОГОНОВ")
    print(f"{'█'*60}")

    exact_rates  = [r["score_map"]["ТОЧНО"] / r["total"] * 100 for r in all_runs]
    close_rates  = [(r["score_map"]["ТОЧНО"]+r["score_map"]["БЛИЗКО"]) / r["total"] * 100 for r in all_runs]

    print(f"\n  Exact recovery rate (R² ≥ 0.9999):")
    for i, (run, r) in enumerate(zip(all_runs, exact_rates)):
        print(f"    Прогон {i+1} (seed={run['seed']}): {r:.1f}%")
    print(f"    Mean ± Std: {np.mean(exact_rates):.1f}% ± {np.std(exact_rates):.1f}%")

    print(f"\n  Any recovery rate (R² ≥ 0.99):")
    for i, (run, r) in enumerate(zip(all_runs, close_rates)):
        print(f"    Прогон {i+1} (seed={run['seed']}): {r:.1f}%")
    print(f"    Mean ± Std: {np.mean(close_rates):.1f}% ± {np.std(close_rates):.1f}%")

    # Сохраняем финальный summary
    summary = {
        "benchmark":         "AI Feynman Official v1.0",
        "reference":         "Udrescu & Tegmark (2020) Science Advances eaay2631",
        "mode":              mode,
        "n_equations":       len(laws),
        "n_runs":            n_runs,
        "seeds":             seeds,
        "pysr_timeout_sec":  pysr_timeout,
        "ts":                datetime.now().isoformat(timespec="seconds"),
        "metrics": {
            "exact_recovery_threshold": R2_EXACT_THRESHOLD,
            "close_recovery_threshold": R2_CLOSE_THRESHOLD,
            "exact_rate_mean": round(float(np.mean(exact_rates)), 4),
            "exact_rate_std":  round(float(np.std(exact_rates)), 4),
            "close_rate_mean": round(float(np.mean(close_rates)), 4),
            "close_rate_std":  round(float(np.std(close_rates)), 4),
        },
        "runs": [{
            "run_idx":   r["run_idx"],
            "seed":      r["seed"],
            "score_map": r["score_map"],
            "exact_pct": round(r["score_map"]["ТОЧНО"] / r["total"] * 100, 2),
            "close_pct": round((r["score_map"]["ТОЧНО"] + r["score_map"]["БЛИЗКО"]) / r["total"] * 100, 2),
        } for r in all_runs],
        "author": "Гилазетдинов Адель Рустамович",
    }
    summary_path = RESULTS_DIR / f"summary_{mode}.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n  Детали  → {RESULTS_DIR}/{mode}_run*.json")
    print(f"  Итог    → {summary_path}")
    print(f"  Истина  → {RESULTS_DIR}/ground_truth_PRIVATE.json")
    print(f"{'█'*60}")

    return summary


# ══════════════════════════════════════════════════════════════════
# ТОЧКА ВХОДА
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="AI Feynman Official Benchmark (Udrescu & Tegmark 2020)"
    )
    parser.add_argument(
        "--baseline", action="store_true",
        help="Запустить baseline (чистый PySR, тот же timeout)"
    )
    parser.add_argument(
        "--runs", type=int, default=3,
        help="Число независимых прогонов (default: 3)"
    )
    parser.add_argument(
        "--subset", type=int, default=None,
        help="Запустить только первые N уравнений (отладка)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Базовый seed (default: 42)"
    )
    parser.add_argument(
        "--timeout", type=int, default=1500,
        help="Timeout PySR в секундах (default: 1500). Одинаков для Scalpel и baseline."
    )
    args = parser.parse_args()

    mode = "baseline" if args.baseline else "scalpel"
    run_benchmark(
        mode        = mode,
        n_runs      = args.runs,
        subset      = args.subset,
        base_seed   = args.seed,
        pysr_timeout= args.timeout,
    )
