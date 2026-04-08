"""
curriculum.py — Curriculum Learning для Navigator.

Тренирует Navigator на известных формулах от простых к сложным.
Пользователь запускает: python -m scalpel --curriculum
Система проходит 40 датасетов (4 уровня × 10) автоматически.

Никакого ручного переключения фаз — curriculum управляет сам.
"""
from __future__ import annotations

import gc
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Tuple

import numpy as np

from .config import ORACLE_MODEL
from .config import (
    OLLAMA_HOST, OLLAMA_MODEL, SCRIPT_DIR,
    PYSR_N_PROCS, PYSR_POPULATIONS, PYSR_POPULATION_SIZE, PYSR_BATCH_SIZE,
)
from .episodic_memory import get_memory

log = logging.getLogger("scalpel")

# ══════════════════════════════════════════════════════════════════
# CHECKPOINT — сохранение прогресса при долгом запуске
# ══════════════════════════════════════════════════════════════════

CHECKPOINT_PATH = SCRIPT_DIR / "scalpel_vault" / "curriculum_checkpoint.json"


def _save_checkpoint(level: int, attempt: int, dataset_idx: int, r2_history: list, elapsed_sec: float = 0.0) -> None:
    """
    Сохраняет прогресс после каждого датасета.
    Защита от потери 50 часов работы при падении.

    Структура:
      {
        "level":      2,
        "attempt":    1,
        "dataset":    7,       ← следующий датасет для запуска
        "r2_history": [0.91, 0.88, ...],
        "ts":         1234567890.0
      }
    """
    try:
        CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            "level":       level,
            "attempt":     attempt,
            "dataset":     dataset_idx,
            "r2_history":  [round(x, 4) for x in r2_history],
            "elapsed_sec": round(elapsed_sec, 1),
            "ts":          time.time(),
        }
        tmp = CHECKPOINT_PATH.with_suffix(".tmp")
        tmp.write_text(json.dumps(checkpoint, ensure_ascii=False), encoding="utf-8")
        tmp.replace(CHECKPOINT_PATH)
        log.debug("[Checkpoint] Уровень %d, датасет %d сохранён", level, dataset_idx)
    except Exception as e:
        log.warning("[Checkpoint] Ошибка записи: %s", e)


def _load_checkpoint() -> Optional[dict]:
    """
    Загружает checkpoint если существует.
    Возвращает None если файла нет или он повреждён.
    """
    if not CHECKPOINT_PATH.exists():
        return None
    try:
        data = json.loads(CHECKPOINT_PATH.read_text(encoding="utf-8"))
        log.info("[Checkpoint] Найден: уровень=%d, попытка=%d, датасет=%d",
                 data.get("level", 0), data.get("attempt", 1), data.get("dataset", 0))
        return data
    except Exception as e:
        log.warning("[Checkpoint] Ошибка чтения, игнорируем: %s", e)
        return None


def _clear_checkpoint() -> None:
    """Удаляет checkpoint после успешного завершения уровня."""
    try:
        if CHECKPOINT_PATH.exists():
            CHECKPOINT_PATH.unlink()
            log.debug("[Checkpoint] Удалён после завершения уровня")
    except Exception as e:
        log.warning("[Checkpoint] Ошибка удаления: %s", e)


# ══════════════════════════════════════════════════════════════════
# УРОВНИ CURRICULUM
# ══════════════════════════════════════════════════════════════════

@dataclass
class CurriculumLevel:
    level:           int
    name:            str
    noise_level:     float   # доля от std(y)
    maxsize:         int
    fast_fail_sec:   int
    r2_threshold:    float
    datasets:        int = 10
    max_retries:     int = 3
    domain:          str = "curriculum"


LEVELS = [
    CurriculumLevel(
        level=1, name="Линейные",
        noise_level=0.01, maxsize=8, fast_fail_sec=300, r2_threshold=0.90,
        domain="curriculum_linear",
    ),
    CurriculumLevel(
        level=2, name="Степенные",
        noise_level=0.03, maxsize=12, fast_fail_sec=600, r2_threshold=0.85,
        domain="curriculum_power",
    ),
    CurriculumLevel(
        level=3, name="Комбинированные",
        noise_level=0.05, maxsize=15, fast_fail_sec=900, r2_threshold=0.80,
        domain="curriculum_combined",
    ),
    CurriculumLevel(
        level=4, name="Реальные законы",
        noise_level=0.10, maxsize=20, fast_fail_sec=1800, r2_threshold=0.75,
        domain="curriculum_physics",
    ),
    CurriculumLevel(
        level=5, name="Грязные реальные данные",
        noise_level=0.20, maxsize=20, fast_fail_sec=2400, r2_threshold=0.65,
        domain="curriculum_dirty",
    ),
]

# ── С какого уровня начинать по умолчанию ────────────────────────
# Уровни 1-3 учат структуре операторов — это важно для DSPy few-shot.
# Но если уже есть experience — можно стартовать с 4.
# Измени на 1 если хочешь полный прогон.
DEFAULT_START_LEVEL = 4


# ══════════════════════════════════════════════════════════════════
# ГЕНЕРАТОРЫ ДАТАСЕТОВ
# ══════════════════════════════════════════════════════════════════

@dataclass
class CurriculumDataset:
    X_train:     np.ndarray
    y_train:     np.ndarray
    X_test:      np.ndarray
    y_test:      np.ndarray
    feat_names:  List[str]
    dim_codes:   List[int]
    formula_str: str          # истинная формула в строковом виде
    level:       int


def _add_noise(y: np.ndarray, noise_level: float, rng: np.random.Generator) -> np.ndarray:
    """Добавляем нормальный шум: noise_level × std(y)."""
    std = np.std(y)
    if std < 1e-12:
        return y
    return y + rng.normal(0, noise_level * std, len(y))


def generate_level1(noise: float, rng: np.random.Generator) -> List[CurriculumDataset]:
    """Уровень 1 — линейные формулы. dim_codes=0 (просто числа)."""
    datasets = []
    n_train, n_test = 150, 50

    templates = [
        ("y = 2.5*x",      lambda x: 2.5 * x[:, 0]),
        ("y = 3*x + 1.2",  lambda x: 3.0 * x[:, 0] + 1.2),
        ("y = -1.5*x",     lambda x: -1.5 * x[:, 0]),
        ("y = 0.8*x + 5",  lambda x: 0.8 * x[:, 0] + 5.0),
        ("y = 4.2/x",      lambda x: 4.2 / (x[:, 0] + 1e-9)),
        ("y = x1 + x2",    lambda x: x[:, 0] + x[:, 1]),
        ("y = 2*x1 - x2",  lambda x: 2.0 * x[:, 0] - x[:, 1]),
        ("y = x1*x2",      lambda x: x[:, 0] * x[:, 1]),
        ("y = x1/x2",      lambda x: x[:, 0] / (x[:, 1] + 1e-9)),
        ("y = x1 + x2 + x3", lambda x: x[:, 0] + x[:, 1] + x[:, 2]),
    ]

    for formula_str, fn in templates:
        n_feats = 3 if "x3" in formula_str else (2 if "x2" in formula_str else 1)
        X_tr = rng.uniform(1, 10, (n_train, n_feats)).astype(np.float32)
        X_te = rng.uniform(1, 10, (n_test,  n_feats)).astype(np.float32)
        y_tr = fn(X_tr.astype(np.float64))
        y_te = fn(X_te.astype(np.float64))
        y_tr = _add_noise(y_tr, noise, rng)
        feat_names = [f"x{i+1}" for i in range(n_feats)]
        dim_codes  = [0] * n_feats
        datasets.append(CurriculumDataset(
            X_train=X_tr, y_train=y_tr,
            X_test=X_te,  y_test=y_te,
            feat_names=feat_names, dim_codes=dim_codes,
            formula_str=formula_str, level=1,
        ))
    return datasets


def generate_level2(noise: float, rng: np.random.Generator) -> List[CurriculumDataset]:
    """Уровень 2 — степенные формулы."""
    datasets = []
    n_train, n_test = 150, 50

    templates = [
        ("y = x^2",             lambda x: x[:, 0]**2),
        ("y = 2*x^2",           lambda x: 2.0 * x[:, 0]**2),
        ("y = x^3",             lambda x: x[:, 0]**3),
        ("y = sqrt(x)",         lambda x: np.sqrt(np.abs(x[:, 0]))),
        ("y = 3*sqrt(x)",       lambda x: 3.0 * np.sqrt(np.abs(x[:, 0]))),
        ("y = x1^2 + x2",       lambda x: x[:, 0]**2 + x[:, 1]),
        ("y = sqrt(x1) * x2",   lambda x: np.sqrt(np.abs(x[:, 0])) * x[:, 1]),
        ("y = x1^2 / x2",       lambda x: x[:, 0]**2 / (x[:, 1] + 1e-9)),
        ("y = x1^2 + x2^2",     lambda x: x[:, 0]**2 + x[:, 1]**2),
        ("y = sqrt(x1^2+x2^2)", lambda x: np.sqrt(x[:, 0]**2 + x[:, 1]**2)),
    ]

    for formula_str, fn in templates:
        n_feats = 2 if "x2" in formula_str else 1
        X_tr = rng.uniform(0.5, 8, (n_train, n_feats)).astype(np.float32)
        X_te = rng.uniform(0.5, 8, (n_test,  n_feats)).astype(np.float32)
        y_tr = fn(X_tr.astype(np.float64))
        y_te = fn(X_te.astype(np.float64))
        y_tr = _add_noise(y_tr, noise, rng)
        feat_names = [f"x{i+1}" for i in range(n_feats)]
        dim_codes  = [0] * n_feats
        datasets.append(CurriculumDataset(
            X_train=X_tr, y_train=y_tr,
            X_test=X_te,  y_test=y_te,
            feat_names=feat_names, dim_codes=dim_codes,
            formula_str=formula_str, level=2,
        ))
    return datasets


def generate_level3(noise: float, rng: np.random.Generator) -> List[CurriculumDataset]:
    """Уровень 3 — комбинированные формулы."""
    datasets = []
    n_train, n_test = 200, 60

    templates = [
        ("y = x^2 + 2*x",       lambda x: x[:, 0]**2 + 2.0*x[:, 0]),
        ("y = x^2 - 3*x + 1",   lambda x: x[:, 0]**2 - 3.0*x[:, 0] + 1.0),
        ("y = 4/x^2",            lambda x: 4.0 / (x[:, 0]**2 + 1e-9)),
        ("y = sqrt(x) + 1/x",   lambda x: np.sqrt(np.abs(x[:, 0])) + 1.0/(x[:, 0]+1e-9)),
        ("y = 2*sqrt(x) + 3",   lambda x: 2.0*np.sqrt(np.abs(x[:, 0])) + 3.0),
        ("y = x1^2 / sqrt(x2)", lambda x: x[:, 0]**2 / (np.sqrt(np.abs(x[:, 1]))+1e-9)),
        ("y = x1*x2^2",         lambda x: x[:, 0] * x[:, 1]**2),
        ("y = x1^2 + x2^2 + x3", lambda x: x[:, 0]**2 + x[:, 1]**2 + x[:, 2]),
        ("y = (x1+x2)^2",       lambda x: (x[:, 0] + x[:, 1])**2),
        ("y = x1/(x2+x3)",      lambda x: x[:, 0] / (x[:, 1] + x[:, 2] + 1e-9)),
    ]

    for formula_str, fn in templates:
        n_feats = 3 if "x3" in formula_str else (2 if "x2" in formula_str else 1)
        X_tr = rng.uniform(0.5, 6, (n_train, n_feats)).astype(np.float32)
        X_te = rng.uniform(0.5, 6, (n_test,  n_feats)).astype(np.float32)
        y_tr = fn(X_tr.astype(np.float64))
        y_te = fn(X_te.astype(np.float64))
        y_tr = _add_noise(y_tr, noise, rng)
        feat_names = [f"x{i+1}" for i in range(n_feats)]
        dim_codes  = [0] * n_feats
        datasets.append(CurriculumDataset(
            X_train=X_tr, y_train=y_tr,
            X_test=X_te,  y_test=y_te,
            feat_names=feat_names, dim_codes=dim_codes,
            formula_str=formula_str, level=3,
        ))
    return datasets


def generate_level4(noise: float, rng: np.random.Generator) -> List[CurriculumDataset]:
    """
    Уровень 4 — реальные физические законы.
    dim_codes берутся из физического смысла переменных.
    """
    datasets = []
    n_train, n_test = 200, 60

    # Кеплер: T = a^1.5 (a — большая полуось, T — период)
    # dim: a→2(длина), T→8(время)
    a = rng.uniform(0.5, 10, n_train).astype(np.float64)
    T_tr = a**1.5
    T_tr = _add_noise(T_tr, noise, rng)
    a_te = rng.uniform(0.5, 10, n_test).astype(np.float64)
    T_te = a_te**1.5
    datasets.append(CurriculumDataset(
        X_train=a.reshape(-1,1).astype(np.float32),
        y_train=T_tr,
        X_test=a_te.reshape(-1,1).astype(np.float32),
        y_test=T_te,
        feat_names=["f0"], dim_codes=[2],
        formula_str="L4_001", level=4,
    ))

    # Кеплер 2: T = 2*a^1.5 (другая константа)
    a2 = rng.uniform(1, 15, n_train).astype(np.float64)
    T2_tr = 2.0 * a2**1.5
    T2_tr = _add_noise(T2_tr, noise, rng)
    a2_te = rng.uniform(1, 15, n_test).astype(np.float64)
    datasets.append(CurriculumDataset(
        X_train=a2.reshape(-1,1).astype(np.float32),
        y_train=T2_tr,
        X_test=a2_te.reshape(-1,1).astype(np.float32),
        y_test=2.0 * a2_te**1.5,
        feat_names=["f0"], dim_codes=[2],
        formula_str="L4_002", level=4,
    ))

    # Ньютон: F = m*a  (m — масса, a — ускорение)
    # dim: m→3(масса), a→6(ускорение), F→6(сила)
    m = rng.uniform(0.5, 20, n_train).astype(np.float64)
    acc = rng.uniform(0.1, 10, n_train).astype(np.float64)
    F_tr = m * acc
    F_tr = _add_noise(F_tr, noise, rng)
    m_te  = rng.uniform(0.5, 20, n_test).astype(np.float64)
    acc_te = rng.uniform(0.1, 10, n_test).astype(np.float64)
    X_newton_tr = np.column_stack([m, acc]).astype(np.float32)
    X_newton_te = np.column_stack([m_te, acc_te]).astype(np.float32)
    datasets.append(CurriculumDataset(
        X_train=X_newton_tr, y_train=F_tr,
        X_test=X_newton_te,  y_test=m_te*acc_te,
        feat_names=["f0", "f1"], dim_codes=[3, 6],
        formula_str="L4_003", level=4,
    ))

    # Маятник: T = 2*pi*sqrt(L/g)   (L — длина, g = 9.81)
    # dim: L→2(длина), T→8(время)
    L = rng.uniform(0.1, 5, n_train).astype(np.float64)
    g = 9.81
    T_pend_tr = 2 * np.pi * np.sqrt(L / g)
    T_pend_tr = _add_noise(T_pend_tr, noise, rng)
    L_te = rng.uniform(0.1, 5, n_test).astype(np.float64)
    datasets.append(CurriculumDataset(
        X_train=L.reshape(-1,1).astype(np.float32),
        y_train=T_pend_tr,
        X_test=L_te.reshape(-1,1).astype(np.float32),
        y_test=2*np.pi*np.sqrt(L_te/g),
        feat_names=["f0"], dim_codes=[2],
        formula_str="L4_004", level=4,
    ))

    # Газ: P*V = n*R*T → P = n*R*T/V  (n=1 моль, R=8.314)
    # dim: T→4(температура), V→5(объём), P→6(давление)
    R = 8.314
    T_gas = rng.uniform(200, 500, n_train).astype(np.float64)
    V_gas = rng.uniform(0.5, 5, n_train).astype(np.float64)
    P_gas_tr = R * T_gas / V_gas
    P_gas_tr = _add_noise(P_gas_tr, noise, rng)
    T_gas_te = rng.uniform(200, 500, n_test).astype(np.float64)
    V_gas_te = rng.uniform(0.5, 5, n_test).astype(np.float64)
    X_gas_tr = np.column_stack([T_gas, V_gas]).astype(np.float32)
    X_gas_te = np.column_stack([T_gas_te, V_gas_te]).astype(np.float32)
    datasets.append(CurriculumDataset(
        X_train=X_gas_tr, y_train=P_gas_tr,
        X_test=X_gas_te,  y_test=R*T_gas_te/V_gas_te,
        feat_names=["f0", "f1"], dim_codes=[4, 5],
        formula_str="L4_005", level=4,
    ))

    # Кулон: F = k/r^2  (k = 8.99e9, r — расстояние)
    r_coulomb = rng.uniform(0.1, 3, n_train).astype(np.float64)
    F_coulomb_tr = 1.0 / r_coulomb**2   # нормализованный
    F_coulomb_tr = _add_noise(F_coulomb_tr, noise, rng)
    r_te = rng.uniform(0.1, 3, n_test).astype(np.float64)
    datasets.append(CurriculumDataset(
        X_train=r_coulomb.reshape(-1,1).astype(np.float32),
        y_train=F_coulomb_tr,
        X_test=r_te.reshape(-1,1).astype(np.float32),
        y_test=1.0/r_te**2,
        feat_names=["f0"], dim_codes=[2],
        formula_str="L4_006", level=4,
    ))

    # Гравитация: g = G*M/r^2 → нормализованная
    r_grav = rng.uniform(1, 10, n_train).astype(np.float64)
    g_grav_tr = 100.0 / r_grav**2
    g_grav_tr = _add_noise(g_grav_tr, noise, rng)
    r_grav_te = rng.uniform(1, 10, n_test).astype(np.float64)
    datasets.append(CurriculumDataset(
        X_train=r_grav.reshape(-1,1).astype(np.float32),
        y_train=g_grav_tr,
        X_test=r_grav_te.reshape(-1,1).astype(np.float32),
        y_test=100.0/r_grav_te**2,
        feat_names=["f0"], dim_codes=[2],
        formula_str="L4_007", level=4,
    ))

    # Снаряд: y = v*t - 0.5*g*t^2
    v0 = rng.uniform(5, 30, n_train).astype(np.float64)
    t  = rng.uniform(0.1, 3, n_train).astype(np.float64)
    proj_tr = v0 * t - 0.5 * 9.81 * t**2
    proj_tr = _add_noise(proj_tr, noise, rng)
    v0_te = rng.uniform(5, 30, n_test).astype(np.float64)
    t_te  = rng.uniform(0.1, 3, n_test).astype(np.float64)
    X_proj_tr = np.column_stack([v0, t]).astype(np.float32)
    X_proj_te = np.column_stack([v0_te, t_te]).astype(np.float32)
    datasets.append(CurriculumDataset(
        X_train=X_proj_tr, y_train=proj_tr,
        X_test=X_proj_te,  y_test=v0_te*t_te - 0.5*9.81*t_te**2,
        feat_names=["f0", "f1"], dim_codes=[6, 8],
        formula_str="L4_008", level=4,
    ))

    # Закон Хука: F = k*x  (k = 5.0)
    x_hook = rng.uniform(0.1, 10, n_train).astype(np.float64)
    F_hook_tr = 5.0 * x_hook
    F_hook_tr = _add_noise(F_hook_tr, noise, rng)
    x_hook_te = rng.uniform(0.1, 10, n_test).astype(np.float64)
    datasets.append(CurriculumDataset(
        X_train=x_hook.reshape(-1,1).astype(np.float32),
        y_train=F_hook_tr,
        X_test=x_hook_te.reshape(-1,1).astype(np.float32),
        y_test=5.0*x_hook_te,
        feat_names=["f0"], dim_codes=[2],
        formula_str="L4_009", level=4,
    ))

    # Кинетическая энергия: E = 0.5*m*v^2
    m_ke = rng.uniform(0.5, 20, n_train).astype(np.float64)
    v_ke = rng.uniform(0.5, 15, n_train).astype(np.float64)
    E_ke_tr = 0.5 * m_ke * v_ke**2
    E_ke_tr = _add_noise(E_ke_tr, noise, rng)
    m_ke_te = rng.uniform(0.5, 20, n_test).astype(np.float64)
    v_ke_te = rng.uniform(0.5, 15, n_test).astype(np.float64)
    X_ke_tr = np.column_stack([m_ke, v_ke]).astype(np.float32)
    X_ke_te = np.column_stack([m_ke_te, v_ke_te]).astype(np.float32)
    datasets.append(CurriculumDataset(
        X_train=X_ke_tr, y_train=E_ke_tr,
        X_test=X_ke_te,  y_test=0.5*m_ke_te*v_ke_te**2,
        feat_names=["f0", "f1"], dim_codes=[3, 6],
        formula_str="L4_010", level=4,
    ))

    return datasets[:10]  # ровно 10


def generate_level5(noise: float, rng: np.random.Generator) -> List[CurriculumDataset]:
    """
    Уровень 5 — Грязные реальные данные.

    ЧТО ОТЛИЧАЕТ ЭТОТ УРОВЕНЬ от предыдущих:
      1. МНОГО ПРИЗНАКОВ (8-15) — большинство ненужные «шумовые»
         Система учится отбирать 2-3 важных из 10+
         Navigator не должен хватать первые попавшиеся

      2. КОРРЕЛИРУЮЩИЕ ПРИЗНАКИ — некоторые x_i линейно связаны
         Мультиколлинеарность: PySR может найти x3 вместо x1
         если x3 = 0.9*x1 + шум
         Физик должен уметь это замечать

      3. РАЗНЫЕ ДИАПАЗОНЫ TRAIN И TEST (OOD)
         Train: x в [1, 5], Test: x в [6, 10]
         Система должна находить формулу которая ЭКСТРАПОЛИРУЕТ
         R² на OOD — главная метрика реального качества

      4. ВЫСОКИЙ ШУМ (15-25%) — реальные полевые измерения
         Данные из реального эксперимента, не из учебника
         Система не должна переобучаться под шум

      5. АДДИТИВНЫЕ ПОМЕХИ — слагаемое от нерелевантного признака
         y = f(x1, x2) + 0.15 * x7
         Скептик должен замечать: «это слагаемое — артефакт?»

      6. СМЕШАННЫЕ ЕДИНИЦЫ / МАСШТАБЫ — признаки в разных порядках
         x1 ~ [0.001, 0.01], x2 ~ [1000, 10000]
         Surgery и Diffusion должны правильно обрабатывать

    ЧТО СИСТЕМА ДОЛЖНА ВЫНЕСТИ:
      - Navigator: не берёт все признаки подряд, отбирает осторожно
      - Скептик: «высокий R² на train при шуме 20% — подозрительно»
      - Физик: «x3 и x1 коррелируют — формула нестабильна»
      - Матрёшка в целом: R²=0.65-0.72 на грязных данных = хорошо
        (до этого она видела только R²>0.80 и считала это нормой)
    """
    datasets = []
    n_train, n_test = 200, 80
    noise_hi = noise        # 20% — базовый шум уровня
    noise_lo = noise * 0.6  # 12% — для более стабильных датасетов

    # ── 1. Закон с шумовыми признаками (8 признаков, 2 важных) ──
    # y = x1 / x2, остальные 6 — шум
    X_tr = rng.uniform(1, 8, (n_train, 8)).astype(np.float64)
    X_te = rng.uniform(1, 8, (n_test,  8)).astype(np.float64)
    y_tr = X_tr[:, 0] / X_tr[:, 1]
    y_te = X_te[:, 0] / X_te[:, 1]
    y_tr = y_tr + rng.normal(0, noise_lo * np.std(y_tr), n_train)
    datasets.append(CurriculumDataset(
        X_train=X_tr.astype(np.float32), y_train=y_tr,
        X_test=X_te.astype(np.float32),  y_test=y_te,
        feat_names=[f"x{i+1}" for i in range(8)],
        dim_codes=[0]*8,
        formula_str="L5_001", level=5,
    ))

    # ── 2. Степенной закон + коррелирующие признаки ──────────────
    # x3 = 0.85*x1 + noise → система может перепутать x1 и x3
    x1 = rng.uniform(0.5, 6, n_train).astype(np.float64)
    x2 = rng.uniform(0.5, 6, n_train).astype(np.float64)
    x3 = 0.85 * x1 + rng.normal(0, 0.3, n_train)  # почти копия x1
    x4 = rng.uniform(0.5, 6, n_train).astype(np.float64)
    X_tr = np.column_stack([x1, x2, x3, x4]).astype(np.float32)
    y_tr = x1**2 / x2
    y_tr = y_tr + rng.normal(0, noise_lo * np.std(y_tr), n_train)
    x1t = rng.uniform(0.5, 6, n_test).astype(np.float64)
    x2t = rng.uniform(0.5, 6, n_test).astype(np.float64)
    X_te = np.column_stack([x1t, x2t,
                             0.85*x1t + rng.normal(0, 0.3, n_test),
                             rng.uniform(0.5, 6, n_test)]).astype(np.float32)
    datasets.append(CurriculumDataset(
        X_train=X_tr, y_train=y_tr,
        X_test=X_te,  y_test=x1t**2/x2t,
        feat_names=["x1","x2","x3_corr","x4"],
        dim_codes=[0, 0, 0, 0],
        formula_str="L5_002", level=5,
    ))

    # ── 3. OOD — train [1,4], test [5,10] ────────────────────────
    # Экстраполяция: знает ли система формулу или просто интерполирует?
    x_tr = rng.uniform(1, 4, n_train).astype(np.float64)
    x_te = rng.uniform(5, 10, n_test).astype(np.float64)
    y_tr_ = 3.0 * x_tr**1.5
    y_te_ = 3.0 * x_te**1.5
    y_tr_ = y_tr_ + rng.normal(0, noise_hi * np.std(y_tr_), n_train)
    datasets.append(CurriculumDataset(
        X_train=x_tr.reshape(-1,1).astype(np.float32), y_train=y_tr_,
        X_test=x_te.reshape(-1,1).astype(np.float32),  y_test=y_te_,
        feat_names=["x"], dim_codes=[0],
        formula_str="L5_003", level=5,
    ))

    # ── 4. Аддитивная помеха от нерелевантного признака ──────────
    # y = sqrt(x1)*x2 + 0.15*x5 — x5 «просачивается» в формулу
    x1 = rng.uniform(1, 9, n_train).astype(np.float64)
    x2 = rng.uniform(1, 9, n_train).astype(np.float64)
    x5 = rng.uniform(0, 5, n_train).astype(np.float64)
    extras = rng.uniform(1, 5, (n_train, 3)).astype(np.float64)
    y_clean = np.sqrt(x1) * x2
    y_dirty = y_clean + 0.15 * x5 + rng.normal(0, noise_hi * np.std(y_clean), n_train)
    X_tr = np.column_stack([x1, x2, extras[:, 0], extras[:, 1], x5]).astype(np.float32)
    x1t  = rng.uniform(1, 9, n_test).astype(np.float64)
    x2t  = rng.uniform(1, 9, n_test).astype(np.float64)
    X_te = np.column_stack([x1t, x2t,
                             rng.uniform(1,5,n_test),
                             rng.uniform(1,5,n_test),
                             rng.uniform(0,5,n_test)]).astype(np.float32)
    datasets.append(CurriculumDataset(
        X_train=X_tr, y_train=y_dirty,
        X_test=X_te,  y_test=np.sqrt(x1t)*x2t,
        feat_names=["x1","x2","x3","x4","x5_noise"],
        dim_codes=[0]*5,
        formula_str="L5_004", level=5,
    ))

    # ── 5. Разные масштабы признаков ──────────────────────────────
    # x1 ~ [0.001, 0.01], x2 ~ [1000, 5000] — Diffusion/Surgery тест
    x1 = rng.uniform(0.001, 0.01, n_train).astype(np.float64)
    x2 = rng.uniform(1000,  5000, n_train).astype(np.float64)
    y_tr_ = x2 * x1**2  # y = x2 * x1^2
    y_tr_ = y_tr_ + rng.normal(0, noise_lo * np.std(y_tr_), n_train)
    x1t = rng.uniform(0.001, 0.01, n_test).astype(np.float64)
    x2t = rng.uniform(1000,  5000, n_test).astype(np.float64)
    datasets.append(CurriculumDataset(
        X_train=np.column_stack([x1, x2]).astype(np.float32), y_train=y_tr_,
        X_test=np.column_stack([x1t, x2t]).astype(np.float32), y_test=x2t*x1t**2,
        feat_names=["micro","macro"],
        dim_codes=[0, 0],
        formula_str="L5_005", level=5,
    ))

    # ── 6. Реальный закон + высокий шум + OOD ────────────────────
    # Идеальный газ P=RT/V, шум 25%, test в другом диапазоне T
    R = 8.314
    T_tr = rng.uniform(200, 400, n_train).astype(np.float64)
    V_tr = rng.uniform(1, 4, n_train).astype(np.float64)
    P_tr = R * T_tr / V_tr
    P_tr = P_tr + rng.normal(0, 0.25 * np.std(P_tr), n_train)  # 25% шум
    T_te = rng.uniform(400, 600, n_test).astype(np.float64)    # OOD температура
    V_te = rng.uniform(1, 4, n_test).astype(np.float64)
    datasets.append(CurriculumDataset(
        X_train=np.column_stack([T_tr, V_tr]).astype(np.float32), y_train=P_tr,
        X_test=np.column_stack([T_te, V_te]).astype(np.float32),  y_test=R*T_te/V_te,
        feat_names=["f0", "f1"], dim_codes=[4, 5],
        formula_str="L5_006", level=5,
    ))

    # ── 7. Кеплер с 10 признаками, один важный ───────────────────
    X_all = rng.uniform(0.5, 10, (n_train, 10)).astype(np.float64)
    y_tr_ = X_all[:, 2]**1.5   # только x3 важен, остальные 9 — шум
    y_tr_ = y_tr_ + rng.normal(0, noise_hi * np.std(y_tr_), n_train)
    X_all_te = rng.uniform(0.5, 10, (n_test, 10)).astype(np.float64)
    datasets.append(CurriculumDataset(
        X_train=X_all.astype(np.float32), y_train=y_tr_,
        X_test=X_all_te.astype(np.float32), y_test=X_all_te[:, 2]**1.5,
        feat_names=[f"x{i+1}" for i in range(10)],
        dim_codes=[0]*10,
        formula_str="L5_007", level=5,
    ))

    # ── 8. Нелинейная зависимость со смещением ───────────────────
    # y = (x1 + x2)^2 / (x3 + 2) — сложный знаменатель
    x1 = rng.uniform(0.5, 4, n_train).astype(np.float64)
    x2 = rng.uniform(0.5, 4, n_train).astype(np.float64)
    x3 = rng.uniform(0.5, 4, n_train).astype(np.float64)
    x4 = rng.uniform(0.5, 4, n_train).astype(np.float64)  # шум
    y_tr_ = (x1 + x2)**2 / (x3 + 2.0)
    y_tr_ = y_tr_ + rng.normal(0, noise_lo * np.std(y_tr_), n_train)
    x1t = rng.uniform(0.5, 4, n_test).astype(np.float64)
    x2t = rng.uniform(0.5, 4, n_test).astype(np.float64)
    x3t = rng.uniform(0.5, 4, n_test).astype(np.float64)
    datasets.append(CurriculumDataset(
        X_train=np.column_stack([x1,x2,x3,x4]).astype(np.float32), y_train=y_tr_,
        X_test=np.column_stack([x1t,x2t,x3t,rng.uniform(0.5,4,n_test)]).astype(np.float32),
        y_test=(x1t+x2t)**2/(x3t+2.0),
        feat_names=["x1","x2","x3","x4_noise"], dim_codes=[0]*4,
        formula_str="L5_008", level=5,
    ))

    # ── 9. Логистическое насыщение ────────────────────────────────
    # y = Vmax * x / (Km + x) — Михаэлис-Ментен с шумом и помехами
    Vmax, Km = 10.0, 2.0
    x1 = rng.uniform(0.1, 15, n_train).astype(np.float64)
    x2 = rng.uniform(0.1, 10, n_train).astype(np.float64)  # шум
    y_tr_ = Vmax * x1 / (Km + x1)
    y_tr_ = y_tr_ + rng.normal(0, noise_hi * np.std(y_tr_), n_train)
    x1t = rng.uniform(0.1, 15, n_test).astype(np.float64)
    datasets.append(CurriculumDataset(
        X_train=np.column_stack([x1, x2]).astype(np.float32), y_train=y_tr_,
        X_test=np.column_stack([x1t, rng.uniform(0.1,10,n_test)]).astype(np.float32),
        y_test=Vmax*x1t/(Km+x1t),
        feat_names=["substrate","noise_feat"], dim_codes=[0, 0],
        formula_str="L5_009", level=5,
    ))

    # ── 10. Гравитация OOD + коррелирующие ───────────────────────
    # g = GM/r² — обратный квадрат, test в экстремальных r
    r_tr = rng.uniform(1, 5, n_train).astype(np.float64)
    r_corr = r_tr * 0.9 + rng.normal(0, 0.2, n_train)     # почти копия r
    y_tr_ = 100.0 / r_tr**2
    y_tr_ = y_tr_ + rng.normal(0, noise_hi * np.std(y_tr_), n_train)
    r_te = rng.uniform(6, 15, n_test).astype(np.float64)  # OOD далёкие расстояния
    datasets.append(CurriculumDataset(
        X_train=np.column_stack([r_tr, r_corr]).astype(np.float32), y_train=y_tr_,
        X_test=np.column_stack([r_te, r_te*0.9]).astype(np.float32),
        y_test=100.0/r_te**2,
        feat_names=["radius","radius_corr"], dim_codes=[2, 2],
        formula_str="L5_010", level=5,
    ))

    return datasets[:10]


GENERATORS = {
    1: generate_level1,
    2: generate_level2,
    3: generate_level3,
    4: generate_level4,
    5: generate_level5,
}


# ══════════════════════════════════════════════════════════════════
# РЕЗУЛЬТАТ ОДНОГО ДАТАСЕТА
# ══════════════════════════════════════════════════════════════════

@dataclass
class DatasetResult:
    formula_true:  str
    formula_found: str
    r2_blind:      float
    passed:        bool
    level:         int
    noise_level:   float
    maxsize:       int
    fast_fail_sec: int
    domain:        str
    elapsed_sec:   float = 0.0
    error:         str   = ""


# ══════════════════════════════════════════════════════════════════
# ЯДРО CURRICULUM — ОДИН ДАТАСЕТ
# ══════════════════════════════════════════════════════════════════

def run_curriculum_dataset(
    ds:      CurriculumDataset,
    level:   CurriculumLevel,
    model:   str = OLLAMA_MODEL,
    host:    str = OLLAMA_HOST,
    depth:   int = 0,   # v10.14 БАГ 7: глубина дерева вариаций
) -> DatasetResult:
    """
    Полный цикл pysr → llm для одного curriculum датасета.
    Управляет фазами автоматически — пользователь не нужен.
    """
    from .engine import run_engine, ollama_stop
    from .config import PHASE_RESULT_PATH

    t_start = time.time()
    formula_found = ""
    r2_blind      = 0.0
    error_msg     = ""

    print(f"\n  {'─'*58}")
    print(f"  Датасет: {ds.formula_str}")
    print(f"  Уровень {level.level} | maxsize={level.maxsize} | "
          f"noise={level.noise_level:.0%} | timeout={level.fast_fail_sec}s")
    print(f"  {'─'*58}")

    result_pysr = None   # v10.14: инициализируем до try, иначе NameError при падении PySR
    result_llm  = None
    try:
        # ── Фаза 1: PySR ──────────────────────────────────────────
        print(f"  [Curriculum] → phase=pysr")
        result_pysr = run_engine(
            X_train     = ds.X_train,
            y_train     = ds.y_train,
            X_test      = ds.X_test,
            y_test      = ds.y_test,
            feat_names  = ds.feat_names,
            target_col  = "y",
            timeout_sec = level.fast_fail_sec,
            domain_type = level.domain,
            model       = model,
            host        = host,
            phase       = "pysr",
            dim_codes   = ds.dim_codes,  # v10.14: без этого engine спросит пользователя интерактивно
            skip_heritage = True,
            noise_hint    = level.noise_level,  # FIX: передаём реальный шум чтобы не переоценивало
        )

        # Небольшая пауза чтобы Julia освободила память
        ollama_stop(model)
        gc.collect()
        time.sleep(2)

        # ── Фаза 2: LLM ───────────────────────────────────────────
        print(f"  [Curriculum] → phase=llm")
        from .engine import run_llm_phase
        result_llm = run_llm_phase(model=model, host=host)

        formula_found = (
            (result_llm.formula_shadow  if result_llm  else "") or
            (result_pysr.formula_shadow if result_pysr else "") or
            ""
        )
        r2_blind      = result_llm.r2_blind

        ollama_stop(model)
        gc.collect()
        time.sleep(1)

    except Exception as e:
        error_msg = str(e)[:200]
        log.warning("[Curriculum] Ошибка датасета '%s': %s", ds.formula_str, error_msg)
        # Пытаемся освободить ресурсы
        try:
            from .engine import ollama_stop as _stop
            _stop(model)
        except Exception:
            pass
        gc.collect()

    elapsed = time.time() - t_start
    passed  = r2_blind >= level.r2_threshold and not error_msg

    status_icon = "✓" if passed else "✗"
    print(f"  [Curriculum] {status_icon} R²_blind={r2_blind:.4f} "
          f"(порог {level.r2_threshold}) t={elapsed:.0f}s")
    if error_msg:
        print(f"  [Curriculum] ⚠ Ошибка: {error_msg[:80]}")

    # Записываем в episodic_memory
    try:
        get_memory().remember_curriculum(
            level         = level.level,
            formula_true  = ds.formula_str,
            formula_found = formula_found,
            r2_blind      = r2_blind,
            noise_level   = level.noise_level,
            maxsize       = level.maxsize,
            fast_fail_sec = level.fast_fail_sec,
            domain        = level.domain,
            passed        = passed,
            depth         = depth,   # v10.14 БАГ 7
        )
    except Exception as _mem_err:
        log.debug("[Curriculum] Ошибка памяти: %s", _mem_err)

    return DatasetResult(
        formula_true  = ds.formula_str,
        formula_found = formula_found,
        r2_blind      = r2_blind,
        passed        = passed,
        level         = level.level,
        noise_level   = level.noise_level,
        maxsize       = level.maxsize,
        fast_fail_sec = level.fast_fail_sec,
        domain        = level.domain,
        elapsed_sec   = elapsed,
        error         = error_msg,
    )


# ══════════════════════════════════════════════════════════════════
# ОДИН УРОВЕНЬ
# ══════════════════════════════════════════════════════════════════

@dataclass
class LevelResult:
    level:       int
    name:        str
    mean_r2:     float
    passed:      bool
    attempt:     int
    results:     List[DatasetResult] = field(default_factory=list)
    elapsed_sec: float = 0.0


def run_level(
    level:   CurriculumLevel,
    model:   str = OLLAMA_MODEL,
    host:    str = OLLAMA_HOST,
    seed:    int = 42,
) -> LevelResult:
    """
    Запускает один уровень curriculum.
    Если средний R² < порог — повторяет (max 3 попытки).
    """
    print(f"\n{'═'*62}")
    print(f"  CURRICULUM LEVEL {level.level} — {level.name}")
    print(f"  {level.datasets} датасетов | "
          f"шум={level.noise_level:.0%} | "
          f"maxsize={level.maxsize} | "
          f"timeout={level.fast_fail_sec}s")
    print(f"  Порог R²: {level.r2_threshold}")
    print(f"{'═'*62}")

    generator = GENERATORS[level.level]
    best_result: Optional[LevelResult] = None

    # ── Resume: проверяем checkpoint ────────────────────────────
    _cp = _load_checkpoint()
    _resume_attempt = _cp["attempt"] if _cp and _cp["level"] == level.level else 1
    _resume_dataset = _cp["dataset"] if _cp and _cp["level"] == level.level else 0
    _resume_r2      = _cp.get("r2_history", []) if _cp and _cp["level"] == level.level else []
    if _resume_dataset > 0:
        print(f"\n  [Resume] ▶ Продолжаем с датасета {_resume_dataset + 1}/"
              f"{level.datasets} (попытка {_resume_attempt})")
        print(
            f"  [Resume] Уже выполнено: {_resume_dataset} датасетов, "
            f"R²_history={[round(x, 3) for x in _resume_r2]}"
        )

    for attempt in range(1, level.max_retries + 1):
        # При resume начинаем с сохранённой попытки; при новых — с 1
        if attempt < _resume_attempt:
            print(f"\n  [Resume] Пропускаем попытку {attempt} (уже выполнена)")
            continue

        print(f"\n  [Уровень {level.level}] Попытка {attempt}/{level.max_retries}")
        t_start  = time.time()
        rng      = np.random.default_rng(seed + attempt * 1000)
        datasets = generator(level.noise_level, rng)
        results: List[DatasetResult] = []

        # Восстанавливаем R²-историю из checkpoint если та же попытка
        if attempt == _resume_attempt and _resume_r2:
            print(f"  [Resume] Восстанавливаем {len(_resume_r2)} результатов из checkpoint")
            # Создаём заглушки для уже выполненных датасетов
            for _old_r2 in _resume_r2:
                results.append(DatasetResult(
                    formula_true  = "(resume — пропущен)",
                    formula_found = "(resume — из checkpoint)",
                    r2_blind      = _old_r2,
                    passed        = _old_r2 >= level.r2_threshold,
                    level         = level.level,
                    noise_level   = level.noise_level,
                    maxsize       = level.maxsize,
                    fast_fail_sec = level.fast_fail_sec,
                    domain        = level.domain,
                    elapsed_sec   = 0.0,   # Дыра 10: время не учитывается для пропущенных
                    error         = "",
                ))

        # v10.14: start_from = только для resume попытки
        start_from = _resume_dataset if attempt == _resume_attempt else 0
        # БАГ D исправлен: сброс ПОСЛЕ установки start_from, не ДО
        # и условие не "not results" (заглушки уже добавлены) — а просто факт попытки
        _is_resume_attempt = (attempt == _resume_attempt)

        for i, ds in enumerate(datasets, 1):
            if i <= start_from:
                print(f"  [Resume] Пропускаем датасет {i}/{level.datasets} (из checkpoint)")
                continue
            print(f"\n  [{i}/{level.datasets}] {ds.formula_str}")
            res = run_curriculum_dataset(ds, level, model, host)
            results.append(res)
            # ── Checkpoint после каждого датасета ─────────────────
            _r2_hist = [r.r2_blind for r in results if not r.error]
            _elapsed_so_far = (
                _cp.get("elapsed_sec", 0.0) if _cp and _cp.get("level") == level.level
                and attempt == _resume_attempt else 0.0
            ) + (time.time() - t_start)
            _save_checkpoint(level.level, attempt, i, _r2_hist, elapsed_sec=_elapsed_so_far)

        # БАГ D: после завершения датасетного цикла сбрасываем resume-состояние
        # Следующая попытка (attempt+1) должна начать с датасета 0, не с checkpoint
        if _is_resume_attempt:
            _resume_attempt = attempt + 1  # следующая попытка не будет resume
            _resume_dataset = 0
            _resume_r2      = []

        # Средний R² по всем датасетам (пропускаем ошибки и заглушки)
        valid_r2 = [r.r2_blind for r in results
                    if not r.error and r.formula_true != "(resume — пропущен)"]
        mean_r2  = float(np.mean(valid_r2)) if valid_r2 else 0.0
        passed   = mean_r2 >= level.r2_threshold
        elapsed  = time.time() - t_start

        lr = LevelResult(
            level       = level.level,
            name        = level.name,
            mean_r2     = mean_r2,
            passed      = passed,
            attempt     = attempt,
            results     = results,
            elapsed_sec = elapsed,
        )

        icon = "✓" if passed else "✗"
        print(f"\n  [Уровень {level.level}] {icon} "
              f"Средний R²={mean_r2:.4f} (порог {level.r2_threshold}) "
              f"t={elapsed/60:.1f} мин")

        # Сохраняем лучший результат
        if best_result is None or mean_r2 > best_result.mean_r2:
            best_result = lr

        if passed:
            # Компилируем DSPy на накопленных примерах
            _recompile_dspy(model, host)
            _clear_checkpoint()   # уровень пройден — чистим прогресс
            return lr

        if attempt < level.max_retries:
            print(f"  [Уровень {level.level}] Повторяем с новыми данными…")

    # Исчерпали попытки — возвращаем лучший
    print(f"\n  [Уровень {level.level}] ⚠ Не прошёл после {level.max_retries} "
          f"попыток. Лучший R²={best_result.mean_r2:.4f}. Идём дальше.")
    _recompile_dspy(model, host)
    _clear_checkpoint()   # уровень завершён (не прошёл) — чистим checkpoint
    return best_result


def _recompile_dspy(model: str, host: str) -> None:
    """Перекомпилирует DSPy на накопленных примерах после уровня."""
    try:
        from .dspy_optimizer import DSPyOrchestrator
        orch = DSPyOrchestrator(model=model, host=host)
        ok   = orch.siege_compile()
        if ok:
            print(f"  [Curriculum] ✓ DSPy перекомпилирован на новых примерах")
        else:
            print(f"  [Curriculum] ○ DSPy: нет новых примеров для компиляции")
        from .engine import ollama_stop
        ollama_stop(model)
        gc.collect()
    except Exception as e:
        log.warning("[Curriculum] DSPy перекомпиляция: %s", e)


# ══════════════════════════════════════════════════════════════════
# ГЛАВНАЯ ФУНКЦИЯ
# ══════════════════════════════════════════════════════════════════

def run_curriculum(
    level_filter: Optional[int] = None,
    start_level:  Optional[int] = None,
    end_level:    Optional[int] = None,
    model:        str = OLLAMA_MODEL,
    host:         str = OLLAMA_HOST,
    seed:         int = 42,
) -> List[LevelResult]:
    """
    Запускает curriculum learning.

    Режимы запуска:
      --curriculum                  → уровни DEFAULT_START_LEVEL..5
      --curriculum --level 4        → только уровень 4
      --curriculum --start 1        → уровни 1..5 (полный прогон)
      --curriculum --start 4 --end 5 → уровни 4 и 5
      --curriculum --start 1 --end 3 → уровни 1, 2, 3

    DEFAULT_START_LEVEL = 4 (задан в начале файла).
    Меняй его если хочешь начинать с другого уровня по умолчанию.

    ПОЧЕМУ УРОВНИ 1-3 ТОЖЕ ПОЛЕЗНЫ — важный нюанс:
      Уровни 1-3 учат DSPy Navigator конкретным парам
      (условие → правильный ответ). Это few-shot supervision signal.
      Navigator видит: «при noise=1%, maxsize=8, линейные данные
      → правильно пробовать x1+x2, x1*x2, x1/x2».
      Это ценно даже если формулы простые — потому что система учится
      СТРУКТУРЕ рассуждения, а не конкретным числам.
      Уровни 4-5 — самые важные для физики, но 1-3 не бесполезны.
    """
    # Определяем какие уровни запускать
    if level_filter is not None:
        levels_to_run = [l for l in LEVELS if l.level == level_filter]
    else:
        _start = start_level if start_level is not None else DEFAULT_START_LEVEL
        _end   = end_level   if end_level   is not None else max(l.level for l in LEVELS)
        levels_to_run = [l for l in LEVELS if _start <= l.level <= _end]

    print(f"\n{'═'*62}")
    print(f"  SCALPEL CURRICULUM LEARNING")
    if level_filter:
        print(f"  Уровень {level_filter} из {len(LEVELS)}")
    else:
        _start_l = levels_to_run[0].level  if levels_to_run else DEFAULT_START_LEVEL
        _end_l   = levels_to_run[-1].level if levels_to_run else DEFAULT_START_LEVEL
        print(f"  Уровни {_start_l}–{_end_l} → {sum(l.datasets for l in levels_to_run)} датасетов")
        if _start_l > 1:
            print(f"  (уровни 1–{_start_l-1} пропущены — DEFAULT_START_LEVEL={DEFAULT_START_LEVEL})")
    _est = {1: 50, 2: 100, 3: 150, 4: 300, 5: 400}
    total_est = sum(_est.get(l.level, 60) for l in levels_to_run)
    print(f"  Примерное время: ~{total_est//60}ч {total_est%60}мин")
    print(f"{'═'*62}")

    # v10.14: ДЫРА 8 — проверяем что Ollama живая перед 50-часовым прогоном
    try:
        import urllib.request as _ureq
        _req = _ureq.Request(f"{host.rstrip('/')}/api/tags")
        with _ureq.urlopen(_req, timeout=5):
            print(f"  [Curriculum] ✓ Ollama живая ({host})")
    except Exception as _ollama_err:
        print(f"  [Curriculum] ✗ ОШИБКА: Ollama не отвечает на {host}")
        print(f"  [Curriculum]   Запусти: ollama serve")
        print(f"  [Curriculum]   Ошибка: {_ollama_err}")
        raise RuntimeError(
            f"[Curriculum] Ollama недоступна. Запусти 'ollama serve' и повтори."
        )

    # Проверяем checkpoint при старте
    _cp_at_start = _load_checkpoint()
    if _cp_at_start:
        print(f"  [Curriculum] ▶ Найден checkpoint: уровень={_cp_at_start.get('level')}, "
              f"датасет={_cp_at_start.get('dataset')}, "
              f"время={_cp_at_start.get('elapsed_sec', 0)/3600:.1f}ч")
    else:
        print(f"  [Curriculum] ○ Checkpoint не найден — начинаем с начала")

    t_total = time.time()
    all_results: List[LevelResult] = []

    for lv in levels_to_run:
        lr = run_level(lv, model=model, host=host, seed=seed)
        all_results.append(lr)
        _save_curriculum_log(lr)

        # v10.14: ЛЕТОПИСЕЦ — финальная запись после каждого уровня
        try:
            from .episodic_memory import get_memory as _lv_chr_mem
            # Берём лучший R² по оригинальным (не resume) датасетам
            _lv_best_r2 = lr.mean_r2
            _lv_formula = next(
                (r.formula_found for r in lr.results
                 if r.formula_found and r.formula_found != "(resume)"),
                "(не найдено)",
            )
            _lv_chr_mem().remember_chronicle_final(
                level          = lr.level,
                formula_final  = _lv_formula,
                r2_blind       = _lv_best_r2,
                total_attempts = len([r for r in lr.results if r.formula_found != "(resume)"]),
                chronicle_text = (
                    f"Curriculum уровень {lr.level} ({lr.name}): "
                    f"средний R²={lr.mean_r2:.4f}, "
                    f"прошёл={'да' if lr.passed else 'нет'}, "
                    f"попытка {lr.attempt}/{lv.max_retries}"
                ),
                domain         = lv.domain,
                passed         = lr.passed,
            )
            log.info("[Летописец] Финал уровня %d записан (R²=%.4f, passed=%s)",
                     lr.level, lr.mean_r2, lr.passed)
        except Exception as _lv_chr_err:
            log.debug("[Летописец/Level] %s", _lv_chr_err)

    total_elapsed = time.time() - t_total

    # Итоговый отчёт
    print(f"\n{'═'*62}")
    print(f"  CURRICULUM ЗАВЕРШЁН — {total_elapsed/3600:.1f}ч")
    print(f"{'═'*62}")
    for lr in all_results:
        icon = "✓" if lr.passed else "✗"
        print(f"  {icon} Уровень {lr.level} ({lr.name}): "
              f"R²={lr.mean_r2:.4f} | "
              f"попытка {lr.attempt} | "
              f"{lr.elapsed_sec/60:.1f} мин")
    print(f"{'═'*62}")

    return all_results


# ══════════════════════════════════════════════════════════════════
# СОХРАНЕНИЕ ЛОГА
# ══════════════════════════════════════════════════════════════════

def _save_curriculum_log(lr: LevelResult) -> None:
    """Сохраняет итог уровня в curriculum_log.jsonl."""
    log_path = SCRIPT_DIR / "scalpel_vault" / "curriculum_log.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "ts":          datetime.now().isoformat(),
        "level":       lr.level,
        "name":        lr.name,
        "mean_r2":     round(lr.mean_r2, 4),
        "passed":      lr.passed,
        "attempt":     lr.attempt,
        "elapsed_min": round(lr.elapsed_sec / 60, 1),
        "datasets": [
            {
                "formula_true":  r.formula_true,
                "formula_found": r.formula_found,
                "r2_blind":      round(r.r2_blind, 4),
                "passed":        r.passed,
                "elapsed_sec":   round(r.elapsed_sec, 1),
                "error":         r.error,
            }
            for r in lr.results
        ],
    }
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    log.info("[Curriculum] Уровень %d сохранён → %s", lr.level, log_path)
    print(f"  [Curriculum] ✓ Лог → {log_path}")
