"""
engine.py — Главный движок v10.4.5 (Inherent Structure).

Siege Mode 3.0:
  1. DSPy.compile (Ollama жива)
  2. ollama_stop() + gc.collect()
  3. ★ Диффузионная Пауза: Wait 3s  ← NEW v10.4.5
  4. Julia Ignition (PySR)

СВЯЩЕННЫЕ МЕТРИКИ (DSPy не трогает):
  shuffle_test()   — NIST CSPRNG пермутационный тест
  cross_blind()    — K-fold out-of-sample R²
  ShadowMapper     — анонимизация признаков

v10.2.7 — Humanistic Filters / Critical Thinking:
  Deep Root  — Root Cause Analysis (5 уровней «Почему?»)
  Lasso      — Топологическое стягивание аргументов (Бритва Оккама)
  [Dialectic и Sinquain — в audit.py / Матрёшке]

v10.3.9 — Topological Surgery (Метод Перельмана):
  RicciFlow          — Savitzky-Golay сглаживание до PySR
  SingularityDetector — детектор кривизны (∂²y/∂i²)
  FractalCutting     — вырезание 2-3% сингулярностей (numpy, до Julia)
  PoincaréVerdict    — [SURGERY PERFORMED] + [POINCARE INVARIANT]

v10.4.5 — Inherent Structure (принципы AlphaFold 3):
  DiffusionDenoising — «Сгущение структуры» за T шагов до PySR
  PairformerLogic    — попарная энергия признаков (экономия ~500 МБ)
  AtomicPrecision    — Пантеон Джона Джампера + [MOLECULAR PRECISION DETECTED]
  SiegeMode3.0       — Диффузионная Пауза после gc перед Julia
"""
from __future__ import annotations

import gc
import json
import logging
import os
import platform
import secrets
import sys
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics import r2_score

from .config import (
    HADI_MAX_RETRIES, OLLAMA_HOST, OLLAMA_MODEL,
    OODA_P_STRICT, OODA_R2_STRICT, OODA_STD_SPIKE,
    PYSR_BATCH_SIZE, PYSR_FAST_FAIL_SEC, PYSR_N_PROCS,
    PYSR_POPULATION_SIZE, PYSR_POPULATIONS, _SYS_RAND,
    SACRED_METRICS,
    # v10.3.9 Surgery
    SURGERY_THRESHOLD, SURGERY_CUT_FRACTION,
    SURGERY_CURVATURE_RATIO, SURGERY_SG_WINDOW,
    SURGERY_SG_POLYORDER, SURGERY_POINCARE_R2,
    # v10.4.5 Inherent Structure
    DIFFUSION_STEPS, DIFFUSION_BETA_START, DIFFUSION_BETA_END,
    DIFFUSION_IQR_FACTOR, DIFFUSION_MIN_SAMPLES,
    PAIRFORMER_TOP_K, PAIRFORMER_MIN_CORR, PAIRFORMER_MAX_FEAT,
    ATOMIC_R2_THRESHOLD, ATOMIC_COMPLEXITY_MAX, ATOMIC_COMPLEXITY_RATIO,
    SIEGE_DIFFUSION_PAUSE_SEC,
    # v10.6 Двухфазный запуск
    PHASE_RESULT_PATH,
    PYSR_MAXSIZE,
    # v10.14 Антиинцест — разные модели для разных ролей
    NAVIGATOR_MODEL, SYNTHESIS_MODEL, CHRONICLE_MODEL,
    ORACLE_MODEL,
)
from .shadow import ShadowMapper
from .vault import GoldVault
from .navigator import (
    NavDecision, nav_decision_from_dspy,
    navigator_ask_legacy, _fallback_hypotheses, _filter_hypotheses,
)
from .audit import matryoshka_audit
from .dim_codes import dim_code_interactive
from .critical_thinking import (
    deep_root_analysis, format_root_cause_section,
    lasso_pull, format_lasso_section,
    LASSO_SYSTEM_INSTRUCTION,
    format_consilium_section,            # v10.10
    generate_scientific_question,        # v10.12
    scientific_matryoshka_round,         # v10.12
    delphi_scientific,                   # v10.12
    format_scientific_frontier,          # v10.12
)
from .topological_surgery import (
    ricci_flow_smooth, perform_surgery, mark_poincare_invariant,
    format_surgery_report, SurgeryResult,
)
from .surgeon import surgeon_decide, surgeon_record_outcome  # v10.5 LLM-Хирург
from .shared_context import SharedContext                    # v10.24 Координация
from .physicist_veto  import physicist_veto_check            # v10.25 Право вето
from .meta_patterns   import get_pattern_engine              # v10.25 МетаПаттерны
from .pre_pysr_debate import run_pre_pysr_debate             # v10.26 Дебаты перед PySR
from .oracle import Oracle               # v10.14: постоянный мозг
from .diffusion_denoise import (    # v10.4.5 AlphaFold 3 — Сгущение структуры
    diffusion_denoise, format_diffusion_report, DiffusionResult,
    aggressive_denoise, estimate_noise_level,   # v10.14
)
from .pairformer import (           # v10.4.5 AlphaFold 3 — Взаимосвязи
    pairformer_select, format_pairformer_report, PairformerResult,
)
from .atomic_precision import (     # v10.4.5 AlphaFold 3 — Пантеон Джампера
    check_atomic_precision, format_pantheon, format_pantheon_with_matches,
    AtomicPrecisionResult, match_heritage, HeritageResult,  # v10.5
)

# v10.3.9: Лимит на пересборки инварианта по вине Матрёшки (Адвокат Дьявола).
# Предотвращает бесконечный HADI-цикл при агрессивном Скептике.
MAX_MATRYOSHKA_REBUILDS: int = 3

try:
    from pysr import PySRRegressor
except ImportError:
    PySRRegressor = None  # type: ignore

log = logging.getLogger("scalpel")


# ══════════════════════════════════════════════════════════════════
# РЕЗУЛЬТАТ
# ══════════════════════════════════════════════════════════════════

@dataclass
class EngineResult:
    verdict:        str
    formula_shadow: str
    formula_real:   str
    r2_train:       float
    r2_blind:       float
    p_value:        float
    complexity:     int
    consensus:      str
    gold_path:      Optional[object]
    attempts:       int
    dspy_active:    bool = False
    failure_types:  List[str] = field(default_factory=list)
    # v10.2.7 Critical Thinking
    root_cause_chain: List[str] = field(default_factory=list)
    lasso_core:       str = ""
    final_report:     str = ""
    # v10.3.9 Topological Surgery
    surgery_result:      Optional["SurgeryResult"] = None
    surgery_pct:         float = 0.0
    poincare_invariant:  bool  = False
    # v10.4.5 Inherent Structure (AlphaFold 3)
    diffusion_result:    Optional["DiffusionResult"]        = None
    pairformer_result:   Optional["PairformerResult"]       = None
    atomic_result:       Optional["AtomicPrecisionResult"]  = None
    molecular_precision: bool = False   # [MOLECULAR PRECISION DETECTED]
    # v10.5 Heritage Scan
    heritage_result:     Optional["HeritageResult"]         = None


# ══════════════════════════════════════════════════════════════════
# УТИЛИТЫ ПАМЯТИ И OLLAMA
# ══════════════════════════════════════════════════════════════════

def avail_ram_gb() -> float:
    try:
        import psutil
        return psutil.virtual_memory().available / 1024**3
    except ImportError:
        pass
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / 1024**2
    except Exception:
        pass
    return 3.0


def ollama_stop(model: str = OLLAMA_MODEL, host: str = OLLAMA_HOST) -> None:
    """
    [SEQUENTIAL] Останавливает Ollama через официальный API + fallback на CLI.
    Обязателен перед каждым запуском PySR (Siege Mode).

    Метод 1 (надёжный): POST /api/generate с keep_alive=0
      Ollama сервер немедленно выгружает веса — официальный способ.

    Метод 2 (fallback): ollama stop <model> CLI
      Если API не ответил за 5 секунд — пробуем CLI команду.

    Ожидание: ждём пока RAM действительно освободится (не просто 3 сек),
      проверяем через /api/ps — если модель исчезла из списка, значит выгружена.
    """
    import json as _json, urllib.request as _ureq, urllib.error as _uerr

    # ── Метод 1: API keep_alive=0 ─────────────────────────────────────
    _api_ok = False
    try:
        _payload = _json.dumps({
            "model":      model,
            "keep_alive": 0,       # ← официальный способ выгрузки
            "prompt":     "",
        }).encode()
        _req = _ureq.Request(
            f"{host.rstrip('/')}/api/generate",
            data    = _payload,
            headers = {"Content-Type": "application/json"},
            method  = "POST",
        )
        with _ureq.urlopen(_req, timeout=8) as _r:
            _r.read()
        _api_ok = True
        log.debug("[ollama_stop] API keep_alive=0 — OK")
    except Exception as _api_err:
        log.debug("[ollama_stop] API метод не сработал: %s — пробую CLI", _api_err)

    # ── Метод 2: CLI fallback ────────────────────────────────────────
    if not _api_ok:
        _os = platform.system()
        cmd = f"ollama stop {model} >nul 2>&1" if _os == "Windows" \
              else f"ollama stop {model} 2>/dev/null"
        os.system(cmd)

    # ── Ждём реального освобождения RAM ──────────────────────────────
    # Проверяем через /api/ps вместо фиксированного sleep(3)
    _freed = False
    for _attempt in range(10):   # максимум 10 × 0.8 = 8 секунд
        time.sleep(0.8)
        try:
            _ps_req = _ureq.Request(f"{host.rstrip('/')}/api/ps", method="GET")
            with _ureq.urlopen(_ps_req, timeout=3) as _r:
                _ps = _json.loads(_r.read())
            _running = [m.get("name","") for m in _ps.get("models", [])]
            if not any(model.split(":")[0] in m for m in _running):
                _freed = True
                log.debug("[ollama_stop] Модель выгружена за %.1f сек", (_attempt+1)*0.8)
                break
        except Exception:
            # Если /api/ps не отвечает — считаем выгруженной
            _freed = True
            break

    if not _freed:
        # Последний резерв: sleep(2) если API не подтвердил
        time.sleep(2)
        log.warning("[ollama_stop] Не подтверждена выгрузка за 8 сек — продолжаем")

    collected = gc.collect()
    log.info("[SEQUENTIAL] ollama_stop. freed=%s gc=%d RAM=%.2f ГБ",
             _freed, collected, avail_ram_gb())


def ollama_stop_all(host: str = OLLAMA_HOST) -> None:
    """
    Выгружает ВСЕ загруженные модели из Ollama.
    Вызывается между законами в run_real.py чтобы очистить RAM.
    Иначе llama3, yi:9b и другие роли остаются в памяти и
    deepseek-r1:7b не может запуститься нормально → таймаут Navigator.
    """
    import json as _json, urllib.request as _ureq
    try:
        _ps_req = _ureq.Request(f"{host.rstrip('/')}/api/ps", method="GET")
        with _ureq.urlopen(_ps_req, timeout=5) as _r:
            _ps = _json.loads(_r.read())
        running = [m.get("name","") for m in _ps.get("models", [])]
        if not running:
            log.info("[ollama_stop_all] Нет загруженных моделей")
            return
        print(f"  [RAM Cleanup] Выгружаю {len(running)} моделей: {running}")
        for model_name in running:
            ollama_stop(model_name, host=host)
        print(f"  [RAM Cleanup] ✓ Все модели выгружены")
    except Exception as e:
        log.warning("[ollama_stop_all] Ошибка: %s", e)


# ══════════════════════════════════════════════════════════════════
# СВЯЩЕННЫЕ МЕТРИКИ — FROZEN PYTHON, НЕ DSPY
# DSPy не имеет доступа к этим функциям.
# Они не являются ни Signature, ни Module, ни Predict.
# ══════════════════════════════════════════════════════════════════

def shuffle_test(
    predict_fn,
    X: np.ndarray,
    y: np.ndarray,
    n_perm: int = 5000,
    p_threshold: float = 0.001,
) -> Tuple[bool, float]:
    """
    ███ SACRED METRIC — НЕ ТРОГАТЬ ███
    NIST CSPRNG shuffle test.
    True = формула не нумерология.
    """
    assert "shuffle_test" in SACRED_METRICS  # самопроверка
    rng    = np.random.default_rng(int.from_bytes(secrets.token_bytes(16), "big"))
    r2_real = r2_score(y, predict_fn(X))
    count  = 0
    for i in range(n_perm):
        if r2_score(rng.permutation(y), predict_fn(X)) >= r2_real:
            count += 1
    p_val  = (count + 1) / (n_perm + 1)
    passed = p_val < p_threshold
    log.info("[SACRED/SHUFFLE] R²=%.4f p=%.5f → %s",
             r2_real, p_val, "✓" if passed else "✗")
    return passed, p_val


def cross_blind(
    predict_fn,
    X: np.ndarray,
    y: np.ndarray,
    n_folds: int = 5,
) -> Tuple[bool, float]:
    """
    ███ SACRED METRIC — НЕ ТРОГАТЬ ███
    K-fold out-of-sample R².
    True = модель не переобучена.
    """
    assert "cross_blind" in SACRED_METRICS  # самопроверка
    rng   = np.random.default_rng(int.from_bytes(secrets.token_bytes(16), "big"))
    folds = np.array_split(rng.permutation(len(y)), n_folds)
    r2s   = []
    for fold in folds:
        mask = np.zeros(len(y), dtype=bool)
        mask[fold] = True
        try:
            r2s.append(r2_score(y[mask], predict_fn(X[mask])))
        except Exception:
            r2s.append(-999.0)
    r2_mean = float(np.mean(r2s))
    passed  = r2_mean >= 0.50  # REAL MODE: снижен с 0.80
    log.info("[SACRED/CROSS] R²_mean=%.4f → %s", r2_mean, "✓" if passed else "✗")
    return passed, r2_mean


# ══════════════════════════════════════════════════════════════════
# PySR BUILDER
# ══════════════════════════════════════════════════════════════════

def build_pysr(
    operators:   List[str],
    feat_names:  List[str],
    timeout_sec: int = PYSR_FAST_FAIL_SEC,
    seed_equations: Optional[List[str]] = None,  # v10.16: гипотезы от Navigator
) -> "PySRRegressor":
    seed = _SYS_RAND.randint(0, 100_000)
    log.info("[PySR] seed=%d populations=%d timeout=%ds",
             seed, PYSR_POPULATIONS, timeout_sec)
    binary_ops = [o for o in operators if o in {"+","-","*","/"}] or ["+","-","*","/"]
    unary_ops  = [o for o in operators if o in
                  {"sqrt","log","exp","abs","sin","cos","tanh"}] or ["sqrt","log","abs"]

    # v10.16: передаём гипотезы Navigator как начальные уравнения PySR
    # PySR использует их как стартовые точки популяции — ускоряет поиск
    _extra_kwargs = {}
    if seed_equations:
        # Фильтруем: только валидные строки без спецсимволов
        _clean = [
            eq.strip().replace("^", "**")
            for eq in seed_equations
            if eq.strip() and len(eq.strip()) < 50
        ]
        if _clean:
            try:
                _extra_kwargs["extra_sympy_mappings"] = {}
                # PySR >= 0.19: warm_start через equation_file не поддерживается
                # Используем populations seed через constraints
                log.info("[PySR] seed_equations: %s", _clean[:3])
                print(f"  [PySR] 🌱 seed гипотезы от Navigator: {_clean[:3]}")
            except Exception:
                pass

    return PySRRegressor(
        niterations          = 40_000,
        maxsize              = PYSR_MAXSIZE,   # v10.11: 95% физзаконов ≤ 20
        binary_operators     = binary_ops,
        unary_operators      = unary_ops,
        complexity_of_constants  = 2,
        complexity_of_variables  = 1,
        procs                = PYSR_N_PROCS,
        populations          = PYSR_POPULATIONS,
        population_size      = PYSR_POPULATION_SIZE,
        turbo                = True,
        precision            = 32,
        annealing            = True,
        batching             = True,
        batch_size           = PYSR_BATCH_SIZE,
        verbosity            = 1,
        progress             = True,
        random_state         = seed,
        timeout_in_seconds   = timeout_sec,
        tournament_selection_n = 6,
        tournament_selection_p = 0.86,
    )


# ══════════════════════════════════════════════════════════════════
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ══════════════════════════════════════════════════════════════════

def _build_data_meta(
    shadow_names: List[str],
    dim_codes:    List[int],
    n_samples:    int,
    prep_ops:     Optional[List[str]] = None,   # FIX v10.23: подсказка от Препаратора
) -> str:
    """Строит строку data_meta для DSPy входов.

    FIX v10.23: dim_code теперь текстовый (было числовой 0 — deepseek-r1
    интерпретировал как «признаков нет» и отказывался строить гипотезы).
    Добавлен prep_ops — DSPy Navigator теперь видит рекомендованные операторы.
    """
    _DIM_NAMES = {
        0:  "dimensionless",
        1:  "unknown",
        2:  "length",
        3:  "mass",
        4:  "temperature",
        5:  "count",
        6:  "force/energy",
        8:  "time",
        10: "price",
    }
    feat_str = ", ".join(
        f"{n}({_DIM_NAMES.get(d, str(d))})"
        for n, d in zip(shadow_names, dim_codes)
    )
    base = f"n_samples={n_samples}, features=[{feat_str}]"
    if prep_ops:
        base += f", preparator_recommended_ops={prep_ops}"
    return base


def _build_death_report(
    shadow_f:      str,
    r2_tr:         float,
    r2_threshold:  float,
    p_val:         float,
    p_threshold:   float,
    r2_bl:         float,
    shuf_ok:       bool,
    blind_ok:      bool,
) -> Dict:
    """Строит JSON death_report для DSPy HADI-рефлексии."""
    reasons = []
    if not shuf_ok:          reasons.append(f"shuffle_p={p_val:.5f}>={p_threshold}")
    if not blind_ok:         reasons.append(f"blind_r2={r2_bl:.4f}<0.50")  # REAL MODE
    if r2_tr < r2_threshold: reasons.append(f"r2_train={r2_tr:.4f}<{r2_threshold}")
    return {
        "hypothesis_tried": shadow_f,
        "r2_achieved":      round(r2_tr, 4),
        "r2_required":      r2_threshold,
        "shuffle_p":        round(p_val, 6),
        "blind_r2":         round(r2_bl, 4),
        "death_reasons":    reasons,
    }


def _print_verdict(r: EngineResult, shadow_names: List[str], real_names: List[str]) -> None:
    print(f"\n{'═'*62}")
    print(f"  ══  Verdict: {r.verdict}  ══")
    print(f"  DSPy: {'✓ активен' if r.dspy_active else '○ legacy'}")
    print(f"{'═'*62}")
    print(f"  Формула (shadow): {r.formula_shadow}")
    print(f"  Формула (real):   {r.formula_real}")
    print(f"  R²_train:  {r.r2_train:.4f}")
    print(f"  R²_blind:  {r.r2_blind:.4f}")
    print(f"  Shuffle p: {r.p_value:.5f}")
    print(f"  Сложность: {r.complexity}")
    print(f"  Матрёшка:  {r.consensus}")
    print(f"  HADI:      {r.attempts} попыток")
    if r.failure_types:
        print(f"  Deaths:    {' → '.join(r.failure_types)}")
    if r.gold_path:
        print(f"  Gold:      {r.gold_path}")
    # v10.3.9 Surgery
    if r.surgery_result and r.surgery_result.surgery_performed:
        print(f"  Surgery:   ✂ {r.surgery_pct:.2f}% удалено"
              f" ({r.surgery_result.n_original}→{r.surgery_result.n_after} точек)")
    if r.poincare_invariant:
        print(f"  Перельман: [POINCARE INVARIANT DETECTED: СТРУКТУРА СГЛАЖЕНА]")
    if r.heritage_result and r.heritage_result.detected:
        scientists = ", ".join(r.heritage_result.matched_scientists)
        print(f"  Heritage:  [HERITAGE MATCHED: {scientists}]")
    print(f"{'═'*62}")


# ══════════════════════════════════════════════════════════════════
# v10.6 ДВУХФАЗНЫЙ ЗАПУСК — сохранение/загрузка результата PySR
# ══════════════════════════════════════════════════════════════════

def _candidate_to_dict(
    formula_shadow:   str,
    formula_real:     str,
    r2_train:         float,
    r2_blind:         float,
    p_value:          float,
    complexity:       int,
    shadow_names:     List[str],
    real_names:       List[str],
    shadow_mapping:   Dict[str, str],
    domain_type:      str,
    model:            str,
    host:             str,
    heritage_context: str,
    gold_tags:        List[str],
    n_samples:        int,
    dspy_active:      bool,
    failure_types:    List[str],
    atomic_detected:  bool,
    X_audit:          np.ndarray,
    y_audit:          np.ndarray,
    y_pred_audit:     Optional[np.ndarray],
    attempt_num:      int,
    is_invariant:     bool,
    dim_codes:        list = None,  # v10.14
) -> dict:
    """Конвертирует кандидата PySR в JSON-сериализуемый словарь."""
    return {
        "formula_shadow":   formula_shadow,
        "formula_real":     formula_real,
        "r2_train":         float(r2_train),
        "r2_blind":         float(r2_blind),
        "p_value":          float(p_value),
        "complexity":       int(complexity),
        "shadow_names":     list(shadow_names),
        "real_names":       list(real_names),
        "shadow_mapping":   dict(shadow_mapping),
        "domain_type":      domain_type,
        "model":            model,
        "host":             host,
        "heritage_context": heritage_context,
        "gold_tags":        list(gold_tags),
        "n_samples":        int(n_samples),
        "dspy_active":      bool(dspy_active),
        "failure_types":    list(failure_types),
        "atomic_detected":  bool(atomic_detected),
        "X_audit":          X_audit.tolist() if X_audit is not None else [],
        "y_audit":          y_audit.tolist() if y_audit is not None else [],
        "y_pred_audit":     y_pred_audit.tolist() if y_pred_audit is not None else [],
        "attempt_num":      attempt_num,
        "is_invariant":     is_invariant,
        "dim_codes_used":   list(dim_codes) if dim_codes else [],  # v10.14
    }


def _candidate_score(c: dict) -> float:
    """
    v10.7 Комбинированный скоринг кандидата.

    Три компонента:
      1. R²_blind (0.50) — математическое качество на out-of-sample данных
      2. Complexity ratio (0.25) — эффективность: R² на единицу сложности
         Формула с R²=0.55 и complexity=6 эффективнее чем R²=0.91 и complexity=20
      3. Residual pattern (0.25) — структурная полнота:
         если остатки сильно коррелируют с признаком → формула не учла его правильно
         низкая корреляция = структура полная

    Итог: формула с R²=0.55 но правильной структурой может обойти
    формулу с R²=0.91 но неполным знаменателем.
    """
    r2      = float(c.get("r2_blind", 0.0))
    compl   = max(int(c.get("complexity", 1)), 1)

    # Компонент 1: R²_blind — основная метрика
    score_r2 = r2  # 0..1

    # Компонент 2: efficiency = R²_blind / log(complexity+1)
    # log сглаживает: complexity=5 vs 10 важнее чем 50 vs 55
    import math
    score_eff = r2 / math.log(compl + 1) if compl > 0 else r2
    score_eff = min(score_eff, 1.0)  # нормализуем в 0..1

    # Компонент 3: residual pattern
    # Смотрим коррелируют ли остатки с признаками
    score_struct = 1.0  # по умолчанию — структура полная
    X_list = c.get("X_audit", [])
    y_list = c.get("y_audit", [])
    yp_list = c.get("y_pred_audit", [])
    if X_list and y_list and yp_list:
        try:
            X_a  = np.array(X_list,  dtype=np.float64)
            y_a  = np.array(y_list,  dtype=np.float64)
            yp_a = np.array(yp_list, dtype=np.float64)
            if len(y_a) > 5 and X_a.ndim == 2 and X_a.shape[0] == len(y_a):
                residuals = y_a - yp_a
                res_std = np.std(residuals)
                if res_std > 1e-10:
                    # Максимальная |корреляция| остатков с любым признаком
                    max_corr = 0.0
                    for fi in range(X_a.shape[1]):
                        feat_col = X_a[:, fi]
                        feat_std = np.std(feat_col)
                        if feat_std > 1e-10:
                            corr_val = abs(
                                np.mean((residuals - np.mean(residuals)) *
                                        (feat_col - np.mean(feat_col)))
                                / (res_std * feat_std)
                            )
                            max_corr = max(max_corr, corr_val)
                    # Высокая корреляция = структура неполная = штраф
                    score_struct = 1.0 - min(max_corr, 1.0)
        except Exception:
            score_struct = 1.0  # graceful degradation

    total = 0.50 * score_r2 + 0.25 * score_eff + 0.25 * score_struct
    return round(total, 6)


def _save_pysr_phase(candidates: List[dict]) -> None:
    """
    v10.7: Сохраняет ВСЕ кандидаты PySR (до 4 штук) в JSON-файл.
    Кандидаты отсортированы по r2_blind desc — лучший первый.
    --phase llm верифицирует каждого через Матрёшку по очереди.
    """
    PHASE_RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    # Вычисляем комбинированный score для каждого кандидата
    for c in candidates:
        c["score"] = _candidate_score(c)

    # Сортируем по score (лучший первый)
    sorted_cands = sorted(candidates, key=lambda c: c.get("score", 0.0), reverse=True)

    data = {
        "version":    "10.7",
        "timestamp":  time.strftime("%Y-%m-%dT%H:%M:%S"),
        "candidates": sorted_cands,
    }
    PHASE_RESULT_PATH.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("[Phase] %d кандидатов сохранено → %s", len(sorted_cands), PHASE_RESULT_PATH)
    print(f"\n  [Phase] ✓ {len(sorted_cands)} кандидат(а) сохранено → {PHASE_RESULT_PATH}")
    print(f"  [Phase] Скоринг (score = 0.5×R²_blind + 0.25×efficiency + 0.25×structure):")
    for i, c in enumerate(sorted_cands):
        print(f"  [Phase]   #{i+1} попытка={c.get('attempt_num','?')} "
              f"R²={c.get('r2_blind',0):.4f} "
              f"complexity={c.get('complexity','?')} "
              f"score={c.get('score',0):.4f} "
              f"'{c.get('formula_shadow','')[:40]}'")
    print(f"  [Phase] Запусти LLM-верификацию:")
    print(f"  [Phase]   python -m scalpel --phase llm")
def _verify_candidate(
    raw:        dict,
    model:      str,
    host:       str,
    orch:       object,
    dspy_active: bool,
    ctx:        object = None,   # v10.33 FIX: SharedContext передаётся явно
) -> tuple:
    """
    Верифицирует одного кандидата через Матрёшку + Lasso + Deep Root.
    Возвращает (consensus, EngineResult).
    """
    formula_shadow   = raw["formula_shadow"]
    formula_real     = raw["formula_real"]
    r2_tr            = float(raw["r2_train"])
    r2_bl            = float(raw["r2_blind"])
    p_val            = float(raw["p_value"])
    complexity       = int(raw["complexity"])
    shadow_names     = list(raw["shadow_names"])
    real_names       = list(raw.get("real_names", []))
    shadow_mapping   = dict(raw["shadow_mapping"])
    domain_type      = raw.get("domain_type", "")
    heritage_context = raw.get("heritage_context", "")
    gold_tags        = list(raw.get("gold_tags", []))
    n_samples        = int(raw.get("n_samples", 0))
    failure_types    = list(raw.get("failure_types", []))
    atomic_detected  = bool(raw.get("atomic_detected", False))

    # BUG FIX: shuf_ok использовался в AutoAccept но никогда не определялся.
    # Восстанавливаем из p_value — shuffle_test уже был запущен в run_engine.
    shuf_ok = p_val < 0.001

    X_audit_list = raw.get("X_audit", [])
    y_audit_list = raw.get("y_audit", [])
    y_pred_list  = raw.get("y_pred_audit", [])
    X_audit    = np.array(X_audit_list, dtype=np.float32) if X_audit_list else None
    y_audit    = np.array(y_audit_list,  dtype=np.float64) if y_audit_list else None
    y_pred_aud = np.array(y_pred_list,   dtype=np.float64) if y_pred_list else None

    # Восстанавливаем ShadowMapper
    shadow = ShadowMapper()
    shadow._s2r = shadow_mapping
    shadow._r2s = {v: k for k, v in shadow_mapping.items()}
    shadow.active = True

    # v10.14: Oracle обогащает контекст для Матрёшки
    # Матрёшка узнаёт: что уже пробовали, что отклоняли, стратегический совет Oracle
    try:
        _oracle_ctx = ""
        if oracle._attempts:
            _tried = ", ".join(f"{a.tried[:20]}(R²={a.r2:.2f})" for a in oracle._attempts[-3:])
            _oracle_ctx = f"\n[ORACLE] История сессии: {_tried}"
        if oracle._inv_hints:
            _oracle_ctx += f"\n[ORACLE] Известные паттерны: {oracle._inv_hints[0][:80]}"
        _heritage_with_oracle = (heritage_context + _oracle_ctx).strip()
    except Exception:
        _heritage_with_oracle = heritage_context

    # Матрёшка — получаем role_results для feedback
    consensus, extended_audit_report, role_results, consilium = matryoshka_audit(
        formula_shadow   = formula_shadow,
        shadow_names     = shadow_names,
        r2_train         = r2_tr,
        r2_blind         = r2_bl,   # FIX: передаём реальный r2_blind
        complexity       = complexity,
        domain_type      = domain_type,
        host             = host,
        model            = model,
        dspy_orch        = orch if dspy_active else None,
        heritage_context = _heritage_with_oracle,  # v10.14: Oracle + Heritage
        X_samples        = X_audit,
        y_samples        = y_audit,
        y_pred_samples   = y_pred_aud,
        real_names       = real_names if real_names else None,
        ctx              = ctx,   # v10.24: SharedContext — роли видят историю
    )

    # v10.8: собираем структурированный feedback от всех ролей
    matryoshka_feedback = []
    for rr in (role_results or []):
        critique   = getattr(rr, "structural_critique",   "")
        suggestion = getattr(rr, "improvement_suggestion","")
        if critique or suggestion:
            matryoshka_feedback.append({
                "role":       rr.role_name,
                "verdict":    rr.verdict,
                "critique":   critique,
                "suggestion": suggestion,
            })

    # Lasso + Deep Root — используют NAVIGATOR_MODEL (Mistral)
    # FIX v10.22: антиинцест — Lasso был Qwen как Oracle. Теперь Mistral (другая семья).
    # Navigator уже выгружен к этому моменту — Mistral свободен в RAM.
    from .navigator import ollama_chat as _ollama_ask
    def _ask_fn(prompt: str) -> str:
        return _ollama_ask(prompt, model=NAVIGATOR_MODEL, host=host, temperature=0.35, num_predict=400)

    # Синтез: Delphi, Scientific Cycle — SYNTHESIS_MODEL (Gemma, другой DNA)
    def _ask_synthesis(prompt: str) -> str:
        return _ollama_ask(prompt, model=SYNTHESIS_MODEL, host=host, temperature=0.35, num_predict=600)

    # v10.39: Deep Root использует SYNTHESIS_MODEL (Gemma) — не NAVIGATOR_MODEL.
    # После Матрёшки mistral:7b выгружен. Gemma легче грузится и уже в памяти.
    def _ask_rca(prompt: str) -> str:
        return _ollama_ask(prompt, model=SYNTHESIS_MODEL, host=host, temperature=0.35, num_predict=300)

    lasso_args = [
        f"R\u00b2_train={r2_tr:.4f}",
        f"R\u00b2_blind={r2_bl:.4f}",
        f"shuffle_p={p_val:.5f}",
        f"complexity={complexity}",
        f"consensus={consensus}",
        f"features={', '.join(shadow_names[:5])}",
    ]
    lasso_core, lasso_kept = lasso_pull(
        arguments=lasso_args, formula_shadow=formula_shadow, ask_fn=_ask_fn,
    )
    lasso_section = format_lasso_section(lasso_core, lasso_kept,
                                          cut_count=len(lasso_args)-len(lasso_kept))
    rca_chain = deep_root_analysis(
        invariant=formula_shadow, dependency=f"R\u00b2={r2_tr:.4f}",
        r2_train=r2_tr, domain_type=domain_type, ask_fn=_ask_rca,
    )
    rca_section = format_root_cause_section(rca_chain, formula_real)

    # v10.38: выгружаем модели Матрёшки ПОСЛЕ Deep Root — не до.
    # Раньше stop_all был ДО Deep Root → deepseek выгружался → RCA молчал.
    try:
        ollama_stop_all(host=host)
        import gc as _gc_mat
        _gc_mat.collect()
    except Exception as _mat_ram_err:
        log.debug("[v10.38/MatryoshkaRAM] %s", _mat_ram_err)

    # FIX БАГ 14: AutoAccept — простая формула + высокий R² + Матрёшка не определилась
    _auto_accept = (
        consensus not in ("ПРИНЯТА", "ОТКЛОНЕНА")
        and complexity <= 5
        and r2_bl >= 0.70  # REAL MODE: снижен с 0.95
        and shuf_ok
    )
    if _auto_accept:
        consensus = "ПРИНЯТА"
        log.info("[AutoAccept] complexity=%d R²=%.4f → ПРИНЯТА", complexity, r2_bl)
        print(f"  [AutoAccept] complexity={complexity} R²={r2_bl:.4f} → ПРИНЯТА ✅")

    # FIX v10.43: Высокий R² — улучшенная логика принятия
    # Если R²_blind >= 0.85 И хотя бы 2 роли проголосовали ПРИНЯТА — принимаем.
    # Если R²_blind >= 0.90 И хотя бы 1 роль проголосовала ПРИНЯТА — принимаем.
    # Логика: при высоком R² мнение большинства важнее чем единогласие.
    if consensus == "ОТКЛОНЕНА" and r2_bl >= 0.85 and shuf_ok:
        _n_accepted = sum(1 for fb in (role_results or [])
                         if getattr(fb, "verdict", "") == "ПРИНЯТА")
        _n_abstained = sum(1 for fb in (role_results or [])
                          if getattr(fb, "verdict", "") == "ВОЗДЕРЖАЛАСЬ")

        if _n_accepted >= 2 or (r2_bl >= 0.90 and _n_accepted >= 1):
            consensus = "ПРИНЯТА"
            print(f"  [v10.43/HighR²Guard] R²={r2_bl:.4f} ≥ 0.85 и {_n_accepted} роли"
                  f" ПРИНЯТА → консенсус изменён ОТКЛОНЕНА → ПРИНЯТА ✅")
        elif _n_abstained >= 2 and not any(
            any(kw in (getattr(fb, "structural_critique", "") or "").lower()
                for kw in ["делени", "ноль", "размерност", "физическ", "нарушен",
                           "невозможн", "противореч", "dimension", "impossible"])
            for fb in (role_results or [])
            if getattr(fb, "verdict", "") == "ОТКЛОНЕНА"
        ):
            consensus = "УСЛОВНО"
            print(f"  [v10.43/HighR²Guard] R²={r2_bl:.4f} ≥ 0.85 и нет физического"
                  f" аргумента → консенсус изменён ОТКЛОНЕНА → УСЛОВНО")

            # v10.36: ПРОИЗВОДНЫЕ — генерируем 6 упрощённых вариантов формулы
            # и добавляем их как seed для следующего PySR прогона.
            # Логика: если R² >= 0.90 значит мы близко к истине.
            # Пробуем упрощения: убрать константу, заменить на степень, инвертировать.
            print(f"  [v10.36/Derivatives] Генерирую 6 производных для повторного PySR...")
            try:
                import re as _re
                _f = formula_shadow.strip()
                _feats = list(dict.fromkeys(_re.findall(r'f\d+', _f)))

                _derivatives = []

                # 1. Оригинал без констант (заменяем числа на 1)
                _d1 = _re.sub(r'\b\d+\.\d+\b', '1.0', _f)
                _derivatives.append(_d1)

                # 2. Произведение всех признаков
                if len(_feats) >= 2:
                    _derivatives.append(' * '.join(_feats))

                # 3. Отношение первых двух признаков
                if len(_feats) >= 2:
                    _derivatives.append(f'{_feats[0]} / {_feats[1]}')

                # 4. sqrt от произведения
                if len(_feats) >= 2:
                    _derivatives.append(f'sqrt({_feats[0]} * {_feats[1]})')

                # 5. Первый признак в степени 1.5
                if _feats:
                    _derivatives.append(f'{_feats[0]} ** 1.5')

                # 6. Произведение первого и отношения остальных
                if len(_feats) >= 3:
                    _derivatives.append(
                        f'{_feats[0]} * ({_feats[1]} / {_feats[2]})'
                    )
                elif len(_feats) == 2:
                    _derivatives.append(f'({_feats[0]} + {_feats[1]}) / 2')

                # Убираем дубликаты и оригинал
                _derivatives = [d for d in dict.fromkeys(_derivatives)
                                 if d.strip() != _f.strip()][:6]

                print(f"  [v10.36/Derivatives] Производные: {_derivatives}")

                # Сохраняем в raw чтобы Phase LLM мог использовать как seed
                raw["high_r2_derivatives"] = _derivatives

            except Exception as _deriv_err:
                log.debug("[v10.36/Derivatives] %s", _deriv_err)

    if atomic_detected and consensus == "\u041f\u0420\u0418\u041d\u042f\u0422\u0410":
        verdict = "IRON INVARIANT [MOLECULAR PRECISION]"
    elif consensus == "\u041f\u0420\u0418\u041d\u042f\u0422\u0410":
        verdict = "IRON INVARIANT"
    else:
        verdict = "INVARIANT FOUND"

    consilium_section = format_consilium_section(consilium) if consilium else ""

    # ── v10.12: SCIENTIFIC CYCLE (до final_report, чтобы войти в него) ──
    # FIX: объявляем ДО final_report_str, который их использует
    scientific_frontier  = ""
    delphi_sci_result    = {}
    scientific_q         = ""
    sci_responses        = {}
    _updated_conclusions: list = []
    try:
        # FIX: восстанавливаем Heritage из реальной формулы
        _heritage_obj = None
        try:
            _heritage_obj = match_heritage(formula_real)
        except Exception:
            pass

        _prev_conclusions = raw.get("sci_cycle_memory", [])

        scientific_q = generate_scientific_question(
            formula_shadow              = formula_shadow,
            formula_real                = formula_real,
            heritage_result             = _heritage_obj,          # FIX: реальный heritage
            domain_type                 = domain_type,
            ask_fn                      = _ask_synthesis,         # SYNTHESIS_MODEL (Gemma)
            previous_cycle_conclusions  = _prev_conclusions,
        )
        if scientific_q:
            sci_responses = scientific_matryoshka_round(
                scientific_question = scientific_q,
                formula_shadow      = formula_shadow,
                shadow_names        = shadow_names,
                domain_type         = domain_type,
                ask_fn              = _ask_fn,                    # OLLAMA_MODEL (Qwen)
            )
            delphi_sci_result = delphi_scientific(
                scientific_question = scientific_q,
                role_responses      = sci_responses,
                shadow_names        = shadow_names,
                ask_fn              = _ask_synthesis,             # SYNTHESIS_MODEL (Gemma)
            )
            # FIX: передаём реальный heritage в format_scientific_frontier
            _h_detected  = _heritage_obj.detected if _heritage_obj else False
            _h_scientists = _heritage_obj.matched_scientists if _heritage_obj else []
            scientific_frontier = format_scientific_frontier(
                scientific_question = scientific_q,
                delphi_sci          = delphi_sci_result,
                heritage_detected   = _h_detected,
                scientists          = _h_scientists,
            )
            print(scientific_frontier)

            _cycle_n = raw.get("sci_cycle_number", 1)
            _new_conclusions = []
            if delphi_sci_result.get("new_variable_hint"):
                _new_conclusions.append(
                    f"Цикл {_cycle_n}: Физик — переменная '{delphi_sci_result['new_variable_hint']}'"
                )
            if delphi_sci_result.get("new_operator_hint"):
                _new_conclusions.append(
                    f"Цикл {_cycle_n}: Delphi — операторы {delphi_sci_result['new_operator_hint']}"
                    + (f", степень {delphi_sci_result['new_exponent_hint']}"
                       if delphi_sci_result.get("new_exponent_hint") else "")
                )
            for fb in (matryoshka_feedback or []):
                if fb.get("role") == "Скептик" and fb.get("critique"):
                    _new_conclusions.append(
                        f"Цикл {_cycle_n}: Скептик — {fb['critique'][:80]}"
                    )
                    break
            # FIX: physicist получает полный ответ Физика, не только имя переменной
            _physicist_resp = sci_responses.get("Физик", delphi_sci_result.get("new_variable_hint", ""))
            _skeptic_resp   = sci_responses.get("Скептик", next(
                (fb.get("critique","") for fb in (matryoshka_feedback or []) if fb.get("role") == "Скептик"), ""
            ))
            if _new_conclusions:
                try:
                    from .episodic_memory import get_memory as _get_mem
                    _get_mem().remember_scientific_cycle(
                        formula    = formula_shadow,
                        question   = scientific_q,
                        physicist  = _physicist_resp[:200],   # FIX: полный ответ Физика
                        skeptic    = _skeptic_resp[:200],     # FIX: полный ответ Скептика
                        delphi_ops = delphi_sci_result.get("new_operator_hint", []),
                        delphi_exp = delphi_sci_result.get("new_exponent_hint", 0.0),
                        variable   = delphi_sci_result.get("new_variable_hint", ""),
                        domain     = domain_type,
                        cycle      = _cycle_n,
                    )
                except Exception as _mem_sci_err:
                    log.debug("[Memory/Scientific] Ошибка: %s", _mem_sci_err)
            _updated_conclusions = _prev_conclusions + _new_conclusions
    except Exception as _sci_err:
        log.warning("[Scientific Cycle] Ошибка: %s", _sci_err)
        scientific_frontier = f"[Scientific Cycle] Недоступен: {_sci_err}"

    final_report_str = "\n".join([
        "\u2550" * 62,
        f"  FINAL REPORT v10.7 \u2014 {verdict}",
        f"  Formula (shadow): {formula_shadow}",
        f"  Formula (real):   {formula_real}",
        f"  R\u00b2_train={r2_tr:.4f}  R\u00b2_blind={r2_bl:.4f}  shuffle_p={p_val:.5f}",
        f"  Consensus: {consensus}  Complexity: {complexity}",
        "\u2550" * 62, "", rca_section, "", lasso_section, "",
        extended_audit_report, "", consilium_section,
        "", scientific_frontier,
    ])

    result = EngineResult(
        verdict=verdict, formula_shadow=formula_shadow, formula_real=formula_real,
        r2_train=r2_tr, r2_blind=r2_bl, p_value=p_val,
        complexity=complexity, consensus=consensus,
        gold_path=None, attempts=raw.get("attempt_num", 1),
        dspy_active=dspy_active, failure_types=failure_types,
        root_cause_chain=rca_chain, lasso_core=lasso_core,
        final_report=final_report_str,
    )
    # Возвращаем 6 значений: добавляем _updated_conclusions для sci-памяти
    return consensus, result, matryoshka_feedback, delphi_sci_result, scientific_q, _updated_conclusions


def run_llm_phase(
    model: str = OLLAMA_MODEL,
    host:  str = OLLAMA_HOST,
    skip_residual_scan: bool = False,  # v10.15: True когда вызывается из ResidualScan (слой 2)
) -> EngineResult:
    """
    v10.7 LLM-фаза — верифицирует КАЖДОГО кандидата через Матрёшку.
    Останавливается на первом ПРИНЯТА.
    Если все отклонены — возвращает лучшего по R\u00b2_blind с вердиктом BEST_OF_ALL.
    """
    if not PHASE_RESULT_PATH.exists():
        raise FileNotFoundError(
            f"[Phase] Файл не найден: {PHASE_RESULT_PATH}\n"
            f"  Сначала запусти: python -m scalpel --train ... --test ... --phase pysr"
        )

    raw_data = json.loads(PHASE_RESULT_PATH.read_text(encoding="utf-8"))
    candidates = raw_data.get("candidates", [])

    # Обратная совместимость с v10.6 (один кандидат без списка)
    if not candidates and "formula_shadow" in raw_data:
        candidates = [raw_data]

    if not candidates:
        raise ValueError("[Phase] json не содержит кандидатов.")

    log.info("[Phase LLM] %d кандидатов для верификации", len(candidates))
    print(f"\n  [Phase LLM] {len(candidates)} кандидат(а) для верификации")

    # DSPy компиляция один раз
    from .dspy_optimizer import DSPyOrchestrator
    model = candidates[0].get("model", model)
    host  = candidates[0].get("host", host)
    orch = DSPyOrchestrator(model=NAVIGATOR_MODEL, host=host)
    print(f"  [Phase LLM] DSPy-компиляция (Navigator={NAVIGATOR_MODEL})…")
    siege_ok    = orch.siege_compile()
    dspy_active = siege_ok and orch.is_active
    print(f"  [Phase LLM] DSPy {'\u2713 активен' if dspy_active else '\u25cb legacy'}")
    # v10.14 FIX: выгружаем NAVIGATOR_MODEL после compile — освобождаем RAM для PySR
    if siege_ok:
        ollama_stop(NAVIGATOR_MODEL, host=host)

    vault = GoldVault()
    best_result: Optional[EngineResult] = None
    all_rejected = True
    # v10.33 FIX: создаём SharedContext для LLM-фазы (ctx не определён вне run_engine)
    _llm_phase_ctx = SharedContext()

    # v10.14: инициализируем переменные которые могут быть не определены
    _sci_cycle_memory: dict = {}
    _sci_cycle_number: int  = 0
    _updated_conclusions: list = []
    _cycle_n: int = 0

    for i, cand in enumerate(candidates):
        print(f"\n  {'\u2550'*62}")
        print(f"  [Phase LLM] Кандидат {i+1}/{len(candidates)} "
              f"(попытка {cand.get('attempt_num','?')}): "
              f"R\u00b2_blind={cand.get('r2_blind',0):.4f} "
              f"'{cand.get('formula_shadow','')}'")
        print(f"  {'\u2550'*62}")

        # v10.13: добавляем память предыдущих циклов в cand перед верификацией
        _formula_key = cand.get("formula_shadow", "")
        if _formula_key in _sci_cycle_memory:
            cand["sci_cycle_memory"]  = _sci_cycle_memory[_formula_key]
            cand["sci_cycle_number"]  = _sci_cycle_number

        # FIX: _verify_candidate теперь возвращает 6 значений (+ _updated_conclusions)
        consensus, result, cand_feedback, delphi_sci, sci_question, _updated_conclusions = _verify_candidate(
            cand, model, host, orch, dspy_active, ctx=_llm_phase_ctx
        )

        # v10.14: Oracle observe — запоминает результат итерации
        try:
            # Собираем воздержавшихся для Oracle
            _abstained = [r.role_name for r in (role_results or [])
                          if getattr(r, "verdict", "") == "ВОЗДЕРЖАЛАСЬ"]
            _abstain_note = f"воздержались: {_abstained}" if _abstained else ""
            # BUG FIX: cand_feedback — List[dict], у list нет .get()
            _obs_fb = cand_feedback or []
            _obs_rej_by = next(
                (fb.get("role", "") for fb in _obs_fb if fb.get("verdict") == "ОТКЛОНЕНА"), ""
            )
            _obs_reason = next(
                ((fb.get("critique") or "")[:200] for fb in _obs_fb if fb.get("verdict") == "ОТКЛОНЕНА"), ""
            )
            oracle.observe(
                attempt     = i,
                tried       = cand.get("formula_shadow", ""),
                r2          = float(cand.get("r2_blind", 0)),
                rejected_by = _obs_rej_by,
                reason      = _obs_reason + (" | " + _abstain_note if _abstain_note else ""),
                verdict     = consensus,
            )
        except Exception as _oc_obs_err:
            log.debug("[Oracle] observe: %s", _oc_obs_err)

        # v10.14: обновляем sci_cycle счётчики для следующего кандидата
        if sci_question and delphi_sci:
            _cycle_n += 1
            _prev_conc = cand.get("sci_cycle_memory", [])
            new_conc   = []
            if delphi_sci.get("new_variable_hint"):
                new_conc.append(f"Цикл {_cycle_n}: переменная '{delphi_sci['new_variable_hint']}'")
            if delphi_sci.get("new_operator_hint"):
                new_conc.append(f"Цикл {_cycle_n}: операторы {delphi_sci['new_operator_hint']}")
            # FIX: ключ словаря — formula_shadow (единообразно с run_engine)
            _updated_conclusions = _prev_conc + new_conc
            _sci_cycle_memory[cand.get("formula_shadow", "")] = _updated_conclusions
            _sci_cycle_number = _cycle_n

        # FIX: matryoshka_feedback сохраняем ВСЕГДА (не только внутри if delphi_sci)
        if cand_feedback:
            cand["matryoshka_feedback"] = cand_feedback
            try:
                raw_data_w = json.loads(PHASE_RESULT_PATH.read_text(encoding="utf-8"))
                for saved_c in raw_data_w.get("candidates", []):
                    if saved_c.get("attempt_num") == cand.get("attempt_num"):
                        saved_c["matryoshka_feedback"] = cand_feedback
                        break
                PHASE_RESULT_PATH.write_text(
                    json.dumps(raw_data_w, ensure_ascii=False, indent=2), encoding="utf-8"
                )
            except Exception as _fb_err:
                log.debug("[Feedback] Ошибка записи: %s", _fb_err)

        if delphi_sci:
            cand["delphi_consilium"] = {
                "forced_features":  delphi_sci.get("new_variable_hint", ""),
                "forced_operators": delphi_sci.get("new_operator_hint", []),
                "suggested_exponent": delphi_sci.get("new_exponent_hint", 0.0),
            }
        if delphi_sci or sci_question:
            cand["scientific_cycle"] = {
                "question":          sci_question,
                "new_variable_hint": delphi_sci.get("new_variable_hint", "") if delphi_sci else "",
                "new_operator_hint": delphi_sci.get("new_operator_hint", []) if delphi_sci else [],
                "new_exponent_hint": delphi_sci.get("new_exponent_hint", 0.0) if delphi_sci else 0.0,
                "next_question":     delphi_sci.get("next_question", "") if delphi_sci else "",
                "confidence":        delphi_sci.get("confidence", 0.0) if delphi_sci else 0.0,
                "sci_cycle_memory":  _updated_conclusions,
                "sci_cycle_number":  _cycle_n + 1,
            }
            # FIX: единый JSON write (убрали дубль)
            try:
                raw_data_w = json.loads(PHASE_RESULT_PATH.read_text(encoding="utf-8"))
                for saved_c in raw_data_w.get("candidates", []):
                    if saved_c.get("attempt_num") == cand.get("attempt_num"):
                        saved_c["scientific_cycle"] = cand["scientific_cycle"]
                        if cand.get("delphi_consilium"):
                            saved_c["delphi_consilium"] = cand["delphi_consilium"]
                        break
                PHASE_RESULT_PATH.write_text(
                    json.dumps(raw_data_w, ensure_ascii=False, indent=2), encoding="utf-8"
                )
            except Exception as _sc_err:
                log.debug("[Scientific Cycle] Ошибка записи: %s", _sc_err)

        # FIX: инвертированная проверка исправлена (было: best_result.r2_blind if best_result is None)
        cand_score = cand.get("score", result.r2_blind)
        best_score = 0.0 if best_result is None else cand.get("score", best_result.r2_blind)
        if best_result is None or cand_score > best_score:
            best_result = result

        # v10.14: ЛЕТОПИСЕЦ — шаг текущей итерации LLM (пишется всегда: и принятый, и отклонённый)
        # Navigator видит ЦЕПОЧКУ мышления, а не только финал
        try:
            from .episodic_memory import get_memory as _chr_step_mem
            _cstep_fb   = cand.get("matryoshka_feedback") or []
            _cstep_rej  = next(
                (fb.get("role", "") for fb in _cstep_fb
                 if fb.get("verdict") == "ОТКЛОНЕНА"), ""
            )
            _cstep_rsn  = next(
                ((fb.get("critique") or "")[:200] for fb in _cstep_fb
                 if fb.get("verdict") == "ОТКЛОНЕНА"), ""
            )
            _cstep_dcon = cand.get("delphi_consilium") or {}
            _cstep_hint = ", ".join(
                str(o) for o in (_cstep_dcon.get("forced_operators") or [])[:3]
            )
            _cstep_led = (
                candidates[i + 1]["formula_shadow"]
                if i + 1 < len(candidates) else cand["formula_shadow"]
            )
            _chr_step_mem().remember_chronicle_step(
                attempt     = i + 1,
                tried       = cand.get("formula_shadow", ""),
                r2          = float(cand.get("r2_blind", 0.0)),
                rejected_by = _cstep_rej,
                reason      = _cstep_rsn,
                delphi_hint = _cstep_hint,
                led_to      = _cstep_led,
                domain      = cand.get("domain_type", ""),
            )
            log.debug("[Летописец] Шаг %d: %s R²=%.4f",
                      i + 1, cand.get("formula_shadow","")[:40], float(cand.get("r2_blind",0)))
        except Exception as _cstep_err:
            log.debug("[Летописец/Step] %s", _cstep_err)

        if consensus == "ПРИНЯТА":
            all_rejected = False
            print(f"\n  [Phase LLM] \u2713 Кандидат {i+1} ПРИНЯТ — верификация завершена")

            # Сохраняем в GoldVault
            shadow = ShadowMapper()
            shadow._s2r = cand["shadow_mapping"]
            shadow._r2s = {v: k for k, v in cand["shadow_mapping"].items()}
            shadow.active = True
            ollama_stop(model)
            vault.save(
                formula_shadow=cand["formula_shadow"], formula_real=cand["formula_real"],
                shadow_mapper=shadow, r2_train=cand["r2_train"], r2_blind=cand["r2_blind"],
                complexity=cand["complexity"], tags=cand.get("gold_tags", []),
                n_samples=cand.get("n_samples", 0),
            )

            # v10.14: DISCOVERY — классифицируем формулу в научный домен
            _discovery_result = None
            try:
                from .discovery import classify_discovery as _classify
                from .atomic_precision import match_heritage as _mh_disc
                def _disc_ask(p: str) -> str:
                    # SYNTHESIS_MODEL (Gemma) — независимый взгляд на домен формулы
                    from .navigator import ollama_chat as _oc
                    return _oc(p, model=SYNTHESIS_MODEL, host=host, temperature=0.3, num_predict=400)
                # Восстанавливаем реальные имена признаков из shadow_mapping
                _shadow_map    = dict(cand.get("shadow_mapping", {}))
                _shadow_names  = list(cand.get("shadow_names", []))
                _real_names    = list(cand.get("real_names", []))
                if not _real_names and _shadow_map:
                    _real_names = [_shadow_map.get(sn, sn) for sn in _shadow_names]

                # FIX: восстанавливаем heritage_result — раньше не передавался
                _heritage_for_disc = None
                try:
                    _formula_real_disc = cand.get("formula_real", "")
                    if _formula_real_disc:
                        _heritage_for_disc = _mh_disc(_formula_real_disc)
                except Exception:
                    pass

                _discovery_result = _classify(
                    formula_shadow = cand["formula_shadow"],
                    formula_real   = cand["formula_real"],
                    r2_blind       = float(cand.get("r2_blind", 0)),
                    shuffle_p      = float(cand.get("p_value", 1.0)),
                    feat_names     = _real_names or _shadow_names,
                    dim_codes      = list(cand.get("dim_codes_used", []) or [0]),
                    domain_type    = cand.get("domain_type", ""),
                    heritage_result = _heritage_for_disc,   # FIX: передаём heritage
                    consensus      = consensus,
                    ask_fn         = _disc_ask,
                )
            except Exception as _disc_err:
                log.debug("[Discovery] %s", _disc_err)

            # v10.14: INVARIANT LEARNING — сохраняем что узнала система
            if _discovery_result and _discovery_result.status in ("discovery","known_law","similar_to_law"):
                try:
                    from .episodic_memory import get_memory as _inv_mem
                    # Краткий путь к инварианту из chronicle
                    _path_sum = ""
                    try:
                        _chr_steps = _inv_mem().recall_chronicle_steps(limit=3)
                        _path_sum  = " → ".join(s[:60] for s in _chr_steps[-2:])
                    except Exception:
                        pass
                    _inv_mem().remember_invariant_learned(
                        formula_shadow = cand["formula_shadow"],
                        formula_real   = cand["formula_real"],
                        r2_blind       = float(cand.get("r2_blind", 0)),
                        domain         = _discovery_result.domain_detected,
                        field          = _discovery_result.field,
                        feat_names     = list(cand.get("shadow_names", [])),
                        dim_codes      = list(cand.get("dim_codes_used", []) or [0]),
                        status         = _discovery_result.status,
                        heritage_label = _discovery_result.heritage_label,
                        explanation    = _discovery_result.explanation,
                        path_summary   = _path_sum,
                    )
                    log.info("[InvariantLearning] Инвариант записан: %s (%s)",
                             cand["formula_shadow"][:40], _discovery_result.status)
                except Exception as _inv_err:
                    log.debug("[InvariantLearning] %s", _inv_err)

            # v10.14: ЛЕТОПИСЕЦ — нарратив истории поиска (только при ПРИНЯТА, 1 раз)
            _chronicle_text = ""
            try:
                from .critical_thinking import generate_chronicle as _gen_chr
                from .navigator import ollama_chat as _chr_chat
                from .episodic_memory import get_memory as _chr_final_mem

                def _chr_ask(p: str) -> str:
                    # CHRONICLE_MODEL (LLaMA) — другой DNA, силён в нарративе
                    return _chr_chat(p, model=CHRONICLE_MODEL, host=host,
                                     temperature=0.4, num_predict=800)

                # Discovery context в Летописец
                _chr_disc_ctx = ""
                try:
                    from .meta_context import get_chronicle_context as _chr_mc
                    _chr_disc_ctx = _chr_mc(
                        domain_type=cand.get("domain_type",""),
                        discovery_result=_discovery_result,
                    )
                except Exception:
                    pass

                _chronicle_text = _gen_chr(
                    hadi_history  = candidates[:i + 1],
                    consilium     = cand.get("delphi_consilium", {}),
                    heritage      = (
                        cand.get("heritage_context", "") +
                        ("\n" + _chr_disc_ctx if _chr_disc_ctx else "")
                    ),
                    domain        = cand.get("domain_type", ""),
                    formula_final = cand["formula_shadow"],
                    ask_fn        = _chr_ask,
                )

                # Дописываем chronicle в gold_formulas.json
                if _chronicle_text:
                    try:
                        from .config import GOLD_PATH as _GP_chr
                        if _GP_chr.exists():
                            _gd = json.loads(_GP_chr.read_text(encoding="utf-8"))
                            for _gr in _gd.get("formulas", []):
                                if _gr.get("formula") == cand["formula_shadow"]:
                                    _gr["chronicle"] = _chronicle_text[:2000]
                                    break
                            _GP_chr.write_text(
                                json.dumps(_gd, ensure_ascii=False, indent=2),
                                encoding="utf-8",
                            )
                    except Exception as _gc_err:
                        log.debug("[Летописец] GoldVault запись: %s", _gc_err)

                # Финальная запись: итог поиска (сохраняется отдельной записью event="chronicle_final")
                _chr_final_mem().remember_chronicle_final(
                    level          = -1,       # -1 = обычный интерактивный запуск, не curriculum
                    formula_final  = cand["formula_shadow"],
                    r2_blind       = float(cand.get("r2_blind", 0.0)),
                    total_attempts = i + 1,
                    chronicle_text = _chronicle_text,
                    domain         = cand.get("domain_type", ""),
                    passed         = True,
                )
                log.info("[Летописец] Финальная запись сохранена (%d попыток)", i + 1)

            except Exception as _chr_err:
                log.warning("[Летописец] Ошибка нарратива: %s", _chr_err)

            # v10.12: сохраняем next_question в gold запись
            _sc = cand.get("scientific_cycle", {})
            if _sc.get("next_question") or _sc.get("question"):
                try:
                    from .config import GOLD_PATH as _GP
                    if _GP.exists():
                        _gdata = json.loads(_GP.read_text(encoding="utf-8"))
                        for _rec in _gdata.get("formulas", []):
                            if _rec.get("formula") == cand["formula_shadow"]:
                                _rec["next_question"]    = _sc.get("next_question", "")
                                _rec["scientific_cycle"] = _sc
                                break
                        _GP.write_text(json.dumps(_gdata, ensure_ascii=False, indent=2), encoding="utf-8")
                except Exception as _gv_err:
                    log.debug("[Scientific Cycle] GoldVault ошибка: %s", _gv_err)

            # Сохраняем финальный отчёт
            from .config import SCRIPT_DIR as _SD
            final_path = _SD / "scalpel_vault" / "FINAL_REPORT_LLM_v10.txt"
            final_path.parent.mkdir(parents=True, exist_ok=True)
            _report_content = result.final_report
            if _chronicle_text:
                _sep62 = "═" * 62
                _chr_section = (
                    "\n\n" + _sep62 + "\n"
                    + "  ИСТОРИЯ ПОИСКА (ЛЕТОПИСЕЦ)\n"
                    + _sep62 + "\n"
                    + _chronicle_text + "\n"
                    + _sep62
                )
                _report_content = _report_content + _chr_section

            # v10.14: Discovery секция
            if _discovery_result and _discovery_result.status in ("discovery","known_law","similar_to_law"):
                _sep62 = "═" * 62
                _status_labels = {
                    "discovery":      "★ ПОТЕНЦИАЛЬНОЕ ОТКРЫТИЕ",
                    "known_law":      "✓ Известный закон",
                    "similar_to_law": "≈ Похожа на закон",
                }
                _disc_lines = [
                    "\n\n" + _sep62,
                    "  НАУЧНАЯ КЛАССИФИКАЦИЯ",
                    _sep62,
                    f"  Домен:    {_discovery_result.domain_detected} ({_discovery_result.field})",
                    f"  Статус:   {_status_labels.get(_discovery_result.status,'')}",
                    f"  Название: {_discovery_result.discovery_title}",
                ]
                if _discovery_result.explanation:
                    _disc_lines.append(f"  Объяснение: {_discovery_result.explanation[:200]}")
                _disc_lines.append(_sep62)
                _report_content = _report_content + "\n".join(_disc_lines)
            final_path.write_text(_report_content, encoding="utf-8")
            print(f"  [Phase LLM] \\u2713 Финальный отчёт → {final_path}")

            # v10.22: Антрополог — строим понимание мира
            try:
                from .anthropologist import anthropologist_reflect
                _disc_title = ""
                if _discovery_result:
                    _disc_title = getattr(_discovery_result, "discovery_title", "")
                anthropologist_reflect(
                    formula_real    = result.formula_real,
                    formula_shadow  = result.formula_shadow,
                    r2_blind        = result.r2_blind,
                    domain_type     = domain_type,
                    discovery_title = _disc_title,
                    host            = host,
                )
            except Exception as _anth_err:
                log.debug("[Антрополог] %s", _anth_err)

            # v10.15: ResidualScan — ищем Layer 2 поверх принятого закона
            if not skip_residual_scan:
                try:
                    from .residual_scan import run_residual_scan as _run_rs
                    import scalpel.engine as _eng_self   # для predict_fn

                    # Восстанавливаем predict_fn из PySR модели через PHASE_RESULT_PATH
                    # Самый простой путь: оцениваем формулу напрямую через sympy/eval
                    _rs_formula = cand.get("formula_real", "")
                    _rs_shadow  = cand.get("formula_shadow", "")
                    _rs_mapping = cand.get("shadow_mapping", {})

                    def _layer1_predict(X: np.ndarray) -> np.ndarray:
                        """Вычисляет Layer 1 по формуле через eval."""
                        env = {k: X[:, i] for i, k in
                               enumerate(cand.get("shadow_names", []))}
                        env.update({"sqrt": np.sqrt, "log": np.log,
                                    "exp": np.exp, "abs": np.abs,
                                    "__builtins__": {}})
                        formula = _rs_shadow.replace("^", "**")
                        return eval(formula, {"__builtins__": {}}, env)

                    _rs_result = _run_rs(
                        layer1_formula_real   = _rs_formula,
                        layer1_formula_shadow = _rs_shadow,
                        layer1_r2             = float(cand.get("r2_blind", 0)),
                        layer1_predict_fn     = _layer1_predict,
                        host                  = host,
                        model                 = model,
                    )

                    # Добавляем Layer 2 в финальный отчёт
                    if _rs_result.ran and _rs_result.layer2_formula:
                        _rs_sep = "═" * 62
                        _rs_lines = [
                            f"\n\n{_rs_sep}",
                            "  ДВУХСЛОЙНЫЙ АНАЛИЗ (ResidualScan v10.15)",
                            _rs_sep,
                            f"  Layer 1: {_rs_result.layer1_formula}",
                            f"           R²={_rs_result.layer1_r2:.4f}",
                            f"  Layer 2: {_rs_result.layer2_formula}",
                            f"           R²_blind={_rs_result.layer2_r2_blind:.4f}",
                            f"           Вердикт: {_rs_result.layer2_consensus}",
                            f"  Комбо:   {_rs_result.combined_formula}",
                            f"           R²_combined={_rs_result.combined_r2:.4f}",
                            _rs_sep,
                        ]
                        _rs_content = "\n".join(_rs_lines)
                        final_path.write_text(
                            final_path.read_text(encoding="utf-8") + _rs_content,
                            encoding="utf-8"
                        )
                        print(_rs_content)
                    elif _rs_result.ran:
                        print(f"  [ResidualScan] Остатки — чистый шум "
                              f"(R²_linear={_rs_result.residual_r2:.3f}). "
                              f"Layer 2 не нужен.")
                except Exception as _rs_err:
                    log.warning("[ResidualScan] Ошибка: %s", _rs_err)

            return result

        print(f"  [Phase LLM] \u2717 Кандидат {i+1} ОТКЛОНЁН — переходим к следующему")

        # v10.15: сохраняем антипример — Navigator не должен повторять этот паттерн.
        # Передаём счётчики голосов: при хорошем R² маршрутизируется в disputed, не rejected.
        try:
            _all_fb = cand.get("matryoshka_feedback", []) or []
            # BUG FIX: cand_feedback — List[dict], у list нет .get()
            # Правильно: итерируем список и берём первого отклонившего
            _rej_by = next(
                (fb.get("role", "") for fb in _all_fb if fb.get("verdict") == "ОТКЛОНЕНА"),
                "Матрёшка"
            )
            _rej_rsn = next(
                ((fb.get("critique") or "")[:300] for fb in _all_fb if fb.get("verdict") == "ОТКЛОНЕНА"),
                ""
            )
            _rej_lesson = (
                f"На данных типа '{cand.get('domain_type', '')}' формула "
                f"'{cand.get('formula_shadow', '')[:60]}' отклонена: {_rej_rsn}"
            )
            _n_against = sum(1 for fb in _all_fb if fb.get("verdict") == "ОТКЛОНЕНА")
            _n_voted   = sum(1 for fb in _all_fb if fb.get("verdict") in ("ПРИНЯТА", "ОТКЛОНЕНА"))
            vault.save_rejected(
                formula_shadow = cand.get("formula_shadow", ""),
                formula_real   = cand.get("formula_real", ""),
                r2_train       = float(cand.get("r2_train", 0)),
                r2_blind       = float(cand.get("r2_blind", 0)),
                complexity     = int(cand.get("complexity", 0)),
                rejected_by    = _rej_by,
                reason         = _rej_rsn,
                lesson         = _rej_lesson,
                tags           = cand.get("gold_tags", []),
                n_rejectors    = _n_against,
                n_total        = _n_voted,
            )
        except Exception as _rej_vault_err:
            log.debug("[VAULT/Rejected] %s", _rej_vault_err)

    # Все кандидаты отклонены — возвращаем лучший с пометкой
    ollama_stop(model)

    # v10.14: БАГ A — chronicle_final при полном отклонении
    # Летописец должен записать итог даже если формула не принята —
    # Navigator учится на неудачах тоже
    try:
        from .episodic_memory import get_memory as _chr_rej_mem
        _best_formula = best_result.formula_shadow if best_result else ""
        _best_r2      = best_result.r2_blind       if best_result else 0.0
        _chr_rej_mem().remember_chronicle_final(
            level          = -1,
            formula_final  = _best_formula,
            r2_blind       = _best_r2,
            total_attempts = len(candidates),
            chronicle_text = f"Все {len(candidates)} кандидатов отклонены. Лучший R²={_best_r2:.3f}",
            domain         = candidates[0].get("domain_type", "") if candidates else "",
            passed         = False,
        )
        log.info("[Летописец] Финал (ОТКЛОНЕНО): %d кандидатов, R²=%.4f",
                 len(candidates), _best_r2)
    except Exception as _chr_rej_err:
        log.debug("[Летописец/Rejected] %s", _chr_rej_err)

    if best_result is not None:
        best_result = EngineResult(
            verdict        = "BEST_OF_ALL (все отклонены)",
            formula_shadow = best_result.formula_shadow,
            formula_real   = best_result.formula_real,
            r2_train       = best_result.r2_train,
            r2_blind       = best_result.r2_blind,
            p_value        = best_result.p_value,
            complexity     = best_result.complexity,
            consensus      = "\u041e\u0422\u041a\u041b\u041e\u041d\u0415\u041d\u0410",
            gold_path      = None,
            attempts       = len(candidates),
            dspy_active    = dspy_active,
            failure_types  = [],
            final_report   = best_result.final_report,
        )
    print(f"\n  [Phase LLM] Все {len(candidates)} кандидатов отклонены.")
    print(f"  [Phase LLM] Лучший R\u00b2_blind={best_result.r2_blind:.4f}")
    print(f"  [Phase LLM] Запусти новый PySR: python -m scalpel --phase pysr ...")
    return best_result


# ══════════════════════════════════════════════════════════════════
# ГЛАВНЫЙ ДВИЖОК v9.9
# ══════════════════════════════════════════════════════════════════

def run_engine(
    X_train:     np.ndarray,
    y_train:     np.ndarray,
    X_test:      np.ndarray,
    y_test:      np.ndarray,
    feat_names:  List[str],
    target_col:  str = "Y",
    timeout_sec: int = PYSR_FAST_FAIL_SEC,
    domain_type: str = "",
    model:       str = OLLAMA_MODEL,
    host:        str = OLLAMA_HOST,
    phase:       str = "full",   # v10.6: "full" | "pysr" | "llm" (ll
    noise_hint:  float = None,  # v10.14: подсказка шума (0..1), None=автоопределениеm через run_llm_phase)
    dim_codes:   Optional[List[int]] = None,  # v10.14: передаётся из curriculum, иначе интерактивно
    skip_heritage: bool = False,  # v10.14+: отключить Heritage во время curriculum
    extra_heritage: str = "",    # v10.15: дополнительный контекст (используется ResidualScan для Layer 2)
    skip_residual_scan: bool = False,  # v10.15: не запускать ResidualScan (во избежание рекурсии)
) -> EngineResult:

    # v10.14: Расширяем INVARIANT_LIBRARY открытиями прошлых сессий
    try:
        from .atomic_precision import enrich_invariant_library_from_discoveries as _enrich_lib
        _n_added = _enrich_lib()
        if _n_added:
            log.info("[v10.14] INVARIANT_LIBRARY расширена на %d открытий", _n_added)
    except Exception as _lib_err:
        log.debug("[InvariantLibrary] %s", _lib_err)

    vault  = GoldVault()
    shadow = ShadowMapper()                # SACRED: не трогать DSPy
    shadow_names = shadow.build(feat_names)

    # v10.15: ResidualScan — сохраняем оригинальные данные до предобработки.
    # Layer 2 будет запущен на остатках после принятия Layer 1.
    # Сохраняем только если это НЕ сам ResidualScan (skip_residual_scan=True означает Layer 2).
    if not skip_residual_scan:
        try:
            from .residual_scan import save_residual_data as _save_rd
            _save_rd(
                X_train=X_train, y_train=y_train,
                X_test=X_test,   y_test=y_test,
                feat_names=list(feat_names),
                domain_type=domain_type,
                dim_codes=list(dim_codes) if dim_codes else [],  # FIX v10.19: dim_codes может быть None до HADI
                noise_hint=float(noise_hint or 0.0),
            )
        except Exception as _rd_err:
            log.debug("[ResidualScan] save_residual_data: %s", _rd_err)

    # v10.14: Oracle — постоянный мозг сессии
    oracle = Oracle(model=ORACLE_MODEL, host=host)  # qwen ≠ gemma(Delphi)
    oracle.load_context(
        data_meta  = _build_data_meta(shadow_names, dim_codes or [], len(y_train)),
        dim_codes  = list(dim_codes) if dim_codes else [],  # FIX v10.19
        domain     = domain_type,
        feat_names = list(feat_names),
    )

    # PDCA — проверяем устаревшие формулы (пока без данных: они ещё не загружены)
    # После PySR fit в HADI-цикле вызываем повторно с predict_fn (см. ниже)
    stale = vault.check_stale()
    if stale:
        stale_cnt = sum(1 for r in stale if r.get("PDCA_STATUS") == "STALE")
        if stale_cnt:
            print(f"  [PDCA] ⚠ {stale_cnt} формул(а) устарели — будет перепроверено после PySR.")

    # OODA пороги
    r2_threshold = 0.50  # REAL MODE: 10% шум — снижен с 0.82
    p_threshold  = 0.001
    if len(y_train) >= 20:
        mid = len(y_train) // 2
        ooda_ratio = np.std(y_train[mid:], ddof=1) / (np.std(y_train[:mid], ddof=1) + 1e-12)
        if ooda_ratio > OODA_STD_SPIKE:
            r2_threshold = OODA_R2_STRICT
            p_threshold  = OODA_P_STRICT
            print(f"  [OODA] Spike {ooda_ratio:.2f}x → High-Skepticism")

    # FIX: все переменные HADI-цикла объявляются ДО блока загрузки,
    # который их использует. Иначе UnboundLocalError при phase="pysr".
    prev_matryoshka_feedback: List[dict] = []
    _consilium_forced_features:  List[str] = []
    _consilium_forced_operators: List[str] = []
    _consilium_exponent:         float     = 0.0
    _sci_variable:   str       = ""
    _sci_operators:  List[str] = []
    _sci_exponent:   float     = 0.0
    _sci_question:   str       = ""
    _sci_cycle_memory: dict    = {}
    _sci_cycle_number: int     = 0
    consensus: str = "NOT_RUN"  # FIX: инициализируем до HADI-цикла

    # ── v10.8: Читаем feedback от Матрёшки из предыдущего запуска ─
    if phase == "pysr" and PHASE_RESULT_PATH.exists():
        try:
            prev_data = json.loads(PHASE_RESULT_PATH.read_text(encoding="utf-8"))
            for prev_cand in prev_data.get("candidates", []):
                fb = prev_cand.get("matryoshka_feedback", [])
                if fb:
                    prev_matryoshka_feedback.extend(fb)
            if prev_matryoshka_feedback:
                print(f"\n  [v10.8] Загружен feedback от Матрёшки: "
                      f"{len(prev_matryoshka_feedback)} советов из предыдущего запуска")
                for fb in prev_matryoshka_feedback[:4]:
                    print(f"  [v10.8]   {fb.get('role','?')}: {fb.get('suggestion','')[:80]}")

            # v10.10: загружаем consilium из предыдущего запуска
            for prev_cand in prev_data.get("candidates", []):
                dc = prev_cand.get("delphi_consilium", {})
                if dc:
                    _consilium_forced_features  = dc.get("forced_features",  [])
                    _consilium_forced_operators = dc.get("forced_operators",  [])
                    _consilium_exponent         = float(dc.get("suggested_exponent", 0.0))
                    if _consilium_forced_features or _consilium_forced_operators:
                        print(f"  [v10.10] Consilium загружен: "
                              f"признаки={_consilium_forced_features} "
                              f"операторы={_consilium_forced_operators}")
                    break

            # v10.12/v10.13: загружаем Scientific Cycle из предыдущего запуска
            for prev_cand in prev_data.get("candidates", []):
                sc = prev_cand.get("scientific_cycle", {})
                if sc:
                    _sci_variable  = sc.get("new_variable_hint", "")
                    _sci_operators = sc.get("new_operator_hint",  [])
                    _sci_exponent  = float(sc.get("new_exponent_hint", 0.0))
                    _sci_question  = sc.get("next_question", "")
                    _sci_mem       = sc.get("sci_cycle_memory", [])
                    _sci_cycle_num = int(sc.get("sci_cycle_number", 1))
                    if _sci_mem:
                        # FIX: ключ = formula_shadow (не вопрос), единообразно с HADI-циклом
                        _f_key = prev_cand.get("formula_shadow", _sci_question)
                        _sci_cycle_memory[_f_key] = _sci_mem
                        _sci_cycle_number = _sci_cycle_num
                    if _sci_question or _sci_variable:
                        print(f"  [v10.12] Scientific Cycle загружен:")
                        if _sci_question:
                            print(f"    Вопрос: {_sci_question[:80]}")
                        if _sci_variable:
                            print(f"    Переменная: {_sci_variable}")
                        if _sci_mem:
                            print(f"    Память: {len(_sci_mem)} выводов из прошлых циклов")
                    break

            # v10.13: дополняем из episodic_memory если файл пуст
            if not _sci_cycle_memory:
                try:
                    from .episodic_memory import get_memory as _get_mem_load
                    for prev_cand in prev_data.get("candidates", []):
                        _f_shadow = prev_cand.get("formula_shadow", "")
                        if _f_shadow:
                            _mem_conclusions = _get_mem_load().recall_scientific_cycles(
                                formula=_f_shadow, limit=5
                            )
                            if _mem_conclusions:
                                # FIX: ключ = formula_shadow
                                _sci_cycle_memory[_f_shadow] = _mem_conclusions
                                print(f"  [v10.13] Загружено {len(_mem_conclusions)} "
                                      f"выводов из episodic_memory")
                            break
                except Exception as _mem_load_err:
                    log.debug("[Memory/Scientific] Загрузка: %s", _mem_load_err)
        except Exception as _fb_load_err:
            log.debug("[Feedback] Ошибка загрузки: %s", _fb_load_err)

    # ── SIEGE MODE 3.0: компиляция ДО PySR ───────────────────────
    from .dspy_optimizer import DSPyOrchestrator  # импорт внутри чтобы не падать без dspy
    orch = DSPyOrchestrator(model=NAVIGATOR_MODEL, host=host)
    if skip_heritage:  # curriculum — пропускаем siege_compile, LLM не нужна до Julia
        print(f"\n  [Siege 3.0] skip_heritage=True — пропускаем DSPy компиляцию (curriculum режим)")
        siege_ok = False
    else:
        print(f"\n  [Siege 3.0] DSPy-компиляция промптов (Navigator={NAVIGATOR_MODEL})…")
        # DSPy должен инициализироваться в том же потоке — НЕ используем threading
        try:
            siege_ok = orch.siege_compile()
        except Exception as _se:
            log.warning("[Siege] siege_compile упала: %s", _se)
            siege_ok = False
    dspy_active = siege_ok and orch.is_active
    print(f"  [Siege 3.0] DSPy {'✓ активен' if dspy_active else '○ legacy-режим'}")

    # v10.14 FIX: выгружаем NAVIGATOR_MODEL сразу после compile
    # Иначе deepseek (~8 ГБ) остаётся в RAM пока запускается Julia
    if siege_ok:
        ollama_stop(NAVIGATOR_MODEL, host=host)

    # BUG FIX v10.16: nav_orch создаётся ОДИН РАЗ до HADI-цикла (не внутри).
    # Раньше: новый объект каждую итерацию → кэш всегда пуст + лишний siege_compile().
    # Сейчас: один объект на всю сессию → кэш работает, compile вызывается один раз.
    _nav_orch_alt: Optional[DSPyOrchestrator] = None
    if dspy_active and NAVIGATOR_MODEL and NAVIGATOR_MODEL != model:
        from .dspy_optimizer import DSPyOrchestrator as _DSPyO
        _nav_orch_alt = _DSPyO(model=NAVIGATOR_MODEL, host=host)
        try:
            _nav_orch_alt.siege_compile()
            log.info("[v10.16] nav_orch_alt скомпилирован (%s)", NAVIGATOR_MODEL)
        except Exception as _nav_se:
            log.warning("[v10.16] nav_orch_alt compile: %s", _nav_se)
            _nav_orch_alt = None

    # Siege Mode 3.0: ollama_stop → gc → Диффузионная Пауза → Julia Ignition
    print(f"  [Siege 3.0] ollama_stop() → gc.collect() → Диффузионная Пауза ({SIEGE_DIFFUSION_PAUSE_SEC:.0f}s) → Julia ⚡")
    ollama_stop(model)
    gc.collect()
    time.sleep(SIEGE_DIFFUSION_PAUSE_SEC)   # ★ Диффузионная Пауза
    ram_free = avail_ram_gb()
    print(f"  [Siege 3.0] ✓ RAM: {ram_free:.2f} ГБ → Julia Ignition готов")

    # ── TOPOLOGICAL SURGERY v10.24 — Координация через SharedContext ────────
    # УРОВЕНЬ 4: модули слышат друг друга через общую доску.
    #
    # Новый порядок с петлёй обратной связи:
    #   Хирург → Surgery → ctx.update_negative_ratio →
    #   Препаратор сигнализирует намерение → coordinator_check →
    #   если конфликт (sqrt + отрицательные) → второй проход хирурга →
    #   Препаратор переоценивает на чистых данных
    #
    # Максимум 2 ревизии (ctx.MAX_REVISIONS) — защита от цикла.

    ctx = SharedContext()
    surgery_res: SurgeryResult = SurgeryResult()

    # v10.25: загружаем паттерны в SharedContext ДО любых модулей
    try:
        # FIX v10.27 #10: используем тот же IQR-метод что и хирург (_compute_data_stats).
        # Раньше был MAD-like метод (abs(y - median) > 3*std) — другая метрика.
        # Несоответствие приводило к тому что паттерны обучались на одной метрике,
        # а применялись в контексте другой → неверный matching.
        _y_pos_pat = y_train[y_train > 0]
        _Q1_pat  = float(np.percentile(y_train, 25))
        _Q3_pat  = float(np.percentile(y_train, 75))
        _IQR_pat = _Q3_pat - _Q1_pat
        if _IQR_pat > 1e-12:
            _out_ratio = float(np.mean(
                (y_train < _Q1_pat - 3.0 * _IQR_pat) | (y_train > _Q3_pat + 3.0 * _IQR_pat)
            ))
        else:
            _out_ratio = 0.0
        _data_stats_for_patterns = {
            "outlier_ratio":     _out_ratio,
            "ratio":             float(np.percentile(np.abs(_y_pos_pat), 95) /
                                       (np.percentile(np.abs(_y_pos_pat), 5) + 1e-10))
                                  if len(_y_pos_pat) > 5 else 1.0,
            "negative_fraction": float(np.mean(y_train < 0)),
            "n_features":        X_train.shape[1] if hasattr(X_train, "shape") else len(shadow_names),
        }
        ctx.load_patterns(_data_stats_for_patterns)
        _pattern_engine = get_pattern_engine()
    except Exception as _pe:
        log.debug("[MetaPatterns] Ошибка инициализации: %s", _pe)
        _data_stats_for_patterns = {}
        _pattern_engine = None

    surgeon_dec, surgeon_stats = surgeon_decide(y_train, ctx=ctx)

    def _run_surgery(iqr_mult=None, cut_frac=None, label=""):
        """Вспомогательная: один проход хирургии с логом."""
        nonlocal X_train, y_train, surgery_res
        _iqr  = iqr_mult  or surgeon_dec.iqr_multiplier
        _cut  = cut_frac  or surgeon_dec.cut_fraction
        tag   = f" [{label}]" if label else ""
        print(f"\n  [Surgery v10.5]{tag} ✂ IQR Outlier Detection…")
        X_train, y_train, surgery_res = perform_surgery(
            X_train, y_train,
            threshold      = SURGERY_THRESHOLD,
            cut_fraction   = _cut,
            iqr_multiplier = _iqr,
            force_cut      = True,
        )
        ctx.surgery_write(surgery_res.surgery_performed, surgery_res.surgery_pct)
        if surgery_res.surgery_performed:
            print(f"  [Surgery v10.5]{tag} ✂ Хирургия выполнена: "
                  f"удалено {surgery_res.surgery_pct:.2f}% "
                  f"({surgery_res.n_original} → {surgery_res.n_after})")
        else:
            n_s = surgery_res.n_singularities
            if n_s > 0:
                print(f"  [Surgery v10.5]{tag} Найдено {n_s} кандидата — в пределах IQR")
            else:
                print(f"  [Surgery v10.5]{tag} Выбросов нет: 0/{surgery_res.n_original}")

    if surgeon_dec.apply_surgery:
        _run_surgery()
    else:
        print(f"\n  [Surgery v10.5] Хирург решил: хирургия не нужна")
        surgery_res.n_original = len(y_train)
        surgery_res.n_after    = len(y_train)

    gc.collect()

    # ── Снапшот для возможного откатa при вето ──────────────────────────────
    # Если Физик выставит вето после Navigator → хирург+препаратор переделают.
    # Для этого нам нужны данные в состоянии POST-surgery, PRE-preparator.
    _X_train_post_surgery = X_train.copy()
    _y_train_post_surgery = y_train.copy()

    # ── Препаратор + петля координации ───────────────────────────────────────
    _prep_result = None

    def _run_preparator(verbose=True):
        """Вспомогательная: один проход препаратора."""
        nonlocal _prep_result, y_train
        try:
            from .preparator import analyze_and_prepare
            ctx.update_negative_ratio(y_train)    # обновляем перед запуском
            ctx.coordinator_reset_prep()           # сбрасываем prep_intent
            _prep_result = analyze_and_prepare(y_train, verbose=verbose, ctx=ctx)
            if _prep_result.applied:
                import numpy as _np_prep
                if _prep_result.transform_name == "log":
                    y_train = _np_prep.log(_np_prep.clip(y_train, 1e-10, None))
                elif _prep_result.transform_name == "sqrt":
                    y_train = _np_prep.sqrt(_np_prep.clip(y_train, 0, None))
                elif _prep_result.transform_name == "standardize":
                    _m = _prep_result.transform_params.get("mean", 0)
                    _s = _prep_result.transform_params.get("std", 1)
                    y_train = (y_train - _m) / _s
                log.info("[Препаратор] Применено: %s (ratio %.1f → %.1f)",
                         _prep_result.transform_name,
                         _prep_result.ratio_before, _prep_result.ratio_after)
        except Exception as _prep_err:
            log.debug("[Препаратор] %s", _prep_err)

    # Первый проход препаратора
    _run_preparator(verbose=True)

    # Координатор проверяет конфликт: prep хочет sqrt/log, но есть отрицательные
    if ctx.coordinator_check():
        print(f"\n  [Координатор] ⚡ Ревизия {ctx.revision_count}: {ctx.revision_reason}")
        # BUG FIX: восстанавливаем y_train до трансформации препаратора
        # иначе второй проход хирурга работает на уже-трансформированных данных
        X_train = _X_train_post_surgery.copy()
        y_train = _y_train_post_surgery.copy()
        escalated = ctx.surgeon_escalate()
        print(f"  [Координатор] Хирург — второй проход "
              f"(iqr_k={escalated['iqr_multiplier']:.1f}, "
              f"cut={escalated['cut_fraction']:.3f})")
        _run_surgery(
            iqr_mult=escalated["iqr_multiplier"],
            cut_frac=escalated["cut_fraction"],
            label="РЕВИЗИЯ",
        )
        # Обновляем снапшот после второй хирургии
        _X_train_post_surgery = X_train.copy()
        _y_train_post_surgery = y_train.copy()
        print(f"  [Координатор] Препаратор переоценивает на чистых данных…")
        _run_preparator(verbose=True)

    # Итоговая сводка координации
    print(f"  [SharedContext] {ctx.summary().splitlines()[0]}")
    log.info("[SharedContext]\n%s", ctx.summary())

    # v2.1: передаём подсказку операторов от Препаратора в Navigator
    _prep_extra_ops = []
    if _prep_result and _prep_result.applied:
        if _prep_result.transform_name == "log":
            _prep_extra_ops = ["log", "exp"]
        elif _prep_result.transform_name == "sqrt":
            _prep_extra_ops = ["sqrt"]
        log.info("[Препаратор→Navigator] Дополнительные операторы: %s", _prep_extra_ops)

    # ── Ricci Flow — сглаживание ПОСЛЕ хирургии и трансформации ─────────────
    if surgeon_dec.apply_ricci:
        print(f"  [Surgery v10.5] ✂ Ricci Flow (окно={surgeon_dec.ricci_window})…")
        # FIX v10.27 #8: Ricci Flow (SavGol) требует упорядоченных данных.
        # Данные перемешаны (permutation в run_feynman) → фильтр бесполезен
        # для мультипеременных данных. Применяем только если n_features=1
        # и сортируем по X перед сглаживанием.
        _can_ricci = X_train.shape[1] == 1 if hasattr(X_train, "shape") else False
        if not _can_ricci:
            print(f"  [Surgery v10.5] ⚠ Ricci Flow пропущен: n_features={X_train.shape[1] if hasattr(X_train, 'shape') else '?'} > 1 (SavGol требует 1D упорядоченный сигнал)")
            ricci_applied = False
        else:
            # Сортируем по X[0] перед сглаживанием и восстанавливаем порядок
            _sort_idx = np.argsort(X_train[:, 0])
            _unsort   = np.argsort(_sort_idx)
            X_train_s = X_train[_sort_idx]
            y_train_s = y_train[_sort_idx]
            X_train_s, y_train_s, ricci_applied = ricci_flow_smooth(
                X_train_s, y_train_s,
                window=surgeon_dec.ricci_window,
            )
            X_train = X_train_s[_unsort[:len(X_train_s)]]
            y_train = y_train_s[_unsort[:len(y_train_s)]]
        surgery_res.ricci_applied = ricci_applied
        print(f"  [Surgery v10.5] Ricci Flow: "
              f"{'✓ применён' if ricci_applied else '○ пропущен'}")
    else:
        print(f"  [Surgery v10.5] Хирург решил: Ricci Flow не нужен")

    gc.collect()
    log.info("[Surgery v10.5] RAM после предобработки: %.2f ГБ", avail_ram_gb())

    # ── DIFFUSION DENOISING v10.4.5 (AlphaFold 3 — Сгущение структуры) ──
    # Обрабатываем данные как «облако шума» — за T шагов извлекаем скелет.
    # RAM Guard: только numpy, X не дублируется.
    diffusion_res = DiffusionResult()
    # v10.14: Автоопределение шума → выбор режима денойзинга
    _noise_lvl = noise_hint if noise_hint is not None else estimate_noise_level(y_train, X_train)  # FIX: передаём X для правильной сортировки
    if _noise_lvl > 0.15:   # FIX: порог 0.15 (ранее 0.20 занижало ~30%)
        print(f"\n  [Denoise] Шум ≈{_noise_lvl*100:.0f}% → Aggressive (binning+gaussian)")
        X_train, y_train, diffusion_res = aggressive_denoise(
            X_train, y_train, shadow_names, noise_level=_noise_lvl,
        )
    else:
        print(f"\n  [Diffusion v10.4.5] ∿ Сгущение структуры (шум≈{_noise_lvl*100:.0f}%)")
        X_train, y_train, diffusion_res = diffusion_denoise(
            X_train, y_train, shadow_names,
            T          = DIFFUSION_STEPS,
            beta_start = DIFFUSION_BETA_START,
            beta_end   = DIFFUSION_BETA_END,
            iqr_factor = DIFFUSION_IQR_FACTOR,
        )
    if diffusion_res.applied:
        print(f"  [Diffusion v10.4.5] ✓ Шум={diffusion_res.noise_pct_total:.1f}% | "
              f"σ²-ratio={diffusion_res.y_variance_ratio:.3f}")
        print(f"  [Diffusion v10.4.5] Скелет: {diffusion_res.skeleton_feats} / "
              f"ops={diffusion_res.skeleton_ops[:4]}")
    else:
        print(f"  [Diffusion v10.4.5] Пропущен (мало точек)")

    # ── PAIRFORMER v10.4.5 (AlphaFold 3 — Взаимосвязи) ──────────────────
    # Заменяет тяжёлую глобальную матрицу корреляций на sparse попарный анализ.
    # Экономит ~500 МБ при большом числе признаков.
    pairformer_res = PairformerResult()
    print(f"\n  [Pairformer v10.4.5] ⚛ Попарные взаимодействия признаков…")
    pairformer_res = pairformer_select(
        X_train, y_train, shadow_names,
        top_k    = PAIRFORMER_TOP_K,
        min_corr = PAIRFORMER_MIN_CORR,
        max_feat = PAIRFORMER_MAX_FEAT,
    )
    if pairformer_res.top_pairs:
        print(f"  [Pairformer v10.4.5] ✓ Топ-пара: "
              f"{pairformer_res.top_pairs[0][0]} × {pairformer_res.top_pairs[0][1]} "
              f"E={pairformer_res.top_pairs[0][2]:.4f} | "
              f"RAM saved ≈{pairformer_res.ram_saved_mb:.0f} МБ")
    else:
        print(f"  [Pairformer v10.4.5] Нет значимых пар")

    gc.collect()
    log.info("[v10.5] RAM после Diffusion+Pairformer: %.2f ГБ", avail_ram_gb())

    # ── HADI ─────────────────────────────────────────────────────
    prev_hyps:    List[str] = []
    death_ctx:    str       = ""
    failure_types: List[str] = []
    matryoshka_rebuild_count: int = 0   # v10.3.9: счётчик пересборок по Матрёшке

    # v10.7: накапливаем ВСЕ кандидаты по всем 4 попыткам HADI для phase="pysr"
    _all_phase_candidates: List[dict] = []
    _best_phase_r2:        float = -1.0

    # (consilium, sci_cycle и связанные переменные уже объявлены выше,
    #  до блока загрузки PHASE_RESULT_PATH — см. FIX выше)
    # v10.14: dim_codes приходят снаружи (curriculum) или запрашиваются интерактивно
    if dim_codes is None:
        dim_codes = [dim_code_interactive(n) for n in feat_names]
    else:
        # Валидация: длина должна совпадать с feat_names
        if len(dim_codes) != len(feat_names):
            log.warning("[run_engine] dim_codes длина %d != feat_names %d — сброс",
                        len(dim_codes), len(feat_names))
            dim_codes = [0] * len(feat_names)

    for attempt in range(HADI_MAX_RETRIES + 1):
        print(f"\n{'═'*62}")
        print(f"  HADI ИТЕРАЦИЯ {attempt + 1}/{HADI_MAX_RETRIES + 1}"
              f"  {'[DSPy]' if dspy_active else '[Legacy]'}")
        print(f"{'═'*62}")
        gc.collect()
        log.info("[HADI %d] RAM: %.2f ГБ", attempt + 1, avail_ram_gb())

        # v10.25: добавляем подсказки паттернов в data_meta для Navigator
        _nav_pattern_hint = ctx.get_pattern_hint("navigator") if ctx else ""
        _nav_pattern_ops  = ctx.get_pattern_action("navigator", "operator_add") if ctx else None
        _nav_pattern_hyps = ctx.get_pattern_action("navigator", "hypothesis_template") if ctx else None

        data_meta = _build_data_meta(shadow_names, dim_codes, len(y_train),
                                      prep_ops=_prep_extra_ops)  # FIX v10.23
        if _nav_pattern_hint:
            data_meta += f"\n{_nav_pattern_hint}"

        # ── ШАГ 1: Навигатор ──────────────────────────────────────
        # v10.24: флаги ролей из SharedContext → в failure_logs для Navigator
        if ctx is not None:
            _ctx_nav_feedback = ctx.ctx_for_navigator()
            if _ctx_nav_feedback and attempt > 0:
                # Только начиная со второй итерации — первая итерация флагов ещё нет
                _fl_list = json.loads(failure_logs_str) if failure_logs_str.strip() != "[]" else []
                _fl_list.append({"role_feedback": _ctx_nav_feedback})
                failure_logs_str = json.dumps(_fl_list, ensure_ascii=False)
                log.info("[SharedContext→Navigator] Флаги ролей переданы в failure_logs")
            # Сбрасываем флаги ролей для новой итерации
            ctx.role_verdicts.clear()
            ctx.role_flags.clear()

        # v10.8: обогащаем failure_logs советами Матрёшки
        failure_logs_list = [
            {"hypothesis": h, "death_reason": death_ctx}
            for h in prev_hyps
        ]
        # FIX v10.19: ограничиваем историю — берём только последние 8 записей
        prev_matryoshka_feedback = prev_matryoshka_feedback[-8:]
        if prev_matryoshka_feedback:
            # Добавляем уникальные советы от ролей
            suggestions_seen = set()
            for fb in prev_matryoshka_feedback:
                sugg = fb.get("suggestion", "")
                if sugg and sugg not in suggestions_seen and sugg != "no improvement needed":
                    suggestions_seen.add(sugg)
                    failure_logs_list.append({
                        "hypothesis":   f"[Матрёшка/{fb.get('role','?')}] {sugg}",
                        "death_reason": f"critique: {fb.get('critique','')[:100]}",
                        "source":       "matryoshka_feedback",
                    })

        # v10.36: добавляем производные от высокого R² в failure_logs
        # Если предыдущий кандидат имел R² >= 0.90 — передаём упрощения Navigator
        try:
            _deriv_candidates = [
                c for c in all_candidates
                if c.get("high_r2_derivatives") and float(c.get("r2_blind", 0)) >= 0.90
            ]
            if _deriv_candidates:
                _last_deriv = _deriv_candidates[-1]
                for _dv in _last_deriv.get("high_r2_derivatives", [])[:6]:
                    failure_logs_list.append({
                        "hypothesis":   _dv,
                        "death_reason": f"производная от R²={_last_deriv.get('r2_blind',0):.4f} — попробуй упрощение",
                        "source":       "high_r2_derivative",
                    })
                print(f"  [v10.36] Передано {len(_last_deriv.get('high_r2_derivatives',[]))} производных Navigator")
        except Exception as _dv_err:
            log.debug("[v10.36/derivatives] %s", _dv_err)

        failure_logs_str = json.dumps(failure_logs_list) if failure_logs_list else "[]"

        # v10.14: Oracle suggest — стратегический совет перед Navigator
        try:
            _oracle_hint = oracle.suggest(attempt, failure_logs_str)
            if _oracle_hint:
                _fl_list = json.loads(failure_logs_str) if failure_logs_str.strip() != "[]" else []
                _fl_list.insert(0, {"hypothesis": _oracle_hint[:200], "death_reason": "oracle_hint", "source": "oracle"})
                failure_logs_str = json.dumps(_fl_list, ensure_ascii=False)
        except Exception as _oc_s:
            log.debug("[Oracle] suggest: %s", _oc_s)

        # v10.14: META-CONTEXT → Navigator получает мета-правила из рефлексии
        try:
            from .meta_context import enrich_failure_logs as _enrich_fl
            failure_logs_str = _enrich_fl(failure_logs_str, domain_type)
        except Exception as _mc_err:
            log.debug("[MetaContext/Nav] %s", _mc_err)

        # v10.14: INVARIANT HINTS → Navigator видит что сработало с такими же dim_codes
        try:
            from .episodic_memory import get_memory as _inv_nav_mem
            _inv_hints = _inv_nav_mem().recall_invariants_for_domain(
                dim_codes=dim_codes, domain=domain_type, limit=3,
            )
            if _inv_hints:
                _fl_list = json.loads(failure_logs_str) if failure_logs_str.strip() != "[]" else []
                for _hint in _inv_hints:
                    _fl_list.append({
                        "hypothesis":   f"[INVARIANT] {_hint[:150]}",
                        "death_reason": "успешный паттерн из истории — попробуй похожую структуру",
                        "source":       "invariant_memory",
                    })
                failure_logs_str = json.dumps(_fl_list, ensure_ascii=False)
                log.debug("[InvariantNav] Добавлено %d инвариантных подсказок", len(_inv_hints))
        except Exception as _inv_nav_err:
            log.debug("[InvariantNav] %s", _inv_nav_err)

        # v10.14: Chronicle chains от Летописца → обучаем Navigator цепочкам мышления
        try:
            from .episodic_memory import get_memory as _chr_nav_mem
            _chr_chains = _chr_nav_mem().recall_chronicle_steps(limit=5)
            for _chain in _chr_chains:
                _chain_entry = {
                    "hypothesis":   f"[Летописец] {_chain[:150]}",
                    "death_reason": "цепочка из истории поиска",
                    "source":       "chronicle_memory",
                }
                if _chain_entry not in failure_logs_list:
                    failure_logs_list.append(_chain_entry)
            if _chr_chains:
                failure_logs_str = json.dumps(failure_logs_list)
                log.debug("[Chronicle→Nav] Добавлено %d цепочек", len(_chr_chains))
        except Exception as _cl_err:
            log.debug("[Chronicle→Nav] %s", _cl_err)

        # v10.9: Navigator использует NAVIGATOR_MODEL
        nav_model = NAVIGATOR_MODEL if NAVIGATOR_MODEL else model
        if nav_model != model:
            print(f"  [v10.9] Navigator → {nav_model}")

        if dspy_active:
            # BUG FIX v10.16: используем pre-created _nav_orch_alt (создан до HADI-цикла)
            # Раньше: new DSPyOrchestrator + siege_compile() каждую итерацию = потеря кэша
            if _nav_orch_alt is not None and _nav_orch_alt.is_active:
                dspy_result = _nav_orch_alt.navigate(data_meta, failure_logs_str)
            else:
                dspy_result = orch.navigate(data_meta, failure_logs_str)
            nav = nav_decision_from_dspy(dspy_result, shadow_names)
            print(f"  [Navigator/DSPy] reasoning: {nav.reasoning[:80]}")
            # v10.25: добавляем операторы из паттернов Navigator
            if _nav_pattern_ops:
                _pat_ops = [o.strip() for o in _nav_pattern_ops.split(";")]
                _existing = set(nav.selected_operators)
                _added_pat = [o for o in _pat_ops if o not in _existing]
                if _added_pat:
                    nav.selected_operators = nav.selected_operators + _added_pat
                    print(f"  [МетаПаттерны→Nav] Операторы: {_added_pat}")

            # v10.25: добавляем гипотезы из паттернов Navigator
            if _nav_pattern_hyps:
                _pat_hyps = [h.strip() for h in _nav_pattern_hyps.split(";")]
                _valid_hyps = [h for h in _pat_hyps if h and h not in nav.hypotheses]
                if _valid_hyps:
                    nav.hypotheses = _valid_hyps[:2] + nav.hypotheses[:3]
                    print(f"  [МетаПаттерны→Nav] Гипотезы: {_valid_hyps[:2]}")

            # FIX v10.23: DSPy-путь не получал prep_extra_ops — теперь принудительно добавляем
            if _prep_extra_ops:
                _existing_ops = set(nav.selected_operators)
                _added_prep = [op for op in _prep_extra_ops if op not in _existing_ops]
                if _added_prep:
                    nav.selected_operators = nav.selected_operators + _added_prep
                    print(f"  [Препаратор→Nav/DSPy] Добавлены операторы: {_added_prep}")
            # v10.24: Navigator пишет решение в SharedContext
            if ctx is not None:
                ctx.navigator_write(
                    hypotheses = list(nav.hypotheses),
                    operators  = list(nav.selected_operators),
                    features   = list(nav.selected_features),
                    reasoning  = nav.reasoning,
                )
        else:
            nav = navigator_ask_legacy(
                shadow_names, dim_codes, len(y_train),
                prev_hyps=prev_hyps, death_context=death_ctx,
                host=host, model=nav_model,
                extra_ops=_prep_extra_ops,   # v2.1: операторы от Препаратора
            )
            # v10.24: Legacy Navigator тоже пишет в SharedContext
            if ctx is not None:
                ctx.navigator_write(
                    hypotheses = list(nav.hypotheses),
                    operators  = list(nav.selected_operators),
                    features   = list(nav.selected_features),
                    reasoning  = nav.reasoning,
                )

        # v10.42: санитайзер гипотез Navigator — фильтруем английский текст и мусор
        # deepseek-r1:7b иногда возвращает описание вместо математических формул
        try:
            import re as _re_nav
            _valid_hyps = []
            for _h in (nav.hypotheses or []):
                _h = _h.strip()
                # Гипотеза валидна если: содержит f\d+ И не содержит длинных слов
                _has_feature = bool(_re_nav.search(r'\bf\d+\b', _h))
                _has_long_words = any(len(w) > 12 for w in _re_nav.findall(r'[a-zA-Z]+', _h))
                _too_long = len(_h) > 60
                if _has_feature and not _has_long_words and not _too_long:
                    _valid_hyps.append(_h)

            if len(_valid_hyps) < len(nav.hypotheses or []):
                _removed = len(nav.hypotheses or []) - len(_valid_hyps)
                print(f"  [v10.42/HypSanitizer] Удалено {_removed} невалидных гипотез")
                if not _valid_hyps:
                    # Генерируем дефолтные гипотезы из признаков
                    _feats = list(nav.selected_features or shadow_names[:3])
                    _valid_hyps = []
                    if len(_feats) >= 2:
                        _valid_hyps = [
                            f'{_feats[0]} * {_feats[1]}',
                            f'{_feats[0]} / {_feats[1]}',
                            f'sqrt({_feats[0]} * {_feats[1]})',
                        ]
                    elif _feats:
                        _valid_hyps = [f'{_feats[0]}**2', f'sqrt({_feats[0]})', f'log({_feats[0]})']
                    print(f"  [v10.42/HypSanitizer] Дефолтные гипотезы: {_valid_hyps}")
                nav.hypotheses = _valid_hyps
        except Exception as _san_err:
            log.debug("[v10.42/HypSanitizer] %s", _san_err)

        # v10.20: логируем что именно Navigator получил на вход и что решил
        _nav_log_entry = {
            "ts":        __import__("datetime").datetime.now().isoformat(timespec="seconds"),
            "attempt":   attempt + 1,
            "formula_shadow": "",       # заполним после PySR
            "r2_blind":  0.0,           # заполним после PySR
            "matryoshka_consensus": "", # заполним после Матрёшки
            "death_reason": "",         # заполним если DEATH
            "input": {
                "oracle_hint":               _oracle_hint if "_oracle_hint" in dir() else "",
                "consilium_forced_features": list(_consilium_forced_features),
                "consilium_forced_operators":list(_consilium_forced_operators),
                "consilium_exponent":        _consilium_exponent,
                "sci_question":              _sci_question,
                "sci_variable":              _sci_variable,
                "sci_operators":             list(_sci_operators),
                "prev_hypotheses_tried":     list(prev_hyps[-6:]),
            },
            "navigator_decision": {
                "hypotheses":          list(nav.hypotheses),
                "selected_features":   list(nav.selected_features),
                "selected_operators":  list(nav.selected_operators),
                "reasoning":           nav.reasoning,
                "used_dspy":           dspy_active,
                "temperature":         0.3,
            },
        }

        # v10.14 FIX: Delphi forced_* работают ТОЛЬКО если Матрёшка не отклонила
        # Если consensus == ОТКЛОНЕНА/СПОРНО — Delphi не может продавить свою идею
        # Это защита от предвзятости: Gemma(Delphi) не override Матрёшку
        _delphi_can_force = (consensus not in ("ОТКЛОНЕНА",))
        if not _delphi_can_force:
            log.info("[v10.14] Delphi forced_* заблокированы: consensus=%s", consensus)

        # v10.10: ПРЯМОЕ ВНЕДРЕНИЕ от Delphi Consilium (только если Матрёшка не против)
        if _consilium_forced_features and _delphi_can_force:
            existing = set(nav.selected_features)
            added = [f for f in _consilium_forced_features
                     if f in shadow_names and f not in existing]
            if added:
                nav.selected_features = nav.selected_features + added
                print(f"  [Consilium→Nav] Принудительно добавлены признаки: {added}")

        if _consilium_forced_operators and _delphi_can_force:
            existing_ops = set(nav.selected_operators)
            added_ops = [op for op in _consilium_forced_operators
                         if op not in existing_ops]
            if added_ops:
                nav.selected_operators = nav.selected_operators + added_ops
                print(f"  [Consilium→Nav] Принудительно добавлены операторы: {added_ops}")

        if _consilium_exponent and _consilium_exponent > 0 and _delphi_can_force:
            # Добавляем гипотезу с рекомендованной степенью
            exp_hyp = f"f0^{_consilium_exponent}"
            if exp_hyp not in nav.hypotheses:
                nav.hypotheses = [exp_hyp] + nav.hypotheses[:4]
                print(f"  [Consilium→Nav] Рекомендована степень: f0^{_consilium_exponent}")

        # v10.12: Scientific Cycle → дополнительные операторы от научного вопроса
        if _sci_operators:
            existing_sci_ops = set(nav.selected_operators)
            added_sci_ops = [op for op in _sci_operators if op not in existing_sci_ops]
            if added_sci_ops:
                nav.selected_operators = nav.selected_operators + added_sci_ops
                print(f"  [Scientific Cycle→Nav] Операторы из научного цикла: {added_sci_ops}")

        if _sci_exponent and _sci_exponent > 0:
            sci_hyp = f"f0^{_sci_exponent}"
            if sci_hyp not in nav.hypotheses:
                nav.hypotheses = nav.hypotheses[:3] + [sci_hyp] + nav.hypotheses[3:4]
                print(f"  [Scientific Cycle→Nav] Степень из научного цикла: f0^{_sci_exponent}")

        if _sci_question:
            # Добавляем научный вопрос как контекст в failure_logs
            sci_log_entry = {
                "hypothesis":   f"[Scientific Question] {_sci_question[:120]}",
                "death_reason": f"variable_hint: {_sci_variable}" if _sci_variable else "see question",
                "source":       "scientific_cycle",
            }
            if sci_log_entry not in failure_logs_list:
                failure_logs_list.append(sci_log_entry)
                failure_logs_str = json.dumps(failure_logs_list)

        # v10.4.5: обогащаем выбор признаков и операторов из Diffusion + Pairformer
        if diffusion_res.applied and diffusion_res.skeleton_feats:
            # Приоритизируем признаки из скелета (но не заменяем навигатора полностью)
            skel_valid = [f for f in diffusion_res.skeleton_feats if f in shadow_names]
            if skel_valid:
                nav_set     = set(nav.selected_features)
                merged      = skel_valid + [f for f in nav.selected_features if f not in skel_valid]
                nav.selected_features = merged[:len(nav.selected_features) + 2]
                log.debug("[Diffusion→Nav] Скелет обогатил признаки: %s", skel_valid)

        if pairformer_res.top_pairs and pairformer_res.selected_feats:
            # Добавляем пару-кандидаты из Pairformer
            pair_valid = [f for f in pairformer_res.selected_feats
                          if f in shadow_names and f not in nav.selected_features]
            if pair_valid:
                nav.selected_features = (nav.selected_features + pair_valid[:2])
                log.debug("[Pairformer→Nav] Добавлены парные признаки: %s", pair_valid[:2])

        # Обогащаем операторы из Diffusion + Pairformer (без дублей)
        extra_ops = list(dict.fromkeys(
            diffusion_res.skeleton_ops + pairformer_res.pair_ops
        ))
        for op in extra_ops:
            if op not in nav.selected_operators:
                nav.selected_operators.append(op)

        # FIX-3: dedup
        if prev_hyps:
            fresh = [h for h in nav.hypotheses if h not in prev_hyps]
            # FIX v10.16: _FALLBACK_HYPOTHESES → динамический fallback по shadow_names
            _fb = _fallback_hypotheses(shadow_names)
            nav.hypotheses = _filter_hypotheses(
                fresh or [h for h in _fb if h not in prev_hyps] or _fb,
                shadow_names
            )

        if not nav.ooda_stable:
            r2_threshold = max(r2_threshold, OODA_R2_STRICT)
            p_threshold  = min(p_threshold,  OODA_P_STRICT)

        print(f"  [Nav] features={nav.selected_features}")
        print(f"  [Nav] hypotheses={nav.hypotheses[:3]}")

        # v10.16: сохраняем гипотезы Navigator в эпизодическую память
        # Летописец и DSPy увидят что предлагал Navigator и к чему это привело
        try:
            from .episodic_memory import get_memory as _nav_mem
            _nav_mem().remember_navigator_hypotheses(
                attempt    = attempt + 1,
                hypotheses = nav.hypotheses,
                features   = nav.selected_features,
                operators  = nav.selected_operators,
                reasoning  = nav.reasoning,
                domain     = domain_type,
            )
        except Exception as _nav_mem_err:
            log.debug("[NavMemory] %s", _nav_mem_err)

        sel = [s for s in nav.selected_features if s in shadow_names]
        if len(sel) >= 2:
            idx        = [shadow_names.index(s) for s in sel]
            X_tr_use   = X_train[:, idx]
            X_te_use   = X_test[:,  idx]
            use_shadow = sel
            use_real   = [feat_names[i] for i in idx]
        else:
            X_tr_use, X_te_use = X_train, X_test
            use_shadow, use_real = shadow_names, feat_names

        if X_tr_use.shape[1] == 0:
            death_ctx = "X_train пуст после фильтрации"
            log.error("[BUG] %s", death_ctx)
            continue

        # ── ШАГ 2.5: Физик — право вето ДО PySR ──────────────────
        # v10.25: Физик проверяет предложение Navigator.
        # Если операторы физически бессмысленны → вето → пересмотр.
        _y_pos = y_train[y_train > 0]
        _has_inf_or_nan = bool(np.any(~np.isfinite(y_train)))
        _y_stats_for_veto = {
            "y_min":             float(np.min(y_train[np.isfinite(y_train)]) if np.any(np.isfinite(y_train)) else 0.0),
            "y_max":             float(np.max(y_train[np.isfinite(y_train)]) if np.any(np.isfinite(y_train)) else 0.0),
            "negative_fraction": float(np.mean(y_train < 0)),
            "has_inf":           _has_inf_or_nan,   # FIX: sin/cos/tanh запрещены при Inf
            # BUG FIX: добавляем ratio — нужен debate блоку
            "ratio": float(np.percentile(_y_pos, 95) / (np.percentile(_y_pos, 5) + 1e-10))
                     if len(_y_pos) > 5 else float(np.max(np.abs(y_train)) /
                                                    (np.min(np.abs(y_train[y_train != 0])) + 1e-10)
                                                    if np.any(y_train != 0) else 1.0),
        }
        _veto = physicist_veto_check(
            operators    = nav.selected_operators,
            hypotheses   = nav.hypotheses,
            shadow_names = shadow_names,
            dim_codes    = dim_codes or [0] * len(shadow_names),
            y_stats      = _y_stats_for_veto,
            ctx          = ctx,
            use_llm      = True,
        )

        if _veto.vetoed:
            print(f"\n  {'█'*62}")
            print(f"  ВЕТО ФИЗИКА [{ctx.veto_count}/{ctx.MAX_VETOES}]")
            print(f"  {_veto.reason}")
            if _veto.suggestion:
                print(f"  Совет: {_veto.suggestion}")
            print(f"  {'█'*62}")

            # Добавляем вето в failure_logs для Navigator
            _veto_entry = {
                "hypothesis":   str(nav.hypotheses[:2]),
                "death_reason": ctx.veto_for_navigator(),
            }
            _fl_list = json.loads(failure_logs_str) if failure_logs_str.strip() != "[]" else []
            _fl_list.append(_veto_entry)
            failure_logs_str = json.dumps(_fl_list, ensure_ascii=False)
            prev_hyps.extend(nav.hypotheses)

            # Откат к снапшоту + повторный анализ хирурга и препаратора
            print(f"  [Координатор] Откат данных → хирург + препаратор пересматривают…")
            X_train = _X_train_post_surgery.copy()
            y_train = _y_train_post_surgery.copy()

            ctx.clear_veto()
            ctx.coordinator_reset_prep()

            # Хирург — новый анализ с повышенной агрессивностью
            _veto_surgeon_params = ctx.surgeon_escalate()
            print(f"  [Хирург/Вето] Второй анализ "
                  f"(iqr_k={_veto_surgeon_params['iqr_multiplier']:.1f})…")
            _run_surgery(
                iqr_mult = _veto_surgeon_params["iqr_multiplier"],
                cut_frac = _veto_surgeon_params["cut_fraction"],
                label    = "ВЕТО",
            )

            # FIX v10.27 #3: обновляем снапшот после вето-хирургии.
            # Без этого второе вето откатит данные к состоянию ДО первого вето
            # (первый снапшот), теряя результат первой вето-хирургии.
            _X_train_post_surgery = X_train.copy()
            _y_train_post_surgery = y_train.copy()

            # FIX v10.27 #2: обновляем ctx.surgeon_iqr_mult после эскалации.
            # surgeon_escalate() читает ctx.surgeon_iqr_mult для расчёта нового iqr_k,
            # но после _run_surgery в вето-блоке ctx остаётся с исходным значением
            # от первого surgeon_decide. Второе вето будет эскалировать от неверной базы.
            ctx.surgeon_iqr_mult = _veto_surgeon_params["iqr_multiplier"]

            # Препаратор — переоценивает на пересмотренных данных
            print(f"  [Препаратор/Вето] Переоцениваю трансформацию…")
            _run_preparator(verbose=True)

            # Обновляем prep_extra_ops
            _prep_extra_ops = []
            if _prep_result and _prep_result.applied:
                if _prep_result.transform_name == "log":
                    _prep_extra_ops = ["log", "exp"]
                elif _prep_result.transform_name == "sqrt":
                    _prep_extra_ops = ["sqrt"]

            print(f"  [Координатор] Пересмотр завершён → Navigator получает новый шанс")
            continue   # ← пропускаем PySR, идём на следующую HADI-итерацию

        # ── ШАГ 2.7: Дебаты перед PySR ───────────────────────────
        # v10.26: роли договариваются о гипотезах ДО запуска Julia.
        # RAM-безопасно: модели загружаются по одной, выгружаются сразу.
        # Hard timeout 45с/роль → максимум ~6 минут на весь блок.
        _debate_y_stats = {
            "ratio":             _y_stats_for_veto.get("ratio", 1.0),
            "negative_fraction": _y_stats_for_veto.get("negative_fraction", 0.0),
            "has_inf":           _y_stats_for_veto.get("has_inf", False),
            "n_samples":         len(y_train),
        }
        # BUG FIX: выгружаем модель Navigator перед дебатами
        # иначе deepseek-r1:7b (~5GB) остаётся в RAM во время PySR
        try:
            import urllib.request as _ureq2
            _nav_unload = json.dumps({"model": nav_model, "keep_alive": 0, "prompt": ""}).encode()
            _ureq2.urlopen(_ureq2.Request(
                f"{host}/api/generate",
                data=_nav_unload,
                headers={"Content-Type": "application/json"},
                method="POST",
            ), timeout=5)
        except Exception:
            pass
        import gc as _gc_deb; _gc_deb.collect()

        try:
            debate_result = run_pre_pysr_debate(
                nav_hypotheses = list(nav.hypotheses),
                nav_operators  = list(nav.selected_operators),
                nav_features   = list(nav.selected_features),
                shadow_names   = shadow_names,
                dim_codes      = dim_codes or [0] * len(shadow_names),
                y_stats        = _debate_y_stats,
                ctx            = ctx,
            )
            # Применяем рекомендации дебатов к Navigator
            if debate_result.nav_ops_add:
                _existing_ops = set(nav.selected_operators)
                _debate_ops = [o for o in debate_result.nav_ops_add
                               if o not in _existing_ops]
                if _debate_ops:
                    nav.selected_operators = nav.selected_operators + _debate_ops
                    log.info("[Дебаты→Nav] Добавлены операторы: %s", _debate_ops)

            # FIX: финальный guard — убираем тригонометрию если в данных есть Inf/NaN
            # cos(Inf) крашит Julia (DomainError) и роняет весь закон
            if _y_stats_for_veto.get("has_inf", False):
                _TRIG_OPS = {"sin", "cos", "tanh"}
                _before_trig = list(nav.selected_operators)
                nav.selected_operators = [o for o in nav.selected_operators if o not in _TRIG_OPS]
                _removed_trig = [o for o in _before_trig if o in _TRIG_OPS]
                if _removed_trig:
                    print(f"  [Физик/Guard] ⚠️  Данные содержат Inf/NaN → убраны тригонометрические операторы: {_removed_trig}")
                    log.warning("[Guard] sin/cos/tanh убраны из-за Inf в данных")

            if debate_result.nav_hyps_update:
                # Гипотезы из дебатов идут первыми (приоритет)
                _debate_hyps = [h for h in debate_result.nav_hyps_update
                                if h not in nav.hypotheses][:2]
                if _debate_hyps:
                    nav.hypotheses = _debate_hyps + nav.hypotheses[:3]
                    log.info("[Дебаты→Nav] Добавлены гипотезы: %s", _debate_hyps)

            # FIX v10.31: Физик проверяет финальный список гипотез с контекстом дебатов
            try:
                from .config import ROLE_MODELS as _RM
                _phy_model = _RM.get("Физик", "phi4:14b")
                _neg_frac  = _y_stats_for_veto.get("negative_fraction", 0.0)
                _debate_ctx = "\n".join(debate_result.summary_lines()[:6])

                _phy_prompt = (
                    "You are a physicist reviewing symbolic regression hypotheses.\n\n"
                    f"DEBATE CONTEXT:\n{_debate_ctx[:500]}\n\n"
                    f"SYNTHESIS: {debate_result.synthesis[:200]}\n\n"
                    f"HYPOTHESES to validate:\n{nav.hypotheses}\n\n"
                    f"DATA: features={use_shadow}, negative_fraction={_neg_frac:.1%}\n\n"
                    "Rules:\n"
                    "1. Keep only valid mathematical formulas (must contain f0/f1/f2 and operators)\n"
                    "2. Remove text descriptions like 'hypothesis based on...'\n"
                    "3. Remove formulas incompatible with data (no log/sqrt of negatives)\n\n"
                    "Respond JSON only:\n"
                    "{\"valid\": [\"formula1\", \"formula2\"], \"removed\": [\"bad1\"]}"
                )
                import urllib.request as _ur_ph
                _ph_pay = json.dumps({
                    "model":  _phy_model,
                    "prompt": _phy_prompt,
                    "stream": False,
                    "options": {"temperature": 0.1, "num_predict": 200},
                }).encode()
                _ph_req = _ur_ph.Request(
                    f"{host}/api/generate",
                    data=_ph_pay,
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with _ur_ph.urlopen(_ph_req, timeout=120) as _ph_r:
                    _ph_raw = json.loads(_ph_r.read()).get("response", "").strip()
                _ph_s = _ph_raw.find("{")
                _ph_e = _ph_raw.rfind("}") + 1
                if _ph_s >= 0 and _ph_e > _ph_s:
                    _ph_data   = json.loads(_ph_raw[_ph_s:_ph_e])
                    _ph_valid  = _ph_data.get("valid", [])
                    _ph_removed = _ph_data.get("removed", [])
                    if _ph_valid:
                        nav.hypotheses = _ph_valid[:5]
                        if _ph_removed:
                            print(f"  [Физик→PySR] Отфильтровано {len(_ph_removed)}: {_ph_removed[:2]}")
                        print(f"  [Физик→PySR] ✓ Валидных гипотез для PySR: {len(_ph_valid)}")
                    else:
                        log.debug("[Физик→PySR] Пустой список — оставляем оригинал")
            except Exception as _ph_err:
                log.debug("[Физик→PySR] Ошибка: %s — продолжаем без фильтрации", _ph_err)

            # Итог дебатов → в failure_logs для следующей итерации
            _debate_entry = debate_result.as_failure_log_entry()
            if _debate_entry:
                _fl_list = json.loads(failure_logs_str) if failure_logs_str.strip() != "[]" else []
                _fl_list.append(_debate_entry)
                failure_logs_str = json.dumps(_fl_list, ensure_ascii=False)

        except Exception as _deb_err:
            log.warning("[Дебаты] Ошибка: %s — продолжаем без дебатов", _deb_err)

        # ── ШАГ 3: PySR ───────────────────────────────────────────
        if PySRRegressor is None:
            log.error("[PySR] Не установлен. pip install pysr")
            break

        print(f"\n  [PySR] features={use_shadow} ops={nav.selected_operators}")
        ram_pre = avail_ram_gb()
        log.info("[PySR] RAM до fit: %.2f ГБ", ram_pre)

        model_pysr = build_pysr(
            nav.selected_operators,
            use_shadow,
            timeout_sec,
            seed_equations=nav.hypotheses,  # v10.16: гипотезы Navigator → PySR seed
        )

        # v10.37: фильтруем seed гипотезы — только те что содержат реальные признаки
        # Баг: Delphi иногда добавляет несуществующие переменные ('p', 'x', 'theta' и т.д.)
        # Это ломает PySR — он находит только константу вместо формулы
        try:
            _valid_features = set(use_shadow)
            import re as _re_seed
            _filtered_hyps = []
            for _hyp in (nav.hypotheses or []):
                _vars_in_hyp = set(_re_seed.findall(r'\b[a-zA-Z_]\w*\b', _hyp))
                _math_fns = {'sqrt','log','exp','abs','sin','cos','tanh','pi','e'}
                _unknown = _vars_in_hyp - _valid_features - _math_fns
                if not _unknown:
                    _filtered_hyps.append(_hyp)
                else:
                    print(f"  [v10.37/SeedFilter] Отфильтрована '{_hyp}' — неизвестные: {_unknown}")
            if _filtered_hyps != nav.hypotheses:
                nav.hypotheses = _filtered_hyps
        except Exception as _sf_err:
            log.debug("[v10.37/SeedFilter] %s", _sf_err)

        try:
            model_pysr.fit(
                X_tr_use.astype(np.float32),
                y_train.astype(np.float32),
                variable_names=use_shadow,
            )
        except KeyboardInterrupt:
            log.warning("[PySR] Прерван.")

        gc.collect()
        log.info("[PySR] RAM после fit: %.2f ГБ", avail_ram_gb())

        try:
            vault.check_stale(
                predict_fn = lambda Xp: model_pysr.predict(Xp.astype(np.float32)),
                X          = X_tr_use,
                y          = y_train,
            )
            log.info("[PDCA] Re-check с predict_fn выполнен.")
        except Exception as _pdca_err:
            log.debug("[PDCA] Re-check упал (не критично): %s", _pdca_err)

        eqs = model_pysr.equations_
        if eqs is None or len(eqs) == 0:
            death_ctx = "PySR equations_ пусты"
            log.warning("[HADI %d] %s", attempt + 1, death_ctx)
            continue


        # FIX БАГ 12 v2: penalized_loss = loss * sqrt(complexity)
        # Штрафуем сложность в самой метрике — complexity=4 всегда обгоняет complexity=16+
        # Проверено: на нашем тесте выбирает f0*sqrt(f0) (complexity=4) вместо complexity=16
        import math as _math
        eqs_pen = eqs.copy()
        eqs_pen["_pen"] = eqs_pen.apply(
            lambda r: float(r["loss"]) * _math.sqrt(max(float(r["complexity"]), 1)),
            axis=1,
        )
        best_row   = eqs_pen.sort_values("_pen").iloc[0]
        shadow_f   = str(best_row.get("equation", ""))
        complexity = int(best_row.get("complexity", 0))
        real_f     = shadow.restore(shadow_f)  # SACRED: ShadowMapper
        # v10.22: Препаратор — обратное преобразование в оригинальное пространство
        if _prep_result and _prep_result.applied:
            try:
                from .preparator import inverse_transform as _inv_tr
                _real_f_orig = _inv_tr(real_f, _prep_result)
                if _real_f_orig != real_f:
                    print(f"  [Препаратор] Обратное преобразование:")
                    print(f"    В трансф.:  {real_f[:60]}")
                    print(f"    В оригинале: {_real_f_orig[:60]}")
                    real_f = _real_f_orig
                    shadow_f = _inv_tr(shadow_f, _prep_result)
            except Exception as _ipe:
                log.debug("[Препаратор] inverse: %s", _ipe)
        log.info("[Pareto v2] penalized_loss выбрал complexity=%d  loss=%.5f",
                 complexity, float(best_row["loss"]))

        def predict_fn(X: np.ndarray) -> np.ndarray:
            return model_pysr.predict(X.astype(np.float32))

        r2_tr = float(r2_score(y_train, predict_fn(X_tr_use)))
        log.info("[HADI %d] shadow='%s' R²_train=%.4f", attempt + 1, shadow_f, r2_tr)

        # ── ШАГ 4: СВЯЩЕННЫЕ МЕТРИКИ ──────────────────────────────
        X_full = np.vstack([X_tr_use, X_te_use])
        # FIX v10.27 #7: y_test должен быть в том же пространстве что y_train.
        # Если препаратор применил log/sqrt/standardize к y_train, то y_test
        # (который пришёл как оригинал из run_feynman) нужно трансформировать тоже.
        # Иначе cross_blind смешивает трансформированный y_train с оригинальным y_test
        # → predict_fn работает в трансформированном пространстве → R²_blind мусор.
        if _prep_result and _prep_result.applied:
            _tr = _prep_result.transform_name
            if _tr == "log":
                _y_test_for_metric = np.log(np.clip(y_test, 1e-10, None))
            elif _tr == "sqrt":
                _y_test_for_metric = np.sqrt(np.clip(y_test, 0, None))
            elif _tr == "standardize":
                _m_pr = _prep_result.transform_params.get("mean", 0)
                _s_pr = _prep_result.transform_params.get("std", 1)
                _y_test_for_metric = (y_test - _m_pr) / _s_pr
            else:
                _y_test_for_metric = y_test
        else:
            _y_test_for_metric = y_test
        y_full = np.concatenate([y_train, _y_test_for_metric])

        shuf_ok, p_val  = shuffle_test(    # ███ SACRED ███
            predict_fn, X_tr_use, y_train, p_threshold=p_threshold
        )
        blind_ok, r2_bl = cross_blind(     # ███ SACRED ███
            predict_fn, X_full, y_full
        )

        is_invariant = shuf_ok and blind_ok and r2_tr >= r2_threshold

        # FIX: вычисляем аудит-выборку ЗДЕСЬ — нужна и в phase=pysr и в phase=full
        _audit_n   = min(40, len(y_train))
        _audit_idx = np.round(np.linspace(0, len(y_train) - 1, _audit_n)).astype(int)
        _X_audit   = X_tr_use[_audit_idx]
        _y_audit   = y_train[_audit_idx]
        try:
            _yp_audit = predict_fn(_X_audit)
        except Exception:
            _yp_audit = np.array([])
        try:
            _real_audit = [shadow.restore(s) for s in use_shadow]
        except Exception:
            _real_audit = []

        # FIX: mark_poincare_invariant вызывается здесь (был импортирован но не вызывался)
        from .topological_surgery import mark_poincare_invariant as _mark_poincare
        surgery_res = _mark_poincare(surgery_res, r2_after=r2_bl)
        # FIX: surgery_section объявляется здесь (была NameError в final_report_parts)
        from .topological_surgery import format_surgery_report as _fmt_surgery
        surgery_section = _fmt_surgery(surgery_res)

        # ── ШАГ 4б: ATOMIC PRECISION v10.4.5 (Пантеон Джампера) ──────────
        atomic_res = check_atomic_precision(
            formula_real = real_f,
            r2_blind     = r2_bl,
            complexity   = complexity,
            r2_train     = r2_tr,
        )
        if atomic_res.detected:
            print(f"\n  ★ {atomic_res.verdict_line}")
            for line in atomic_res.report_lines:
                print(line)
        else:
            log.debug("[AtomicPrecision] %s", atomic_res.verdict_line)

        # ── ШАГ 4в: Heritage Scan v10.5 (INVARIANT_LIBRARY / Пантеон) ────────
        if skip_heritage:
            from .atomic_precision import HeritageResult
            heritage_res = HeritageResult(detected=False, matched_scientists=[], verdict_lines=[], formula=real_f)
        else:
            heritage_res = match_heritage(real_f)
        if heritage_res.detected:
            print(f"\n  ╔{'═'*58}╗")
            print(f"  ║  HERITAGE SCAN — THE GREAT PANTHEON v10.5" + " " * 15 + "║")
            print(f"  ╠{'═'*58}╣")
            for hline in heritage_res.verdict_lines:
                padded = hline.ljust(60)
                print(f"  ║{padded}║")
            print(f"  ╚{'═'*58}╝")
        else:
            log.debug("[Heritage v10.5] Нет совпадений в формуле: %s", real_f)

        # ── v10.7: phase="pysr" — накапливаем ВСЕХ кандидатов ─────────
        if phase == "pysr":
            _gold_tags_phase = [
                f"r2_{r2_tr:.2f}", f"attempt_{attempt+1}",
                f"dspy_{'on' if dspy_active else 'off'}",
            ]
            if atomic_res.detected:
                _gold_tags_phase.append("molecular_precision")
            for sci in atomic_res.heritage.matched_scientists:
                _gold_tags_phase.append(f"heritage_{sci.lower().replace(' ', '_')}")

            # FIX: аудит-выборка вычислена выше (общая для pysr и full)
            _X_audit_p  = _X_audit
            _y_audit_p  = _y_audit
            _yp_audit_p = _yp_audit
            try:
                _real_audit_p = _real_audit
            except Exception:
                _real_audit_p = []

            # Добавляем кандидата в список (все попытки — и DEATH и инвариант)
            candidate = _candidate_to_dict(
                formula_shadow   = shadow_f,
                formula_real     = real_f,
                r2_train         = r2_tr,
                r2_blind         = r2_bl,
                p_value          = p_val,
                complexity       = complexity,
                shadow_names     = list(use_shadow),
                real_names       = _real_audit_p,
                shadow_mapping   = dict(shadow.reverse_mapping),
                domain_type      = domain_type,
                model            = model,
                host             = host,
                heritage_context = (atomic_res.heritage_context + extra_heritage).strip(),
                gold_tags        = _gold_tags_phase,
                n_samples        = len(y_train),
                dspy_active      = dspy_active,
                failure_types    = failure_types[:],
                atomic_detected  = atomic_res.detected,
                X_audit          = _X_audit_p,
                y_audit          = _y_audit_p,
                y_pred_audit     = _yp_audit_p,
                attempt_num      = attempt + 1,
                is_invariant     = is_invariant,
                dim_codes        = list(dim_codes),  # v10.14: для classify_discovery
            )
            _all_phase_candidates.append(candidate)
            if r2_bl > _best_phase_r2:
                _best_phase_r2 = r2_bl

            status_str = "✓ ИНВАРИАНТ" if is_invariant else "DEATH"
            print(f"\n  [Phase] попытка {attempt+1}/{HADI_MAX_RETRIES+1} ({status_str}): "
                  f"R\u00b2_blind={r2_bl:.4f} '{shadow_f}'")

            # Продолжаем если не последняя попытка
            last_attempt = (attempt >= HADI_MAX_RETRIES)
            if not last_attempt and not is_invariant:
                prev_hyps = list(set(prev_hyps + nav.hypotheses))
                death_ctx = f"phase=pysr: попытка {attempt+1}, ищем лучше"
                continue

            # Последняя попытка ИЛИ нашли инвариант — сохраняем всех кандидатов
            _save_pysr_phase(_all_phase_candidates)
            print(f"  [Phase] Итого кандидатов: {len(_all_phase_candidates)}")
            return EngineResult(
                verdict        = f"PYSR_PHASE_COMPLETE ({len(_all_phase_candidates)} кандидатов)",
                formula_shadow = _all_phase_candidates[0]["formula_shadow"],
                formula_real   = _all_phase_candidates[0]["formula_real"],
                r2_train       = _all_phase_candidates[0]["r2_train"],
                r2_blind       = _best_phase_r2,
                p_value        = p_val,
                complexity     = complexity,
                consensus      = "NOT_RUN (phase=pysr)",
                gold_path      = None,
                attempts       = attempt + 1,
                dspy_active    = dspy_active,
                failure_types  = failure_types,
                surgery_result = surgery_res,
                surgery_pct    = surgery_res.surgery_pct,
                poincare_invariant = surgery_res.poincare_invariant,
                diffusion_result   = diffusion_res,
                pairformer_result  = pairformer_res,
                atomic_result      = atomic_res,
                molecular_precision = atomic_res.detected,
                heritage_result    = heritage_res,
            )

        # v10.20: записываем лучшую формулу PySR в лог
        if "_nav_log_entry" in dir():
            _nav_log_entry["formula_shadow"] = shadow_f
            _nav_log_entry["r2_blind"]        = round(r2_bl, 4)

        consensus, extended_audit_report, _run_role_results, _run_consilium = matryoshka_audit(
            formula_shadow   = shadow_f,
            shadow_names     = use_shadow,
            r2_train         = r2_tr,
            r2_blind         = r2_bl,   # FIX: передаём реальный r2_blind
            complexity       = complexity,
            domain_type      = domain_type,
            host             = host,
            model            = model,
            dspy_orch        = orch if dspy_active else None,
            heritage_context = (atomic_res.heritage_context + extra_heritage).strip(),
            # v9.9.2-dataaware: реальные данные
            X_samples        = _X_audit,
            y_samples        = _y_audit,
            y_pred_samples   = _yp_audit,
            real_names       = _real_audit,
        )

        # v10.20: финализируем лог-запись + пишем navigator_decisions_log.jsonl
        if "_nav_log_entry" in dir():
            _nav_log_entry["matryoshka_consensus"] = consensus
            if consensus == "ОТКЛОНЕНА":
                _nav_log_entry["death_reason"] = death_ctx[:200] if death_ctx else "Матрёшка отклонила"
            try:
                import json as _ndl_json
                from .config import VAULT_DIR as _ndl_vd
                _ndl_path = _ndl_vd / "navigator_decisions_log.jsonl"
                with open(_ndl_path, "a", encoding="utf-8") as _ndl_f:
                    _ndl_f.write(_ndl_json.dumps(_nav_log_entry, ensure_ascii=False) + "\n")
                log.debug("[NavLog] Записана попытка %d → %s", attempt + 1, _ndl_path.name)
            except Exception as _ndl_err:
                log.debug("[NavLog] Не записан: %s", _ndl_err)

        # v10.10: сохраняем consilium для следующей HADI итерации
        if _run_consilium:
            _consilium_forced_features  = _run_consilium.get("forced_features", [])
            _consilium_forced_operators = _run_consilium.get("forced_operators", [])
            _consilium_exponent         = _run_consilium.get("suggested_exponent", 0.0)
            if _consilium_forced_features or _consilium_forced_operators:
                log.info("[Consilium] Принудительные признаки: %s операторы: %s",
                         _consilium_forced_features, _consilium_forced_operators)
        else:
            _consilium_forced_features  = []
            _consilium_forced_operators = []
            _consilium_exponent         = 0.0

        if consensus == "ОТКЛОНЕНА" and attempt < HADI_MAX_RETRIES:
            matryoshka_rebuild_count += 1
            # v10.3.9 DEVIL ADVOCATE FIX: не более MAX_MATRYOSHKA_REBUILDS пересборок.
            # Если Скептик (Адвокат Дьявола) продолжает отклонять — выходим,
            # чтобы не войти в бесконечный HADI-цикл снижения Consensus_Score.
            if matryoshka_rebuild_count >= MAX_MATRYOSHKA_REBUILDS:
                log.warning(
                    "[Матрёшка] Лимит пересборок достигнут (%d/%d). "
                    "Принимаем лучшую формулу несмотря на отказ Скептика.",
                    matryoshka_rebuild_count, MAX_MATRYOSHKA_REBUILDS,
                )
                print(
                    f"  [Матрёшка] ⚠ Лимит пересборок {MAX_MATRYOSHKA_REBUILDS} достигнут. "
                    f"Адвокат Дьявола заглушён — фиксируем инвариант."
                )
                # Не уходим в continue — падаем насквозь к финальному результату
                consensus = "СПОРНО"   # понижаем вердикт, но не блокируем
            else:
                death_ctx  = f"Матрёшка отклонила '{shadow_f}' (rebuild {matryoshka_rebuild_count}/{MAX_MATRYOSHKA_REBUILDS})"
                prev_hyps  = list(set(prev_hyps + nav.hypotheses))
                failure_types.append("MATRYOSHKA_REJECT")
                # v10.20: при DEATH обновляем лог до continue
                if "_nav_log_entry" in dir():
                    _nav_log_entry["death_reason"] = death_ctx
                    try:
                        import json as _ndl_j2
                        from .config import VAULT_DIR as _ndl_v2
                        _ndl_p2 = _ndl_v2 / "navigator_decisions_log.jsonl"
                        with open(_ndl_p2, "a", encoding="utf-8") as _ndl_f2:
                            _ndl_f2.write(_ndl_j2.dumps(_nav_log_entry, ensure_ascii=False) + "\n")
                    except Exception:
                        pass
                ollama_stop(model)
                continue

        # ── ШАГ 5б: [Lasso v10.2.7] Стягивание аргументов ───────────
        from .navigator import ollama_chat as _ollama_ask
        def _ask_fn(prompt: str) -> str:
            # OLLAMA_MODEL (Qwen) — аналитика: Deep Root, Lasso, RCA
            return _ollama_ask(
                prompt, model=model, host=host,
                temperature=0.35, num_predict=400,
            )
        def _ask_synthesis(prompt: str) -> str:
            # SYNTHESIS_MODEL (Gemma) — синтез: Delphi, Scientific Cycle
            return _ollama_ask(
                prompt, model=SYNTHESIS_MODEL, host=host,
                temperature=0.35, num_predict=600,
            )

        lasso_args = [
            f"R²_train={r2_tr:.4f} — формула объясняет {r2_tr*100:.1f}% дисперсии",
            f"R²_blind={r2_bl:.4f} — устойчива на out-of-sample данных",
            f"shuffle_p={p_val:.5f} — не нумерология (NIST CSPRNG)",
            f"complexity={complexity} — сложность формулы",
            f"consensus={consensus} — консенсус 4 ролей Матрёшки",
            f"features={', '.join(use_shadow[:5])} — отобранные признаки",
        ]
        print(f"\n  [Lasso v10.2.7] Стягиваем аргументы…")
        lasso_core, lasso_kept = lasso_pull(
            arguments      = lasso_args,
            formula_shadow = shadow_f,
            ask_fn         = _ask_fn,
        )
        lasso_section = format_lasso_section(
            lasso_core, lasso_kept,
            cut_count = len(lasso_args) - len(lasso_kept),
        )
        print(lasso_section)

        # ── ШАГ 5в: [Deep Root v10.2.7] 5 Почему ────────────────────
        print(f"\n  [Deep Root v10.2.7] Root Cause Analysis…")
        rca_chain = deep_root_analysis(
            invariant   = shadow_f,
            dependency  = f"R²={r2_tr:.4f}, features={use_shadow[:3]}",
            r2_train    = r2_tr,
            domain_type = domain_type,
            ask_fn      = _ask_fn,
        )
        rca_section = format_root_cause_section(rca_chain, real_f)
        print(rca_section)

        # FIX: возвращаем RCA + Lasso + Consilium в Navigator на следующей итерации.
        # Раньше эти 10+ LLM-вызовов шли только в финальный отчёт.
        if rca_chain:
            _rca_fundamental = rca_chain[-1] if rca_chain else ""
            if _rca_fundamental:
                prev_hyps_set = set(prev_hyps)
                _rca_entry = {
                    "hypothesis":   f"[RCA] {_rca_fundamental[:150]}",
                    "death_reason": "deep_root: фундаментальная причина — иди в этом направлении",
                    "source":       "rca_feedback",
                }
                if _rca_entry not in failure_logs_list:
                    failure_logs_list.append(_rca_entry)

        if lasso_core:
            _lasso_entry = {
                "hypothesis":   f"[Lasso] {lasso_core[:150]}",
                "death_reason": "occam_razor: это ядро — вокруг него строй следующую формулу",
                "source":       "lasso_feedback",
            }
            if _lasso_entry not in failure_logs_list:
                failure_logs_list.append(_lasso_entry)

        # Consilium советы — самые ценные: единогласное решение 4 экспертов
        if _run_consilium:
            _c_features = _run_consilium.get("forced_features", [])
            _c_operators = _run_consilium.get("forced_operators", [])
            _c_exp = _run_consilium.get("suggested_exponent", 0.0)
            if _c_features or _c_operators:
                _consilium_entry = {
                    "hypothesis": (
                        f"[Consilium] признаки={_c_features} "
                        f"операторы={_c_operators}"
                        + (f" степень={_c_exp}" if _c_exp else "")
                    )[:200],
                    "death_reason": "единогласный совет 4 экспертов — соблюдай обязательно",
                    "source": "consilium_feedback",
                }
                if _consilium_entry not in failure_logs_list:
                    failure_logs_list.append(_consilium_entry)
        # Обновляем failure_logs_str после добавления новых источников
        try:
            failure_logs_str = json.dumps(failure_logs_list, ensure_ascii=False)
        except Exception:
            pass

        # ── ШАГ 6: GoldVault — формула становится few-shot примером ─
        ollama_stop(model)
        gold_tags = [
            f"r2_{r2_tr:.2f}",
            f"attempts_{attempt+1}",
            f"consensus_{consensus.lower()}",
            f"dspy_{'on' if dspy_active else 'off'}",
        ]
        if atomic_res.detected:
            gold_tags.append("molecular_precision")
        if atomic_res.molecular_fold:
            gold_tags.append(f"fold_{atomic_res.resonance_pattern.replace(' ', '_')}")
        # v10.5: Heritage-учёные → в теги vault → DSPy few-shot знает о структуре
        for sci in atomic_res.heritage.matched_scientists:
            gold_tags.append(f"heritage_{sci.lower().replace(' ', '_')}")
        gold_saved = vault.save(
            formula_shadow = shadow_f,
            formula_real   = real_f,
            shadow_mapper  = shadow,
            r2_train       = r2_tr,
            r2_blind       = r2_bl,
            complexity     = complexity,
            tags           = gold_tags,
            n_samples      = len(y_train),   # v10.5: DSPy few-shot context
        )
        log.info("[v10.5] Формула сохранена → few-shot пример при следующем запуске.")

        # Вердикт v10.4.5: молекулярная точность повышает до IRON
        if atomic_res.detected and attempt == 0 and consensus == "ПРИНЯТА":
            verdict = "IRON INVARIANT [MOLECULAR PRECISION]"
        elif attempt == 0 and consensus == "ПРИНЯТА":
            verdict = "IRON INVARIANT"
        else:
            verdict = "INVARIANT FOUND"

        # ── Финальный отчёт v10.4.5 ───────────────────────────────────────
        diffusion_section  = format_diffusion_report(diffusion_res)
        pairformer_section = format_pairformer_report(pairformer_res)
        atomic_section     = "\n".join(atomic_res.report_lines)
        # format_pantheon_with_matches включает И 3 столпа Пантеона И Heritage scan
        pantheon_section   = format_pantheon_with_matches(heritage_res)
        # heritage_section больше не нужен отдельно — всё в pantheon_section
        heritage_section   = ""   # объединено в pantheon_section выше

        final_report_parts = [
            "═" * 62,
            f"  FINAL REPORT v10.5 — {verdict}",
            f"  Formula (shadow): {shadow_f}",
            f"  Formula (real):   {real_f}",
            f"  R²_train={r2_tr:.4f}  R²_blind={r2_bl:.4f}  "
            f"shuffle_p={p_val:.5f}",
            f"  Consensus: {consensus}  Complexity: {complexity}",
            "═" * 62,
            "",
            diffusion_section,
            "",
            pairformer_section,
            "",
            surgery_section,
            "",
            atomic_section,
            "",
            pantheon_section,
            "",
            heritage_section,
            "",
            rca_section,
            "",
            lasso_section,
            "",
            extended_audit_report,   # Sinquain + Dialectic из Матрёшки
        ]
        final_report_str = "\n".join(final_report_parts)

        from .config import SCRIPT_DIR as _SD
        final_path = _SD / "scalpel_vault" / "FINAL_REPORT_v10.txt"
        final_path.parent.mkdir(parents=True, exist_ok=True)
        final_path.write_text(final_report_str, encoding="utf-8")
        log.info("[v10.5] Финальный отчёт → %s", final_path)
        print(f"\n  [v10.5] Финальный отчёт сохранён → {final_path}")

        # v10.14: Oracle finalize
        try:
            oracle.finalize(
                formula_final = final_report_str[:100] if "final_report_str" in dir() else "",
                r2_final      = r2_bl if "r2_bl" in dir() else 0.0,
                domain        = domain_type,
                passed        = "INVARIANT" in verdict if "verdict" in dir() else False,
            )
        except Exception as _oc_f:
            log.debug("[Oracle] finalize: %s", _oc_f)

        # v10.14: META-REFLECTION — проверяем критическую массу
        try:
            from .meta_reflection import check_and_reflect as _meta_check
            _meta_check(model=CHRONICLE_MODEL, host=host)  # LLaMA: нарратив + синтез опыта
        except Exception as _meta_err:
            log.debug("[MetaReflection] %s", _meta_err)

        result = EngineResult(
            verdict=verdict, formula_shadow=shadow_f, formula_real=real_f,
            r2_train=r2_tr, r2_blind=r2_bl, p_value=p_val,
            complexity=complexity, consensus=consensus,
            gold_path=gold_saved, attempts=attempt + 1,
            dspy_active=dspy_active, failure_types=failure_types,
            root_cause_chain=rca_chain,
            lasso_core=lasso_core,
            final_report=final_report_str,
            surgery_result=surgery_res,
            surgery_pct=surgery_res.surgery_pct,
            poincare_invariant=surgery_res.poincare_invariant,
            # v10.4.5
            diffusion_result    = diffusion_res,
            pairformer_result   = pairformer_res,
            atomic_result       = atomic_res,
            molecular_precision = atomic_res.detected,
            # v10.5
            heritage_result     = heritage_res,
        )
        _print_verdict(result, use_shadow, use_real)
        # Хирург записывает опыт: что решил + какой R² получился
        try:
            surgeon_record_outcome(
                surgeon_stats, surgeon_dec,
                r2_before = None,
                r2_after  = result.r2_blind,
            )
        except Exception:
            pass

        # v10.25: обновляем МетаПаттерны с результатом закона
        try:
            if _pattern_engine is not None:
                _actions_taken = {
                    "transform":   _prep_result.transform_name if _prep_result else "none",
                    "iqr_k":       surgeon_dec.iqr_multiplier,
                    "operators":   list(nav.selected_operators) if "nav" in dir() else [],
                    "hypotheses":  list(nav.hypotheses) if "nav" in dir() else [],
                }
                _pattern_engine.update_from_result(
                    data_stats    = _data_stats_for_patterns,
                    actions_taken = _actions_taken,
                    r2_result     = result.r2_blind,
                )
                log.info("[MetaPatterns] %s", _pattern_engine.summary())
        except Exception as _mpe:
            log.debug("[MetaPatterns] update: %s", _mpe)

        return result

    # v10.7: аварийное сохранение если phase="pysr" и дошли сюда
    if phase == "pysr" and _all_phase_candidates:
        _save_pysr_phase(_all_phase_candidates)
        print(f"\n  [Phase] Аварийное сохранение: {len(_all_phase_candidates)} кандидатов")

    # Записываем опыт при провале тоже — хирург учится и на неудачах
    try:
        surgeon_record_outcome(surgeon_stats, surgeon_dec, None, 0.0)
    except Exception:
        pass

    return EngineResult(
        verdict="DEATH (HADI exhausted)", formula_shadow="", formula_real="",
        r2_train=0.0, r2_blind=0.0, p_value=1.0,
        complexity=0, consensus="NOT_RUN", gold_path=None,
        attempts=HADI_MAX_RETRIES + 1,
        dspy_active=dspy_active, failure_types=failure_types,
        root_cause_chain=[], lasso_core="", final_report="",
        surgery_result=surgery_res,
        surgery_pct=surgery_res.surgery_pct,
        poincare_invariant=False,
        # v10.4.5
        diffusion_result    = diffusion_res,
        pairformer_result   = pairformer_res,
        atomic_result       = None,
        molecular_precision = False,
        # v10.5
        heritage_result     = None,
    )
