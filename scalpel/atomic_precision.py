"""
atomic_precision.py — Atomic Precision v10.5 (The Great Pantheon).

══════════════════════════════════════════════════════════════════
АРХИТЕКТУРА v10.5 — три уровня в check_atomic_precision():
══════════════════════════════════════════════════════════════════

  Уровень 1 — Structural Quality:
      R²_blind, complexity, density → [MOLECULAR PRECISION DETECTED]
      + нелинейные складки (_FOLD_PATTERNS)

  Уровень 2 — Heritage Structural Match (NEW):
      sympy.nsimplify + preorder_traversal → скелеты INVARIANT_LIBRARY
      → [HERITAGE MATCHED: Кеплер — T ~ a^(3/2)]
      Fallback на regex если sympy недоступен.
      Результат → .heritage_context для инъекции в Delphi-промпты.

  Уровень 3 — Pantheon Display:
      format_pantheon()               — 3 столпа (всегда)
      format_pantheon_with_matches()  — + подсветка совпавших учёных ★

INVARIANT_LIBRARY учёные:
  Тесла    — константы {3,6,9,...}
  ДаВинчи  — log(x)/log(y), φ≈1.618
  Бах      — f(g(x)) глубина ≥ 2
  Кюри     — exp(-λt)
  Гаусс    — exp(-x²)
  Кеплер   — x**(3/2)

Интеграция с Delphi:
  AtomicPrecisionResult.heritage_context → прямая вставка в промпт
  Скептика и Физика в socratic_cross_examination().
"""
from __future__ import annotations

import re
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

log = logging.getLogger("scalpel")


# ══════════════════════════════════════════════════════════════════════════════
# INVARIANT_LIBRARY
# ══════════════════════════════════════════════════════════════════════════════

INVARIANT_LIBRARY: Dict = {

    # ══ МЕХАНИКА ══════════════════════════════════════════════════════════
    "Ньютон_F=ma": {
        "label":       "Newton — Second Law",
        "skeleton":    "x * y",
        "description": "Второй закон Ньютона: F = m·a",
        "context": (
            "HERITAGE: формула соответствует второму закону Ньютона F = m·a. "
            "Произведение двух переменных — сила = масса × ускорение. "
            "Физик: проверь размерности: [кг · м/с²] = [Н]. "
            "Скептик: не является ли это случайным произведением без физического смысла?"
        ),
        "patterns": [
            re.compile(r"^f\d+\s*\*\s*f\d+$"),
        ],
        "dim_hint": [3, 6],   # масса × сила/ускорение
    },

    "Ньютон_гравитация": {
        "label":       "Newton — Gravitation F ~ 1/r²",
        "skeleton":    "x * y / z**2",
        "description": "Закон тяготения: F = G·m₁·m₂ / r²",
        "context": (
            "HERITAGE: формула содержит обратный квадрат — закон тяготения Ньютона "
            "F = G·m₁·m₂/r². Сила убывает как квадрат расстояния. "
            "Физик: показатель степени ДОЛЖЕН быть ровно 2.0, не 1.9 или 2.1. "
            "Скептик: обратный квадрат может быть геометрическим артефактом — проверь."
        ),
        "patterns": [
            re.compile(r"f\d+\s*\*\s*f\d+\s*/\s*f\d+\s*\*\*\s*2"),
            re.compile(r"/\s*f\d+\s*\*\*\s*2\.0"),
            re.compile(r"f\d+\s*/\s*f\d+\s*\*\*\s*2(?![\d.])"),
        ],
        "dim_hint": [3, 3, 2],  # масса × масса / расстояние²
    },

    "Кеплер": {
        "label":       "Kepler — Third Law T ~ a^(3/2)",
        "skeleton":    "x ** 1.5",
        "description": "Третий закон Кеплера: T ~ a^(3/2)",
        "context": (
            "HERITAGE: формула содержит степень 3/2 — структуру Третьего закона Кеплера: "
            "T² ∝ a³, т.е. T ~ a^(3/2). Ньютон вывел его из закона тяготения. "
            "Скептик: степень 3/2 могла возникнуть из геометрии — проверь физический смысл. "
            "Физик: если подтверждён — какая центральная сила стоит за этим законом?"
        ),
        "patterns": [
            # Кеплер: только если формула = c * f0**1.5 или просто f0**1.5
            # НЕ f0*f1**1.5 (два признака перемножены — другой закон)
            re.compile(r"^\s*(?:\d+\.?\d*\s*\*\s*)?f\d+\s*\*\*\s*1\.5(?![\d.])\s*$"),
            re.compile(r"^\s*(?:\d+\.?\d*\s*\*\s*)?f\d+\s*\*\*\s*\(\s*3\s*/\s*2\s*\)\s*$"),
            re.compile(r"^\s*sqrt\s*\(\s*f\d+\s*\*\*\s*3\s*\)\s*$"),
        ],
        "dim_hint": [2, 8],
    },

    "Хук": {
        "label":       "Hooke — F = k·x",
        "skeleton":    "c * x",
        "description": "Закон Гука: сила пропорциональна деформации F = k·x",
        "context": (
            "HERITAGE: линейная зависимость силы от деформации — закон Гука F = k·x. "
            "Справедлив в области упругости до предела текучести. "
            "Скептик: проверь — нет ли нелинейности при больших деформациях. "
            "Физик: k — жёсткость системы. Какова её физическая природа?"
        ),
        "patterns": [
            re.compile(r"^-?\s*\d*\.?\d*\s*\*?\s*f\d+$"),
        ],
        "dim_hint": [6, 2],
    },

    # ══ ТЕРМОДИНАМИКА И КИНЕТИКА ═══════════════════════════════════════════
    "Аррениус": {
        "label":       "Arrhenius — k = A·exp(-Ea/RT)",
        "skeleton":    "c * exp(-x / y)",
        "description": "Уравнение Аррениуса: скорость реакции ~ exp(-Ea/RT)",
        "context": (
            "HERITAGE: формула содержит структуру уравнения Аррениуса k = A·exp(-Ea/RT). "
            "Скорость химической реакции экспоненциально зависит от температуры. "
            "Физик: аргумент показателя — это Ea/RT, где Ea — энергия активации. "
            "Скептик: убедись, что знаменатель действительно температура, а не другая переменная."
        ),
        "patterns": [
            re.compile(r"exp\s*\(\s*-\s*\w+\s*/\s*\w+"),
        ],
        "dim_hint": [4],
    },

    "Кюри": {
        "label":       "Curie — Radioactive decay exp(-λt)",
        "skeleton":    "exp(-c · t)",
        "description": "Экспоненциальный распад: N = N₀·exp(-λt)",
        "context": (
            "HERITAGE: формула содержит экспоненциальный распад — exp(-λt). "
            "Мария Кюри: каждый изотоп имеет строгий период полураспада. "
            "Применим для: радиоактивности, RC-цепей, охлаждения, химических реакций 1-го порядка. "
            "Скептик: убедись что зависимость истинно экспоненциальная, а не степенная. "
            "Физик: определи λ = ln(2)/T₁/₂ и проверь соответствие периоду полураспада."
        ),
        "patterns": [
            re.compile(r"exp\s*\(\s*-\s*(?:f\d+|\d+\.?\d*)\s*(?:\*\s*f\d+)?\s*\)"),
            re.compile(r"e\s*\*\*\s*\(\s*-"),
        ],
        "dim_hint": [8],
    },

    "Больцман": {
        "label":       "Boltzmann — P ~ exp(-E/kT)",
        "skeleton":    "exp(-x / y)",
        "description": "Больцман: заселённость уровня ~ exp(-E/kT)",
        "context": (
            "HERITAGE: распределение Больцмана — вероятность состояния убывает "
            "экспоненциально с энергией: P ~ exp(-E/kT). "
            "Физик: если аргумент exp = -E/kT, то это статистическая механика. "
            "Скептик: отличие от Аррениуса — здесь E/kT безразмерно, T в знаменателе."
        ),
        "patterns": [
            re.compile(r"exp\s*\(\s*-\s*f\d+\s*/\s*f\d+\s*\)"),
        ],
        "dim_hint": [4],
    },

    "Гаусс": {
        "label":       "Gauss — Normal distribution exp(-x²)",
        "skeleton":    "exp(-x²)  или  exp(-c · x²)",
        "description": "Нормальное распределение: exp(-x²)",
        "context": (
            "HERITAGE: формула содержит гауссову структуру — exp(-x²) или exp(-c·x²). "
            "Гаусс: ошибки измерений подчиняются нормальному распределению (ЦПТ). "
            "Встречается в: диффузии, квантовой механике (волновой пакет), оптике (гауссов пучок). "
            "Скептик: является ли гауссовость реальной физикой или артефактом выборки? "
            "Физик: какой диссипативный процесс порождает нормальное распределение?"
        ),
        "patterns": [
            re.compile(r"exp\s*\(\s*-.*?f\d+\s*\*\*\s*2\s*\)"),
            # Removed: exp(-f0*f1) is Curie/Arrhenius, not Gauss
        ],
        "dim_hint": [0],
    },

    # ══ БИОЛОГИЯ И БИОХИМИЯ ════════════════════════════════════════════════
    "Михаэлис-Ментен": {
        "label":       "Michaelis–Menten — v = Vmax·S/(Km+S)",
        "skeleton":    "x * y / (c + y)",
        "description": "Ферментативная кинетика: v = Vmax·[S]/(Km+[S])",
        "context": (
            "HERITAGE: формула содержит гиперболическое насыщение — структуру "
            "уравнения Михаэлиса-Ментен v = Vmax·[S]/(Km+[S]). "
            "Физик: при [S] >> Km скорость → Vmax (насыщение). При [S] << Km — линейный рост. "
            "Скептик: проверь — действительно ли насыщение есть в данных, а не линейная зависимость."
        ),
        "patterns": [
            re.compile(r"f\d+\s*\*\s*f\d+\s*/\s*\(\s*(?:f\d+|\d+\.?\d*)\s*\+\s*f\d+\s*\)"),
        ],
        "dim_hint": [0, 5],
    },

    "Лотка-Вольтерра": {
        "label":       "Lotka–Volterra — logistic/predator-prey",
        "skeleton":    "r * N * (1 - N/K)",
        "description": "Логистический рост / уравнения хищник-жертва",
        "context": (
            "HERITAGE: формула содержит структуру логистического роста или Лотка-Вольтерра. "
            "Логистика: dN/dt = r·N·(1 - N/K), где K — ёмкость среды. "
            "При малых N — экспоненциальный рост, при N→K — насыщение. "
            "Скептик: если насыщение отсутствует в данных — может быть просто экспонента. "
            "Физик: определи r (скорость роста) и K (несущая ёмкость)."
        ),
        "patterns": [
            re.compile(r"f\d+\s*\*\s*\(.*?1\s*-\s*f\d+\s*/\s*f\d+"),
            re.compile(r"f\d+\s*\*\s*f\d+\s*\*\s*\(.*?1\s*-"),
        ],
        "dim_hint": [0, 0, 0],
    },

    "Клейбер": {
        "label":       "Kleiber — Metabolic scaling M^0.75",
        "skeleton":    "x ** 0.75",
        "description": "Закон Клейбера: метаболизм ~ масса^(3/4)",
        "context": (
            "HERITAGE: показатель степени ~0.75 — закон масштабирования Клейбера. "
            "Метаболическая мощность P ~ M^(3/4), где M — масса тела. "
            "Универсален для всех живых организмов от бактерий до китов. "
            "Физик: показатель 3/4 объясняется фракталоподобной сетью кровеносных сосудов. "
            "Скептик: степень ДОЛЖНА быть близка к 0.75, а не просто между 0.5 и 1."
        ),
        "patterns": [
            re.compile(r"\*\*\s*0\.7[45]\d*"),
            re.compile(r"\*\*\s*\(\s*3\s*/\s*4\s*\)"),
        ],
        "dim_hint": [3],
    },

    # ══ СТЕПЕННЫЕ И УНИВЕРСАЛЬНЫЕ ЗАКОНЫ ══════════════════════════════════
    "Степенной_закон": {
        "label":       "Power law — scale-free y ~ x^α",
        "skeleton":    "c * x ** alpha",
        "description": "Степенной закон: y ~ x^α (масштабная инвариантность)",
        "context": (
            "HERITAGE: формула — степенной закон y ~ c·x^α. "
            "Встречается в: распределении богатства (Парето), сейсмологии (Гутенберг-Рихтер), "
            "сетях (Барабаши-Альберт), лингвистике (Зипф). "
            "Физик: степень α — критический показатель, может указывать на фазовый переход. "
            "Скептик: на log-log графике должна быть прямая линия."
        ),
        "patterns": [
            # Нецелые показатели (не 0.5, 0.75, 1.5 — они покрыты другими законами)
            re.compile(r"f\d+\s*\*\*\s*(?!(?:0\.5|0\.75|1\.5|1\.0|2\.0|3\.0)(?![\d.]))(?:0\.[1-9]\d+|[1-9]\.\d*[1-9]\d*)(?![\d.])"),
        ],
        "dim_hint": [0],
    },

}


# ── Пантеон (3 столпа) ─────────────────────────────────────────────────────
PANTHEON = [
    ("Колмогоров–Арнольд", "KAM-теорема: сложные системы сохраняют инвариантные торы"),
    ("Перельман",          "Топологическая хирургия: сингулярности удаляются, скелет остаётся"),
    ("Джон Джампер",       "AlphaFold 3: атомарная точность через диффузию структуры"),
]

ATOMIC_R2_THRESHOLD:     float = 0.88
ATOMIC_COMPLEXITY_MAX:   int   = 25
ATOMIC_COMPLEXITY_RATIO: float = 0.04
RESONANCE_R2:            float = 0.85

_FOLD_PATTERNS = [
    (re.compile(r"(sin|cos)\s*\(.*?(exp|log)\s*\("), "trig×exp fold"),
    (re.compile(r"(log|sqrt)\s*\(.*?(log|sqrt)\s*\("), "nested log fold"),
    (re.compile(r"exp\s*\(.*?exp\s*\("),              "double-exp resonance"),
    (re.compile(r"\*\*\s*[3-9]"),                      "high-power nonlinear fold"),
    (re.compile(r"(sin|cos)\s*\(.*?\*\*"),              "trig-power coupling"),
]


# ══════════════════════════════════════════════════════════════════════════════
# DATACLASSES
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class HeritageMatch:
    scientist:   str
    label:       str
    skeleton:    str
    description: str
    method:      str    # "sympy" | "regex"
    verdict:     str    # "[HERITAGE MATCHED: ...]"


@dataclass
class HeritageResult:
    detected:           bool                = False
    matches:            List[HeritageMatch] = field(default_factory=list)
    matched_scientists: List[str]           = field(default_factory=list)
    verdict_lines:      List[str]           = field(default_factory=list)
    # Блок для инъекции в Delphi-промпты (Скептик + Физик)
    heritage_context:   str                 = ""


@dataclass
class AtomicPrecisionResult:
    # Уровень 1
    detected:          bool      = False
    molecular_fold:    bool      = False
    resonance_pattern: str       = ""
    r2_atomic_ok:      bool      = False
    complexity_ok:     bool      = False
    density_ok:        bool      = False
    pantheon_ref:      str       = ""
    verdict_line:      str       = ""
    report_lines:      List[str] = field(default_factory=list)
    # Уровень 2 — Heritage
    heritage:          HeritageResult = field(default_factory=HeritageResult)
    heritage_context:  str            = ""   # копия для удобства в engine.py


# ══════════════════════════════════════════════════════════════════════════════
# УРОВЕНЬ 1 — нелинейные складки (regex)
# ══════════════════════════════════════════════════════════════════════════════

def _detect_resonance(formula: str) -> Optional[str]:
    f = formula.lower()
    for pattern, name in _FOLD_PATTERNS:
        if pattern.search(f):
            return name
    return None


# ══════════════════════════════════════════════════════════════════════════════
# УРОВЕНЬ 2A — Sympy Structural Match
# ══════════════════════════════════════════════════════════════════════════════

def _sympy_parse(formula: str):
    try:
        import sympy as sp
        from sympy.parsing.sympy_parser import (
            parse_expr, standard_transformations,
            implicit_multiplication_application,
        )
        trans = standard_transformations + (implicit_multiplication_application,)
        ld = {c: sp.Symbol(c) for c in "abcdefghijklmnopqrstuvwxyz"}
        ld.update({"exp": sp.exp, "log": sp.log, "sqrt": sp.sqrt,
                   "sin": sp.sin, "cos": sp.cos, "tan": sp.tan,
                   "abs": sp.Abs, "tanh": sp.tanh})
        return parse_expr(formula, local_dict=ld, transformations=trans)
    except Exception as e:
        log.debug("[HeritageSympy] parse failed: %s", e)
        return None


def _is_approx(val, target: float, tol: float = 1e-9) -> bool:
    try:
        return abs(float(val) - target) < tol
    except Exception:
        return False


def _sympy_structural_match(formula: str) -> List[Tuple[str, str]]:
    """
    Structural match через sympy дерево.
    Возвращает [(scientist_key, description), ...].

    Детекторы:
      Кеплер   — x**(3/2) после nsimplify/powdenest
      Гаусс    — exp(отрицательный квадратичный аргумент)
      Кюри     — exp(отрицательный линейный), не Гаусс
      Бах      — глубина вложенности функций ≥ 2
      Тесла    — числа из {3,6,9,12,18,27,36,-3,-6,-9}
      ДаВинчи  — log(x)·log(y)^-1 или φ≈1.618
    """
    try:
        import sympy as sp
        from sympy import preorder_traversal, Rational
    except ImportError:
        return []

    expr = _sympy_parse(formula)
    if expr is None:
        return []

    try:
        expr_ns = sp.nsimplify(expr, rational=True)
        expr_pd = sp.powdenest(sp.simplify(expr), force=True)
    except Exception:
        expr_ns = expr
        expr_pd = expr

    results: List[Tuple[str, str]] = []

    # ── Кеплер ────────────────────────────────────────────────────────────
    def _has_kepler(e) -> bool:
        for s in preorder_traversal(e):
            if s.is_Pow:
                if _is_approx(s.args[1], 1.5) or s.args[1] == Rational(3, 2):
                    return True
        return False

    if _has_kepler(expr) or _has_kepler(expr_ns) or _has_kepler(expr_pd):
        results.append(("Кеплер", "T ~ a^(3/2) — Третий закон Кеплера (sympy)"))

    # ── Гаусс ─────────────────────────────────────────────────────────────
    def _has_gauss(e) -> bool:
        for s in preorder_traversal(e):
            if s.func == sp.exp:
                arg = s.args[0]
                if arg.could_extract_minus_sign():
                    for sub in preorder_traversal(arg):
                        if sub.is_Pow and _is_approx(sub.args[1], 2.0):
                            return True
        return False

    if _has_gauss(expr) or _has_gauss(expr_ns):
        results.append(("Гаусс", "exp(-x²) — нормальная кривая Гаусса (sympy)"))

    # ── Кюри (не Гаусс) ───────────────────────────────────────────────────
    gauss_found = any(r[0] == "Гаусс" for r in results)

    def _has_curie(e) -> bool:
        for s in preorder_traversal(e):
            if s.func == sp.exp:
                arg = s.args[0]
                if arg.could_extract_minus_sign():
                    is_quadratic = any(
                        sub.is_Pow and _is_approx(sub.args[1], 2.0)
                        for sub in preorder_traversal(arg)
                    )
                    if not is_quadratic:
                        return True
        return False

    if not gauss_found and (_has_curie(expr) or _has_curie(expr_ns)):
        results.append(("Кюри", "exp(-λt) — экспоненциальный распад Кюри (sympy)"))

    # ── Бах ───────────────────────────────────────────────────────────────
    _FUNC_SET = {sp.exp, sp.log, sp.sin, sp.cos, sp.sqrt, sp.tan, sp.tanh, sp.Abs}

    def _depth(e, d: int = 0) -> int:
        if e.func in _FUNC_SET:
            return max((_depth(a, d + 1) for a in e.args), default=d + 1)
        return max((_depth(a, d) for a in e.args), default=d)

    if _depth(expr) >= 2:
        results.append(("Бах", "f(g(x)) — рекурсивная вложенность ≥ 2 (sympy)"))

    # ── Тесла ─────────────────────────────────────────────────────────────
    _TESLA = {3, 6, 9, 12, 18, 27, 36, -3, -6, -9}

    def _has_tesla(e) -> bool:
        for s in preorder_traversal(e):
            if s.is_Number and not s.is_infinite:
                try:
                    v = int(round(float(s)))
                    if v in _TESLA:
                        return True
                except Exception:
                    pass
        return False

    if _has_tesla(expr) or _has_tesla(expr_ns):
        results.append(("Тесла", "константа ∈ {3,6,9,...} — симметрия Теслы (sympy)"))

    # ── Да Винчи ──────────────────────────────────────────────────────────
    def _has_davinci(e) -> bool:
        for s in preorder_traversal(e):
            if s.is_Mul:
                has_log = any(a.func == sp.log for a in s.args)
                has_inv = any(
                    a.is_Pow and a.args[0].func == sp.log and a.args[1] == -1
                    for a in s.args
                )
                if has_log and has_inv:
                    return True
            if s.is_Number:
                try:
                    v = float(s)
                    if abs(v - 1.6180339887) < 1e-3 or abs(v - 0.6180339887) < 1e-3:
                        return True
                except Exception:
                    pass
        return False

    if _has_davinci(expr) or _has_davinci(expr_ns):
        results.append(("ДаВинчи", "log(x)/log(y) или φ≈1.618 — Да Винчи (sympy)"))

    log.debug("[HeritageSympy] matches=%s", [r[0] for r in results])
    return results


# ══════════════════════════════════════════════════════════════════════════════
# УРОВЕНЬ 2B — Regex Fallback
# ══════════════════════════════════════════════════════════════════════════════

def _regex_structural_match(formula: str) -> List[Tuple[str, str]]:
    results: List[Tuple[str, str]] = []
    f, fl = formula, formula.lower()
    for key, info in INVARIANT_LIBRARY.items():
        for pattern in info["patterns"]:
            if pattern.search(f) or pattern.search(fl):
                results.append((key, f"{info['label']} (regex)"))
                break
    log.debug("[HeritageRegex] matches=%s", [r[0] for r in results])
    return results


# ══════════════════════════════════════════════════════════════════════════════
# PUBLIC: match_heritage
# ══════════════════════════════════════════════════════════════════════════════

def match_heritage(formula: str, prefer_sympy: bool = True) -> HeritageResult:
    """
    Heritage Scan: sympy structural match → regex fallback.

    Возвращает HeritageResult с:
      .detected           — True если хотя бы один учёный найден
      .matches            — List[HeritageMatch]
      .matched_scientists — List[str]  ("Кеплер", "Гаусс", ...)
      .verdict_lines      — List[str]  [HERITAGE MATCHED: ...]
      .heritage_context   — str для инъекции в Delphi-промпты
    """
    res = HeritageResult()

    raw: List[Tuple[str, str]] = []
    if prefer_sympy:
        raw = _sympy_structural_match(formula)
    if not raw:
        raw = _regex_structural_match(formula)

    for key, method_desc in raw:
        if key not in INVARIANT_LIBRARY:
            log.debug("[Heritage] Ключ '%s' не найден в INVARIANT_LIBRARY — пропускаем", key)
            continue
        info = INVARIANT_LIBRARY[key]
        hm = HeritageMatch(
            scientist   = key,
            label       = info["label"],
            skeleton    = info["skeleton"],
            description = info["description"],
            method      = "sympy" if "sympy" in method_desc else "regex",
            verdict     = f"[HERITAGE MATCHED: {info['label']}] {info['skeleton']}",
        )
        res.matches.append(hm)
        res.matched_scientists.append(key)
        res.verdict_lines.append(
            f"  [HERITAGE MATCHED: {info['label']}]  "
            f"Скелет: {info['skeleton']}  |  {method_desc}"
        )

    res.detected = bool(res.matches)

    # ── Собираем heritage_context для Delphi ──────────────────────────────
    if res.detected:
        ctx_parts = ["\n  ━━━ HERITAGE CONTEXT (The Great Pantheon) ━━━"]
        for hm in res.matches:
            ctx_parts.append(f"  ★ {INVARIANT_LIBRARY[hm.scientist]['context']}")
        ctx_parts.append("  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        res.heritage_context = "\n".join(ctx_parts)
    else:
        res.heritage_context = ""

    log.info("[Heritage v10.5] detected=%s scientists=%s",
             res.detected, res.matched_scientists)
    return res


# ══════════════════════════════════════════════════════════════════════════════
# check_atomic_precision — интегрирует оба уровня
# ══════════════════════════════════════════════════════════════════════════════

def check_atomic_precision(
    formula_real: str,
    r2_blind:     float,
    complexity:   int,
    r2_train:     float = 0.0,
) -> AtomicPrecisionResult:
    """
    Полная проверка атомарной точности v10.5.

    Уровень 1 — качество формулы:
      R²_blind ≥ 0.88, complexity ≤ 25, density ≥ 0.04, нелинейные складки
      → [MOLECULAR PRECISION DETECTED]

    Уровень 2 — Heritage Structural Match (sympy → regex fallback):
      → [HERITAGE MATCHED: Кеплер — T ~ a^(3/2)]
      → .heritage_context готов для Delphi-промптов
    """
    res = AtomicPrecisionResult()

    # ── Уровень 1 ──────────────────────────────────────────────────────────
    res.r2_atomic_ok  = r2_blind >= ATOMIC_R2_THRESHOLD
    res.complexity_ok = 0 < complexity <= ATOMIC_COMPLEXITY_MAX
    res.density_ok    = (r2_blind / max(complexity, 1)) >= ATOMIC_COMPLEXITY_RATIO

    resonance             = _detect_resonance(formula_real)
    res.molecular_fold    = resonance is not None
    res.resonance_pattern = resonance or ""

    structural_ok = res.r2_atomic_ok and res.complexity_ok and res.density_ok
    molecular_ok  = res.r2_atomic_ok and res.molecular_fold
    res.detected  = structural_ok or molecular_ok

    # ── Уровень 2: Heritage ────────────────────────────────────────────────
    heritage          = match_heritage(formula_real, prefer_sympy=True)
    res.heritage      = heritage
    res.heritage_context = heritage.heritage_context   # для engine.py

    # ── Строим report_lines ────────────────────────────────────────────────
    if res.detected:
        res.pantheon_ref = (
            "Джон Джампер (AlphaFold 3): "
            "структура найдена с атомарной точностью через диффузию"
        )
        if res.molecular_fold:
            res.verdict_line = (
                f"[MOLECULAR PRECISION DETECTED] "
                f"Паттерн: {res.resonance_pattern} | "
                f"R²_blind={r2_blind:.4f} | complexity={complexity}"
            )
        else:
            res.verdict_line = (
                f"[MOLECULAR PRECISION DETECTED] "
                f"Плотность: R²/complexity={r2_blind/max(complexity,1):.4f} | "
                f"R²_blind={r2_blind:.4f}"
            )

        report = [
            "═" * 62,
            "  ★ ATOMIC PRECISION — John Jumper Pantheon ★",
            f"  {res.verdict_line}",
            f"  {res.pantheon_ref}",
            "─" * 62,
            f"  R²_blind={r2_blind:.4f} ≥ {ATOMIC_R2_THRESHOLD}  "
            f"→ {'✓' if res.r2_atomic_ok else '✗'}",
            f"  Complexity={complexity} ≤ {ATOMIC_COMPLEXITY_MAX}  "
            f"→ {'✓' if res.complexity_ok else '✗'}",
            f"  Density={r2_blind/max(complexity,1):.4f} ≥ {ATOMIC_COMPLEXITY_RATIO}  "
            f"→ {'✓' if res.density_ok else '✗'}",
            f"  Молекулярная складка: "
            f"{'✓ ' + res.resonance_pattern if res.molecular_fold else '✗'}",
        ]
        if heritage.detected:
            report += ["─" * 62, "  ★ HERITAGE MATCHED — The Great Pantheon v10.5 ★"]
            for hm in heritage.matches:
                report.append(
                    f"  ◆ [{hm.label}]  Скелет: {hm.skeleton}  [{hm.method}]"
                )
        report.append("═" * 62)
        res.report_lines = report

    else:
        res.verdict_line = (
            f"[Atomic] Не обнаружен: R²={r2_blind:.4f}/{ATOMIC_R2_THRESHOLD} "
            f"cpx={complexity}/{ATOMIC_COMPLEXITY_MAX} "
            f"fold={'✓' if res.molecular_fold else '✗'}"
        )
        base = [f"  {res.verdict_line}"]
        if heritage.detected:
            base += [
                "─" * 62,
                "  ★ HERITAGE MATCHED (качество ниже порога, но структура узнана) ★",
            ]
            for hm in heritage.matches:
                base.append(f"  ◆ [{hm.label}]  Скелет: {hm.skeleton}  [{hm.method}]")
        res.report_lines = base

    log.info(
        "[AtomicPrecision v10.5] detected=%s fold=%s heritage=%s r2_blind=%.4f",
        res.detected, res.resonance_pattern,
        heritage.matched_scientists, r2_blind,
    )
    return res


# ══════════════════════════════════════════════════════════════════════════════
# ФОРМАТИРОВАНИЕ ПАНТЕОНА
# ══════════════════════════════════════════════════════════════════════════════

def format_pantheon() -> str:
    """3 столпа Пантеона (всегда одинаково)."""
    lines = ["  ПАНТЕОН СТРУКТУРЫ:"]
    for name, desc in PANTHEON:
        lines.append(f"    ◆ {name}: {desc}")
    return "\n".join(lines)


def format_pantheon_with_matches(heritage: HeritageResult) -> str:
    """
    3 столпа + Heritage Scan с подсветкой совпавших учёных.

    Пример:
      ПАНТЕОН СТРУКТУРЫ:
        ◆ Колмогоров–Арнольд: KAM-теорема...
        ◆ Перельман: ...
        ◆ Джон Джампер: AlphaFold 3...
      ────────────────────────────────────────────────────
      НАСЛЕДИЕ УЧЁНЫХ (Heritage Scan v10.5):
        ★ [Kepler — Третий закон]  T ~ a^(3/2)  [sympy]
        ○ Da Vinci — не обнаружен
        ...
    """
    lines = ["  ПАНТЕОН СТРУКТУРЫ:"]
    for name, desc in PANTHEON:
        lines.append(f"    ◆ {name}: {desc}")
    lines.append("  " + "─" * 58)
    lines.append("  НАСЛЕДИЕ УЧЁНЫХ (Heritage Scan v10.5):")
    matched_keys = set(heritage.matched_scientists)
    for key, info in INVARIANT_LIBRARY.items():
        if key in matched_keys:
            hm = next(m for m in heritage.matches if m.scientist == key)
            lines.append(
                f"    ★ [{info['label']}]  "
                f"Скелет: {info['skeleton']}  [{hm.method}]"
            )
        else:
            lines.append(f"    ○ {info['label']} — не обнаружен")
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════
# v10.14: ДИНАМИЧЕСКОЕ РАСШИРЕНИЕ INVARIANT_LIBRARY
# ══════════════════════════════════════════════════════════════════

def enrich_invariant_library_from_discoveries() -> int:
    """
    Загружает discoveries.jsonl и добавляет новые инварианты в INVARIANT_LIBRARY.
    Вызывается при старте движка — так система "помнит" что нашла в прошлых сессиях.

    Возвращает число добавленных инвариантов.
    """
    try:
        from .config import VAULT_DIR
        disc_path = VAULT_DIR / "discoveries.jsonl"
        if not disc_path.exists():
            return 0

        import json as _j
        added = 0
        for line in disc_path.read_text(encoding="utf-8").strip().splitlines():
            try:
                rec = _j.loads(line)
                formula  = rec.get("formula_shadow", "")
                title    = rec.get("title", "")
                domain   = rec.get("domain", "")
                expl     = rec.get("explanation", "")
                r2       = float(rec.get("r2_blind", 0))

                if not formula or not title:
                    continue

                # Ключ для библиотеки
                key = f"Discovery_{domain}_{formula[:20]}"
                if key in INVARIANT_LIBRARY:
                    continue

                INVARIANT_LIBRARY[key] = {
                    "label":       title,
                    "skeleton":    formula,
                    "description": expl[:150] or f"Найдено системой Scalpel R²={r2:.3f}",
                    "context": (
                        f"DISCOVERY: система ранее нашла этот закон в домене '{domain}'. "
                        f"Формула: {formula}. R²={r2:.3f}. "
                        f"{expl[:200] if expl else ''} "
                        f"Мистик: проверь аналогии. Физик: подтверди структуру."
                    ),
                    "patterns": [],
                }
                added += 1

            except Exception:
                continue

        if added > 0:
            import logging
            logging.getLogger("scalpel").info(
                "[InvariantLibrary] Добавлено %d инвариантов из discoveries.jsonl", added
            )
        return added

    except Exception as e:
        import logging
        logging.getLogger("scalpel").debug("[InvariantLibrary] %s", e)
        return 0
