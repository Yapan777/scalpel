"""
scalpel — Universal Scalpel v10.5.1 (The Great Pantheon + DataAware Audit)

v10.2.7: Humanistic Filters — Critical Thinking через гуманитарные фильтры.
v10.3.8: Topological Surgery (Метод Перельмана).
v10.4.5: Inherent Structure (принципы AlphaFold 3):
  DiffusionDenoising  — «Сгущение структуры» за T шагов до PySR.
  PairformerLogic     — попарная энергия признаков (экономия ~500 МБ RAM).
  AtomicPrecision     — Пантеон Джона Джампера + [MOLECULAR PRECISION DETECTED].
  SiegeMode3.0        — Диффузионная Пауза: ollama_stop→gc→Wait3s→Julia.

v10.5 (The Great Pantheon):
  INVARIANT_LIBRARY   — Heritage Scan: 12 реальных законов (Ньютон, Кеплер, Хук, Аррениус, Кюри, Гаусс, Больцман, Михаэлис-Ментен, Лотка-Вольтерра, Клейбер, степенной закон).
  match_heritage()    — sympy structural match → [HERITAGE MATCHED: ...].
  heritage_context    — Heritage инжектируется в Delphi (Скептик + Физик).
  PDCA re-check       — check_stale() с реальным predict_fn после PySR fit.
  Sinquain+Dialectic  — extended_audit_report сохраняется в FINAL REPORT.
"""
from .config import VERSION, SACRED_METRICS
from .shadow import ShadowMapper
from .dim_codes import dim_code, dim_code_interactive
from .navigator import NavDecision, nav_decision_from_dspy, ollama_chat, navigator_ask_legacy
from .vault import GoldVault
from .audit import matryoshka_audit
from .ram_queue import MatryoshkaQueue, RoleResult, ram_status_report
from .episodic_memory import EpisodicMemory, get_memory
from .engine import EngineResult, run_engine, shuffle_test, cross_blind
from .data import load_csv
from .critical_thinking import (
    deep_root_analysis, format_root_cause_section,
    socratic_cross_examination, format_dialectic_section,
    generate_sinquain, format_sinquain_section,
    lasso_pull, format_lasso_section,
    LASSO_SYSTEM_INSTRUCTION,
    generate_chronicle,        # v10.14: Летописец
)
from .curriculum import run_curriculum           # v10.14: Curriculum Learning
from .meta_reflection import (                   # v10.14: Мета-рефлексия
    check_and_reflect, run_forced_reflection, load_meta_examples,
)
from .discovery import (                         # v10.14: Детектор открытий
    classify_discovery, load_discoveries_as_examples,
    load_discoveries_summary, DOMAIN_SIGNATURES,
)
from .meta_context import (                      # v10.14: Инжекция мета-знаний
    get_navigator_context, get_matryoshka_context,
    get_delphi_context, get_chronicle_context,
    enrich_failure_logs, get_full_meta_summary,
)
from .dspy_optimizer import DSPyOrchestrator     # v10.14: доступен напрямую
from .topological_surgery import (
    detect_outliers_iqr, ricci_flow_smooth, perform_surgery,
    mark_poincare_invariant, format_surgery_report, SurgeryResult,
)
from .diffusion_denoise import (    # v10.4.5
    diffusion_denoise, format_diffusion_report, DiffusionResult,
)
from .pairformer import (           # v10.4.5
    pairformer_select, format_pairformer_report, PairformerResult,
)
from .atomic_precision import (     # v10.5 The Great Pantheon
    check_atomic_precision, AtomicPrecisionResult,
    format_pantheon, format_pantheon_with_matches,
    match_heritage, HeritageResult, HeritageMatch,
    PANTHEON, INVARIANT_LIBRARY,
)

__all__ = [
    "VERSION", "SACRED_METRICS",
    "ShadowMapper",
    "dim_code", "dim_code_interactive",
    "NavDecision", "nav_decision_from_dspy", "ollama_chat", "navigator_ask_legacy",
    "GoldVault",
    "matryoshka_audit",
    "MatryoshkaQueue", "RoleResult", "ram_status_report",
    "EpisodicMemory", "get_memory",
    "EngineResult", "run_engine",
    "shuffle_test", "cross_blind",
    "load_csv",
    # v10.2.7 Critical Thinking
    "deep_root_analysis", "format_root_cause_section",
    "socratic_cross_examination", "format_dialectic_section",
    "generate_sinquain", "format_sinquain_section",
    "lasso_pull", "format_lasso_section",
    "LASSO_SYSTEM_INSTRUCTION",
    # v10.14 Летописец + Curriculum
    "generate_chronicle",
    "run_curriculum",
    "DSPyOrchestrator",
    "check_and_reflect", "run_forced_reflection", "load_meta_examples",
    "classify_discovery", "load_discoveries_as_examples", "load_discoveries_summary", "DOMAIN_SIGNATURES",
    "get_navigator_context", "get_matryoshka_context", "get_full_meta_summary",
    # v10.3.8 Topological Surgery
    "detect_outliers_iqr", "ricci_flow_smooth", "perform_surgery",
    "mark_poincare_invariant", "format_surgery_report", "SurgeryResult",
    # v10.4.5 Inherent Structure (AlphaFold 3)
    "diffusion_denoise", "format_diffusion_report", "DiffusionResult",
    "pairformer_select", "format_pairformer_report", "PairformerResult",
    # v10.5 The Great Pantheon
    "check_atomic_precision", "AtomicPrecisionResult",
    "format_pantheon", "format_pantheon_with_matches",
    "match_heritage", "HeritageResult", "HeritageMatch",
    "PANTHEON", "INVARIANT_LIBRARY",
]

# v10.22: Антрополог — понимание структуры мира
from .anthropologist import anthropologist_reflect, load_world_model_for_oracle

# v10.22: Препаратор — математическое масштабирование данных
from .preparator import analyze_and_prepare, inverse_transform, prepare_report
