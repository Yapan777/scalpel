"""
cli.py — Точка входа: python -m scalpel или scalpel в терминале.
"""
import sys
import argparse
import logging
import os
import urllib.request
from pathlib import Path
from datetime import datetime

from .config import (
    VERSION, OLLAMA_HOST, OLLAMA_MODEL,
    PYSR_FAST_FAIL_SEC, PYSR_POPULATIONS, PYSR_N_PROCS,
    DEBUG_MODE, LOG_PATH,
)
from .data import load_csv
from .engine import run_engine, avail_ram_gb, run_llm_phase

log = logging.getLogger("scalpel")


def setup_logging(debug: bool = True) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    fmt = "%(asctime)s [%(levelname)s] %(message)s"
    logging.basicConfig(
        level=logging.DEBUG if debug else logging.INFO,
        format=fmt,
        handlers=[
            logging.FileHandler(LOG_PATH, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )


def _check_ollama(host: str = OLLAMA_HOST, timeout: int = 3) -> bool:
    try:
        req = urllib.request.Request(f"{host}/api/tags")
        with urllib.request.urlopen(req, timeout=timeout):
            return True
    except Exception:
        return False


def _julia_env_ready() -> bool:
    home = Path(os.environ.get("USERPROFILE", os.path.expanduser("~")))
    candidates = [
        home / ".julia" / "environments" / "pyjuliapkg" / "Manifest.toml",
        Path.home() / ".julia" / "environments" / "pyjuliapkg" / "Manifest.toml",
    ]
    depot = os.environ.get("JULIA_DEPOT_PATH", "")
    if depot:
        candidates.append(Path(depot) / "environments" / "pyjuliapkg" / "Manifest.toml")
    return any(p.exists() and p.stat().st_size > 500 for p in candidates)


def startup_checklist(debug: bool) -> None:
    """Печатает таблицу состояния перед стартом."""
    print("\n" + "═" * 62)
    print("  STARTUP CHECKLIST v10.4.5 — Inherent Structure")
    print("═" * 62)

    julia_ready = _julia_env_ready()
    print(f"  Julia-окружение  : {'✓ ГОТОВО' if julia_ready else '⚠ НЕ УСТАНОВЛЕНО (нужен интернет)'}")
    print(f"  JULIAPKG_OFFLINE : {os.environ.get('JULIAPKG_OFFLINE', 'НЕ ВЫСТАВЛЕН')}")

    ollama_alive = _check_ollama()
    print(f"  Ollama           : {'✓ живая' if ollama_alive else '✗ не отвечает (ollama serve)'}")

    ram = avail_ram_gb()
    print(f"  Доступная RAM    : {ram:.2f} ГБ {'✓' if ram >= 2.0 else '⚠ МАЛО (<2 ГБ)'}")

    try:
        from pysr import PySRRegressor
        print(f"  PySR             : ✓ OK")
    except Exception as e:
        print(f"  PySR             : ✗ ОШИБКА ({e})")

    try:
        import numpy as np
        print(f"  numpy            : ✓ OK ({np.__version__})")
    except Exception:
        print(f"  numpy            : ✗ ОШИБКА")

    try:
        from scipy.signal import savgol_filter
        print(f"  scipy            : ✓ OK (Diffusion + Surgery)")
    except Exception:
        print(f"  scipy            : ✗ ОШИБКА")

    # v10.4.5 модули
    try:
        from scalpel.diffusion_denoise import diffusion_denoise
        print(f"  DiffusionDenoise : ✓ OK (AlphaFold 3 — Сгущение структуры)")
    except Exception as e:
        print(f"  DiffusionDenoise : ✗ ({e})")

    try:
        from scalpel.pairformer import pairformer_select
        print(f"  Pairformer       : ✓ OK (AlphaFold 3 — Взаимосвязи, ~500 МБ saved)")
    except Exception as e:
        print(f"  Pairformer       : ✗ ({e})")

    try:
        from scalpel.atomic_precision import check_atomic_precision, PANTHEON, INVARIANT_LIBRARY
        print(f"  AtomicPrecision  : ✓ OK (Пантеон: {len(PANTHEON)} столпов)")
        print(f"  Heritage Library : ✓ OK ({len(INVARIANT_LIBRARY)} учёных: {', '.join(INVARIANT_LIBRARY.keys())})")
    except Exception as e:
        print(f"  AtomicPrecision  : ✗ ({e})")

    try:
        import sympy
        print(f"  sympy            : ✓ OK {sympy.__version__} (Heritage structural match)")
    except ImportError:
        print(f"  sympy            : ⚠ НЕ УСТАНОВЛЕН — Heritage fallback на regex (pip install sympy)")

    print(f"  Siege Mode       : 3.0 (Диффузионная Пауза {PYSR_FAST_FAIL_SEC//1}s → Julia)")
    print(f"  DEBUG_MODE       : {'ON' if debug else 'OFF'}")

    # v10.14: Летописец и Curriculum
    try:
        from scalpel.episodic_memory import get_memory, MEMORY_DIR
        mem   = get_memory()
        stats = mem.stats("Скептик")
        chron_path = MEMORY_DIR / "chronicle_steps.jsonl"
        n_chronicle = 0
        if chron_path.exists():
            n_chronicle = sum(1 for l in chron_path.read_text(encoding="utf-8").strip().splitlines() if l)
        curr_path = MEMORY_DIR / "curriculum_memory.jsonl"
        n_curriculum = 0
        if curr_path.exists():
            n_curriculum = sum(1 for l in curr_path.read_text(encoding="utf-8").strip().splitlines() if l)
        print(f"  Летописец        : ✓ {n_chronicle} записей в хронике")
        print(f"  Curriculum память: ✓ {n_curriculum} датасетов обучено")
        from scalpel.config import VAULT_DIR
        cp_path = VAULT_DIR / "curriculum_checkpoint.json"
        if cp_path.exists():
            import json as _json
            cp = _json.loads(cp_path.read_text(encoding="utf-8"))
            print(f"  Checkpoint       : ▶ уровень={cp.get('level')} датасет={cp.get('dataset')} "
                  f"(не завершён — используй --curriculum для продолжения)")
    except Exception as _chr_e:
        print(f"  Летописец        : ⚠ ({_chr_e})")

    # v10.14 Антиинцест: проверяем наличие всех моделей
    try:
        from .config import NAVIGATOR_MODEL, SYNTHESIS_MODEL, CHRONICLE_MODEL, ROLE_MODELS
        import urllib.request as _ureq
        _models_response = _ureq.urlopen(f"{OLLAMA_HOST}/api/tags", timeout=3)
        _tags = __import__('json').loads(_models_response.read())
        _available = {m['name'].split(':')[0] for m in _tags.get('models', [])}
        _required = {
            "Navigator":   NAVIGATOR_MODEL,
            "Синтез":      SYNTHESIS_MODEL,
            "Летописец":   CHRONICLE_MODEL,
            **{f"Матрёшка/{k}": v for k,v in ROLE_MODELS.items()},
        }
        _missing = []
        for role, model in _required.items():
            base = model.split(':')[0]
            if base not in _available:
                _missing.append(f"{role}: {model}")
        if _missing:
            print(f"  ⚠ МОДЕЛИ НЕ СКАЧАНЫ ({len(_missing)} из {len(_required)}):")
            for m in _missing:
                print(f"      ollama pull {m.split(': ')[1]}")
        else:
            uniq = set(_required.values())
            print(f"  Модели (антиинцест): ✓ {len(uniq)} из {len(uniq)} доступны")
    except Exception as _model_check_err:
        print(f"  Модели           : ⚠ не проверить ({_model_check_err})")

    print("═" * 62 + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=f"Universal Scalpel v{VERSION}")
    parser.add_argument("--train",    default="universe_train.csv")
    parser.add_argument("--test",     default="universe_test.csv")
    parser.add_argument("--target",   default="",
                        help="целевой столбец (авто если пусто)")
    parser.add_argument("--timeout",  type=int, default=PYSR_FAST_FAIL_SEC,
                        help="таймаут PySR в секундах")
    parser.add_argument("--model",    default=OLLAMA_MODEL)
    parser.add_argument("--domain",   default="",
                        help="Finance / Physics / Biology / Generic")
    parser.add_argument("--no-debug", action="store_true",
                        help="Отключить расширенное логирование")
    parser.add_argument("--phase",    default="full",
                        choices=["full", "pysr", "llm"],
                        help=(
                            "Режим запуска: "
                            "'full' — полный (по умолчанию), "
                            "'pysr' — только PySR, сохранить результат, "
                            "'llm'  — только LLM-верификация сохранённого результата"
                        ))
    parser.add_argument("--retry",    action="store_true",
                        help=(
                            "Только с --phase llm: если Матрёшка вернула ОТКЛОНЕНА, "
                            "автоматически запустить ещё один PySR-цикл и повторить "
                            "LLM-верификацию. Финальный вердикт всегда из LLM-фазы."
                        ))
    parser.add_argument("--curriculum",  action="store_true",
                            help="Запустить curriculum learning (все уровни автоматически).")
    parser.add_argument("--level",       type=int, default=None,
                            choices=[1, 2, 3, 4, 5],
                            help="Только с --curriculum: запустить конкретный уровень (1-5).")
    parser.add_argument("--start",       type=int, default=None,
                            choices=[1, 2, 3, 4, 5],
                            metavar="N",
                            help="Только с --curriculum: начать с уровня N (по умолчанию 4).")
    parser.add_argument("--end",         type=int, default=None,
                            choices=[1, 2, 3, 4, 5],
                            metavar="N",
                            help="Только с --curriculum: закончить уровнем N (по умолчанию 5).")
    parser.add_argument("--reflect",     action="store_true",
                            help=(
                                "Принудительный мета-анализ: читает всю историю памяти, "
                                "генерирует выводы по ошибкам/успехам/правилам, "
                                "обновляет DSPy обучение."
                            ))
    args = parser.parse_args()

    # ── Meta-Reflection (принудительная) ─────────────────────────
    if args.reflect:
        setup_logging(debug=not args.no_debug)
        from .meta_reflection import run_forced_reflection
        report = run_forced_reflection(model=args.model, host=OLLAMA_HOST)
        if report:
            print(f"\n  Отчёт: {report}")
            return 0
        print("\n  Рефлексия не запущена (мало данных или ошибка).")
        return 1

    # ── Curriculum Learning ───────────────────────────────────────
    if args.curriculum:
        from .curriculum import run_curriculum
        results = run_curriculum(
            level_filter = args.level,
            start_level  = args.start,
            end_level    = args.end,
            model        = args.model,
            host         = OLLAMA_HOST,
        )
        passed = sum(1 for r in results if r.passed)
        print(f"\n  Curriculum завершён: {passed}/{len(results)} уровней пройдено.")
        return 0 if passed == len(results) else 1

    # ── Быстрый путь: --phase llm ─────────────────────────────────
    if args.phase == "llm":
        debug = not args.no_debug
        setup_logging(debug)
        print("\n" + "═" * 62)
        print(f"  Universal Scalpel v{VERSION} — LLM-фаза")
        if args.retry:
            print("  Режим: LLM + авто-retry при ОТКЛОНЕНА")
        print("═" * 62)
        startup_checklist(debug)

        # ── Раунд 1: LLM-верификация сохранённого PySR-результата ──
        try:
            result = run_llm_phase(model=args.model, host=OLLAMA_HOST)
        except FileNotFoundError as e:
            sys.exit(str(e))

        log.info("[LLM] Раунд 1: verdict=%s consensus=%s",
                 result.verdict, result.consensus)

        # ── Retry: если все кандидаты отклонены — один новый PySR-цикл ─
        all_rejected = result.verdict.startswith("BEST_OF_ALL") or result.consensus == "ОТКЛОНЕНА"
        if args.retry and all_rejected:
            print("\n" + "═" * 62)
            print("  [Retry] Матрёшка отклонила формулу.")
            print("  [Retry] Запускаем новый PySR-цикл…")
            print("═" * 62)

            # Нужны CSV для нового PySR-цикла
            try:
                X_train, y_train, X_test, y_test, feat_names, target_col = load_csv(
                    train_path=args.train,
                    test_path=args.test,
                    target_col=args.target,
                )
            except FileNotFoundError as e:
                print(f"  [Retry] Не найдены CSV файлы: {e}")
                print("  [Retry] Укажи --train и --test для retry.")
                print("  [Retry] Финальный вердикт: ОТКЛОНЕНА (retry невозможен)")
                log.warning("[Retry] CSV не найдены, retry пропущен.")
            else:
                # Запускаем новый PySR-цикл (--phase pysr)
                retry_engine = run_engine(
                    X_train, y_train, X_test, y_test,
                    feat_names  = feat_names,
                    target_col  = target_col,
                    timeout_sec = args.timeout,
                    domain_type = args.domain,
                    phase       = "pysr",   # сохраняет новый json
                )
                log.info("[Retry] PySR-цикл: verdict=%s R²_blind=%.4f",
                         retry_engine.verdict, retry_engine.r2_blind)

                print("\n" + "═" * 62)
                print("  [Retry] PySR завершён. Запускаем LLM-верификацию (раунд 2)…")
                print("═" * 62)

                # Раунд 2: LLM-верификация нового результата
                try:
                    result = run_llm_phase(model=args.model, host=OLLAMA_HOST)
                except FileNotFoundError as e:
                    print(f"  [Retry] Ошибка LLM-фазы: {e}")
                else:
                    log.info("[LLM] Раунд 2 (retry): verdict=%s consensus=%s",
                             result.verdict, result.consensus)

        # ── Финальный вердикт ─────────────────────────────────────
        print("\n" + "═" * 62)
        print(f"  ОКОНЧАТЕЛЬНЫЙ ВЕРДИКТ: {result.verdict}")
        print(f"  Consensus: {result.consensus}")
        print(f"  R²_train={result.r2_train:.4f}  R²_blind={result.r2_blind:.4f}")
        print("═" * 62)

        log.info("=== LLM-ФАЗА ЗАВЕРШЕНА verdict=%s ===", result.verdict)
        accepted = "INVARIANT" in result.verdict and "BEST_OF_ALL" not in result.verdict
        return 0 if accepted else 1

    debug = not args.no_debug
    setup_logging(debug)

    print("==" * 31)
    print(f"  Universal Scalpel v{VERSION} — Inherent Structure")
    print(f"  NIST CSPRNG | Shadow | OODA | Diffusion | Pairformer | AtomicPrecision")
    print(f"  Siege 3.0: Ollama→stop→gc→Wait3s→Julia ⚡")
    print(f"  PySR: populations={PYSR_POPULATIONS} n_procs={PYSR_N_PROCS}")
    print(f"  Model: {args.model} | Timeout: {args.timeout}s")
    print(f"  Domain: {args.domain or '(не указан)'}")
    phase_label = {"full": "ПОЛНЫЙ", "pysr": "ТОЛЬКО PySR → сохранить", "llm": "ТОЛЬКО LLM"}.get(args.phase, args.phase)
    if args.phase == "llm" and args.retry:
        phase_label = "LLM + авто-retry"
    print(f"  Режим:  {phase_label}")
    print("==" * 31)

    startup_checklist(debug)

    try:
        X_train, y_train, X_test, y_test, feat_names, target_col = load_csv(
            train_path=args.train,
            test_path=args.test,
            target_col=args.target,
        )
    except FileNotFoundError as e:
        sys.exit(f"[FATAL] {e}")

    result = run_engine(
        X_train, y_train, X_test, y_test,
        feat_names  = feat_names,
        target_col  = target_col,
        timeout_sec = args.timeout,
        domain_type = args.domain,
        phase       = args.phase,
        noise_hint  = getattr(args, 'noise_level', None),  # v10.14: --noise-level
    )

    log.info("=== ЗАВЕРШЕНО [%s] verdict=%s ===",
             datetime.now().isoformat(), result.verdict)
    return 0 if "INVARIANT" in result.verdict else 1


if __name__ == "__main__":
    sys.exit(main())
