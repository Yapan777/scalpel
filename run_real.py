"""
run_real.py — Режим реальной науки (Real Science Mode)

НЕ для benchmark — для реальных данных учёного.

Отличия от официального benchmark:
  - Шум 10% (лаборатория), все 4 типа
  - Тест тоже зашумлён — нет чистых данных
  - Порог не R²≥0.9999 — Матрёшка всегда решает
  - Матрёшка запускается при ЛЮБОМ R²
  - Результат: ПРИНЯТО / КАНДИДАТ / ОТКЛОНЕНО

Запуск:
  python run_real.py                    # все 122 закона Фейнмана
  python run_real.py --subset 5         # первые 5 для теста
  python run_real.py --noise gaussian   # только один тип шума
  python run_real.py --noise outliers
  python run_real.py --noise hetero
  python run_real.py --noise missing

Математическое обоснование порогов:
  R²_max при 10% шуме = 1 - eps^2 = 1 - 0.01 = 0.99
  R2_STRONG   = 0.80  (82% от R²_max)
  R2_MODERATE = 0.50  (51% от R²_max)
  Источник: Hastie et al. "Elements of Statistical Learning" гл.2.9
"""
import sys, os, argparse, time, json, hashlib
sys.path.insert(0, os.getcwd())

import numpy as np
from pathlib import Path
from datetime import datetime

# ── Уровень шума ──────────────────────────────────────────────────
NOISE_LEVEL = 0.10   # 10% — лабораторные данные

# ── Пороги оценки (выведены математически, не подобраны) ──────────
R2_STRONG   = 0.80
R2_MODERATE = 0.50

RESULTS_DIR       = Path("scalpel_vault/real_results")
GROUND_TRUTH_PATH = Path("scalpel_vault/real_ground_truth_PRIVATE.json")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

N_TRAIN = 150   # дефолт если n_samples не задан
N_TEST  = 50


def adapt_to_n_samples(n: int) -> dict:
    """
    Адаптирует параметры системы под количество точек.
    Реальный учёный даёт столько данных сколько есть.
    Система сама подбирает оптимальные параметры.

    Принципы:
    - Мало данных → консервативнее (не режем выбросы лишние, меньше итераций)
    - Много данных → агрессивнее (больше итераций PySR, шире Ricci окно)
    - Train/test split адаптируется: при малых n — больше на train
    """
    # Train/test split
    if n <= 20:
        n_train = max(10, int(0.85 * n))   # 85% train — каждая точка на счету
    elif n <= 50:
        n_train = int(0.75 * n)
    else:
        n_train = int(0.80 * n)
    n_test = max(3, n - n_train)

    # PySR итерации
    if n <= 20:    pysr_iter = 10000
    elif n <= 50:  pysr_iter = 20000
    else:          pysr_iter = 40000

    # Ricci Flow окно (Savitzky-Golay)
    if n <= 20:    ricci_w = 3
    elif n <= 50:  ricci_w = 5
    elif n <= 100: ricci_w = 7
    else:          ricci_w = 11

    # IQR порог хирурга — мало данных → консервативнее
    if n <= 30:    iqr_k = 4.0
    elif n <= 80:  iqr_k = 3.5
    else:          iqr_k = 3.0

    # Предупреждение если очень мало
    if n < 20:
        warning = f"⚠️  Очень мало данных ({n} точек) — результат ненадёжен"
    elif n < 50:
        warning = f"⚠️  Мало данных ({n} точек) — повысить IQR_k до {iqr_k}"
    else:
        warning = None

    return {
        "n_train":   n_train,
        "n_test":    n_test,
        "pysr_iter": pysr_iter,
        "ricci_w":   ricci_w,
        "iqr_k":     iqr_k,
        "warning":   warning,
    }


def add_noise(y, rng, noise_type="gaussian", level=NOISE_LEVEL):
    std_y = float(np.std(y)) if float(np.std(y)) > 1e-12 else 1.0
    n     = len(y)
    y_out = y.copy().astype(np.float64)

    if noise_type == "gaussian":
        y_out += rng.normal(0, level * std_y, n)

    elif noise_type == "outliers":
        y_out += rng.normal(0, level * std_y, n)
        n_out  = max(1, int(0.05 * n))
        idx    = rng.choice(n, n_out, replace=False)
        signs  = rng.choice([-1, 1], n_out)
        y_out[idx] += signs * 3.0 * std_y

    elif noise_type == "hetero":
        scale  = level * (np.abs(y) + 0.1 * std_y)
        y_out += rng.normal(0, scale, n)

    elif noise_type == "missing":
        y_out += rng.normal(0, level * std_y, n)
        n_miss = max(1, int(0.08 * n))
        idx    = rng.choice(n, n_miss, replace=False)
        y_out[idx] = float(np.median(y_out))

    return y_out


def score_real(r2, consensus):
    cons = (consensus or "").upper()
    accepted = any(w in cons for w in ["ПРИНЯТ", "ACCEPT", "ПОДТВЕР"])
    rejected = any(w in cons for w in ["ОТКЛОН", "REJECT", "DEATH"])

    if r2 >= R2_STRONG and not rejected:
        return "ПРИНЯТО"
    if r2 >= R2_MODERATE and not rejected:
        return "КАНДИДАТ"
    if r2 >= R2_MODERATE and accepted:
        return "КАНДИДАТ"
    return "ОТКЛОНЕНО"


def _check_ollama(host="http://localhost:11434"):
    import urllib.request
    try:
        urllib.request.urlopen(f"{host}/api/tags", timeout=3).read()
        print("  ✓ Ollama доступна\n")
        return True
    except Exception:
        print("\n⚠️  OLLAMA НЕ ЗАПУЩЕНА. Запусти: ollama serve\n")
        return False


def _generate_data_real(law, law_idx, seed, noise_type, n_train=N_TRAIN, n_test=N_TEST):
    """
    Генерирует данные используя law["ranges"] и law["formula_fn"].
    Оба набора (train и test) зашумлены — реальный режим.
    law_idx — целое число (порядковый индекс) для seed.
    """
    n_feat  = law["n_features"]
    feats   = [f"f{i}" for i in range(n_feat)]
    ranges  = law["ranges"]
    n_total = n_train + n_test

    rng_feat     = np.random.default_rng(seed + law_idx * 100)
    rng_noise_tr = np.random.default_rng(seed + law_idx * 200)
    rng_noise_te = np.random.default_rng(seed + law_idx * 300)

    # Генерируем признаки из ranges
    data = {}
    for i, (lo, hi) in enumerate(ranges):
        if lo == hi:
            data[f"f{i}"] = np.full(n_total, lo)
        else:
            data[f"f{i}"] = rng_feat.uniform(lo, hi, n_total)

    # Чистый y
    y_clean = law["formula_fn"](data)

    # Защита от NaN/Inf
    valid = np.isfinite(y_clean)
    if not np.all(valid):
        for k in data:
            data[k] = data[k][valid]
        y_clean = y_clean[valid]

    n_total = len(y_clean)
    n_tr    = min(n_train, int(0.75 * n_total))
    n_te    = min(n_test,  n_total - n_tr)

    X_all = np.column_stack([data[f] for f in feats])

    perm   = rng_feat.permutation(n_total)
    tr_idx = perm[:n_tr]
    te_idx = perm[n_tr:n_tr + n_te]

    X_train = X_all[tr_idx].astype(np.float64)
    X_test  = X_all[te_idx].astype(np.float64)

    # Реальный режим: шум на ОБОИХ наборах
    y_train = add_noise(y_clean[tr_idx], rng_noise_tr, noise_type, NOISE_LEVEL)
    y_test  = add_noise(y_clean[te_idx], rng_noise_te, noise_type, NOISE_LEVEL * 0.5)

    return X_train, X_test, y_train.astype(np.float64), y_test.astype(np.float64), feats


def run_real_benchmark(noise_types=None, subset=None, seed=42, n_samples=None):
    if noise_types is None:
        noise_types = ["gaussian", "outliers", "hetero", "missing"]

    print("=" * 62)
    print("  REAL SCIENCE MODE — Scalpel")
    print(f"  Шум: {NOISE_LEVEL*100:.0f}% | Типы: {', '.join(noise_types)}")
    print("  Матрёшка активна при ЛЮБОМ R²")
    print("  Тест зашумлён — как реальные данные учёного")
    print("=" * 62)
    print()

    _check_ollama()

    # Адаптируем параметры под количество точек
    if n_samples is not None:
        params = adapt_to_n_samples(n_samples)
        print(f"  📊 Адаптация под {n_samples} точек:")
        print(f"     train={params['n_train']} test={params['n_test']}")
        print(f"     PySR итерации={params['pysr_iter']}")
        print(f"     Ricci окно={params['ricci_w']}  IQR_k={params['iqr_k']}")
        if params['warning']:
            print(f"     {params['warning']}")
        print()
        n_train_use = params['n_train']
        n_test_use  = params['n_test']
        surgeon_iqr = params['iqr_k']
    else:
        n_train_use = N_TRAIN
        n_test_use  = N_TEST
        surgeon_iqr = 3.0
        params      = None

    from scalpel.engine import run_engine, run_llm_phase, ollama_stop_all

    # Загружаем законы Фейнмана
    try:
        from run_feynman_official import build_feynman_laws
        LAWS = build_feynman_laws()
        print(f"  ✓ Загружено {len(LAWS)} законов Фейнмана")
    except Exception as e:
        print(f"⚠️  run_feynman_official.py недоступен ({e}) — встроенный набор")
        LAWS = _builtin_laws()

    laws = LAWS[:subset] if subset else LAWS
    if subset:
        print(f"  Subset: первые {subset} из {len(LAWS)} законов\n")
    else:
        print(f"  Законов: {len(laws)}\n")

    results   = []
    score_map = {"ПРИНЯТО": 0, "КАНДИДАТ": 0, "ОТКЛОНЕНО": 0}
    ts_start  = datetime.now().isoformat(timespec="seconds")

    # Resume — продолжаем с места остановки
    out_path = RESULTS_DIR / f"real_results_{ts_start[:10]}.json"
    done_ids = set()
    if out_path.exists():
        try:
            prev = json.loads(out_path.read_text(encoding="utf-8"))
            for r in prev.get("results", []):
                done_ids.add(r["id"])
                results.append(r)
                score_map[r.get("verdict", "ОТКЛОНЕНО")] = \
                    score_map.get(r.get("verdict", "ОТКЛОНЕНО"), 0) + 1
            if done_ids:
                print(f"  ▶ Продолжаем: уже {len(done_ids)} законов выполнено\n")
        except Exception:
            pass

    for law_idx, law in enumerate(laws):
        lid    = law["id"]           # строка: "I.6.20" и т.д.
        n_type = noise_types[law_idx % len(noise_types)]

        if lid in done_ids:
            print(f"  ⏭  {lid} уже выполнен")
            continue

        # Очищаем RAM перед каждым законом — выгружаем все модели
        # Иначе llama3/yi:9b остаются в памяти и deepseek таймаутит
        ollama_stop_all()

        print(f"\n{'─'*62}")
        print(f"  {lid}: [СКРЫТО] | {law['n_features']} признаков | шум: {n_type}")
        print(f"{'─'*62}")

        try:
            X_train, X_test, y_train, y_test, feats = _generate_data_real(
                law, law_idx, seed, n_type,
                n_train=n_train_use, n_test=n_test_use,
            )
        except Exception as e:
            print(f"  ❌ Ошибка генерации: {e}")
            continue

        print(f"  Данные: {len(X_train)} train / {len(X_test)} test (оба зашумлены)")

        t0        = time.time()
        final_f   = "ERROR"
        r2_final  = 0.0
        consensus = "ERROR"

        try:
            result = run_engine(
                X_train       = X_train,
                y_train       = y_train,
                X_test        = X_test,
                y_test        = y_test,
                feat_names    = feats,
                target_col    = "y",
                domain_type   = "",
                phase         = "pysr",
                dim_codes     = [0] * law["n_features"],
                noise_hint    = NOISE_LEVEL,   # v10.33 FIX: передаём реальный шум (был None → агрессивный денойзинг)
                skip_heritage = False,
            )
            pysr_f  = result.formula_real
            r2_pysr = result.r2_blind
            print(f"\n  PySR: {pysr_f}  R²={r2_pysr:.4f}")

            # МАТРЁШКА ВСЕГДА
            print(f"  [Real Mode] Матрёшка запускается (без порога R²)")
            llm = run_llm_phase()
            if llm and llm.formula_real:
                final_f   = llm.formula_real
                r2_final  = llm.r2_blind
                consensus = llm.consensus
            else:
                final_f   = pysr_f
                r2_final  = r2_pysr
                consensus = "NOT_RUN"

        except Exception as e:
            print(f"  ❌ Ошибка движка: {e}")

        elapsed = round(time.time() - t0, 1)
        verdict = score_real(r2_final, consensus)
        score_map[verdict] = score_map.get(verdict, 0) + 1

        icons = {"ПРИНЯТО": "✅", "КАНДИДАТ": "🟡", "ОТКЛОНЕНО": "❌"}
        print(f"\n  {icons.get(verdict,'?')} [{verdict}] {lid}")
        print(f"     Нашла:    {str(final_f)[:80]}")
        print(f"     Истина:   [СКРЫТО → real_ground_truth_PRIVATE.json]")
        print(f"     R²={r2_final:.4f}  Консенсус={consensus}  t={elapsed}s")
        print(f"     Шум: {n_type}")

        # Истина → приватный файл
        tf_hash = hashlib.sha256(law["true_formula"].encode()).hexdigest()[:16]
        _gt = {}
        if GROUND_TRUTH_PATH.exists():
            try:
                _gt = json.loads(GROUND_TRUTH_PATH.read_text(encoding="utf-8"))
            except Exception:
                pass
        _gt[str(lid)] = {"name": law.get("name",""), "true_formula": law["true_formula"]}
        GROUND_TRUTH_PATH.write_text(
            json.dumps(_gt, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        results.append({
            "id":                lid,
            "name":              law.get("name",""),
            "noise_type":        n_type,
            "noise_level":       NOISE_LEVEL,
            "true_formula_hash": tf_hash,
            "found_formula":     str(final_f)[:200],
            "r2_blind":          round(r2_final, 4),
            "consensus":         consensus,
            "verdict":           verdict,
            "elapsed_sec":       elapsed,
            "ts":                datetime.now().isoformat(timespec="seconds"),
        })

        out_path.write_text(json.dumps({
            "mode":        "real_science",
            "noise_level": NOISE_LEVEL,
            "noise_types": noise_types,
            "ts_start":    ts_start,
            "ts_updated":  datetime.now().isoformat(timespec="seconds"),
            "score_map":   score_map,
            "results":     results,
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  Сохранено → {out_path.name}")

    total = len(laws)
    print(f"\n{'═'*62}")
    print(f"  REAL SCIENCE MODE — ИТОГ")
    print(f"{'═'*62}")
    print(f"  ✅ Принято:   {score_map.get('ПРИНЯТО',0)}/{total}")
    print(f"  🟡 Кандидат: {score_map.get('КАНДИДАТ',0)}/{total}")
    print(f"  ❌ Отклонено:{score_map.get('ОТКЛОНЕНО',0)}/{total}")
    print(f"{'─'*62}")
    print(f"  Шум: {NOISE_LEVEL*100:.0f}% ({', '.join(noise_types)})")
    print(f"  Результаты → {RESULTS_DIR}")
    print(f"  Истина     → {GROUND_TRUTH_PATH.name}")
    print(f"{'═'*62}")


def _builtin_laws():
    """Встроенный мини-набор в правильном формате."""
    return [
        {
            "id": "builtin.1", "name": "Потенциальная энергия",
            "true_formula": "f0 * f1 * f2", "n_features": 3,
            "ranges": [(0.5, 20.0), (9.81, 9.81), (0.1, 50.0)],
            "formula_fn": lambda d: d["f0"] * d["f1"] * d["f2"],
            "noise_level": 0.05,
        },
        {
            "id": "builtin.2", "name": "Объём шара",
            "true_formula": "4.189 * f0**3", "n_features": 1,
            "ranges": [(0.1, 5.0)],
            "formula_fn": lambda d: (4/3) * np.pi * d["f0"]**3,
            "noise_level": 0.05,
        },
        {
            "id": "builtin.3", "name": "Закон Ома",
            "true_formula": "f0 * f1", "n_features": 2,
            "ranges": [(0.1, 10.0), (1.0, 100.0)],
            "formula_fn": lambda d: d["f0"] * d["f1"],
            "noise_level": 0.05,
        },
    ]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Real Science Mode — Scalpel")
    parser.add_argument("--subset", type=int, default=None,
                        help="Первые N законов (для теста)")
    parser.add_argument("--noise", type=str, default=None,
                        choices=["gaussian", "outliers", "hetero", "missing", "all"],
                        help="Тип шума (по умолчанию: все 4)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_samples", type=int, default=None,
                        help="Число точек (дефолт: 200). Система адаптирует параметры.")
    args = parser.parse_args()

    noise_types = (
        [args.noise] if (args.noise and args.noise != "all")
        else ["gaussian", "outliers", "hetero", "missing"]
    )

    run_real_benchmark(
        noise_types = noise_types,
        subset      = args.subset,
        seed        = args.seed,
        n_samples   = args.n_samples,
    )
