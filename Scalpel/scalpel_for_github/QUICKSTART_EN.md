# Scalpel — Quick Start

This guide takes you from zero to your first run. Everything works locally — no cloud, no API keys.

---

## Requirements

| Requirement | Minimum | Recommended |
|---|---|---|
| RAM | 16 GB | 32 GB |
| Python | 3.10+ | 3.11 |
| CPU | any | 8+ cores |
| GPU | not needed | — |
| OS | Windows / Linux / macOS | — |

---

## Step 1 — Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/scalpel.git
cd scalpel
```

Or download ZIP via GitHub → **Code → Download ZIP**, unpack and enter the folder.

---

## Step 2 — Install Python dependencies

```bash
pip install -e ".[full]"
```

This installs everything at once: numpy, scipy, scikit-learn, PySR and DSPy.

> **If you only want base dependencies without PySR/DSPy:**
> ```bash
> pip install -e .
> ```

---

## Step 3 — Install Ollama and download models

Ollama is a local server for language models. Scalpel uses 9 models from different architectural families.

### 3.1 Install Ollama

Go to [ollama.ai](https://ollama.ai) and download the installer for your OS. After installation, verify it works:

```bash
ollama list
```

You should see an empty list (or a list of already downloaded models).

### 3.2 Download all models

```bash
ollama pull qwen2.5:7b
ollama pull deepseek-r1:7b
ollama pull phi4:14b
ollama pull mistral:7b
ollama pull granite3.1-dense:8b
ollama pull llama3:8b
ollama pull gemma2:9b
ollama pull yi:9b
ollama pull internlm2:7b
```

> Each model is 4–8 GB. Total ~50 GB. You can leave the download running overnight.
>
> If RAM is limited — the first 4 models (qwen, phi4, mistral, llama3) are enough. The system will substitute them for the remaining roles.

---

## Step 4 — First run

### Quick test (recommended starting point)

```bash
python run_real.py --subset 5 --noise gaussian --seed 42
```

Takes the first 5 Feynman laws, adds 10% Gaussian noise, and runs the full search cycle.

**This will take several hours** — each law takes ~25 minutes of PySR + LLM role time.

If the run is interrupted — just run the same command again. The system will resume from where it stopped.

### All run options

```bash
# All 122 Feynman laws (full run)
python run_real.py

# First N laws
python run_real.py --subset 5

# Specific noise type
python run_real.py --noise gaussian    # 10% Gaussian noise
python run_real.py --noise outliers    # outliers
python run_real.py --noise hetero      # heteroscedasticity
python run_real.py --noise missing     # missing values

# Without --noise: all 4 types cycle round-robin (one type per law)
python run_real.py --subset 5

# Number of data points (system adapts parameters automatically)
python run_real.py --subset 5 --n_samples 200

# Fix randomness for reproducibility
python run_real.py --subset 5 --seed 42
```

---

## Step 5 — What you'll see at the end

For each law the system prints a result:

```
──────────────────────────────────────────────────────────────
  I.12.1: [HIDDEN] | 4 features | noise: gaussian
──────────────────────────────────────────────────────────────
  Data: 150 train / 50 test (both noisy)
  ...
  ✅ [ACCEPTED] I.12.1
     Found:    (f1 / f4) / (f1 + (f4 * (f3 + (3.6387281 - f2))))
     Truth:    [HIDDEN → real_ground_truth_PRIVATE.json]
     R²=0.9874  Consensus=ACCEPTED  t=1823.4s
     Noise: gaussian
```

At the end of the full run — a summary:

```
══════════════════════════════════════════════════════════════
  REAL SCIENCE MODE — RESULTS
══════════════════════════════════════════════════════════════
  ✅ Accepted:  3/5
  🟡 Candidate: 0/5
  ❌ Rejected:  2/5
──────────────────────────────────────────────────────────────
  Noise: 10% (gaussian)
  Results → scalpel_vault/real_results
  Truth   → real_ground_truth_PRIVATE.json
══════════════════════════════════════════════════════════════
```

> `real_ground_truth_PRIVATE.json` — answer key for the 5 Feynman laws from this run. **Does not affect formula search** — no part of the engine reads it during execution. It works like a teacher's answer sheet: the system writes the correct answer into it after each law is processed, once the search is already done.

Result files:

```
scalpel_vault/
├── real_results/
│   └── real_results_2026-04-08.json  ← current run results
├── FINAL_REPORT_LLM_v10.txt          ← full report
├── gold_formulas.json                 ← all accepted formulas
├── reasoning_log.jsonl                ← full reasoning log for each role
└── CONSENSUS_REPORT.txt              ← last Matryoshka consensus
```

---

## Common issues

**Ollama not responding**
```bash
ollama serve   # run in a separate terminal
```

**Julia not installed / PySR crashes**

PySR installs Julia automatically on first run. Internet required. If offline:
```bash
set JULIAPKG_OFFLINE=yes      # Windows
export JULIAPKG_OFFLINE=yes   # Linux / macOS
```

**Low RAM — system crashes on phi4:14b**

phi4 is a 14B parameter model. If RAM < 16 GB:
```bash
python -m scalpel --model mistral:7b
```

**Run interrupted midway**

Just run the same command again — the system finds the intermediate results file and continues from where it stopped.

---

## Project structure

```
scalpel/
├── scalpel/                  ← main package
│   ├── cli.py                ← entry point (python -m scalpel)
│   ├── oracle.py             ← strategic advisor
│   ├── navigator.py          ← hypothesis generator (Qwen)
│   ├── dspy_optimizer.py     ← prompt compilation (DSPy)
│   ├── critical_thinking.py  ← Matryoshka: debates and audit
│   ├── atomic_precision.py   ← Heritage matching (Coulomb, Kepler...)
│   └── ...
├── scalpel_vault/            ← system memory (created on first run)
├── run_real.py               ← main runner
├── run_feynman.py            ← extended runner
├── requirements.txt
└── pyproject.toml
```

---

## Minimal test (install only, no models needed)

```bash
python -c "import scalpel; print('OK, version', scalpel.__version__)"
```

If you get an error — make sure you're running from the repository folder and have run `pip install -e ".[full]"`.

---

*Scalpel v10 · Adel Gilazetdinov · Moscow, 2026 · MIT License*
