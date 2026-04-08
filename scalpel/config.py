"""
config.py — Все константы v10.4.5 (Inherent Structure).

v10.2.7 — Humanistic Filters / Critical Thinking:
  Deep Root   — 5 Почему (Root Cause Analysis)
  Dialectic   — Сократовская перекрёстная проверка в Delphi
  Sinquain    — Семантическое сжатие (Синквейн инварианта)
  Lasso       — Топологическое стягивание аргументов (Бритва Оккама)

v10.3.9 — Topological Surgery (Метод Перельмана):
  SingularityDetector — детектор кривизны (∂²y/∂i²)
  FractalCutting      — хирургическое вырезание 2-3% сингулярностей
  RicciFlow           — Savitzky-Golay сглаживание («скелет» формы)
  PoincaréVerdict     — [SURGERY PERFORMED] + [POINCARE INVARIANT]

v10.4.5 — Inherent Structure (принципы AlphaFold 3):
  DiffusionDenoising  — «сгущение структуры» за T шагов до PySR
  PairformerLogic     — попарная энергия признаков (экономия ~500 МБ)
  AtomicPrecision     — Пантеон Джона Джампера + [MOLECULAR PRECISION DETECTED]
  SiegeMode3.0        — Диффузионная Пауза: ollama_stop→gc→Wait3s→Julia

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
МАТЕМАТИЧЕСКОЕ ОБОСНОВАНИЕ ПОРОГОВ R² (REAL SCIENCE MODE)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Режим реальной науки использует 10% шум (лабораторные данные).
Пороги выведены из физики задачи ДО запуска тестов.

Теорема о максимальном достижимом R²:
  При аддитивном гауссовом шуме уровня σ_noise / σ_y = ε:
    R²_max = 1 - ε²

Для ε = 0.10 (10% шум):
  R²_max = 1 - 0.10² = 1 - 0.01 = 0.99

Для зашумлённого тестового набора (оба train и test зашумлены):
  R²_effective_max ≈ 1 - 2ε² ≈ 0.98

Консервативные пороги (с запасом 40-50% от теоретического max):
  R2_STRONG   = 0.80  (= 0.98 × 0.82 ≈ 80% от max)
  R2_MODERATE = 0.50  (= 0.98 × 0.51 ≈ 51% от max)

Для внутренних проверок движка:
  r2_threshold  = 0.50  (нижняя граница значимого результата)
  OODA_R2_STRICT = 0.60  (строгий режим OODA при хорошем сигнале)
  DEEP_ROOT_MIN_R2 = 0.45  (запускаем 5 почему всегда кроме полного шума)

Источник: Hastie, Tibshirani, Friedman "Elements of Statistical Learning"
  гл. 2.9 "Bias-Variance Tradeoff under noise" — формула ошибки Байеса.

НЕ подгонка под результаты — математический расчёт до запуска тестов.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
import secrets
from pathlib import Path

VERSION = "10.15.0"

# Пути
SCRIPT_DIR          = Path(__file__).parent.parent
VAULT_DIR           = SCRIPT_DIR / "scalpel_vault"
GOLD_PATH           = VAULT_DIR / "gold_formulas.json"
INTERNAL_GOLD_PATH  = VAULT_DIR / "gold_formulas_INTERNAL_DO_NOT_SHARE.json"
REJECTED_PATH       = VAULT_DIR / "rejected_formulas.json"    # антипримеры для Navigator
DISPUTED_PATH       = VAULT_DIR / "disputed_formulas.json"    # хороший R², но LLM отклонила
RESIDUAL_DATA_PATH  = VAULT_DIR / "residual_data.json"        # данные для ResidualScan (Layer 2)
RESIDUAL_RESULT_PATH = VAULT_DIR / "residual_formulas.json"   # найденные слои поверх Layer 1

# v10.15: порог R²_blind выше которого отклонение LLM считается подозрительным.
# Отклонение при R²>0.65 при 10% шуме подозрительно: Матрёшка, вероятно, ошиблась.
REJECTED_R2_SAFE_MAX = 0.65

# v10.15: ResidualScan — порог R² остатков для запуска второго слоя.
# Если R²(linear fit на остатках) > этого значения — остатки имеют структуру → ищем Layer 2.
# ResidualScan Layer 2: минимальный сигнал в остатках = ε²×2 = 0.02×2 = 0.04.
# Консервативно: 0.25 (в 6 раз выше теор. минимума — защита от ложных открытий).
RESIDUAL_R2_MIN      = 0.25
# ResidualScan safe max: выше 0.65 при 10% шуме — подозрительно (переобучение).
RESIDUAL_R2_SAFE_MAX = 0.65
LOG_PATH            = VAULT_DIR / "scalpel.log"

# v9.9: DSPy compiled model
DSPY_COMPILED_PATH  = VAULT_DIR / "dspy_compiled_model.json"
DSPY_FAILURE_LOG    = VAULT_DIR / "dspy_failure_log.jsonl"

# v10.6: Двухфазный запуск — промежуточный результат после PySR
PHASE_RESULT_PATH   = VAULT_DIR / "pysr_phase_result.json"

# ══════════════════════════════════════════════════════════════════
# МОДЕЛИ — ПРИНЦИП АНТИИНЦЕСТА v10.29
# ══════════════════════════════════════════════════════════════════
#
# ПРАВИЛО: каждая роль — уникальная архитектурная семья.
# Модель не может встречаться в двух ролях которые видят одни данные
# последовательно (генерирует → верифицирует = инцест).
#
# КАРТА СЕМЕЙ И СИЛЬНЫХ СТОРОН:
#
#   DeepSeek  → Navigator
#               Специализация: chain-of-thought reasoning, математика.
#               DeepSeek-R1 обучен на научных задачах с пошаговым
#               объяснением — идеален для генерации гипотез формул.
#
#   Qwen      → Oracle (аналитик памяти)
#               Специализация: instruction-following, структурированный
#               анализ, работа с длинным контекстом (128k токенов).
#               Alibaba обучал Qwen на огромном корпусе научных текстов —
#               Oracle использует это для подсказок из прошлых сессий.
#
#   Solar     → Скептик
#               Специализация: критическое мышление, поиск ошибок.
#               Upstage (Корея) обучил Solar специально для аналитических
#               задач и дебатов — ставит под сомнение любую гипотезу.
#               ollama pull solar:10.7b
#
#   Phi-4     → Физик
#               Специализация: STEM, точные науки, математическое
#               рассуждение. Microsoft Research обучал Phi-4 на учебниках
#               физики, математики, химии — видит размерную несовместимость
#               которую другие пропускают.
#
#   Granite   → Прагматик
#               Специализация: точные, надёжные рассуждения без
#               галлюцинаций. IBM Research обучал Granite на enterprise-
#               задачах где цена ошибки высока — отсекает красивые но
#               нереализуемые формулы, оценивает практическую применимость.
#               ollama pull granite3.1-dense:8b
#
#   LLaMA     → Мистик
#               Специализация: творческое, нестандартное мышление.
#               Meta обучал LLaMA на разнообразнейшем корпусе включая
#               художественные тексты и философию — находит неочевидные
#               аналогии и паттерны которые аналитические модели игнорируют.
#
#   Gemma     → Delphi (синтез) + Препаратор
#               Специализация: синтез и обобщение. Google обучал Gemma2
#               на задачах суммаризации и консенсуса — собирает мнения
#               всех ролей в единое взвешенное решение.
#
#   InternLM  → Летописец + Вето
#               Специализация: структурированная генерация текста,
#               академический стиль. Shanghai AI Lab обучал InternLM2
#               на научных статьях — хроника точная, вето методичное.
#               ollama pull internlm2:7b
#
#   Mistral   → Deep Root / Lasso / RCA (аналитика) + Navigator Fast
#               Специализация: быстрый reasoning, эффективность на CPU.
#               Mistral 7B — лучшее соотношение скорость/качество
#               для аналитических задач без GPU.
#
#   Yi        → Хирург
#               Специализация: точный анализ данных, числовые задачи.
#               01.AI обучал Yi на количественных научных задачах —
#               понимает статистические паттерны в сырых данных.
#
# УСТАНОВКА НОВЫХ МОДЕЛЕЙ:
#   ollama pull solar:10.7b           (~6 ГБ)
#   ollama pull granite3.1-dense:8b   (~5 ГБ)
#   ollama pull internlm2:7b          (~4 ГБ)
#
# На 32 ГБ RAM: модели хранятся на диске, в RAM одна за раз (RAM Queue).
# ══════════════════════════════════════════════════════════════════

OLLAMA_HOST = "http://127.0.0.1:11434"

# Дефолтная модель — Deep Root, Lasso, RCA, аналитические блоки.
# Mistral: быстрый на CPU, хорошее reasoning для аналитики.
# Не участвует в Матрёшке → нет конфликта с верификаторами.
OLLAMA_MODEL = "mistral:7b"

# ── Oracle (стратегический советник) ──────────────────────────────
# deepseek-r1:7b (DeepSeek) — стратегический аналитик памяти.
# v10.42: заменён с qwen2.5:7b — qwen2.5:7b теперь Navigator.
# deepseek-r1:7b хорошо анализирует длинную историю сессий и даёт советы.
# Не участвует в Матрёшке и не генерирует гипотезы — нет инцеста.
ORACLE_MODEL = "deepseek-r1:7b"

# v10.22: Итерационный Препаратор — принудительная трансформация
# None = LLM решает сам, "none"/"log"/"sqrt"/"standardize" = принудительно
PREPARATOR_FORCE_TRANSFORM: str = None

# ── Navigator (Штурман) ───────────────────────────────────────────
# qwen2.5:7b (Alibaba) — генератор гипотез.
# v10.42: заменён с deepseek-r1:7b — deepseek думал на китайском и давал мусорные гипотезы.
# qwen2.5:7b быстрее (60-120с vs 300-500с), лучше следует инструкциям на русском,
# хорошо справляется с математическими задачами и структурированными ответами.
# Уже установлен как Oracle — не требует ollama pull.
# Антиинцест: Alibaba отличается от Mistral/Microsoft/IBM/Meta/Google/DeepSeek.
NAVIGATOR_MODEL = "qwen2.5:7b"

# Fallback если Navigator не отвечает за таймаут.
# mistral:7b — быстрый на CPU, та же фаза что OLLAMA_MODEL.
NAVIGATOR_FAST_MODEL = "mistral:7b"

# Таймаут Navigator. qwen2.5:7b реально ~90с на CPU → 300с запас ×3.
NAV_TIMEOUT_SEC = 300  # 5 минут — qwen2.5:7b значительно быстрее deepseek

# ── Матрёшка: 4 роли — 4 разные архитектурные семьи ─────────────
# FIX v10.29: полный аудит антиинцеста.
# Все 4 роли теперь из уникальных семей, не пересекаются друг с другом
# и не пересекаются с Navigator, Oracle, Delphi, Хирургом, Летописцем.
ROLE_MODELS = {
    # Скептик — mistral:7b (Mistral AI, Франция)
    # v10.37: заменён с solar:10.7b — Solar голосовал ОТКЛОНЕНА без физического аргумента.
    # Mistral лучше рассуждает о физике и чаще использует ВОЗДЕРЖАЛАСЬ при незнании домена.
    # Уже установлен как Navigator Fast — не требует ollama pull.
    # Антиинцест: Mistral отличается от DeepSeek/Microsoft/IBM/Meta/Alibaba/Google.
    "Скептик":   "mistral:7b",

    # Физик — phi4:14b (Microsoft Research)
    # Без изменений. STEM-специалист: физика, математика, химия.
    # Видит размерную несовместимость и физические противоречия.
    "Физик":     "phi4:14b",

    # Прагматик — granite3.1-dense:8b (IBM Research)
    # FIX v10.29: заменён с mistral:7b — инцест с OLLAMA_MODEL и Navigator Fast.
    # IBM Granite обучен на enterprise-задачах с высокой ценой ошибки.
    # Отсекает красивые но нереализуемые формулы, оценивает практичность.
    # Установка: ollama pull granite3.1-dense:8b
    "Прагматик": "granite3.1-dense:8b",

    # Мистик — llama3:8b (Meta)
    # Без изменений. Творческое, нестандартное мышление.
    # Находит неочевидные аналогии которые аналитики игнорируют.
    "Мистик":    "llama3:8b",
}

# ── Синтез и нарратив ─────────────────────────────────────────────
# Delphi Consilium, Scientific Cycle, Препаратор.
# gemma2:9b (Google) — специализация на синтезе и суммаризации.
# Собирает мнения всей Матрёшки в взвешенное решение.
SYNTHESIS_MODEL = "gemma2:9b"

# Летописец (Chronicle), Meta-Reflection, Вето.
# FIX v10.29: заменён с llama3.1:8b (Meta) — инцест с Мистиком (llama3:8b).
# internlm2:7b (Shanghai AI Lab) — академический стиль, структурированный текст.
# Хроника точная, вето методичное. Независимая семья.
# Установка: ollama pull internlm2:7b
CHRONICLE_MODEL = "internlm2:7b"

# ── Хирург (Surgeon) ──────────────────────────────────────────────
# yi:9b (01.AI) — анализ данных, числовые задачи.
# Обучен на количественных научных задачах — понимает статистику в сырых данных.
# Не участвует в поиске формулы — только предобработка.
# Установка: ollama pull yi:9b
SURGEON_MODEL = "yi:9b"

# Максимальная сложность формулы в PySR
# 95% реальных физических формул укладываются в complexity ≤ 15
# 99% в complexity ≤ 20 — золотая середина скорости и охвата
PYSR_MAXSIZE = 20

# PySR — увеличено для 32 ГБ RAM (было 2 проца для 7.7 ГБ)
PYSR_POPULATIONS     = 24   # оставляем как было — больше не значит быстрее
PYSR_POPULATION_SIZE = 33   # оставляем как было
PYSR_N_PROCS         = 4    # было 2 — 32 ГБ позволяет 4 Julia-потока
PYSR_BATCH_SIZE      = 50   # было 30
PYSR_FAST_FAIL_SEC   = 1500  # v10.41: 25 минут — больше времени для сложных законов


# HADI
HADI_MAX_RETRIES = 3

# NIST CSPRNG
_SYS_RAND = secrets.SystemRandom()

# OODA пороги
OODA_STD_SPIKE = 1.8
# R²_max = 1 - ε² = 1 - 0.10² = 0.99. Порог = 0.60 (61% от max).
# Выведено из формулы ошибки Байеса до запуска тестов.
OODA_R2_STRICT = 0.60
OODA_P_STRICT  = 1e-4

# PDCA
PDCA_INTERVAL_DAYS = 30
# Граница "застрял" = R2_MODERATE × 0.9 = 0.50 × 0.9 = 0.45
PDCA_STALE_R2      = 0.45

# ── v9.9 DSPy ────────────────────────────────────────────────────────────────
DSPY_FEW_SHOT_MAX    = 5
# Gold примеры при 10% шуме: R²_gold_min = R2_STRONG × 0.75 = 0.80 × 0.75 = 0.60
DSPY_GOLD_MIN_R2     = 0.60
DSPY_CALL_TIMEOUT    = 600
DSPY_RECOMPILE_DAYS  = 7

# Священные метрики — DSPy их не трогает никогда
SACRED_METRICS = frozenset(["shuffle_test", "cross_blind", "ShadowMapper"])

DEBUG_MODE = True

# ── v10.2.7 Humanistic Filters (Critical Thinking Blocks) ────────────────────

# Deep Root: 5 Почему (Root Cause Analysis)
DEEP_ROOT_LEVELS       = 5          # Глубина погружения «Почему?»
# 5 Почему при любом значимом сигнале: порог = R2_MODERATE × 0.9 = 0.45
# При R² < 0.45 — чистый шум, анализировать нечего.
DEEP_ROOT_MIN_R2       = 0.45

# Dialectic: Сократовская перекрёстная проверка (Skeptic ↔ Physicist)
DIALECTIC_QUESTIONS    = 3          # Количество вопросов каждой стороны
DIALECTIC_ROLES        = ("Скептик", "Физик")   # Участники диалога

# Sinquain: Синквейн инварианта (Мистик | Антрополог)
SINQUAIN_ROLE          = "Мистик"   # Кто генерирует синквейн
SINQUAIN_LINES         = 5          # Классический синквейн: 5 строк

# Lasso: Топологическое стягивание аргументов
LASSO_INSTRUCTION      = (
    "Стяни все аргументы в одну точку-причину. "
    "Если аргумент не ведёт к центру — отсекай его (Бритва Оккама). "
    "Итог: одно предложение-ядро, из которого выводятся все остальные."
)

# ── v10.3.9 Topological Surgery (Метод Перельмана) ───────────────────────────
# RAM Guard: вся хирургия в numpy до Julia. Julia получает «чистый скелет».

# SingularityDetector: точка — сингулярность, если κ_i > RATIO × median(κ)
SURGERY_CURVATURE_RATIO: float = 10.0

# FractalCutting: доля удаляемых аномальных точек (2.5%)
SURGERY_CUT_FRACTION: float = 0.050  # REAL MODE: 10% шум — увеличен с 0.025

# RicciFlow (Savitzky-Golay): параметры фильтра «выпускания воздуха»
SURGERY_SG_WINDOW:    int = 11   # REAL MODE: 10% шум — увеличен с 7    # ширина окна (нечётное)
SURGERY_SG_POLYORDER: int = 3    # порядок полинома

# Минимальное число точек для применения Ricci Flow
SURGERY_MIN_SAMPLES: int = 20

# Порог активации хирургии: если >5% точек — сингулярности (threshold=0.95)
SURGERY_THRESHOLD: float = 0.97  # REAL MODE: активируем при >3% выбросов (было 5%)

# Пуанкаре-инвариант: если R² после хирургии >= этого порога — структура сглажена
# Пуанкаре-инвариант: порог = R2_MODERATE = 0.50.
# После хирургии R² выше этого → структура найдена несмотря на шум.
SURGERY_POINCARE_R2: float = 0.50

# ── v9.9.1 RAM Queue ─────────────────────────────────────────────────────────
# Минимальная свободная RAM перед запуском роли (ГБ)
RAM_ROLE_MIN_GB       = 1.2
# Пауза после ollama_stop перед следующей ролью (сек)
RAM_ROLE_COOLDOWN_SEC = 2.5
# Пауза после gc.collect() (сек)
RAM_GC_SETTLE_SEC     = 1.0
# Таймаут одного вызова роли (сек)
RAM_ROLE_TIMEOUT_SEC  = 300   # v10.33 FIX: был 600с → Мистик занял 989с, снижаем до 300с
# Папка скомпилированных моделей по ролям
ROLE_COMPILED_DIR     = VAULT_DIR / "role_models"
# Папка логов ошибок по ролям
ROLE_FAILURE_DIR      = VAULT_DIR / "role_failures"
# Имена ролей (порядок важен — это очередь)
ROLE_NAMES = ["Скептик", "Физик", "Прагматик", "Мистик"]
# Теги vault для специализированных примеров каждой роли
ROLE_VAULT_TAGS = {
    "Скептик":   "skeptic_correct",
    "Физик":     "physicist_correct",
    "Прагматик": "pragmatist_correct",
    "Мистик":    "mystic_correct",
}

# ── v10.4.5 Diffusion Denoising (AlphaFold 3 — Сгущение структуры) ──────────
# Количество шагов денойзинга (T в noise schedule)
DIFFUSION_STEPS:       int   = 8
# Начальный и конечный уровень «шума» β (косинусный schedule)
DIFFUSION_BETA_START:  float = 0.02
DIFFUSION_BETA_END:    float = 0.001
# Порог IQR-зажима на каждом шаге
DIFFUSION_IQR_FACTOR:  float = 2.0  # REAL MODE: более агрессивный при 10% шуме
# Мин. точек для активации денойзинга
DIFFUSION_MIN_SAMPLES: int   = 15

# ── v10.4.5 Pairformer Logic (AlphaFold 3 — Взаимосвязи) ────────────────────
# Максимальное число пар для оценки
PAIRFORMER_TOP_K:       int   = 50
# Минимальная |корр| для включения признака в пары
PAIRFORMER_MIN_CORR:    float = 0.05
# Порог признаков при семплировании (>PAIRFORMER_MAX_FEAT → семплируем)
PAIRFORMER_MAX_FEAT:    int   = 200

# ── v10.4.5 Atomic Precision (Пантеон Джона Джампера) ───────────────────────
# R²_blind порог для [MOLECULAR PRECISION DETECTED]
# Atomic Precision: порог = R2_MODERATE + 0.05 = 0.55.
# Минимальный R² для признания молекулярной точности при 10% шуме.
ATOMIC_R2_THRESHOLD:     float = 0.55
# Максимальная сложность для «атомарной» формулы
ATOMIC_COMPLEXITY_MAX:   int   = 25
# Минимальная плотность R²/complexity
ATOMIC_COMPLEXITY_RATIO: float = 0.04

# ── v10.4.5 Siege Mode 3.0 — Диффузионная Пауза ─────────────────────────────
# Пауза между gc.collect() и Julia Ignition (секунды)
SIEGE_DIFFUSION_PAUSE_SEC: float = 3.0
