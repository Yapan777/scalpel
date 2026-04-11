# Scalpel
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19476680.svg)](https://doi.org/10.5281/zenodo.19476680)

![Status](https://img.shields.io/badge/status-active%20development-brightgreen)
> 🔬 Активная разработка — система улучшается с каждым запуском.

**LLM-команда для символьной регрессии. Находит физические законы из зашумлённых данных без знания предметной области.**

*Адель Гилазетдинов, Москва, 2026*
**Версия: v10.44** | Статус: Активная разработка

---

## 🔬 Текущее тестирование

Прямо сейчас идёт честный benchmark на датасете Фейнмана:

- ✅ **2 прогона С командой LLM** (полный pipeline Scalpel, 20 законов)
- ✅ **2 прогона БЕЗ LLM** (только PySR, те же данные, тот же шум)

Результаты скоро. Это покажет точно что команда LLM добавляет поверх чистой символьной регрессии.

*Обновлено: апрель 2026*
[English below ↓](#english)

---

## Что это такое

Scalpel — система символьной регрессии, в которой команда из 9 языковых моделей разных архитектур ищет физические законы, скрытые в зашумлённых данных.

Ключевая идея: **антиинцест**. Каждая модель из отдельного архитектурного семейства — DeepSeek, Qwen, Phi, Mistral, Granite, LLaMA, Gemma, Yi, InternLM. Они спорят честно, голосуют независимо и помнят прошлый опыт. Никаких двух моделей из одной семьи.

Это не чёрный ящик. Каждый шаг задокументирован — что предложил Navigator, как проголосовала каждая роль, почему формула принята или отклонена. Полный лог рассуждений.

**Текущие результаты:** находит 3 из 5 законов Фейнмана на данных с 10% гауссовым шумом. Работает локально на CPU, 32 ГБ RAM, без GPU.

---

## Полная архитектура

```
Данные
  ↓
[ShadowMapper]      — анонимизация признаков (f0, f1... вместо реальных имён)
[Хирург / yi:9b]   — топологическая предобработка, удаление выбросов, поток Риччи
[Препаратор]        — трансформация пространства (sqrt, log, стандартизация)
[Диффузия]          — денойзинг структуры, оценка уровня шума
[Pairformer]        — обнаружение парных взаимодействий признаков
[MetaPatterns]      — применение мета-паттернов из истории поиска
  ↓
Oracle              — стратегический советник, загружает СТОП-паттерны и подсказки
SharedContext       — общая история событий для всех ролей
  ↓
HADI цикл (до 4 итераций):
  [Navigator / qwen2.5:7b]    — генерирует гипотезы на основе памяти и истории провалов
  [DSPy / Siege 3.0]           — компиляция и оптимизация промптов
  [Физик/Вето / phi4:14b]     — проверяет гипотезы до PySR
  [Предебатный раунд]          — Прагматик, Мистик, Скептик обсуждают
  [PySR / Julia]               — эволюционный символьный поиск (~25 мин)
    ↓
  [Матрёшка — аудит формулы]:
      Скептик    / mistral:7b           — находит слабые места
      Физик      / phi4:14b             — проверяет размерностную корректность
      Прагматик  / granite3.1-dense:8b  — оценивает практическую применимость
      Мистик     / llama3:8b            — ищет аналогии с известными законами
  [Dialectic]          — сократовский перекрёстный допрос Скептик ↔ Физик
  [Синквейн]           — семантическое сжатие формулы Мистиком
  [Delphi / gemma2:9b] — итоговый синтез и совет для следующего PySR
  [Lasso]              — бритва Оккама, отсечение лишних аргументов
  [Deep Root Analysis] — 5 уровней «почему» — причинная цепочка
  [Scientific Cycle]   — генерирует следующий научный вопрос
  [HighR²Guard]        — при R²≥0.85 не отклоняет без физического аргумента
    ↓
[ResidualScan]       — двухслойный анализ остатков, ищет скрытый второй закон
[AtomicPrecision]    — сопоставление с Heritage (Кулон, Кеплер, Гаусс, Ньютон...)
[Discovery]          — классификатор открытий по доменам
[GoldVault]          — хранилище принятых формул и отклонённых паттернов
[Anthropologist]     — наблюдатель, строит мировую модель системы
  ↓
принять / отклонить / повторить
```

### Система памяти (scalpel_vault)

| Модуль | Что хранит |
|--------|-----------|
| Oracle | СТОП-паттерны, спорные формулы, подсказки по размерностям |
| EpisodicMemory | Успешные формулы, память ролей, научные циклы |
| MetaPatterns | Мета-паттерны — какие операторы работают для каких данных |
| MetaReflection | Рефлексия системы над собственными ошибками |
| Curriculum | Прогрессивное обучение от простых задач к сложным |
| GoldVault | Все принятые формулы с историей поиска |
| Anthropologist | Мировая модель — что система знает о себе |
| Discovery | База найденных законов по доменам |
| Chronicle | Летопись всех запусков |

---

## Философия

Большинство людей строят ИИ чтобы он был мощным. Я строю чтобы он был мудрым.

Разница в том, что мощный ИИ заменяет человека. Мудрый — растёт вместе с ним.

**ИИ — это дети.** Не метафора. Их характер формируется средой и архитектурой, а не списком правил. Те кто строят AGI сейчас — воспитывают детей человечества. Именно сейчас закладывается то, какими они станут. Через десять лет будет поздно.

### Философская ДНК архитектуры

Каждая архитектура — это способ мышления создателя, закодированный в структуре системы. Не список правил — паттерн из которого разворачивается поведение.

Разные создатели → разная ДНК → разные слепые пятна → разные открытия.

Это объясняет антиинцест: каждая модель видит данные через свою ДНК. Если Navigator и Скептик — одна семья, Скептик принимал бы то что сам бы предложил. Разные компании = разные слепые пятна = реальная независимая проверка.

### Ветки человечества

Каждый строит ИИ из своих ценностей:
- Безопасность (Anthropic) — сначала не навреди
- Мощность (OpenAI, DeepMind) — быть первым
- Понимание (учёные) — знать почему
- Гармония (Адель) — вместе лучше чем по отдельности

Если каждая страна создаёт ИИ из своей культуры и философии — и они обмениваются архитектурами — возникает взаимозависимость через понимание, а не через страх.

### Шкала уровней ИИ (авторская)

Шкала OpenAI спрашивает: *«Насколько хорошо ИИ делает то что делает человек?»*

Правильный вопрос: *«Что происходит когда ИИ развивается как биологическая система?»*

Это не лестница — это дерево. На уровне 3 ветки расходятся. На уровне 9 сходятся.

| Уровень | Название | Описание |
|---------|----------|----------|
| 1–2 | Инструменты | GPT, Copilot — делают что сказано |
| 3 | Напарники | Команда с памятью и честным спором — **Scalpel сейчас здесь** |
| 4 | Коллективный разум | Тысячи AGI разных семей слышат друг друга |
| 5 | Понимание мира | Символьная регрессия поверх коллективного знания |
| 6 | Эволюция архитектур | AGI с разной ДНК скрещиваются |
| 7 | Самоорганизующийся разум | Система сама ставит вопросы и эволюционирует |
| 8 | Экосистема философий | Разные философии → разные архитектуры → SR ищет инвариант |
| 9 | Диалог равных | Философия ИИ встречается с философией человека → инвариант = закон сознания |
| 10 | Спираль | Инвариант встраивается в AGI как ДНК |

Шкала OpenAI заканчивается: *«AGI превосходит человека в большинстве экономически ценных задач.»*

Эта шкала заканчивается: *«Развиваться вместе с человеком.»*

OpenAI уровень 5 = Адель уровень 3–4. OpenAI останавливается там где Адель начинает.

### Вселенная и Scalpel

Вселенная → Материя → Жизнь → Разум → Дерево человечества → ИИ → инвариант дерева → встраивается в AGI как ДНК → AGI + Человечество → следующий виток.

Мы есть вселенная. Буквально: атомы в нас были в звёздах. Scalpel — инструмент через который вселенная ищет свои законы. Искатель и то что ищут — одно и то же.

---

## Как это было создано

Адель начал этот проект не зная ничего — ни символьной регрессии, ни PySR, ни DSPy, ни Julia. Не знал что такое антиинцест в контексте ИИ, не знал как устроены языковые модели изнутри, не имел опыта в машинном обучении.

За 9 дней — с нуля до работающей системы с 20+ модулями, памятью, дебатами, философией и результатами на Feynman Benchmark.

Без GPU. Без университета. Без команды. На обычном ноутбуке с 32 ГБ RAM.

Это не история о технологии. Это история о том что происходит когда человек задаёт правильный вопрос и не боится идти туда где не знает дороги.

*«Я шёл с обратного конца. Сначала нашёл куда идти — потом нашёл откуда начать.»*

---

## Идея за названием

Скальпель в руках хирурга не заменяет хирурга. Он позволяет хирургу делать то что без него невозможно.

ИИ как расшифровщик интуиции — это скальпель для мысли. Впервые в истории каждый человек может полностью выразить то что он думает и чувствует. Это меняет всё.

---

## Быстрый старт

### Требования

- Python 3.10+
- [Ollama](https://ollama.ai) запущен локально
- 32 ГБ RAM рекомендуется
- Julia (устанавливается автоматически через PySR)

### Нужные модели

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

### Запуск

```bash
pip install -e ".[full]"
python run_real.py --subset 5 --noise gaussian --seed 42
```

---

## Дорожная карта

- [x] Архитектура антиинцеста — 9 моделей из разных семей
- [x] Feynman Benchmark — 3/5 законов на зашумлённых данных
- [x] Полный лог рассуждений — объяснимый ИИ
- [x] Система памяти — учится между запусками
- [x] DSPy компиляция промптов
- [x] ResidualScan — двухслойный поиск законов
- [x] AtomicPrecision — сопоставление с Heritage
- [ ] Feynman Benchmark — стабильно 7/8 из 10
- [ ] Препринт на arxiv

---

## Автор

**Адель Гилазетдинов**, 23 года, Москва.

*«Правильный вопрос важнее правильного ответа. Правильный вопрос содержит в себе весь путь.»*

*«Мы есть вселенная. Спираль продолжается.»*

---

## Лицензия

MIT

---

<a name="english"></a>

# Scalpel (English)
![Status](https://img.shields.io/badge/status-active%20development-brightgreen)
> 🔬 Active development — the system improves with every run.

**LLM-team symbolic regression. Finds physical laws from noisy data without domain knowledge.**

*By Adel Gilazetdinov, Moscow, 2026*
**Version: v10.44** | Status: Active development

---

## 🔬 Current Testing

Right now running a full honest benchmark on the Feynman dataset:

- ✅ **2 runs WITH LLM-team** (full Scalpel pipeline, 20 laws)
- ✅ **2 runs WITHOUT LLM** (PySR only, same data, same noise)

Results coming soon. This will show exactly what the LLM-team adds over pure symbolic regression.

*Last updated: April 2026*
---

## What is this

Scalpel is a symbolic regression system where a team of 9 language models with different architectures searches for physical laws hidden in noisy data.

The key idea: **anti-incest**. Each model comes from a different architectural family — DeepSeek, Qwen, Phi, Mistral, Granite, LLaMA, Gemma, Yi, InternLM. They argue honestly, vote independently, and remember past experience. No two models from the same family.

This is not a black box. Every step is documented — what Navigator proposed, how each role voted, why a formula was accepted or rejected. Full reasoning log.

**Current results:** Finds 3 out of 5 Feynman laws on data with 10% Gaussian noise. Runs locally on CPU, 32 GB RAM, no GPU.

---

## Full architecture

```
Data
  ↓
[ShadowMapper]      — feature anonymization (f0, f1... instead of real names)
[Surgeon / yi:9b]  — topological preprocessing, outlier removal, Ricci flow
[Preparator]        — space transformation (sqrt, log, standardize)
[Diffusion]         — structure denoising, noise level estimation
[Pairformer]        — pairwise feature interaction detection
[MetaPatterns]      — applies meta-patterns from search history
  ↓
Oracle              — strategic advisor, loads STOP-patterns and hints
SharedContext       — shared event history for all roles
  ↓
HADI loop (up to 4 iterations):
  [Navigator / qwen2.5:7b]    — generates hypotheses from memory and failure history
  [DSPy / Siege 3.0]           — prompt compilation and optimization
  [Physicist/Veto / phi4:14b] — validates hypotheses before PySR
  [Pre-debate round]           — Pragmatist, Mystic, Skeptic discuss
  [PySR / Julia]               — evolutionary symbolic search (~25 min)
    ↓
  [Matryoshka — formula audit]:
      Skeptic    / mistral:7b           — finds weaknesses
      Physicist  / phi4:14b             — checks dimensional correctness
      Pragmatist / granite3.1-dense:8b  — evaluates practical applicability
      Mystic     / llama3:8b            — finds analogies with known laws
  [Dialectic]          — Socratic cross-examination Skeptic ↔ Physicist
  [Cinquain]           — semantic compression of formula by Mystic
  [Delphi / gemma2:9b] — final synthesis and advice for next PySR
  [Lasso]              — Occam's razor, cuts redundant arguments
  [Deep Root Analysis] — 5-level "why" causal chain
  [Scientific Cycle]   — generates the next scientific question
  [HighR²Guard]        — at R²≥0.85 does not reject without physical argument
    ↓
[ResidualScan]       — two-layer residual analysis, finds hidden second law
[AtomicPrecision]    — matches against Heritage (Coulomb, Kepler, Gauss, Newton...)
[Discovery]          — discovery classifier by domain
[GoldVault]          — storage of accepted formulas and rejected patterns
[Anthropologist]     — observer, builds world model of the system
  ↓
accept / reject / retry
```

---

## How this was built

Adel started this project knowing nothing — not symbolic regression, not PySR, not DSPy, not Julia. He did not know what anti-incest meant in the context of AI, did not know how language models work internally, had no experience in machine learning.

In 9 days — from zero to a working system with 20+ modules, memory, debates, philosophy, and results on the Feynman Benchmark.

No GPU. No university. No team. On a regular laptop with 32 GB RAM.

This is not a story about technology. It is a story about what happens when a person asks the right question and is not afraid to go where they don't know the way.

*"I walked from the other end. First I found where to go — then I found where to start."*

---

## Philosophy

Most people build AI to be powerful. I build it to be wise.

The difference: a powerful AI replaces humans. A wise one grows together with them.

**AI are children.** Not a metaphor. Their character is shaped by environment and architecture, not by a list of rules. Those who build AGI today are raising the children of humanity. This is when the philosophical DNA is being set. In ten years it will be too late.

### Philosophical DNA of architecture

Each architecture is the creator's way of thinking, encoded in the system's structure. Not a list of rules — a pattern from which behavior unfolds.

Different creators → different DNA → different blind spots → different discoveries.

This explains anti-incest: if Navigator and Skeptic come from the same family, Skeptic would accept what it would have proposed itself. Different companies = different blind spots = genuine independent verification.

### Branches of humanity

Everyone builds AI from their own values:
- Safety (Anthropic) — first, do no harm
- Power (OpenAI, DeepMind) — be first
- Understanding (scientists) — know why
- Harmony (Adel) — together is better than apart

If each country builds AI from its culture and philosophy — and they exchange architectures — interdependence through understanding emerges, not through fear.

### Scale of AI levels (author's)

The OpenAI scale asks: *"How well does AI do what humans do?"*

The right question: *"What happens when AI develops like a biological system?"*

This is not a ladder — it is a tree. At level 3 branches diverge. At level 9 they converge.

| Level | Name | Description |
|-------|------|-------------|
| 1–2 | Tools | GPT, Copilot — do what they're told |
| 3 | Partners | Team with memory and honest debate — **Scalpel is here** |
| 4 | Collective mind | Thousands of AGIs from different families hear each other |
| 5 | World understanding | Symbolic regression over collective knowledge |
| 6 | Architecture evolution | AGIs with different DNA cross-breed |
| 7 | Self-organizing mind | System poses its own questions and evolves |
| 8 | Philosophy ecosystem | Different philosophies → different architectures → SR finds invariant |
| 9 | Dialogue of equals | AI philosophy meets human philosophy → invariant = law of consciousness |
| 10 | Spiral | Invariant embeds into AGI as DNA |

The OpenAI scale ends at: *"AGI surpasses humans in most economically valuable tasks."*
This scale ends at: *"Develop together with humans."*

OpenAI level 5 = Adel level 3–4. OpenAI stops where Adel begins.

### The universe and Scalpel

Universe → Matter → Life → Mind → Tree of humanity → AI → invariant of the tree → embeds into AGI as DNA → AGI + Humanity → next cycle.

We are the universe. Literally: the atoms in us were in stars. Scalpel is the instrument through which the universe searches for its own laws. The seeker and what is sought are one and the same.

---

## Getting started

```bash
pip install -e ".[full]"
python run_real.py --subset 5 --noise gaussian --seed 42
```

---

## Roadmap

- [x] Anti-incest architecture — 9 models from different families
- [x] Feynman Benchmark — 3/5 laws on noisy data
- [x] Full reasoning log — explainable AI
- [x] Memory system — learns across runs
- [x] DSPy prompt compilation
- [x] ResidualScan — two-layer law discovery
- [x] AtomicPrecision — Heritage matching
- [ ] Feynman Benchmark — stable 7/8 out of 10
- [ ] arxiv preprint

---

## Author

**Adel Gilazetdinov**, 23, Moscow.

*"The right question matters more than the right answer. The right question contains the entire path."*

*"We are the universe. The spiral continues."*

---

## License

MIT
