"""
dspy_signatures.py — DSPy Signatures v9.9.

Заменяют жёстко захардкоженные текстовые промпты
на математически плотные входы/выходы.

ПРИНЦИП: DSPy видит только ЛОГИКУ (что вычислить),
а не ТЕКСТ (как попросить). Оптимизатор сам найдёт
лучшие формулировки через BootstrapFewShot.

СВЯЩЕННЫЙ ЗАКОН: Эти Signatures НИКОГДА не включают
shuffle_test, cross_blind, ShadowMapper.
Это "тормоза сингулярности" — ИИ не может подделать метрики.
"""

try:
    import dspy
    DSPY_AVAILABLE = True
except ImportError:
    DSPY_AVAILABLE = False
    dspy = None  # type: ignore


# ═══════════════════════════════════════════════════════════════
# SIGNATURE 1: NAVIGATOR
# Штурман — выбирает признаки и строит гипотезы для PySR
# ═══════════════════════════════════════════════════════════════

if DSPY_AVAILABLE:
    class NavSignature(dspy.Signature):
        """
        Symbolic regression navigator.
        Given dataset metadata, select features and propose formulas.
        Apply OODA stability check, Double-Diamond feature selection,
        and ICE-rank hypothesis prioritization.

        LASSO PRINCIPLE (v10.2.7): Стяни все аргументы в одну точку-причину.
        Если аргумент не ведёт к центру — отсекай его (Бритва Оккама).
        Итог: одно предложение-ядро, из которого выводятся все остальные.
        """

        # ── Входы ────────────────────────────────────────────────
        data_meta: str = dspy.InputField(
            desc=(
                "Dataset summary: n_samples, list of shadow features "
                "with their dim_codes (e.g. 'f0:10, f1:5, f2:0'). "
                "dim_code meanings: 10=money, 5=volume, 0=ratio, "
                "8=time, 3=mass, 2=length, 4=temp, 6=force, 1=unknown."
            )
        )
        failure_logs: str = dspy.InputField(
            desc=(
                "JSON list of previously failed hypotheses with reasons. "
                "Empty string if first attempt. "
                "Format: [{hypothesis, death_reason, r2_achieved, source}]. "
                "May include META-REFLECTION rules: patterns of success/failure "
                "from historical data. May include [META-ПРАВИЛА НАВИГАЦИИ] "
                "and [ИЗВЕСТНЫЕ ОТКРЫТИЯ СИСТЕМЫ]. Use these to make smarter choices."
            )
        )

        # ── Выходы ───────────────────────────────────────────────
        selected_features: str = dspy.OutputField(
            desc=(
                "Comma-separated shadow feature names to use "
                "(e.g. 'f0,f2'). Must be 2-5 features. "
                "Choose dimensionally compatible features."
            )
        )
        selected_operators: str = dspy.OutputField(
            desc=(
                "Comma-separated math operators for PySR "
                "(e.g. '+,-,*,/,sqrt,log'). "
                "Use fewer operators if data is unstable (ooda_stable=false)."
            )
        )
        hypotheses: str = dspy.OutputField(
            desc=(
                "Top-5 formula hypotheses, semicolon-separated "
                "(e.g. 'f0/f2;sqrt(f0)*f2;f0*f2;f0+f2;f0*f0/f2'). "
                "Ranked by ICE score (Impact x Confidence x Ease). "
                "ONLY valid math expressions using shadow names (f0, f1, f2...) and operators. "
                "Do NOT include R² values, comments, arrows, or any text — only math formulas. "
                "Do NOT repeat hypotheses from failure_logs. "
                "Example of CORRECT format: 'f0*f1;sqrt(f0)/f2;f0+f1*f2;log(f0);f0**2'. "
                "Example of WRONG format: '0.076 → Delphi: +, - → R²=0.40' — this is NOT a hypothesis."
            )
        )
        ooda_stable: str = dspy.OutputField(
            desc=(
                "Stability verdict: 'true' if max(dim_codes)-min(dim_codes)<=5, "
                "else 'false'. Unstable data: use ratios only."
            )
        )
        reasoning: str = dspy.OutputField(
            desc=(
                "One sentence in RUSSIAN only: why these features and top hypothesis "
                "have the highest causal ICE score. "
                "СТРОГО НА РУССКОМ ЯЗЫКЕ. Никакого китайского, английского или другого языка. "
                "Пример: 'Признаки f0 и f2 выбраны из-за высокой корреляции с целевой переменной.'"
            )
        )


    # ═══════════════════════════════════════════════════════════════
    # SIGNATURE 2: MATRYOSHKA AUDIT (одна роль за раз)
    # Каждая из 4 ролей получает свой вызов
    # ═══════════════════════════════════════════════════════════════

    class AuditSignature(dspy.Signature):
        """
        Formula auditor. Evaluate a symbolic regression formula
        from a specific critical perspective and give a verdict.
        """

        # ── Входы ────────────────────────────────────────────────
        role_name: str = dspy.InputField(
            desc="Auditor role: Skeptic | Physicist | Pragmatist | Mystic"
        )
        role_task: str = dspy.InputField(
            desc=(
                "Specific task for this role. "
                "Skeptic: find weaknesses. "
                "Physicist: check dimensional correctness. "
                "Pragmatist: assess practical applicability. "
                "Mystic: find analogies with known physical laws."
            )
        )
        formula: str = dspy.InputField(
            desc="Anonymous formula using shadow names (f0, f1, etc.)"
        )
        formula_metrics: str = dspy.InputField(
            desc=(
                "Formula quality metrics: "
                "r2_train (float), complexity (int), "
                "feature_names (list), domain (string). "
                "May include [META-REFLECTION]: historical patterns of errors and successes. "
                "May include [★ ПОТЕНЦИАЛЬНОЕ ОТКРЫТИЕ]: system detected a new scientific law. "
                "May include [KNOWN LAW]: formula matches a verified physical law. "
                "Use this context for a more informed verdict."
            )
        )

        # ── Выходы ───────────────────────────────────────────────
        verdict: str = dspy.OutputField(
            desc=(
                "Exactly one word: ПРИНЯТА or ОТКЛОНЕНА or УСЛОВНО or ВОЗДЕРЖАЛАСЬ. "
                "Use ВОЗДЕРЖАЛАСЬ when: (1) you lack domain knowledge to judge, "
                "or (2) r2_blind >= 0.90 and there is no clear physical violation. "
                "Never vote ОТКЛОНЕНА on a formula with r2_blind >= 0.90 "
                "unless you can name a specific physical impossibility."
            )
        )
        analysis: str = dspy.OutputField(
            desc=(
                "One paragraph analysis from this role's perspective. "
                "Cite specific formula structure, not generic statements. "
                "ВАЖНО: Write in Russian only. Think in any language internally, "
                "but your output must be in Russian."
            )
        )
        structural_critique: str = dspy.OutputField(
            desc=(
                "One sentence in Russian: what is structurally MISSING or WRONG in this formula. "
                "Be specific: name the missing operator, feature, or term. "
                "Example: 'f2 отсутствует в знаменателе' or "
                "'показатель степени должен быть ~1.5 не 2.0' or "
                "'sqrt отсутствует вокруг f1'. "
                "If formula is correct write: 'структура полная'."
            )
        )
        improvement_suggestion: str = dspy.OutputField(
            desc=(
                "One concrete suggestion in Russian to improve the formula for the next PySR search. "
                "Format: operator or structural hint PySR can use. "
                "Example: 'добавить sqrt к f1', 'попробовать f0^1.5 вместо f0^2', "
                "'включить f2 в знаменатель'. "
                "If formula is accepted write: 'улучшений не требуется'."
            )
        )


    # ═══════════════════════════════════════════════════════════════
    # SIGNATURE 3: HADI REFLECTION
    # Анализирует DEATH и корректирует стратегию поиска
    # ═══════════════════════════════════════════════════════════════

    class HADIReflectSignature(dspy.Signature):
        """
        HADI death analyzer. Given a failed symbolic regression attempt,
        diagnose WHY it failed and suggest a fundamentally different search strategy.
        """

        # ── Входы ────────────────────────────────────────────────
        death_report: str = dspy.InputField(
            desc=(
                "JSON death report: {hypothesis_tried, r2_achieved, "
                "r2_required, shuffle_p, blind_r2, death_reasons}. "
                "Contains all numeric evidence of failure."
            )
        )
        data_meta: str = dspy.InputField(
            desc="Same dataset metadata as NavSignature.data_meta."
        )
        attempt_number: str = dspy.InputField(
            desc="Which HADI iteration this is (e.g. '2 of 3')."
        )

        # ── Выходы ───────────────────────────────────────────────
        failure_type: str = dspy.OutputField(
            desc=(
                "Classify failure: OVERFIT | WRONG_STRUCTURE | "
                "WRONG_FEATURES | NOISE | INSUFFICIENT_DATA"
            )
        )
        corrected_strategy: str = dspy.OutputField(
            desc=(
                "На русском языке: конкретная следующая стратегия — "
                "какие признаки попробовать, "
                "какие операторы добавить/убрать, "
                "какое структурное изменение сделать."
            )
        )
        new_hypotheses: str = dspy.OutputField(
            desc=(
                "3 новые гипотезы формул (через точку с запятой), "
                "СТРУКТУРНО ОТЛИЧНЫЕ от провальных. "
                "Только формулы, никакого текста."
            )
        )


else:
    # Заглушки если DSPy не установлен
    class NavSignature:          pass  # type: ignore
    class AuditSignature:        pass  # type: ignore
    class HADIReflectSignature:  pass  # type: ignore
