"""
data.py — Загрузка и разбивка данных из CSV.
"""
import logging
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

log = logging.getLogger("scalpel")

# FIX БАГ 4: Лимит обновлён под 32 ГБ RAM (было 200 строк для 7.7 ГБ).
# При 32 ГБ PySR + Julia безопасно работают с 50 000 строк.
# Оставляем предупреждение — пользователь сам решает, резать или нет.
MAX_CSV_ROWS: int = 50_000


def load_csv(
    train_path: str = "universe_train.csv",
    test_path:  str = "universe_test.csv",
    target_col: str = "",
    test_size:  float = 0.2,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str], str]:
    """
    Загружает данные из CSV.
    - Если есть оба файла (train + test) — используем их.
    - Если только один — делим автоматически по test_size.
    Возвращает: X_train, y_train, X_test, y_test, feat_names, target_col
    """
    train_p = Path(train_path)
    test_p  = Path(test_path)

    if train_p.exists() and test_p.exists():
        df_train = pd.read_csv(train_p)
        df_test  = pd.read_csv(test_p)
        # v10.3.9-patch: OOM Guard
        total_rows = len(df_train) + len(df_test)
        if total_rows > MAX_CSV_ROWS:
            print(
                f"\n  ⚠  [OOM WARNING] Файл содержит {total_rows} строк "
                f"(лимит: {MAX_CSV_ROWS} для 32 ГБ RAM).\n"
                f"  Риск OOM (Out of Memory). Рекомендуется обрезка данных.\n"
                f"  Совет: df.head({int(MAX_CSV_ROWS * 0.8)}).to_csv('train_cut.csv', index=False)"
            )
            log.warning("[DATA] OOM Risk: %d строк > лимит %d", total_rows, MAX_CSV_ROWS)
        if not target_col:
            target_col = df_train.columns[-1]
        feat_names = [c for c in df_train.columns if c != target_col]
        X_train = df_train[feat_names].values.astype(np.float64)
        y_train = df_train[target_col].values.astype(np.float64)
        X_test  = df_test[feat_names].values.astype(np.float64)
        y_test  = df_test[target_col].values.astype(np.float64)
    else:
        single = train_p if train_p.exists() else None
        if single is None:
            raise FileNotFoundError(
                f"Не найдены файлы: {train_path}, {test_path}\n"
                f"Укажи --train путь к CSV с данными."
            )
        df = pd.read_csv(single)
        # v10.3.9-patch: OOM Guard
        if len(df) > MAX_CSV_ROWS:
            print(
                f"\n  ⚠  [OOM WARNING] Файл содержит {len(df)} строк "
                f"(лимит: {MAX_CSV_ROWS} для 32 ГБ RAM).\n"
                f"  Риск OOM (Out of Memory). Рекомендуется обрезка данных.\n"
                f"  Совет: df.head({int(MAX_CSV_ROWS * 0.8)}).to_csv('train_cut.csv', index=False)"
            )
            log.warning("[DATA] OOM Risk: %d строк > лимит %d", len(df), MAX_CSV_ROWS)
        if not target_col:
            target_col = df.columns[-1]
        feat_names = [c for c in df.columns if c != target_col]
        split = int(len(df) * (1 - test_size))
        X_train = df[feat_names].values[:split].astype(np.float64)
        y_train = df[target_col].values[:split].astype(np.float64)
        X_test  = df[feat_names].values[split:].astype(np.float64)
        y_test  = df[target_col].values[split:].astype(np.float64)

    log.info(
        "[DATA] train=%d test=%d features=%d target='%s'",
        len(y_train), len(y_test), len(feat_names), target_col,
    )
    return X_train, y_train, X_test, y_test, feat_names, target_col
