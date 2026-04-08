"""
dim_codes.py — Определение типа единиц измерения признаков (OODA).
"""
from typing import Dict

_USER_DIM_CACHE: Dict[str, int] = {}

_DIM_MENU = {
    "1": ("деньги/цена",         10),
    "2": ("объём/количество",     5),
    "3": ("безразмерное/доля",    0),
    "4": ("время/дата",           8),
    "5": ("масса/вес",            3),
    "6": ("длина/расстояние",     2),
    "7": ("температура",          4),
    "8": ("сила/энергия/давление",6),
    "9": ("другое/неизвестно",    1),
}


def dim_code(name: str) -> int:
    """Автоматически определяет код размерности по имени признака."""
    n = name.lower()
    # Физические единицы идут ДО финансовых: «gram_value» должен давать 3 (масса),
    # а не 10 (финансы), потому что «gram» специфичнее, чем «value».
    if any(k in n for k in ("mass", "weight", "kg", "gram")):                         return 3
    if any(k in n for k in ("len", "length", "dist", "meter", "km", "radius", "area")): return 2
    if any(k in n for k in ("temp", "kelvin", "celsius", "heat")):                    return 4
    if any(k in n for k in ("force", "newton", "pressure", "energy", "joule")):       return 6
    if any(k in n for k in ("price", "cost", "usd", "eur", "rub", "value", "revenue")): return 10
    if any(k in n for k in ("volume", "qty", "count", "num", "amount")):               return 5
    if any(k in n for k in ("ratio", "rate", "pct", "percent", "share", "rel")):      return 0
    if any(k in n for k in ("time", "date", "day", "hour", "month", "year", "age")):  return 8
    return 1


def dim_code_interactive(name: str, auto_only: bool = False) -> int:
    """Определяет код размерности, при необходимости спрашивает пользователя."""
    code = dim_code(name)
    if code != 1:
        return code
    if name in _USER_DIM_CACHE:
        return _USER_DIM_CACHE[name]
    if auto_only:
        return 1

    print(f"\n  [OODA] Признак '{name}' не распознан автоматически.")
    print(f"  Укажи тип единиц измерения (Enter = пропустить → 'другое'):")
    for k, (label, _) in _DIM_MENU.items():
        print(f"    {k}) {label}")
    try:
        choice = input("  Выбор [1-9]: ").strip()
    except (EOFError, KeyboardInterrupt):
        choice = "9"
    _, code = _DIM_MENU.get(choice, ("другое", 1))
    _USER_DIM_CACHE[name] = code
    return code
