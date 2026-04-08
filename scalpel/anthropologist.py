"""
anthropologist.py — Антрополог v10.22.

Девятая роль. Запускается ПОСЛЕ принятия формулы Матрёшкой.
Не верифицирует — понимает.

Задача: построить связь между найденной формулой и структурой мира.
  - Какой физический принцип за ней стоит?
  - С какими другими законами она связана?
  - Что это говорит об устройстве природы?
  - Где ещё встречается эта математическая структура?

Результат записывается в scalpel_vault/world_model.jsonl.
Oracle читает world_model при следующем запуске — система накапливает
понимание не только формул но и их смысла.

Модель: llama3.1:8b (Chronicle — нарратив, синтез, Meta)
Температура: 0.7 — умеренно творческая, не галлюцинирует но ищет связи
"""
from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from .config import CHRONICLE_MODEL, OLLAMA_HOST, VAULT_DIR

log = logging.getLogger("scalpel")

WORLD_MODEL_PATH = VAULT_DIR / "world_model.jsonl"


def anthropologist_reflect(
    formula_real:    str,
    formula_shadow:  str,
    r2_blind:        float,
    domain_type:     str,
    discovery_title: str = "",
    host:            str = OLLAMA_HOST,
    model:           str = CHRONICLE_MODEL,
) -> Dict:
    """
    Антрополог размышляет о найденной формуле.
    Возвращает dict с полями: principle, connections, world_insight, structure.
    Записывает в world_model.jsonl.
    """
    from .navigator import ollama_chat

    # Загружаем что уже знаем — контекст из прошлых размышлений
    past_insights = _load_recent_insights(limit=5)
    past_str = ""
    if past_insights:
        past_str = "\n\nЧТО МЫ УЖЕ ПОНЯЛИ О МИРЕ (из прошлых открытий):\n"
        for p in past_insights:
            past_str += f"  - {p.get('formula', '?')}: {p.get('world_insight', '')[:100]}\n"

    prompt = (
        f"Ты — Антрополог науки. Ты изучаешь математические законы природы "
        f"чтобы понять глубинную структуру мира.\n\n"
        f"Только что найдена формула: {formula_real}\n"
        f"Домен: {domain_type or 'неизвестен'}\n"
        f"R²_blind = {r2_blind:.4f}\n"
        + (f"Предварительное название: {discovery_title}\n" if discovery_title else "")
        + past_str
        + f"\n\nОтветь на 4 вопроса (каждый — 1-2 предложения, СТРОГО на русском):\n"
        f"1. ПРИНЦИП: Какой физический или математический принцип стоит за этой формулой?\n"
        f"2. СВЯЗИ: С какими другими известными законами или явлениями она связана?\n"
        f"3. МИР: Что эта формула говорит об устройстве природы или вселенной?\n"
        f"4. СТРУКТУРА: Где ещё в природе встречается эта математическая структура?\n\n"
        f"Формат — строго 4 строки начинающиеся с 1. 2. 3. 4."
    )

    print(f"\n  [Антрополог] 🌍 Размышляю о смысле формулы {formula_real}…")
    raw = ollama_chat(
        prompt, model=model, host=host,
        temperature=0.7, num_predict=400,
    ).strip()

    if not raw or raw.startswith("[OLLAMA_ERROR]"):
        log.warning("[Антрополог] Ollama недоступна: %s", raw[:50])
        return {}

    # Парсим 4 ответа
    result = {
        "ts":            datetime.now().isoformat(timespec="seconds"),
        "formula":       formula_real,
        "formula_shadow": formula_shadow,
        "domain":        domain_type,
        "r2_blind":      round(r2_blind, 4),
        "principle":     "",
        "connections":   "",
        "world_insight": "",
        "structure":     "",
        "raw":           raw,
    }

    lines = [l.strip() for l in raw.splitlines() if l.strip()]
    keys = ["principle", "connections", "world_insight", "structure"]
    for line in lines:
        for i, key in enumerate(keys, 1):
            prefix = f"{i}."
            if line.startswith(prefix):
                result[key] = line[len(prefix):].strip()
                break

    # Выводим на экран
    print(f"  [Антрополог] Принцип:    {result['principle'][:100]}")
    print(f"  [Антрополог] Связи:      {result['connections'][:100]}")
    print(f"  [Антрополог] Мир:        {result['world_insight'][:100]}")
    print(f"  [Антрополог] Структура:  {result['structure'][:100]}")

    # Сохраняем в world_model.jsonl
    _save_insight(result)

    return result


def _save_insight(entry: Dict) -> None:
    """Дописывает запись в world_model.jsonl."""
    try:
        WORLD_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(WORLD_MODEL_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        log.info("[Антрополог] Записано в world_model.jsonl: %s", entry.get("formula"))
    except Exception as e:
        log.warning("[Антрополог] Не удалось записать: %s", e)


def _load_recent_insights(limit: int = 5) -> List[Dict]:
    """Загружает последние записи из world_model.jsonl."""
    if not WORLD_MODEL_PATH.exists():
        return []
    try:
        lines = WORLD_MODEL_PATH.read_text(encoding="utf-8").strip().splitlines()
        result = []
        for line in lines[-limit:]:
            try:
                result.append(json.loads(line))
            except Exception:
                continue
        return result
    except Exception:
        return []


def load_world_model_for_oracle(limit: int = 10) -> str:
    """
    Возвращает краткое резюме world_model для Oracle.
    Oracle вставляет это в контекст Navigator перед следующим поиском.
    """
    insights = _load_recent_insights(limit=limit)
    if not insights:
        return ""
    lines = ["[WORLD MODEL — понимание структуры мира из прошлых открытий]"]
    for ins in insights:
        if ins.get("world_insight"):
            lines.append(
                f"  {ins.get('formula', '?')} ({ins.get('domain', '?')}): "
                f"{ins.get('world_insight', '')[:120]}"
            )
        if ins.get("connections"):
            lines.append(f"    Связи: {ins.get('connections', '')[:100]}")
    return "\n".join(lines)
