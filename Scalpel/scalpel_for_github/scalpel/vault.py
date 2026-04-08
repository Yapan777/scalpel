"""
vault.py — GoldVault: сохранение и PDCA-проверка найденных формул.
"""
import hashlib
import json
import logging
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional

from .config import GOLD_PATH, INTERNAL_GOLD_PATH, REJECTED_PATH, DISPUTED_PATH, REJECTED_R2_SAFE_MAX, PDCA_INTERVAL_DAYS, PDCA_STALE_R2
from .shadow import ShadowMapper

log = logging.getLogger("scalpel")

try:
    from sklearn.metrics import r2_score
except ImportError:
    r2_score = None


class GoldVault:
    """Сохраняет формулы и отслеживает их актуальность (PDCA)."""

    def save(
        self,
        formula_shadow: str,
        formula_real:   str,
        shadow_mapper:  ShadowMapper,
        r2_train:       float,
        r2_blind:       float,
        complexity:     int,
        tags:           Optional[List[str]] = None,
        n_samples:      int = 0,          # v10.5: для DSPy few-shot context
        chronicle:      str = "",         # v10.14: ЛЕТОПИСЕЦ — история поиска
    ) -> Path:
        GOLD_PATH.parent.mkdir(parents=True, exist_ok=True)
        _id  = hashlib.sha256(formula_shadow.encode()).hexdigest()[:12]
        _ts  = datetime.now().isoformat()
        _ncd = (datetime.now() + timedelta(days=PDCA_INTERVAL_DAYS)).date().isoformat()

        anon = {
            "id":                 _id,
            "formula":            formula_shadow,
            "skeleton":           self._skeleton(formula_shadow),
            "domain":             "",
            "tags":               tags or [],
            "timestamp":          _ts,
            "r2_train":           round(r2_train, 4),
            "r2_blind":           round(r2_blind, 4),
            "complexity":         complexity,
            "NEXT_CHECK_DATE":    _ncd,
            "n_samples":          n_samples,
            "PDCA_STATUS":        "ACTIVE",
            "PDCA_LAST_CHECKED":  _ts[:10],
            "PDCA_INTERVAL_DAYS": PDCA_INTERVAL_DAYS,
            "chronicle":          chronicle[:2000] if chronicle else "",  # v10.14
        }
        internal = dict(anon)
        internal["real_formula"]   = formula_real
        internal["shadow_to_real"] = shadow_mapper.reverse_mapping

        self._atomic_append(GOLD_PATH, anon)
        self._atomic_append(INTERNAL_GOLD_PATH, internal)
        log.info("[VAULT] Сохранено: %s → %s", formula_shadow, GOLD_PATH)
        return GOLD_PATH

    def save_rejected(
        self,
        formula_shadow: str,
        formula_real:   str,
        r2_train:       float,
        r2_blind:       float,
        complexity:     int,
        rejected_by:    str  = "",
        reason:         str  = "",
        lesson:         str  = "",
        tags:           Optional[List[str]] = None,
        n_rejectors:    int  = 0,   # сколько членов Матрёшки проголосовало ПРОТИВ
        n_total:        int  = 0,   # сколько всего голосовало (без ВОЗДЕРЖАЛАСЬ)
    ) -> Path:
        """Сохраняет отклонённую формулу — в rejected или disputed в зависимости от R².

        v10.15 — два пути:

        REJECTED (r2_blind < REJECTED_R2_SAFE_MAX):
            Математика плохая. Матрёшка права. Navigator запомнит это как СТОП-паттерн.

        DISPUTED (r2_blind >= REJECTED_R2_SAFE_MAX):
            Математика хорошая, но LLM отклонила — возможная ложная негатив.
            НЕ баним паттерн. Сохраняем в disputed_formulas.json для ревью.
            Navigator получит мягкое предупреждение, не жёсткий запрет.

        Дополнительно: единогласное отклонение (n_rejectors == n_total) при хорошем R²
        особенно подозрительно — все могли одновременно ошибиться одинаково.
        """
        _skel = self._skeleton(formula_shadow)
        _id   = hashlib.sha256((formula_shadow + "rejected").encode()).hexdigest()[:12]
        _ts   = datetime.now().isoformat()

        # Определяем куда маршрутизировать
        _is_suspicious = (
            float(r2_blind) >= REJECTED_R2_SAFE_MAX
        )
        _unanimous = (n_total > 0 and n_rejectors == n_total)
        _route_path = DISPUTED_PATH if _is_suspicious else REJECTED_PATH
        _route_label = "СПОРНАЯ" if _is_suspicious else "ОТКЛОНЕНА"

        if _is_suspicious:
            log.warning(
                "[VAULT] ⚠️  СПОРНОЕ отклонение: R²_blind=%.4f >= %.2f "
                "(порог REJECTED_R2_SAFE_MAX). Формула '%s' "
                "не будет занесена в СТОП-паттерны — только в disputed. "
                "%s",
                r2_blind, REJECTED_R2_SAFE_MAX, formula_shadow[:40],
                "(единогласно — особо подозрительно)" if _unanimous else "",
            )
            print(
                f"  [VAULT] ⚠️  СПОРНОЕ отклонение R²={r2_blind:.4f} ≥ {REJECTED_R2_SAFE_MAX}"
                f" → сохранено в disputed_formulas.json (не СТОП-паттерн)"
            )

        # Дедупликация по skeleton в целевом файле
        if _route_path.exists():
            try:
                with _route_path.open(encoding="utf-8") as _f:
                    _existing = json.load(_f)
                _ids   = {r.get("id")      for r in _existing.get("formulas", [])}
                _skels = {r.get("skeleton") for r in _existing.get("formulas", [])}
                if _id in _ids or _skel in _skels:
                    log.debug("[VAULT] Запись уже есть в %s (id=%s) — пропускаем",
                              _route_path.name, _id)
                    return _route_path
            except Exception:
                pass

        _route_path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "id":           _id,
            "formula":      formula_shadow,
            "formula_real": formula_real,
            "skeleton":     _skel,
            "r2_train":     round(r2_train, 4),
            "r2_blind":     round(r2_blind, 4),
            "complexity":   complexity,
            "status":       _route_label,
            "rejected_by":  rejected_by,
            "n_rejectors":  n_rejectors,
            "n_total":      n_total,
            "unanimous":    _unanimous,
            "reason":       reason[:500] if reason else "",
            "lesson":       lesson[:500] if lesson else "",
            "tags":         tags or [],
            "timestamp":    _ts,
        }
        self._atomic_append(_route_path, record)
        log.info("[VAULT] %s сохранена: %s → %s",
                 _route_label, formula_shadow[:40], _route_path.name)
        return _route_path

    def check_stale(self, predict_fn=None, X=None, y=None) -> List[dict]:
        today = datetime.now().date().isoformat()
        if not GOLD_PATH.exists():
            return []
        try:
            with GOLD_PATH.open(encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            return []

        checked, modified = [], False
        for rec in data.get("formulas", []):
            if rec.get("NEXT_CHECK_DATE", "9999") > today:
                continue
            report = {"id": rec.get("id"), "formula": rec.get("formula"),
                      "PDCA_STATUS": rec.get("PDCA_STATUS"), "r2_new": None}
            if predict_fn is not None and X is not None and y is not None and r2_score:
                try:
                    r2 = float(r2_score(y, predict_fn(X)))
                    report["r2_new"] = round(r2, 4)
                    if r2 < PDCA_STALE_R2:
                        rec["PDCA_STATUS"] = "STALE"
                        report["PDCA_STATUS"] = "STALE"
                except Exception:
                    pass
            ncd = (datetime.now() + timedelta(days=PDCA_INTERVAL_DAYS)).date().isoformat()
            rec["NEXT_CHECK_DATE"]   = ncd
            rec["PDCA_LAST_CHECKED"] = today
            checked.append(report)
            modified = True

        if modified:
            data["last_updated"] = datetime.now().isoformat()
            self._atomic_write(GOLD_PATH, data)
        return checked

    @staticmethod
    def _skeleton(formula: str) -> str:
        tokens = re.findall(r"[a-zA-Z_]\w*|[\d.]+|[+\-*/^()]", formula)
        _OPS = {"sin","cos","exp","log","abs","sqrt","tanh","pow"}
        result = []
        for t in tokens:
            if t in _OPS:                      result.append(t)
            elif re.match(r"^[a-zA-Z_]", t):  result.append("v")
            elif re.match(r"^\d",         t):  result.append("c")
            else:                              result.append(t)
        return " ".join(result)

    @staticmethod
    def _atomic_append(path: Path, record: dict) -> None:
        if path.exists():
            try:
                with path.open(encoding="utf-8") as f:
                    data = json.load(f)
                if not isinstance(data.get("formulas"), list):
                    data["formulas"] = []
            except Exception:
                data = {"formulas": []}
        else:
            data = {"formulas": []}
        data["formulas"].append(record)
        data["count"]        = len(data["formulas"])
        data["last_updated"] = datetime.now().isoformat()
        GoldVault._atomic_write(path, data)

    @staticmethod
    def _atomic_write(path: Path, data: dict) -> None:
        tmp = path.with_suffix(".tmp")
        try:
            with tmp.open("w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            tmp.replace(path)
        except Exception as e:
            log.error("[VAULT] Ошибка записи %s: %s", path, e)
            tmp.unlink(missing_ok=True)
