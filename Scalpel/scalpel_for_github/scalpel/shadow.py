"""
shadow.py — Анонимизация имён признаков (ShadowMapper)
NIST SP 800-90B: real_name → f0, f1, f2, …
Маппинг живёт только в RAM.
"""
import hashlib
import json
import re
from typing import Dict, List



def _sanitize_feat_name(name: str) -> str:
    """
    v10.3.9: Санитизация имён признаков перед передачей в Julia/PySR.
    Заменяет спецсимволы / ^ . - пробелы и прочие не-alphanum на _ .
    Пример: df/dx → df_dx, x^2 → x_2, col name → col_name.
    Гарантирует, что Julia-парсер PySR не упадёт на символах операторов.
    """
    import re as _re
    safe = _re.sub(r"[^A-Za-z0-9_]", "_", name)
    if safe and safe[0].isdigit():
        safe = "f_" + safe
    safe = _re.sub(r"_+", "_", safe).strip("_")
    return safe or "feat"

class ShadowMapper:
    """
    Анонимизирует имена признаков: real_name → f0, f1, f2, …
    FIX-V5: restore_formula использует \\b + сортировку по длине (f10 ≠ f1).
    """

    def __init__(self):
        self._r2s: Dict[str, str] = {}
        self._s2r: Dict[str, str] = {}
        self.fingerprint = ""
        self.active = False

    def build(self, feature_names: List[str]) -> List[str]:
        # v10.3.9: санитизируем реальные имена — сохраняем оригинал для restore(),
        # но в Julia передаём только безопасные shadow-имена (f0, f1, …).
        # _sanitize_feat_name нужен если производные df/dx попадут в feat_names.
        self._r2s = {name: f"f{i}" for i, name in enumerate(feature_names)}
        # Дополнительная защита: логируем небезопасные имена
        for name in feature_names:
            if any(c in name for c in "/^. -"):
                import logging as _log
                _log.getLogger("scalpel").warning(
                    "[ShadowMapper] Небезопасное имя признака: %r → будет замаплено в f%d. "
                    "Используйте _sanitize_feat_name() перед передачей в run_engine().",
                    name, list(feature_names).index(name),
                )
        self._s2r = {v: k for k, v in self._r2s.items()}
        raw = json.dumps(self._r2s, sort_keys=True)
        self.fingerprint = hashlib.sha256(raw.encode()).hexdigest()[:16]
        self.active = True
        return list(self._r2s.values())

    def anonymize(self, names: List[str]) -> List[str]:
        return [self._r2s.get(n, n) for n in names]

    def restore(self, formula: str) -> str:
        """shadow формула → реальные имена. FIX-V5: длинные токены первыми."""
        if not self.active:
            return formula
        result = formula
        for shadow, real in sorted(self._s2r.items(), key=lambda x: len(x[0]), reverse=True):
            result = re.sub(r'\b' + re.escape(shadow) + r'\b', real, result)
        return result

    @property
    def reverse_mapping(self) -> Dict[str, str]:
        return dict(self._s2r)
