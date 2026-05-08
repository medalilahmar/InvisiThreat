"""
utils.py — Fonctions utilitaires pures (sans dépendances internes).
"""
from datetime import datetime, timezone
from typing import Any, List, Optional

import numpy as np


def safe_str(val: Any, default: str = "") -> str:
    if val is None or val == "":
        return default
    try:
        if isinstance(val, float) and np.isnan(val):
            return default
        return str(val).strip()
    except (ValueError, TypeError):
        return default


def safe_int(val: Any) -> Optional[int]:
    if val is None or val == "":
        return None
    try:
        if isinstance(val, float) and np.isnan(val):
            return None
        return int(val)
    except (ValueError, TypeError):
        return None


def safe_float(val: Any, default: float = 0.0) -> float:
    if val is None or val == "":
        return default
    try:
        f_val = float(val)
        return default if np.isnan(f_val) else f_val
    except (ValueError, TypeError):
        return default


def _compute_age_days(created_str: Optional[str]) -> Optional[int]:
    if not created_str:
        return None
    try:
        d = datetime.fromisoformat(created_str.replace("Z", "+00:00"))
        return (datetime.now(timezone.utc) - d).days
    except Exception:
        return None


def _parse_tags(raw: Any) -> List[str]:
    if isinstance(raw, list):
        return [str(t).strip() for t in raw if str(t).strip()]
    if isinstance(raw, str):
        return [t.strip() for t in raw.split(",") if t.strip()]
    return []