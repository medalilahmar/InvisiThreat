"""
cache.py — Cache en mémoire générique + cache des scores IA.
"""
import json
import logging
import time
from pathlib import Path
from typing import Any, Callable, Dict, Tuple

from server.config import CACHE_TTL_PRODUCTS, SCORES_CACHE_TTL

logger = logging.getLogger("invisithreat.cache")

# ── Cache générique ───────────────────────────────────────────────────────────
_cache_store: Dict[str, Tuple[Any, float]] = {}

# ── Cache des scores IA ───────────────────────────────────────────────────────
_scores_cache_memory: Dict[str, Any] = {}
_scores_cache_loaded_at: float       = 0.0


def get_cached_or_fetch(
    cache_key: str,
    fetch_func: Callable,
    ttl: int = CACHE_TTL_PRODUCTS,
) -> Any:
    now   = time.time()
    entry = _cache_store.get(cache_key)
    if entry is not None:
        data, ts = entry
        if now - ts < ttl:
            return data
    data = fetch_func()
    _cache_store[cache_key] = (data, now)
    return data


def get_scores_cache() -> Dict[str, Any]:
    global _scores_cache_memory, _scores_cache_loaded_at
    now = time.time()
    if now - _scores_cache_loaded_at < SCORES_CACHE_TTL and _scores_cache_memory:
        return _scores_cache_memory
    cache_file = Path("data/ai_scores_cache.json")
    if cache_file.exists():
        try:
            with open(cache_file) as f:
                _scores_cache_memory = json.load(f)
            _scores_cache_loaded_at = now
            logger.debug(f"Scores cache rechargé : {len(_scores_cache_memory)} entrées")
        except Exception as e:
            logger.warning(f"Impossible de recharger le cache scores : {e}")
    return _scores_cache_memory


def set_scores_cache(cache: Dict[str, Any]) -> None:
    """Met à jour le cache mémoire directement (après scoring au démarrage)."""
    global _scores_cache_memory, _scores_cache_loaded_at
    _scores_cache_memory    = cache
    _scores_cache_loaded_at = time.time()


def invalidate_cache(prefix: str = "") -> int:
    keys = [k for k in list(_cache_store.keys()) if k.startswith(prefix)]
    for k in keys:
        del _cache_store[k]
    return len(keys)


def invalidate_scores_cache() -> None:
    global _scores_cache_loaded_at
    _scores_cache_loaded_at = 0.0