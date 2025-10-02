from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

from ...util.lru_bytes import LRUBytes
from ...cache import ThreadSafeCache
from .config import cfg_get

_T2_CACHE = None
_T2_CACHE_CFG: Optional[Tuple[str, int, int]] = None
_T2_CACHE_KIND: Optional[str] = None


def get_cache(ctx: Any, cfg_t2: Dict[str, Any]):
    global _T2_CACHE, _T2_CACHE_CFG, _T2_CACHE_KIND

    perf_on = bool(cfg_get(ctx, ["cfg", "perf", "enabled"], False))
    max_entries_perf = int(cfg_get(ctx, ["cfg", "perf", "t2", "cache", "max_entries"], 0) or 0)
    max_bytes_perf = int(cfg_get(ctx, ["cfg", "perf", "t2", "cache", "max_bytes"], 0) or 0)

    if perf_on and (max_entries_perf > 0 or max_bytes_perf > 0):
        cfg_tuple = ("bytes", max_entries_perf, max_bytes_perf)
        if _T2_CACHE is None or _T2_CACHE_CFG != cfg_tuple:
            _T2_CACHE = LRUBytes(max_entries=max_entries_perf, max_bytes=max_bytes_perf)
            _T2_CACHE_CFG = cfg_tuple
            _T2_CACHE_KIND = "bytes"
        return _T2_CACHE, _T2_CACHE_KIND

    cache_cfg = cfg_t2.get("cache", {}) or {}
    if not bool(cache_cfg.get("enabled", True)):
        return None, None

    try:
        from ...cache import LRUCache  # local import to avoid unused dependency when disabled
    except Exception:
        return None, None

    max_entries = int(cache_cfg.get("max_entries", 512))
    ttl_s = int(cache_cfg.get("ttl_s", 300))
    cfg_tuple = ("lru", max_entries, ttl_s)
    if _T2_CACHE is None or _T2_CACHE_CFG != cfg_tuple:
        _T2_CACHE = ThreadSafeCache(LRUCache(max_entries=max_entries, ttl_s=ttl_s))  # type: ignore[arg-type]
        _T2_CACHE_CFG = cfg_tuple
        _T2_CACHE_KIND = "lru"
    return _T2_CACHE, _T2_CACHE_KIND
