# clematis/engine/stages/_cache_factory.py (new small module)
from __future__ import annotations
from typing import Any, Tuple, List, Callable
from clematis.engine.cache import ThreadSafeCache, CacheProtocol, merge_caches_deterministic
from clematis.engine.util.lru_bytes import LRUBytes  # or your LRUCache

def make_shared_cache(capacity: int) -> CacheProtocol[bytes, bytes]:
    return ThreadSafeCache(LRUBytes(capacity))

def make_worker_cache(capacity: int) -> CacheProtocol[bytes, bytes]:
    # plain (no lock) per-worker cache to reduce contention
    return LRUBytes(capacity)

# Deterministic merge helper re-export
merge_caches_deterministic_bytes = merge_caches_deterministic
