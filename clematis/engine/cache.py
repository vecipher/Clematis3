from __future__ import annotations
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Dict, Hashable, Tuple
import time
import json


@dataclass
class _Entry:
    ts: float
    value: Any


class _NamespaceCache:
    """Per-namespace LRU cache with TTL, stable eviction order."""
    def __init__(self, max_entries: int, ttl_sec: int, time_fn=time.time) -> None:
        self._max = int(max_entries)
        self._ttl = int(ttl_sec)
        self._time = time_fn
        self._d: "OrderedDict[Hashable, _Entry]" = OrderedDict()

    def _evict_over_cap(self) -> int:
        ev = 0
        while len(self._d) > self._max:
            self._d.popitem(last=False)  # evict oldest
            ev += 1
        return ev

    def get(self, key: Hashable) -> Tuple[bool, Any]:
        now = self._time()
        ent = self._d.get(key)
        if ent is None:
            return False, None
        if self._ttl and (now - ent.ts) > self._ttl:
            # expired → remove and miss
            self._d.pop(key, None)
            return False, None
        # touch (move to most-recent)
        self._d.move_to_end(key, last=True)
        return True, ent.value

    def set(self, key: Hashable, value: Any) -> int:
        self._d[key] = _Entry(ts=self._time(), value=value)
        self._d.move_to_end(key, last=True)
        return self._evict_over_cap()

    def invalidate(self) -> int:
        n = len(self._d)
        self._d.clear()
        return n

    def size(self) -> int:
        return len(self._d)


def stable_key(obj: Any) -> str:
    """JSON-stable key for dicts/lists/tuples when they are not hashable."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


# Backward-compatible LRUCache shim for legacy imports
class LRUCache:
    """
    Backward-compat shim for older stages that import `LRUCache`.
    Internally wraps a single `_NamespaceCache` and exposes a tiny API:
      - get(key) -> (hit, value)
      - set(key, value) / put(key, value) -> None
      - invalidate() / clear() -> int (removed)
      - size() / __len__() -> int
      - stats -> {hits, misses, evicted, size}
    Accepts legacy and new constructor params: max_entries/capacity, ttl_s/ttl_sec/ttl.
    TTL and LRU semantics match the new implementation.
    """
    def __init__(
        self,
        max_entries: int = 1024,
        ttl_s: int | None = None,
        ttl_sec: int | None = None,
        ttl: int | None = None,
        capacity: int | None = None,
        time_fn=time.time,
        **_kwargs,
    ) -> None:
        # Accept both legacy and new names; prefer explicit over defaults.
        effective_max = int(capacity if capacity is not None else max_entries)
        effective_ttl = (
            ttl if ttl is not None
            else (ttl_sec if ttl_sec is not None else (ttl_s if ttl_s is not None else 600))
        )
        self._ns = _NamespaceCache(effective_max, effective_ttl, time_fn)
        self._hits = 0
        self._misses = 0
        self._evicted = 0

    def get(self, key: Any):
        """
        Legacy behavior: return the cached value or None.
        Use get2(key) if you need (hit, value) tuple semantics.
        """
        hk = CacheManager._hashable_or_stable(key)
        hit, val = self._ns.get(hk)
        if hit:
            self._hits += 1
            return val
        else:
            self._misses += 1
            return None

    # Optional tuple-returning variant for newer call sites
    def get2(self, key: Any):
        hk = CacheManager._hashable_or_stable(key)
        hit, val = self._ns.get(hk)
        if hit:
            self._hits += 1
        else:
            self._misses += 1
        return hit, val

    def set(self, key: Any, value: Any) -> None:
        hk = CacheManager._hashable_or_stable(key)
        ev = self._ns.set(hk, value)
        self._evicted += ev

    # Legacy alias sometimes used
    def put(self, key: Any, value: Any) -> None:
        self.set(key, value)

    def invalidate(self) -> int:
        return self._ns.invalidate()

    # Alias used by some codebases
    clear = invalidate

    def size(self) -> int:
        return self._ns.size()

    def __len__(self) -> int:
        return self._ns.size()

    @property
    def stats(self) -> Dict[str, int]:
        return {
            "hits": int(self._hits),
            "misses": int(self._misses),
            "evicted": int(self._evicted),
            "size": int(self._ns.size()),
        }


class CacheManager:
    """
    Simple, process-local cache manager with:
      - Namespaces (e.g., "t2:semantic")
      - LRU eviction with max_entries
      - TTL expiry (seconds)
      - Version-aware keys supported by callers (include version in your `key` tuple)
      - Basic stats: hits/misses/evicted/size
    """

    def __init__(self, max_entries: int = 1024, ttl_sec: int = 600, time_fn=time.time) -> None:
        self._max = int(max_entries)
        self._ttl = int(ttl_sec)
        self._time = time_fn
        self._ns: Dict[str, _NamespaceCache] = {}
        self._hits = 0
        self._misses = 0
        self._evicted = 0

    def _ns_obj(self, namespace: str) -> _NamespaceCache:
        ns = self._ns.get(namespace)
        if ns is None:
            ns = _NamespaceCache(self._max, self._ttl, self._time)
            self._ns[namespace] = ns
        return ns

    @staticmethod
    def _hashable_or_stable(key: Any) -> Hashable:
        try:
            hash(key)
            return key  # already hashable
        except TypeError:
            return stable_key(key)

    def get(self, namespace: str, key: Tuple[Any, ...]) -> Tuple[bool, Any]:
        """Return (hit, value). Expired entries are treated as misses and removed."""
        ns = self._ns_obj(namespace)
        hk = self._hashable_or_stable(key)
        hit, val = ns.get(hk)
        if hit:
            self._hits += 1
        else:
            self._misses += 1
        return hit, val

    def set(self, namespace: str, key: Tuple[Any, ...], value: Any) -> None:
        """Insert or refresh the cached value; may evict oldest entries."""
        ns = self._ns_obj(namespace)
        hk = self._hashable_or_stable(key)
        ev = ns.set(hk, value)
        self._evicted += ev

    def invalidate_namespace(self, namespace: str) -> int:
        """Remove all entries in the given namespace; returns count removed."""
        ns = self._ns.get(namespace)
        if ns is None:
            return 0
        return ns.invalidate()

    def invalidate_all(self) -> int:
        """Remove all entries across all namespaces; returns total count removed."""
        total = 0
        for ns in self._ns.values():
            total += ns.invalidate()
        return total

    @property
    def stats(self) -> Dict[str, int]:
        """Basic counters; size is total live entries across namespaces."""
        size = sum(ns.size() for ns in self._ns.values())
        return {
            "hits": int(self._hits),
            "misses": int(self._misses),
            "evicted": int(self._evicted),
            "size": int(size),
        }
