"""Phase 4 — Deterministic LRU spec (canonical for native mirror).

This module defines two utilities used by Stage T1 and caches.

DeterministicLRUSet (visited bounding):
  • Capacity-bounded membership set; FIFO on first successful insertion.
  • contains(x) is read-only and does NOT refresh recency.
  • add(x) inserts when absent and returns True iff an eviction happened.
  • When cap == 0, structure is disabled (contains→False; add→False).
  • Eviction removes the oldest inserted element; no randomness or time.

DeterministicLRU[K, V] (map cache):
  • Deterministic LRU map with explicit recency updates.
  • Recency changes only when flags allow: update_on_get / update_on_put.
  • contains() never refreshes; only get/put may call _touch when enabled.
  • pop_lru() removes the oldest entry deterministically.
  • When cap == 0, behaves as an always-miss cache.

This file is the authoritative spec for the Rust reimplementation used by the
native T1 kernel (visited LRU). Semantics here must match the native code.
"""
from __future__ import annotations

from collections import deque
from typing import (
    Deque,
    Dict,
    Generic,
    Iterable,
    Iterator,
    MutableMapping,
    Optional,
    Tuple,
    TypeVar,
    Callable,
)

__all__ = ["DeterministicLRUSet", "DeterministicLRU"]

K = TypeVar("K")
V = TypeVar("V")


class DeterministicLRUSet:
    """
    Deterministic, capacity-bounded set with FIFO-on-first-insert semantics.

    Intended for Stage T1 "visited" bounding:
      • Order is by *first successful insertion* only (no clocks, no randomness).
      • contains(x) is O(1) and NEVER refreshes recency.
      • add(x) inserts x iff absent; returns True iff an eviction occurred.
      • When cap == 0, structure is disabled: contains()→False; add()→False.

    Notes:
      • This is intentionally *not* a full LRU; re-contains have no effect.
      • Eviction policy is deterministic FIFO (oldest-in first-out).
      • No logging/side-effects; behavior is fully reproducible.
      • Complexity: contains/add/size O(1) average.
    """

    def __init__(self, cap: int):
        self.cap = max(0, int(cap))
        self.enabled = self.cap > 0
        self._q: Deque[str] = deque()
        self._set: Dict[str, None] = {}

    def contains(self, x: str) -> bool:
        """Return membership without refreshing recency (read-only check)."""
        return self.enabled and (x in self._set)

    __contains__ = contains

    def add(self, x: str) -> bool:
        """Insert x if absent. Returns True iff an eviction occurred. Does NOT refresh on contains."""
        if not self.enabled:
            return False
        if x in self._set:
            return False
        self._q.append(x)
        self._set[x] = None
        evicted = False
        while len(self._set) > self.cap:
            y = self._q.popleft()
            if y in self._set:
                self._set.pop(y, None)
                evicted = True
        return evicted

    def size(self) -> int:
        """Current number of distinct members (≤ cap)."""
        return len(self._set)

    def __len__(self) -> int:
        return len(self._set)

    def clear(self) -> None:
        """Remove all entries; resets order and membership."""
        self._q.clear()
        self._set.clear()


class DeterministicLRU(Generic[K, V]):
    """
    Deterministic LRU map with fixed capacity and explicit, reproducible recency updates.

    Semantics:
      • No clocks or randomness; order changes only on configured events (get/put).
      • contains() / __contains__ NEVER refresh recency.
      • update_on_get / update_on_put control whether those operations call _touch.
      • pop_lru() removes and returns the least-recently-used item.
      • When cap == 0, the structure is disabled and acts as an always-miss cache.

    Complexity: contains/get/put ~O(1) average; deque removals O(cap) worst-case when touching.
    """

    def __init__(
        self,
        cap: int,
        *,
        update_on_get: bool = True,
        update_on_put: bool = True,
        on_evict: Optional[Callable[[K, V], None]] = None,
    ) -> None:
        self.cap = max(0, int(cap))
        self.enabled = self.cap > 0
        self.update_on_get = bool(update_on_get)
        self.update_on_put = bool(update_on_put)
        self.on_evict = on_evict
        self._q: Deque[K] = deque()  # LRU (left) → MRU (right)
        self._map: Dict[K, V] = {}

    def __contains__(self, key: K) -> bool:
        """Read-only membership; does not refresh recency."""
        return self.enabled and (key in self._map)

    def contains(self, key: K) -> bool:
        return key in self

    def __len__(self) -> int:
        return len(self._map) if self.enabled else 0

    def clear(self) -> None:
        """Remove all entries; resets order and contents."""
        self._q.clear()
        self._map.clear()

    # -- Core operations ----------------------------------------------------

    def get(self, key: K, default: Optional[V] = None) -> Optional[V]:
        """Return value for key or default. May refresh recency if update_on_get is True."""
        if not self.enabled:
            return default
        if key not in self._map:
            return default
        val = self._map[key]
        if self.update_on_get:
            self._touch(key)
        return val

    def put(self, key: K, value: V) -> Optional[Tuple[K, V]]:
        """Insert/update key. Returns (evicted_k, evicted_v) if an eviction occurs; None otherwise. Recency may update if configured."""
        if not self.enabled:
            return None
        if key in self._map:
            self._map[key] = value
            if self.update_on_put:
                self._touch(key)
            return None
        # insert new
        self._map[key] = value
        self._q.append(key)
        return self._evict_if_needed()

    def pop_lru(self) -> Optional[Tuple[K, V]]:
        """Remove and return the least-recently-used item, or None if empty/disabled."""
        if not self.enabled or not self._q:
            return None
        k = self._q.popleft()
        if k not in self._map:
            return None
        v = self._map.pop(k)
        if self.on_evict:
            try:
                self.on_evict(k, v)
            except Exception:
                pass
        return (k, v)

    def items(self) -> Iterator[Tuple[K, V]]:
        """Deterministic iteration from LRU→MRU."""
        if not self.enabled:
            return iter(())
        # Deterministic LRU→MRU iteration
        return ((k, self._map[k]) for k in list(self._q) if k in self._map)

    # -- Internal helpers ---------------------------------------------------

    def _touch(self, key: K) -> None:
        """Move key to MRU end deterministically (internal)."""
        try:
            self._q.remove(key)
        except ValueError:
            # If not present in deque (shouldn't happen), append anyway.
            pass
        self._q.append(key)

    def _evict_if_needed(self) -> Optional[Tuple[K, V]]:
        """Evict in LRU order until size ≤ cap. Returns last eviction (k,v) or None."""
        evicted: Optional[Tuple[K, V]] = None
        while len(self._map) > self.cap:
            k = self._q.popleft()
            if k in self._map:
                v = self._map.pop(k)
                evicted = (k, v)
                if self.on_evict:
                    try:
                        self.on_evict(k, v)
                    except Exception:
                        pass
        return evicted
