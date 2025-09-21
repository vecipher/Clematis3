

from __future__ import annotations

from collections import deque
from typing import Deque, Dict, Generic, Iterable, Iterator, MutableMapping, Optional, Tuple, TypeVar, Callable

__all__ = ["DeterministicLRUSet", "DeterministicLRU"]

K = TypeVar("K")
V = TypeVar("V")


class DeterministicLRUSet:
    """
    Deterministic, capacity-bounded set behaving like an LRU on *insert order only*.

    Intended for PR31 'visited' bounding:
      - No time or randomness; order is by first successful insertion (FIFO).
      - `contains()` is O(1).
      - `add(x)` returns True iff an eviction occurred (y was removed).
      - When cap == 0, structure is disabled: contains() is False; add() returns False.

    Note: This is intentionally *not* a full LRU (no recency updates on re-contains), to
    preserve simple, reproducible behavior for one-shot visits.
    """

    def __init__(self, cap: int):
        self.cap = max(0, int(cap))
        self.enabled = self.cap > 0
        self._q: Deque[str] = deque()
        self._set: Dict[str, None] = {}

    def contains(self, x: str) -> bool:
        return self.enabled and (x in self._set)

    __contains__ = contains

    def add(self, x: str) -> bool:
        """
        Insert x if not present.
        Returns True iff an eviction occurred.
        """
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
        return len(self._set)

    def __len__(self) -> int:
        return len(self._set)

    def clear(self) -> None:
        self._q.clear()
        self._set.clear()


class DeterministicLRU(Generic[K, V]):
    """
    Deterministic LRU map with fixed capacity and explicit, reproducible recency updates.

    - No clocks or randomness; recency changes only on deterministic events (get/touch/put).
    - `update_on_get` / `update_on_put` decide whether those events move items to MRU.
    - When `cap == 0`, the structure is disabled and acts as an always-miss cache.

    Complexity: O(1) for contains/get/put average, O(cap) worst-case for deque removals
    when updating recency (acceptable for small perf caps).
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
        self._q: Deque[K] = deque()   # LRU (left) → MRU (right)
        self._map: Dict[K, V] = {}

    def __contains__(self, key: K) -> bool:
        return self.enabled and (key in self._map)

    def contains(self, key: K) -> bool:
        return key in self

    def __len__(self) -> int:
        return len(self._map) if self.enabled else 0

    def clear(self) -> None:
        self._q.clear()
        self._map.clear()

    # -- Core operations ----------------------------------------------------

    def get(self, key: K, default: Optional[V] = None) -> Optional[V]:
        if not self.enabled:
            return default
        if key not in self._map:
            return default
        val = self._map[key]
        if self.update_on_get:
            self._touch(key)
        return val

    def put(self, key: K, value: V) -> Optional[Tuple[K, V]]:
        """
        Insert or update `key`. Returns (evicted_key, evicted_value) if an eviction occurs,
        otherwise None. Eviction order is strictly LRU with no time dependence.
        """
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
        """
        Remove and return the least-recently-used item. Returns None if empty/disabled.
        """
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
        if not self.enabled:
            return iter(())
        # Deterministic LRU→MRU iteration
        return ((k, self._map[k]) for k in list(self._q) if k in self._map)

    # -- Internal helpers ---------------------------------------------------

    def _touch(self, key: K) -> None:
        """
        Move `key` to MRU end deterministically.
        """
        try:
            self._q.remove(key)
        except ValueError:
            # If not present in deque (shouldn't happen), append anyway.
            pass
        self._q.append(key)

    def _evict_if_needed(self) -> Optional[Tuple[K, V]]:
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