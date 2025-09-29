from __future__ import annotations

from collections import deque
from typing import Deque, Dict, Iterable


__all__ = ["DedupeRing", "DeterministicLRU"]

# PR31 guidance:
# - Use DedupeRing for short-horizon push dedupe in T1.
# - For 'visited' bounding, prefer DeterministicLRUSet in util.lru_det.
# - Keep DeterministicLRU here temporarily for potential PR32 cache internals.
# - Do not add side effects (no logging/timing); defaults must preserve disabled-path identity.


class DedupeRing:
    """
    Fixed-capacity ring buffer with O(1) membership via reference counts.

    - When capacity k == 0, the structure is disabled (no-ops).
    - Duplicates are allowed; membership is true if refcount > 0.
    - We avoid O(k) scans by maintaining a deque (order) + refcount map.

    PR31: used by T1 for push deduping; identity preserved when k=0.
    """

    def __init__(self, k: int):
        self.k = max(0, int(k))
        self.enabled = self.k > 0
        # No maxlen: we control eviction to keep refcounts in sync.
        self._q: Deque[str] = deque()
        self._ref: Dict[str, int] = {}

    def contains(self, x: str) -> bool:
        return self.enabled and (self._ref.get(x, 0) > 0)

    __contains__ = contains

    def add(self, x: str) -> None:
        """
        Append an element, evicting from the head when capacity is exceeded.
        Reference counts keep membership queries O(1) even with duplicates.
        """
        if not self.enabled:
            return
        # Evict oldest until there is room
        while self.k and len(self._q) >= self.k:
            old = self._q.popleft()
            c = self._ref.get(old, 0) - 1
            if c <= 0:
                self._ref.pop(old, None)
            else:
                self._ref[old] = c
        self._q.append(x)
        self._ref[x] = self._ref.get(x, 0) + 1

    def extend(self, xs: Iterable[str]) -> None:
        for x in xs:
            self.add(x)

    def discard(self, x: str) -> None:
        """
        Decrease reference count by one if present (does not scan deque).
        The physical entry in the deque will be removed naturally on future evictions.
        """
        if not self.enabled:
            return
        c = self._ref.get(x, 0)
        if c <= 0:
            return
        c -= 1
        if c <= 0:
            self._ref.pop(x, None)
        else:
            self._ref[x] = c

    def clear(self) -> None:
        self._q.clear()
        self._ref.clear()

    def __len__(self) -> int:
        return len(self._q)

    def tolist(self) -> list[str]:
        return list(self._q)


class DeterministicLRU:
    """
    Deterministic LRU (FIFO on first-visit) with fixed capacity for 'visited' bounding.

    - No time or randomness; order is purely by first insertion.
    - When cap == 0, structure is disabled: contains() is False; add() returns False.

    PR31: not used for 'visited' (use DeterministicLRUSet in util.lru_det). Kept temporarily
    for potential PR32 cache internals. Avoid importing this into T1.
    """

    def __init__(self, cap: int):
        self.cap = max(0, int(cap))
        self.enabled = self.cap > 0
        self._q: Deque[str] = deque()
        # Use dict-as-set for possible debug extensions; values unused.
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
