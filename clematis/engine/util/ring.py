"""Phase 4 spec semantics for DedupeRing and DeterministicLRU:

- Window size semantics: fixed capacity k entries.
- contains(x): returns membership without refreshing recency or affecting order.
- add(x): always appends x to the structure; duplicates allowed.
- Overflow evicts oldest entries first (FIFO order).
- This module serves as the canonical spec for native reimplementation.
"""

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
    Fixed-capacity ring buffer with reference-counted membership.

    Capacity/window semantics:
      - Holds up to k entries; duplicates allowed.
      - Membership is true iff refcount > 0.
    Timing:
      - contains(x) is read-only; never refreshes recency.
      - add(x) should be called only after a push is accepted.
      - add(x) always appends; does not consolidate duplicates.
    Eviction policy:
      - FIFO by insertion order into the ring.
      - Eviction removes oldest entries one at a time, decrementing refcounts.
      - Membership drops only when refcount reaches zero.
    Disabled behavior:
      - When k == 0, all methods are no-ops; contains() always returns False.
    Complexity and stability:
      - All operations are O(1) amortized.
      - No side effects such as logging or timing.

    Example:
      >>> dr = DedupeRing(3)
      >>> dr.add("a")
      >>> dr.add("b")
      >>> dr.contains("a")
      True
      >>> dr.add("a")  # duplicate allowed
      >>> len(dr)
      3
      >>> dr.add("c")  # triggers eviction of oldest "a"
      >>> dr.contains("a")
      True  # still present because refcount is 1
      >>> dr.add("d")  # evicts "b"
      >>> dr.contains("b")
      False
    """

    def __init__(self, k: int):
        self.k = max(0, int(k))
        self.enabled = self.k > 0
        # No maxlen: we control eviction to keep refcounts in sync.
        self._q: Deque[str] = deque()
        self._ref: Dict[str, int] = {}

    def contains(self, x: str) -> bool:
        """
        Check membership without refreshing recency.
        Returns True if refcount > 0, else False.
        """
        return self.enabled and (self._ref.get(x, 0) > 0)

    __contains__ = contains

    def add(self, x: str) -> None:
        """
        Append an element after push acceptance.
        Does not consolidate duplicates.
        Evicts oldest entries if capacity exceeded, decrementing refcounts.
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
        """
        Append multiple elements sequentially.
        Each element is added with the same semantics as add().
        """
        for x in xs:
            self.add(x)

    def discard(self, x: str) -> None:
        """
        Decrease reference count by one if present.
        Does not scan or remove from deque immediately.
        Physical entries are removed naturally on future evictions.
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
        """
        Clear all entries and reference counts.
        """
        self._q.clear()
        self._ref.clear()

    def __len__(self) -> int:
        return len(self._q)

    def tolist(self) -> list[str]:
        return list(self._q)


class DeterministicLRU:
    """
    Deterministic FIFO-on-first-visit LRU with fixed capacity.

    Semantics:
      - contains(x) returns membership without refreshing recency.
      - add(x) appends if not present; returns True iff an eviction occurred.
      - Evicts oldest entries first when capacity exceeded.
    Disabled behavior:
      - When cap == 0, contains() always returns False; add() always returns False.
    Notes:
      - T1 uses DeterministicLRUSet in util.lru_det for visited bounding.
      - This class is retained temporarily for potential PR32 cache internals.
      - No side effects or logging.
    """

    def __init__(self, cap: int):
        self.cap = max(0, int(cap))
        self.enabled = self.cap > 0
        self._q: Deque[str] = deque()
        # Use dict-as-set for possible debug extensions; values unused.
        self._set: Dict[str, None] = {}

    def contains(self, x: str) -> bool:
        """
        Check membership without refreshing recency.
        """
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
        """
        Clear all entries.
        """
        self._q.clear()
        self._set.clear()
