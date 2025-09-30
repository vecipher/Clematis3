from __future__ import annotations

from collections import deque
from typing import Callable, Deque, Dict, Generic, Iterator, Optional, Tuple, TypeVar, Iterable

__all__ = ["LRUBytes"]

K = TypeVar("K")
V = TypeVar("V")


class LRUBytes(Generic[K, V]):
    """
    Deterministic LRU cache with *two* capacity constraints:
      - max_entries (count of items)
      - max_bytes   (sum of item costs)

    Identity & determinism requirements for Clematis M6:
      - No clocks, no randomness.
      - Recency is updated only on `get()` and `put()` (update-on-put for inserts and updates).
      - When multiple evictions are required, eviction order is strictly LRU→MRU by deque position.
        Positions are unique; hence tie-breaks are implicit and deterministic.
      - When `max_bytes > 0` and an item `cost_bytes > max_bytes`, the insertion is **rejected**
        (to avoid thrashing), returning (0, 0).

    Notes:
      - If both caps are zero (or the cache is never instantiated), the cache acts as disabled.
      - Keys are kept unique in the recency deque; each `put()` removes old position before re-append.
      - `key_to_str` is accepted for future lex tie-breaks or logging, but is not required for LRU.
    """

    def __init__(
        self,
        max_entries: int,
        max_bytes: int,
        *,
        key_to_str: Optional[Callable[[K], str]] = None,
        on_evict: Optional[Callable[[K, V, int], None]] = None,
    ) -> None:
        self.max_entries = int(max_entries or 0)
        self.max_bytes = int(max_bytes or 0)
        self.key_to_str = key_to_str or (lambda k: str(k))
        self.on_evict = on_evict

        # LRU order: left = LRU, right = MRU
        self._q: Deque[K] = deque()
        # key -> (value, cost_bytes)
        self._map: Dict[K, Tuple[V, int]] = {}
        self._bytes: int = 0

    # ------------------------ Introspection ------------------------
    def size_entries(self) -> int:
        return len(self._map)

    def size_bytes(self) -> int:
        return self._bytes

    def __len__(self) -> int:
        return len(self._map)

    def contains(self, key: K) -> bool:
        return key in self._map and (self.max_entries > 0 or self.max_bytes > 0)

    __contains__ = contains

    def keys(self) -> Iterator[K]:
        # Deterministic LRU→MRU iteration
        return iter(list(self._q))

    def items(self) -> Iterator[Tuple[K, V]]:
        """
        Deterministic snapshot iterator of (key, value) from LRU→MRU without
        mutating recency. This is provided to conform to the CacheProtocol used
        by parallel-safe wrappers. It yields current values (ignores byte costs).
        """
        # Copy order to avoid interference if callers mutate during iteration.
        for k in list(self._q):
            if k in self._map:
                yield (k, self._map[k][0])

    def clear(self) -> None:
        self._q.clear()
        self._map.clear()
        self._bytes = 0

    # ------------------------ Core ops -----------------------------
    def get(self, key: K) -> Optional[V]:
        """
        Returns the value if present and moves the key to MRU deterministically.
        """
        if key not in self._map:
            return None
        val, cost = self._map[key]
        # Move to MRU (deterministic: remove then append)
        try:
            self._q.remove(key)
        except ValueError:
            # Should not happen; tolerate gracefully
            pass
        self._q.append(key)
        return val

    def put(self, key: K, value: V, cost_bytes: int) -> Tuple[int, int]:
        """
        Insert or update an item with a given `cost_bytes`.

        Returns:
          (evicted_count, evicted_total_bytes)

        Behavior:
          - If both caps are zero → acts as disabled cache; returns (0, 0).
          - If `max_bytes > 0` and `cost_bytes > max_bytes` → reject insertion; returns (0, 0).
          - On update: old cost is removed, key is moved to MRU.
          - Evict LRU items until both constraints are satisfied.
        """
        if self.max_entries == 0 and self.max_bytes == 0:
            return (0, 0)

        cost_bytes = int(cost_bytes or 0)
        if cost_bytes < 0:
            cost_bytes = 0

        # Oversized item cannot be cached under current byte cap; reject.
        if self.max_bytes and cost_bytes > self.max_bytes:
            return (0, 0)

        evicted_n = 0
        evicted_b = 0

        # If key exists, adjust bytes and move to MRU
        if key in self._map:
            old_val, old_cost = self._map[key]
            self._bytes -= old_cost
            # Update value/cost
            self._map[key] = (value, cost_bytes)
            # Move to MRU
            try:
                self._q.remove(key)
            except ValueError:
                pass
            self._q.append(key)
        else:
            # Insert new as MRU
            self._map[key] = (value, cost_bytes)
            self._q.append(key)

        # Apply evictions until within caps
        # First, compute the resulting bytes if we keep the new/updated item.
        target_bytes = self._bytes + cost_bytes
        # Evict while exceeding entries cap or byte cap
        while (self.max_entries and len(self._map) > self.max_entries) or (
            self.max_bytes and target_bytes > self.max_bytes
        ):
            # Evict from LRU side
            k0 = self._q.popleft()
            # If k0 was just inserted/updated (same as key), skip and continue;
            # but since we appended 'key' at MRU, normal path won't hit this case.
            if k0 not in self._map:
                continue
            v0, c0 = self._map.pop(k0)
            evicted_n += 1
            evicted_b += c0
            target_bytes -= c0
            self._bytes -= c0
            if self.on_evict:
                try:
                    self.on_evict(k0, v0, c0)
                except Exception:
                    pass

        # Commit final byte count
        self._bytes = target_bytes
        return (evicted_n, evicted_b)
