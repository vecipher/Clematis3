from __future__ import annotations
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Hashable, Tuple
import time, json

@dataclass
class CacheEntry:
    ts: float
    value: Any

class LRUCache:
    def __init__(self, max_entries: int = 256, ttl_s: int = 300) -> None:
        self._max = max_entries
        self._ttl = ttl_s
        self._d: "OrderedDict[Hashable, CacheEntry]" = OrderedDict()

    def _evict(self):
        while len(self._d) > self._max:
            self._d.popitem(last=False)

    def get(self, key: Hashable):
        now = time.time()
        ent = self._d.get(key)
        if not ent: return None
        if self._ttl and now - ent.ts > self._ttl:
            self._d.pop(key, None)
            return None
        # touch
        self._d.move_to_end(key, last=True)
        return ent.value

    def put(self, key: Hashable, value: Any):
        self._d[key] = CacheEntry(ts=time.time(), value=value)
        self._d.move_to_end(key, last=True)
        self._evict()

def stable_key(obj) -> str:
    """JSON-stable key for dicts/lists."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))
