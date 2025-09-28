import pytest
from types import SimpleNamespace

from clematis.engine.cache import LRUCache


class FakeClock:
    def __init__(self, t0: float = 0.0):
        self._t = float(t0)

    def time(self) -> float:
        return self._t

    def advance(self, dt: float) -> None:
        self._t += float(dt)


def test_get_returns_value_or_none_and_get2_tuple():
    c = LRUCache(max_entries=2, ttl_s=1000)
    # Miss returns None (legacy behavior)
    assert c.get("k1") is None
    # Set then get returns the value
    c.set("k1", "v1")
    assert c.get("k1") == "v1"
    # get2 returns (hit, value)
    hit, val = c.get2("k1")
    assert hit is True and val == "v1"


def test_ttl_s_is_honored_and_expires_on_get():
    clock = FakeClock()
    c = LRUCache(max_entries=2, ttl_s=10, time_fn=clock.time)
    c.set("k", "v")
    assert c.get("k") == "v"
    # Advance past TTL -> next get returns None and removes the entry
    clock.advance(11.0)
    assert c.get("k") is None
    assert len(c) == 0


def test_capacity_eviction_and_put_alias():
    c = LRUCache(max_entries=2, ttl_s=1000)
    c.set("a", 1)
    c.set("b", 2)
    # Touch "b" so "a" becomes LRU
    _ = c.get("b")
    # Insert "c" via put alias -> evict "a"
    c.put("c", 3)
    assert c.get("a") is None
    assert c.get("b") == 2
    assert c.get("c") == 3
    # Size remains at capacity
    assert len(c) == 2


def test_clear_invalidate_returns_removed_count():
    c = LRUCache(max_entries=3, ttl_s=1000)
    c.set("x", "X")
    c.set("y", "Y")
    before = len(c)
    removed = c.clear()
    assert removed == before
    assert len(c) == 0
    # idempotent: clearing again removes 0
    assert c.clear() == 0


def test_capacity_alias_via_capacity_param():
    c = LRUCache(capacity=1, ttl_s=1000)
    c.set("u", "U")
    # Adding a second entry evicts the first due to capacity=1
    c.set("v", "V")
    assert c.get("u") is None
    assert c.get("v") == "V"
    assert len(c) == 1
