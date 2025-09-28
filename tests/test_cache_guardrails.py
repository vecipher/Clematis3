from types import SimpleNamespace
import pytest

from clematis.engine.cache import CacheManager, stable_key


class FakeClock:
    def __init__(self, t0: float = 0.0):
        self._t = float(t0)

    def time(self) -> float:
        return self._t

    def advance(self, dt: float) -> None:
        self._t += float(dt)


def test_lru_capacity_and_eviction_order():
    clock = FakeClock()
    cm = CacheManager(max_entries=3, ttl_sec=1000, time_fn=clock.time)
    ns = "ns"

    # Fill capacity
    cm.set(ns, ("v", "a"), "A")
    cm.set(ns, ("v", "b"), "B")
    cm.set(ns, ("v", "c"), "C")
    assert cm.stats["size"] == 3
    assert cm.stats["evicted"] == 0

    # Touch "b" to move it to MRU
    hit, val = cm.get(ns, ("v", "b"))
    assert hit and val == "B"

    # Insert "d" -> should evict the oldest ("a")
    cm.set(ns, ("v", "d"), "D")
    assert cm.stats["size"] == 3
    assert cm.stats["evicted"] == 1

    hit_a, _ = cm.get(ns, ("v", "a"))
    assert not hit_a  # "a" was evicted

    # Remaining keys should be present
    for k, v in [("b", "B"), ("c", "C"), ("d", "D")]:
        hit, val = cm.get(ns, ("v", k))
        assert hit and val == v


def test_ttl_expiry_removes_on_get():
    clock = FakeClock()
    cm = CacheManager(max_entries=4, ttl_sec=10, time_fn=clock.time)
    ns = "ttl"

    cm.set(ns, ("v1", "x"), "X")
    assert cm.stats["size"] == 1

    # Advance past TTL
    clock.advance(11.0)
    hit, val = cm.get(ns, ("v1", "x"))
    assert not hit and val is None

    # Entry should have been removed on access
    assert cm.stats["size"] == 0


def test_invalidate_namespace_and_all():
    clock = FakeClock()
    cm = CacheManager(max_entries=10, ttl_sec=600, time_fn=clock.time)

    # Two namespaces
    for k in ["a", "b"]:
        cm.set("ns1", ("v", k), k.upper())
    cm.set("ns2", ("v", "z"), "Z")

    assert cm.stats["size"] == 3

    removed_ns1 = cm.invalidate_namespace("ns1")
    assert removed_ns1 == 2
    assert cm.stats["size"] == 1

    removed_all = cm.invalidate_all()
    assert removed_all == 1
    assert cm.stats["size"] == 0


def test_stable_key_handles_unhashable_tuples_with_dicts():
    clock = FakeClock()
    cm = CacheManager(max_entries=5, ttl_sec=600, time_fn=clock.time)
    ns = "mix"

    # Tuple containing an unhashable dict -> should be normalized by stable_key
    unhashable_key = ("v2", {"a": 1, "b": [2, 3]})
    cm.set(ns, unhashable_key, 42)

    hit, val = cm.get(ns, ("v2", {"b": [2, 3], "a": 1}))  # different order
    assert hit and val == 42

    # sanity: direct stable_key also produces a deterministic string
    s1 = stable_key(unhashable_key)
    s2 = stable_key(("v2", {"b": [2, 3], "a": 1}))
    assert isinstance(s1, str) and s1 == s2
