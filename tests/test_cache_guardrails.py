import threading
import pytest

from clematis.engine.cache import CacheManager, stable_key, LRUCache, ThreadSafeCache, ThreadSafeBytesCache
from clematis.engine.util.lru_bytes import LRUBytes


class FakeClock:
    def __init__(self, t0: float = 0.0):
        self._t = float(t0)

    def time(self) -> float:
        return self._t

    def advance(self, dt: float) -> None:
        self._t += float(dt)

#
# Legacy LRU behavior: capacity eviction and MRU promotion remain correct.
#
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

#
# TTL pruning occurs on access; expired entries are removed when read.
#
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

#
# Namespaced invalidation removes entries deterministically.
#
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

#
# stable_key normalizes unhashable composite keys (dicts/lists) to a deterministic string.
#
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


# --- PR65 guardrails: thread-safe wrappers are crash-free and deterministic (no timing asserts) ---

def test_threadsafe_cache_basic_concurrency():
    """
    Shared LRUCache wrapped with a lock: concurrent writers/readers must not crash.
    We don't assert exact sizes due to LRU behavior; we only check basic consistency.
    """
    base = LRUCache(max_entries=10_000, ttl_s=600)
    cache = ThreadSafeCache(base)

    N_THREADS = 4
    N_WRITES = 200

    def worker(start: int):
        for i in range(N_WRITES):
            k = ("k", start + i)
            v = f"v-{start+i}"
            cache.put(k, v)
            got = cache.get(k)
            assert got == v

    threads = [threading.Thread(target=worker, args=(i * N_WRITES,)) for i in range(N_THREADS)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # Snapshot should be non-empty and iterable without raising.
    assert len(list(cache.items())) > 0


def test_threadsafe_bytes_cache_basic_concurrency():
    """
    Shared byte-budget cache wrapped with a lock: concurrent puts/gets are safe.
    Evictions may occur based on byte budget; we only require no crashes and that
    items() produces a stable snapshot.
    """
    base = LRUBytes(max_entries=10_000, max_bytes=256_000)
    cache = ThreadSafeBytesCache(base)

    N_THREADS = 4
    N_WRITES = 150

    def worker(start: int):
        for i in range(N_WRITES):
            k = f"kb-{start+i}".encode()
            v = (f"vb-{start+i}").encode()
            cache.put(k, v, cost=len(v))
            _ = cache.get(k)  # may be None if evicted; that's fine

    threads = [threading.Thread(target=worker, args=(i * N_WRITES,)) for i in range(N_THREADS)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # Iteration must be stable and not raise, regardless of internal evictions.
    _ = list(cache.items())
