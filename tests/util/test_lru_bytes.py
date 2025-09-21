

import pytest

from clematis.engine.util.lru_bytes import LRUBytes


def test_put_get_basic_and_sizes():
    c = LRUBytes(max_entries=3, max_bytes=100)
    assert c.size_entries() == 0
    assert c.size_bytes() == 0

    ev = c.put("a", "A", 10)
    assert ev == (0, 0)
    ev = c.put("b", "B", 20)
    assert ev == (0, 0)
    ev = c.put("c", "C", 30)
    assert ev == (0, 0)

    assert c.size_entries() == 3
    assert c.size_bytes() == 60
    assert c.get("a") == "A"
    assert c.get("b") == "B"
    assert c.get("c") == "C"

    # Adding within caps should not evict
    ev = c.put("d", "D", 40)  # entries cap triggers: 4 -> 3 (evict LRU 'a')
    assert ev[0] == 1  # one eviction
    # bytes now should be 60 - 10 + 40 = 90 OR 100 if eviction order removed 'a'
    assert c.size_entries() == 3
    assert c.size_bytes() <= 100
    # 'a' should be gone; 'b','c','d' present
    assert c.get("a") is None
    assert c.get("b") == "B"
    assert c.get("c") == "C"
    assert c.get("d") == "D"


def test_reject_oversize_item():
    c = LRUBytes(max_entries=10, max_bytes=50)
    c.put("x", 1, 10)
    c.put("y", 2, 20)
    c.put("z", 3, 10)
    assert c.size_bytes() == 40
    # Oversized item (> max_bytes) is rejected; cache unchanged
    ev = c.put("big", object(), 60)
    assert ev == (0, 0)
    assert c.size_entries() == 3
    assert c.size_bytes() == 40
    assert c.get("big") is None


def test_multi_eviction_by_bytes_and_entries():
    # Tight caps to force multi-eviction
    c = LRUBytes(max_entries=3, max_bytes=50)
    c.put("a", 1, 10)
    c.put("b", 2, 10)
    c.put("c", 3, 10)
    assert c.size_entries() == 3 and c.size_bytes() == 30
    # Insert d=30 bytes → requires evicting at least one item (LRU='a'), maybe more if needed
    ev_n, ev_b = c.put("d", 4, 30)
    assert ev_n >= 1
    assert ev_b >= 10
    assert c.size_entries() <= 3
    assert c.size_bytes() <= 50
    # Oldest 'a' should be gone; 'b' likely next LRU if further evictions occurred
    assert c.get("a") is None
    assert c.get("d") == 4


def test_update_moves_to_mru_and_adjusts_bytes():
    c = LRUBytes(max_entries=2, max_bytes=100)
    c.put("k1", "v1", 10)
    c.put("k2", "v2", 10)
    assert c.size_entries() == 2
    assert c.size_bytes() == 20
    # Update k1 to larger object and move it to MRU
    ev = c.put("k1", "v1x", 30)
    assert ev == (0, 0)
    assert c.size_bytes() == 40  # 10 (k2) + 30 (k1)
    # Inserting k3 should evict LRU ('k2') since k1 was just touched
    ev_n, ev_b = c.put("k3", "v3", 10)
    assert ev_n == 1
    # k2 should be gone
    assert c.get("k2") is None
    assert c.get("k1") == "v1x"
    assert c.get("k3") == "v3"


def test_get_updates_recency_for_eviction():
    c = LRUBytes(max_entries=2, max_bytes=100)
    c.put("a", 1, 10)
    c.put("b", 2, 10)
    # Touch 'a' so that 'b' becomes LRU
    assert c.get("a") == 1
    # Insert 'c' → should evict 'b'
    ev_n, ev_b = c.put("c", 3, 10)
    assert ev_n == 1
    assert c.get("b") is None
    assert c.get("a") == 1
    assert c.get("c") == 3


def test_clear_resets_cache():
    c = LRUBytes(max_entries=5, max_bytes=100)
    c.put("a", 1, 10)
    c.put("b", 2, 10)
    c.put("c", 3, 10)
    assert len(c) == 3
    assert c.size_bytes() == 30
    c.clear()
    assert len(c) == 0
    assert c.size_bytes() == 0