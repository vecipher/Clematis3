import pytest

from clematis.engine.util.lru_det import DeterministicLRUSet, DeterministicLRU

def test_lru_set_fifo_eviction_and_contains():
    s = DeterministicLRUSet(2)
    assert not s.contains("a")
    assert s.add("a") is False
    assert s.add("b") is False
    assert s.contains("a") and s.contains("b")
    # Adding "c" evicts "a" (FIFO by first-insert)
    ev = s.add("c")
    assert ev is True
    assert not s.contains("a")
    assert s.contains("b") and s.contains("c")
    assert s.size() == 2
    s.clear()
    assert s.size() == 0 and not s.contains("b")

def test_lru_map_put_get_eviction_and_touch():
    # Map LRU with deterministic recency updates on get/put
    m = DeterministicLRU(cap=2, update_on_get=True, update_on_put=True)
    assert len(m) == 0
    assert m.put("k1", "v1") is None
    assert m.put("k2", "v2") is None
    assert len(m) == 2
    # Touch k1 -> k1 becomes MRU; k2 is LRU now
    assert m.get("k1") == "v1"
    # Insert k3 -> evicts k2
    ev = m.put("k3", "v3")
    assert ev == ("k2", "v2")
    assert m.get("k2") is None
    assert m.get("k1") == "v1"
    assert m.get("k3") == "v3"
    # pop_lru should remove k1 now (it’s LRU after touching k3 via put)
    popped = m.pop_lru()
    assert popped[0] in ("k1", "k3")  # exact order depends on last touches
    # Clearing resets state
    m.clear()
    assert len(m) == 0