

import pytest

from clematis.engine.util.ring import DedupeRing


def test_dedupe_ring_basic_membership_and_eviction():
    r = DedupeRing(2)  # capacity 2
    # Initially empty
    assert not r.contains("a")

    # Fill to capacity
    r.add("a")
    r.add("b")
    assert r.contains("a")
    assert r.contains("b")

    # Adding "c" evicts the oldest ("a")
    r.add("c")
    assert not r.contains("a")
    assert r.contains("b")
    assert r.contains("c")

    # Add a duplicate "c": this evicts "b" (to make room), and increases refcount for "c"
    r.add("c")
    assert not r.contains("b")
    assert r.contains("c")

    # Adding "d" evicts one "c" from the queue, but membership persists (refcount > 0)
    r.add("d")
    assert r.contains("c")  # duplicate ensured membership survived a single eviction
    assert r.contains("d")


def test_dedupe_ring_discard_and_clear():
    r = DedupeRing(3)
    # Add two copies of 'z'
    r.add("z")
    r.add("z")
    assert r.contains("z")

    # Discard once -> still present due to refcount>0
    r.discard("z")
    assert r.contains("z")

    # Discard again -> gone
    r.discard("z")
    assert not r.contains("z")

    # Add a few, then clear
    r.add("x"); r.add("y"); r.add("z")
    assert r.contains("x") and r.contains("y") and r.contains("z")
    r.clear()
    assert len(r) == 0
    assert not r.contains("x") and not r.contains("y") and not r.contains("z")