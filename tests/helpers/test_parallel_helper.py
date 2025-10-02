# tests/helpers/test_parallel_helper.py
from __future__ import annotations
from typing import List, Tuple
import pytest

from clematis.engine.util.parallel import run_parallel, ParallelError


def _merge_passthrough(pairs: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    # Identity aggregator used for assertions in tests.
    return pairs


def test_sequential_identity_equivalence():
    tasks = [(k, (lambda v=k: v)) for k in [3, 1, 2, 1]]
    # Sequential path
    seq = run_parallel(tasks, max_workers=1, merge_fn=_merge_passthrough, order_key=lambda k: (k,))
    # Parallel path (workers > 1); result order MUST match deterministic spec
    par = run_parallel(tasks, max_workers=4, merge_fn=_merge_passthrough, order_key=lambda k: (k,))
    assert seq == [(1, 1), (1, 1), (2, 2), (3, 3)]
    assert par == seq


def test_ordering_independent_of_completion():
    # Simulate skewed completion by doing different amounts of work.
    def make_task(k: int):
        def thunk():
            total = 0
            # Busy loop length depends on k; completion order will differ from key order.
            for _ in range(1000 * (4 - k)):
                total += 1
            return k

        return thunk

    tasks = [(k, make_task(k)) for k in [3, 1, 2, 4, 1]]
    out = run_parallel(tasks, max_workers=4, merge_fn=_merge_passthrough, order_key=lambda k: (k,))
    assert out == [(1, 1), (1, 1), (2, 2), (3, 3), (4, 4)]


def test_exceptions_are_deterministic():
    def ok(x: int):
        return x

    def boom(tag: str):
        def _():
            raise ValueError(f"bad-{tag}")

        return _

    tasks = [
        (2, (lambda: ok(2))),  # returns 2
        (1, boom("A")),  # fails
        (3, boom("B")),  # fails
        (1, (lambda: ok(1))),  # returns 1
    ]

    with pytest.raises(ParallelError) as ei:
        _ = run_parallel(
            tasks, max_workers=4, merge_fn=_merge_passthrough, order_key=lambda k: (k,)
        )
    err = ei.value
    # Errors sorted by order_key(key)=1 before key=3; two key=1 errors would be
    # tie-broken by submit index (but here only one key=1 error).
    keys = [e.key for e in err.errors]
    assert keys == [1, 3]
    types = [e.exc_type for e in err.errors]
    assert types == ["ValueError", "ValueError"]
    msgs = [e.message for e in err.errors]
    assert msgs[0].startswith("bad-") and msgs[1].startswith("bad-")


def test_max_workers_normalization_zero_and_negative():
    tasks = [(k, (lambda v=k: v)) for k in [2, 1]]
    a = run_parallel(tasks, max_workers=0, merge_fn=_merge_passthrough, order_key=lambda k: (k,))
    b = run_parallel(tasks, max_workers=-5, merge_fn=_merge_passthrough, order_key=lambda k: (k,))
    assert a == [(1, 1), (2, 2)]
    assert b == a


def test_empty_tasks_calls_merge_with_empty_list():
    called = {"n": 0}

    def merge(xs):
        called["n"] += 1
        assert xs == []
        return []

    out = run_parallel([], max_workers=8, merge_fn=merge, order_key=lambda k: (k,))
    assert out == []
    assert called["n"] == 1
