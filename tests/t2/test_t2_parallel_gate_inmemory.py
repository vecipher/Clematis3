from __future__ import annotations

import numpy as np

from clematis.engine.stages.t2 import _t2_parallel_enabled
from clematis.memory.index import InMemoryIndex


def _add_episode(idx: InMemoryIndex, eid: str) -> None:
    idx.add(
        {
            "id": eid,
            "owner": "A",
            "ts": "2025-01-01T00:00:00Z",
            "text": "stub",
            "vec_full": np.ones(4, dtype=np.float32),
        }
    )


def _cfg(max_workers: int) -> dict:
    return {
        "perf": {
            "parallel": {
                "enabled": True,
                "t2": True,
                "max_workers": max_workers,
            }
        }
    }


def test_t2_parallel_gate_enables_for_multiple_shards():
    idx = InMemoryIndex()
    for i in range(3):
        _add_episode(idx, f"ep{i}")

    assert _t2_parallel_enabled(_cfg(4), "inmemory", idx) is True


def test_t2_parallel_gate_requires_multiple_workers():
    idx = InMemoryIndex()
    for i in range(3):
        _add_episode(idx, f"ep{i}")

    # With only one worker, fan-out remains disabled even with shards available.
    assert _t2_parallel_enabled(_cfg(1), "inmemory", idx) is False
