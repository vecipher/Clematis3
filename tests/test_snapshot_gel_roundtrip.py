import json
import math
import os
import time
import pytest

_gel = pytest.importorskip("clematis.engine.gel")
_snapshot = pytest.importorskip("clematis.engine.snapshot")

observe_retrieval = _gel.observe_retrieval
write_snapshot = _snapshot.write_snapshot
load_latest_snapshot = _snapshot.load_latest_snapshot


class _Ctx:
    def __init__(self, snapshot_dir: str):
        self.config = {
            "t4": {
                "snapshot_dir": snapshot_dir,
            },
            # graph config not required for snapshot I/O, but enable for clarity
            "graph": {"enabled": True},
        }
        self.cfg = self.config


class _State:
    pass


def _edge_map(state):
    return getattr(state, "graph", {}).get("edges", {})


def test_gel_snapshot_roundtrip(tmp_path):
    """Populate GEL edges, write a snapshot to tmp, load it back, and compare."""
    snap_dir = tmp_path
    ctx = _Ctx(str(snap_dir))

    # 1) Build a tiny graph via observe_retrieval
    s1 = _State()
    items = [("a", 0.95), ("b", 0.90), ("c", 0.10)]  # c is below default threshold in gel.py(0.20)
    observe_retrieval(ctx, s1, items, turn=1, agent="A")  # creates edge a→b only

    edges_before = dict(_edge_map(s1))
    assert "a→b" in edges_before and edges_before["a→b"]["weight"] > 0.0

    # 2) Write snapshot using agent id; file name is determined by snapshot layer
    write_snapshot(ctx, s1, "AgentA")
    # Find the snapshot we just wrote
    candidates = sorted(snap_dir.glob("state_*.json"))
    assert candidates, "snapshot file not created"
    snap_path = candidates[-1]

    # quick schema sanity
    payload = json.loads(snap_path.read_text("utf-8"))
    assert payload.get("graph_schema_version") in ("v1", "v1.1", "v1.0")
    assert "gel" in payload

    # 3) Load into a fresh state via loader (which finds latest in dir)
    s2 = _State()
    load_latest_snapshot(ctx, s2)

    edges_after = _edge_map(s2)
    assert "a→b" in edges_after
    # weights should match exactly (no decay applied by loader)
    assert math.isclose(edges_before["a→b"]["weight"], edges_after["a→b"]["weight"], abs_tol=1e-12)

    # both graph and gel should be set by loader
    assert hasattr(s2, "graph")
    assert hasattr(s2, "gel")
    assert s2.graph == s2.gel


def test_loader_tolerates_missing_gel_block(tmp_path):
    """If a legacy snapshot has only a compact `graph` summary, loader should not crash and produce empty maps."""
    snap_dir = tmp_path
    ctx = _Ctx(str(snap_dir))

    legacy = {
        "graph_schema_version": "v1",
        # No 'gel' block here on purpose
        "graph": {"nodes_count": 0, "edges_count": 0, "meta": {"last_update": None}},
        "version_etag": "42",
    }
    p = snap_dir / "state_AgentA.json"
    p.write_text(json.dumps(legacy), encoding="utf-8")

    s = _State()
    load_latest_snapshot(ctx, s)
    assert hasattr(s, "graph") and isinstance(s.graph, dict)
    assert s.graph.get("nodes") == {}
    assert s.graph.get("edges") == {}
    # gel mirror also present
    assert hasattr(s, "gel") and s.gel == s.graph