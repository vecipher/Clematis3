from types import SimpleNamespace
import json
import os

import pytest

from clematis.engine.apply import apply_changes
from clematis.engine.types import T4Result, ProposedDelta


# ------------- helpers -------------


def mk_ctx(t4_overrides=None, turn=0, agent="Ambrose"):
    base = {
        "enabled": True,
        "weight_min": -1.0,
        "weight_max": 1.0,
        "snapshot_every_n_turns": 10_000,  # default: effectively off for tests unless overridden
        "snapshot_dir": "./.data/snapshots",
    }
    if t4_overrides:
        base.update(t4_overrides)
    return SimpleNamespace(turn_id=turn, agent_id=agent, config=SimpleNamespace(t4=base))


def mk_state(store=None, version=None):
    if store is None:
        return SimpleNamespace(store=None, version_etag=version)
    return SimpleNamespace(store=store, version_etag=version)


def mk_delta(key: str, val: float, attr="weight", op_idx=None):
    if key.startswith("n:"):
        kind = "node"
        tid = key.split("n:", 1)[1]
        target_id = f"n:{tid}"
    elif key.startswith("e:"):
        kind = "edge"
        tid = key.split("e:", 1)[1]
        target_id = f"e:{tid}"
    else:
        raise ValueError("key must start with 'n:' or 'e:'")
    return ProposedDelta(target_kind=kind, target_id=target_id, attr=attr, delta=val, op_idx=op_idx)


def mk_t4(deltas):
    return T4Result(approved_deltas=deltas, rejected_ops=[], reasons=[], metrics={})


class FakeStore:
    """
    Minimal in-memory store that supports apply_deltas(graph_id, deltas)
    - Applies additive deltas to an internal weight map (default weight=0.0)
    - Clamps to [wmin, wmax] and counts clamps when clamping occurred
    - Returns dict with {"edits": <num actually changed>, "clamps": <num clamped>}
    - Is idempotent: re-applying same deltas that do not change weights returns edits=0
    """

    def __init__(self, wmin=-1.0, wmax=1.0):
        self.w = {}  # key: (kind,id,attr) -> weight
        self.wmin = float(wmin)
        self.wmax = float(wmax)

    def _ckey(self, d: ProposedDelta):
        return (d.target_kind, d.target_id, d.attr)

    def apply_deltas(self, graph_id, deltas):
        edits = 0
        clamps = 0
        for d in deltas:
            k = self._ckey(d)
            old = self.w.get(k, 0.0)
            proposed = old + float(d.delta)
            clamped = max(self.wmin, min(self.wmax, proposed))
            if clamped != proposed:
                clamps += 1
            # If no change after clamping, it's not an edit (idempotence)
            if clamped != old:
                self.w[k] = clamped
                edits += 1
        return {"edits": edits, "clamps": clamps}


# ------------- tests -------------


def test_apply_clamps_and_increments_version(tmp_path):
    # Configure tight bounds so one delta clamps; set turn=1 so cadence does not fire at turn 0
    ctx = mk_ctx({"weight_min": -0.2, "weight_max": 0.2}, turn=1)
    store = FakeStore(wmin=-0.2, wmax=0.2)
    state = mk_state(store=store, version=None)

    deltas = [
        mk_delta("n:a", 0.5),  # will clamp from 0.0+0.5 -> 0.2
        mk_delta("n:b", -0.1),  # within range
    ]
    res = apply_changes(ctx, state, mk_t4(deltas))

    assert res.applied == 2
    assert res.clamps == 1
    assert res.version_etag == "1"
    assert res.snapshot_path is None  # cadence default is high


def test_apply_additive_semantics_second_apply_edits_again():
    ctx = mk_ctx()
    store = FakeStore()
    state = mk_state(store=store, version="41")

    deltas = [
        mk_delta("n:a", 0.25),
        mk_delta("n:b", -0.50),
    ]
    # First apply: should edit 2
    res1 = apply_changes(ctx, state, mk_t4(deltas))
    assert res1.applied == 2
    assert res1.version_etag == "42"

    # Second apply of same deltas: weights will be incremented again (additive semantics)
    res2 = apply_changes(ctx, state, mk_t4(deltas))
    assert res2.applied == 2
    assert res2.version_etag == "43"
    # Ensure store weights are incremented
    assert store.w[("node", "n:a", "weight")] == pytest.approx(0.50)
    assert store.w[("node", "n:b", "weight")] == pytest.approx(-1.00)


def test_snapshot_written_when_cadence_hits(tmp_path):
    snap_dir = tmp_path / "snaps"
    ctx = mk_ctx(
        {"snapshot_every_n_turns": 1, "snapshot_dir": str(snap_dir)}, turn=5, agent="Kafka"
    )
    store = FakeStore()
    state = mk_state(store=store, version=None)

    deltas = [mk_delta("n:x", 0.1)]
    res = apply_changes(ctx, state, mk_t4(deltas))

    # Verify a snapshot file exists in the directory
    assert res.snapshot_path is not None
    assert os.path.isdir(str(snap_dir))
    assert os.path.isfile(res.snapshot_path)

    # Snapshot should be valid JSON and include version_etag & applied
    with open(res.snapshot_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert data.get("version_etag") == res.version_etag
    assert data.get("applied") == 1
