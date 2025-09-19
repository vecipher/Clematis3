

from types import SimpleNamespace
import json
import os

import pytest

from clematis.engine.apply import apply_changes, load_latest_snapshot
from clematis.engine.types import ProposedDelta, T4Result


# -----------------
# Helper builders
# -----------------

def mk_ctx(turn=0, agent="Tester", snapshot_dir=None, t4_overrides=None):
    cfg = {
        "enabled": True,
        "delta_norm_cap_l2": 1.5,
        "novelty_cap_per_node": 0.3,
        "churn_cap_edges": 64,
        "cooldowns": {},
        "weight_min": -1.0,
        "weight_max": 1.0,
        "snapshot_every_n_turns": 1,  # force snapshot by default
        "snapshot_dir": snapshot_dir or "./.data/snapshots",
        "cache_bust_mode": "on-apply",
    }
    if t4_overrides:
        cfg.update(t4_overrides)
    return SimpleNamespace(turn_id=turn, agent_id=agent, config=SimpleNamespace(t4=cfg))


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


# -----------------
# Fake stores
# -----------------

class ExportImportStore:
    """
    Store that supports export_state / import_state for snapshots.
    Keeps weights in a .w dict keyed by (target_kind, target_id, attr).
    """
    def __init__(self, wmin=-1.0, wmax=1.0):
        self.w = {}
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
            if clamped != old:
                self.w[k] = clamped
                edits += 1
        return {"edits": edits, "clamps": clamps}

    def export_state(self):
        # Serialize weights into a structured dict
        return {
            "weights": [
                {"target_kind": tk, "target_id": tid, "attr": attr, "value": val}
                for (tk, tid, attr), val in self.w.items()
            ]
        }

    def import_state(self, state_obj):
        self.w.clear()
        weights = (state_obj or {}).get("weights", [])
        for item in weights:
            tk = str(item.get("target_kind"))
            tid = str(item.get("target_id"))
            attr = str(item.get("attr"))
            val = float(item.get("value", 0.0))
            self.w[(tk, tid, attr)] = val


class WeightsOnlyStore:
    """
    Store without export/import; snapshot should fall back to .w dict weights.
    """
    def __init__(self, wmin=-1.0, wmax=1.0):
        self.w = {}
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
            if clamped != old:
                self.w[k] = clamped
                edits += 1
        return {"edits": edits, "clamps": clamps}


# -----------------
# Tests
# -----------------

def test_loader_with_export_import_store(tmp_path):
    snap_dir = tmp_path / "snaps"
    ctx = mk_ctx(turn=0, agent="Alice", snapshot_dir=str(snap_dir))
    store = ExportImportStore()
    state = SimpleNamespace(store=store, version_etag=None)

    # Create a snapshot by applying deltas
    deltas = [mk_delta("n:a", 0.2), mk_delta("e:s|rel|d", -0.1)]
    res = apply_changes(ctx, state, mk_t4(deltas))
    assert res.snapshot_path is not None and os.path.isfile(res.snapshot_path)

    # Mutate store and version to ensure loader actually restores
    store.w[("node", "n:a", "weight")] = 999.0
    state.version_etag = "999"

    info = load_latest_snapshot(ctx, state)
    assert info["loaded"] is True
    assert info["path"] == res.snapshot_path
    assert info["version_etag"] == "1"  # from first apply bump

    # Weights restored from snapshot
    assert store.w[("node", "n:a", "weight")] == pytest.approx(0.2)
    assert store.w[("edge", "e:s|rel|d", "weight")] == pytest.approx(-0.1)


def test_loader_with_weights_fallback(tmp_path):
    snap_dir = tmp_path / "snaps"
    ctx = mk_ctx(turn=0, agent="Bob", snapshot_dir=str(snap_dir))
    store = WeightsOnlyStore()
    state = SimpleNamespace(store=store, version_etag=None)

    deltas = [mk_delta("n:x", 0.3)]
    res = apply_changes(ctx, state, mk_t4(deltas))
    assert res.snapshot_path is not None and os.path.isfile(res.snapshot_path)

    # Corrupt weights then load
    store.w[("node", "n:x", "weight")] = -777.0
    info = load_latest_snapshot(ctx, state)

    assert info["loaded"] is True
    assert store.w[("node", "n:x", "weight")] == pytest.approx(0.3)
    assert info["version_etag"] == "1"


def test_loader_missing_snapshot_returns_false(tmp_path):
    snap_dir = tmp_path / "snaps"
    ctx = mk_ctx(turn=0, agent="Ghost", snapshot_dir=str(snap_dir))
    state = SimpleNamespace(store=None, version_etag=None)

    info = load_latest_snapshot(ctx, state)
    assert info["loaded"] is False
    assert info["path"] is None
    assert info["version_etag"] is None


def test_loader_idempotent_twice(tmp_path):
    snap_dir = tmp_path / "snaps"
    ctx = mk_ctx(turn=0, agent="Idem", snapshot_dir=str(snap_dir))
    store = WeightsOnlyStore()
    state = SimpleNamespace(store=store, version_etag=None)

    # Create snapshot
    deltas = [mk_delta("n:z", 0.4)]
    res = apply_changes(ctx, state, mk_t4(deltas))
    assert os.path.isfile(res.snapshot_path)

    # Load twice; second load should be a no-op and not error
    info1 = load_latest_snapshot(ctx, state)
    assert info1["loaded"] is True
    w1 = dict(store.w)

    info2 = load_latest_snapshot(ctx, state)
    assert info2["loaded"] in (True, False)  # loader may report True again depending on path; both acceptable
    assert store.w == w1
    assert state.version_etag == "1"