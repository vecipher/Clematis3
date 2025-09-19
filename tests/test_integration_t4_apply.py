

from types import SimpleNamespace
import json
import os

import pytest

from clematis.engine.stages import t4_filter
from clematis.engine.apply import apply_changes
from clematis.engine.types import ProposedDelta, T4Result
import clematis.engine.orchestrator as orch


# ---- minimal store used for integration ----

class MiniStore:
    """
    Tiny apply target for integration:
    - Maintains weights in-memory keyed by (kind, id, attr)
    - Supports apply_deltas(graph_id, deltas) with additive semantics
    - Reports edits/clamps like a real store would
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


# ---- helpers ----

def mk_ctx(turn, agent, t4_overrides=None, snapshot_dir=None):
    t4_cfg = {
        "enabled": True,
        "delta_norm_cap_l2": 1e6,       # disable L2 scaling for this integration
        "novelty_cap_per_node": 1e6,    # disable novelty clamp
        "churn_cap_edges": 64,          # no trimming
        "weight_min": -1.0,
        "weight_max": 1.0,
        "snapshot_every_n_turns": 1,    # force snapshot
        "snapshot_dir": snapshot_dir or "./.data/snapshots",
        "cache_bust_mode": "on-apply",
        "cooldowns": {},
    }
    if t4_overrides:
        t4_cfg.update(t4_overrides)
    return SimpleNamespace(turn_id=turn, agent_id=agent, config=SimpleNamespace(t4=t4_cfg))


def mk_state(store):
    return SimpleNamespace(store=store, version_etag=None)


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


def mk_plan(ops=None, deltas=None):
    return {"ops": ops or [], "deltas": deltas or []}


# ---- test ----

def test_t4_to_apply_end_to_end(tmp_path):
    # Arrange: temp snapshot dir and a mini in-memory store
    snap_dir = tmp_path / "snaps"
    store = MiniStore()
    ctx = mk_ctx(turn=2, agent="Ambrose", snapshot_dir=str(snap_dir))
    state = mk_state(store)

    # Plan with two deltas: one node, one edge
    ops = [{"kind": "EditGraph"}]
    deltas = [
        mk_delta("n:x", 0.2, op_idx=0),
        mk_delta("e:s|rel|d", -0.1, op_idx=0),
    ]
    plan = mk_plan(ops=ops, deltas=deltas)

    # Act: T4 gate then Apply
    t4 = t4_filter(ctx, state, t1=None, t2=None, plan=plan, utter=None)

    # Expect: with caps disabled (large), both pass; no reasons
    assert len(t4.approved_deltas) == 2
    assert t4.reasons == []

    res = apply_changes(ctx, state, t4)

    # Assert: both edits applied, no clamps, version incremented, snapshot written
    assert res.applied == 2
    assert res.clamps == 0
    assert res.version_etag == "1"
    assert res.snapshot_path is not None
    assert os.path.isfile(res.snapshot_path)

    # Snapshot should be valid JSON and reflect applied count
    with open(res.snapshot_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert data.get("applied") == 2
    assert data.get("version_etag") == res.version_etag

    # Weights updated as expected
    assert store.w[("node", "n:x", "weight")] == pytest.approx(0.2)
    assert store.w[("edge", "e:s|rel|d", "weight")] == pytest.approx(-0.1)

def test_kill_switch_bypasses_t4_and_apply(monkeypatch):
    calls = []

    # Capture any logging attempts
    def fake_append_jsonl(name, payload):
        calls.append(("log", name, payload))

    # Fake T1/T2 stages (return minimal objects)
    monkeypatch.setattr(orch, "append_jsonl", fake_append_jsonl, raising=True)
    monkeypatch.setattr(orch, "t1_propagate", lambda ctx, state, text: SimpleNamespace(metrics={"t1": True}), raising=True)
    monkeypatch.setattr(orch, "t2_semantic", lambda ctx, state, text, t1: SimpleNamespace(metrics={"t2": True}), raising=True)

    # Fake T3 deliberate/dialogue to provide plan + utter
    def fake_t3_deliberate(ctx, state, bundle):
        # Minimal plan with no ops/deltas
        return SimpleNamespace(version="t3-plan-v1", ops=[], deltas=[], reflection=False)

    def fake_t3_dialogue(dialog_bundle, plan):
        return "HELLO_W0RLD"

    # Some codebases hang T3 helpers under orch directly; others import from a t3 module.
    # We patch both names defensively.
    monkeypatch.setattr(orch, "t3_deliberate", fake_t3_deliberate, raising=False)
    monkeypatch.setattr(orch, "t3_dialogue", fake_t3_dialogue, raising=False)

    # Ensure T4 and Apply would explode if called (to prove bypass)
    def explode(*args, **kwargs):
        raise AssertionError("T4/Apply should not be called when t4.enabled is False")

    monkeypatch.setattr(orch, "t4_filter", explode, raising=False)
    monkeypatch.setattr(orch, "apply_changes", explode, raising=False)

    # Build ctx/state with kill switch disabled
    ctx = SimpleNamespace(
        turn_id=3,
        agent_id="Ambrose",
        config=SimpleNamespace(t4={"enabled": False, "snapshot_every_n_turns": 1000}),
    )
    state = SimpleNamespace(store=None)  # store irrelevant when bypassed

    # Act
    result = orch.run_turn(ctx, state, input_text="hi")

    # Assert: utterance comes from T3 path
    assert getattr(result, "line", None) == "HELLO_W0RLD"

    # Assert: no T4 or apply logs were written
    t4_logs = [c for c in calls if c[0] == "log" and c[1] == "t4.jsonl"]
    apply_logs = [c for c in calls if c[0] == "log" and c[1] == "apply.jsonl"]
    assert t4_logs == []
    assert apply_logs == []