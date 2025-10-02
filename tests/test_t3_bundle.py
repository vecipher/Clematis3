import pytest
from datetime import datetime, timezone

from clematis.engine.stages.t3 import make_plan_bundle


class Ctx:
    def __init__(self, now=None, cfg=None, agent_id="agentA", style_prefix="calm", input_text="hi"):
        self.now = now or datetime(2025, 9, 19, 0, 0, 0, tzinfo=timezone.utc)
        self.cfg = cfg or {
            "t3": {"tokens": 256, "max_ops_per_turn": 3, "max_rag_loops": 1, "temp": 0.7},
            "t2": {"owner_scope": "any", "k_retrieval": 3, "sim_threshold": 0.3},
        }
        self.agent_id = agent_id
        # mirror how t3 tries a few places for style
        self.agent = {"style_prefix": style_prefix}
        self.input_text = input_text


class Obj:
    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)


def _mk_t1(num=5):
    # Produce a list of node deltas: id n0..n{num-1}, delta increasing
    deltas = []
    for i in range(num):
        deltas.append({"id": f"n{i:02d}", "label": f"L{i}", "delta": float(i)})
    return Obj(
        graph_deltas=deltas,
        metrics={
            "pops": 10,
            "iters": 3,
            "propagations": 7,
            "radius_cap_hits": 0,
            "layer_cap_hits": 1,
            "node_budget_hits": 0,
        },
    )


def _mk_t2():
    # Mix of dict and object hits; include one missing owner/quarter to test defaults
    retrieved = [
        {"id": "r2", "score": 0.7, "owner": "A", "quarter": "2025Q3"},
        Obj(id="r1", score=0.9, owner="A", quarter="2025Q3"),
        {"id": "r3", "score": 0.7},  # missing owner/quarter
        Obj(id="r4", score=0.2, owner="B", quarter="2025Q2"),
    ]
    metrics = {
        "tier_sequence": ["exact_semantic", "cluster_semantic"],
        "k_returned": len(retrieved),
        "sim_stats": {"mean": 0.625, "max": 0.9},
        "cache_used": True,
    }
    return Obj(retrieved=retrieved, metrics=metrics)


def test_bundle_determinism_and_schema():
    ctx = Ctx()
    state = {"world_hot_labels": ["x", "y", "x"]}
    t1 = _mk_t1(5)
    t2 = _mk_t2()

    b1 = make_plan_bundle(ctx, state, t1, t2)
    b2 = make_plan_bundle(ctx, state, t1, t2)

    assert b1 == b2, "Bundle must be deterministic for identical inputs"

    # Top-level keys
    for key in ["version", "now", "agent", "world", "t1", "t2", "text", "cfg"]:
        assert key in b1

    # Defaults and shapes
    assert isinstance(b1["world"]["hot_labels"], list)
    assert isinstance(b1["t1"]["touched_nodes"], list)
    assert isinstance(b1["t1"]["metrics"], dict)
    assert isinstance(b1["t2"]["retrieved"], list)
    assert isinstance(b1["t2"]["metrics"], dict)
    assert isinstance(b1["text"]["labels_from_t1"], list)


def test_t1_touched_nodes_sorted_and_capped():
    ctx = Ctx()
    state = {}
    # Create 40 nodes; cap is 32 → expect ids n08..n39 in ascending order
    t1 = _mk_t1(40)
    t2 = _mk_t2()

    b = make_plan_bundle(ctx, state, t1, t2)
    nodes = b["t1"]["touched_nodes"]
    assert len(nodes) == 32, "Touched nodes must be capped at 32"

    ids = [n["id"] for n in nodes]
    assert ids[0] == "n08" and ids[-1] == "n39", (
        f"Unexpected ID range/order: {ids[:5]}...{ids[-5:]}"
    )


def test_t2_retrieved_sorted_and_trimmed():
    # Set k_retrieval=2 and ensure we keep top 2 by score, tie-break by id
    ctx = Ctx(
        cfg={
            "t3": {"tokens": 256, "max_ops_per_turn": 3, "max_rag_loops": 1, "temp": 0.7},
            "t2": {"owner_scope": "any", "k_retrieval": 2, "sim_threshold": 0.3},
        }
    )
    state = {}
    t1 = _mk_t1(3)
    t2 = _mk_t2()

    b = make_plan_bundle(ctx, state, t1, t2)
    hits = b["t2"]["retrieved"]

    # Expect r1 (0.9) first, then among (r2, r3) with 0.7 choose id-asc → r2, then r3 is dropped due to k=2
    assert [h["id"] for h in hits] == ["r1", "r2"], f"Unexpected retrieved order/trim: {hits}"


def test_config_snapshot_reflects_ctx_cfg():
    ctx = Ctx(
        cfg={
            "t3": {"tokens": 123, "max_ops_per_turn": 9, "max_rag_loops": 1, "temp": 0.1},
            "t2": {"owner_scope": "world", "k_retrieval": 7, "sim_threshold": 0.01},
        }
    )
    b = make_plan_bundle(ctx, {}, _mk_t1(2), _mk_t2())
    assert b["cfg"]["t3"]["tokens"] == 123
    assert b["cfg"]["t2"]["k_retrieval"] == 7


def test_missing_fields_defaults_in_retrieved():
    ctx = Ctx()
    b = make_plan_bundle(ctx, {}, _mk_t1(2), _mk_t2())
    # Find r3 which lacked owner/quarter and ensure defaults were filled
    r = next((x for x in b["t2"]["retrieved"] if x["id"] == "r3"), None)
    if r:
        assert r["owner"] == "any"
        assert r["quarter"] == ""


# --- PR4 placeholder ---
# The orchestrator imports `make_dialog_bundle` as a forward stub.
# Provide a deterministic, pure placeholder that can be replaced in PR7.
DIALOG_BUNDLE_VERSION = "t3-dialog-bundle-v1"


def make_dialog_bundle(ctx, state, t1, t2, plan=None) -> dict:
    """Minimal deterministic dialogue bundle placeholder.
    Pure: no I/O. For PR4 it mirrors `make_plan_bundle` and projects a small subset.
    """
    base = make_plan_bundle(ctx, state, t1, t2)
    return {
        "version": DIALOG_BUNDLE_VERSION,
        "now": base["now"],
        "agent": base["agent"],
        "text": base["text"],
        # Keep retrieved list as-is (already capped/sorted by make_plan_bundle)
        "retrieved": base["t2"]["retrieved"],
        # Lightweight plan summary if provided (kept minimal to avoid type import cycles)
        "plan_summary": {
            "has_plan": bool(plan is not None),
            "ops": int(len(getattr(plan, "ops", []) or [])),
        },
    }
