import types

import pytest
from dataclasses import asdict
from datetime import datetime, timezone

from clematis.engine.stages.t3 import deliberate, policy


def _bundle(s_max=0.9, labels=None, n_nodes=6, ops_cap=3, tokens=256, owner_scope="any"):
    labels = labels if labels is not None else ["zeta", "alpha", "alpha"]
    # Build touched nodes with ids out of order; deltas >= 0.10 to pass eps_edit
    nodes = []
    for i, nid in enumerate(["b", "a", "c", "e", "d", "f", "g", "h"][:n_nodes]):
        nodes.append({"id": nid, "label": nid.upper(), "delta": 0.2 + 0.01 * i})

    now = datetime(2025, 9, 19, 0, 0, 0, tzinfo=timezone.utc).isoformat()
    return {
        "version": "t3-bundle-v1",
        "now": now,
        "agent": {"id": "agentA", "style_prefix": "", "caps": {"tokens": tokens, "ops": ops_cap}},
        "world": {"hot_labels": [], "k": 0},
        "t1": {
            "touched_nodes": nodes,
            "metrics": {
                "pops": 0,
                "iters": 0,
                "propagations": 0,
                "radius_cap_hits": 0,
                "layer_cap_hits": 0,
                "node_budget_hits": 0,
            },
        },
        "t2": {
            "retrieved": [],
            "metrics": {
                "tier_sequence": [],
                "k_returned": 0,
                "sim_stats": {"mean": s_max / 2, "max": s_max},
                "cache_used": False,
            },
        },
        "text": {"input": "hello world", "labels_from_t1": labels},
        "cfg": {
            "t3": {"max_rag_loops": 1, "tokens": tokens, "temp": 0.7},
            "t2": {"owner_scope": owner_scope, "k_retrieval": 6, "sim_threshold": 0.3},
        },
    }


def _kind_list(plan):
    return [getattr(op, "kind", None) for op in plan.ops]


def _speak_intent(plan):
    for op in plan.ops:
        if getattr(op, "kind", None) == "Speak":
            return getattr(op, "intent", None)
    return None


def _speak_labels(plan):
    for op in plan.ops:
        if getattr(op, "kind", None) == "Speak":
            return list(getattr(op, "topic_labels", []))
    return []


def _edit_ids(plan):
    for op in plan.ops:
        if getattr(op, "kind", None) == "EditGraph":
            return [e.get("id") for e in getattr(op, "edits", [])]
    return []


def test_determinism_same_bundle_same_plan():
    b = _bundle(s_max=0.75)
    p1 = deliberate(b)
    p2 = deliberate(b)
    assert asdict(p1) == asdict(p2)


def test_ops_whitelist_and_cap():
    b = _bundle(s_max=0.9, n_nodes=10, ops_cap=3)
    plan = deliberate(b)
    kinds = _kind_list(plan)
    allowed = {"Speak", "EditGraph", "RequestRetrieve", "CreateGraph", "SetMetaFilter"}
    assert set(kinds).issubset(allowed)
    assert len(kinds) <= 3, f"ops cap exceeded: {kinds}"


def test_intents_and_optional_ops_by_evidence():
    # Strong evidence → summary; no RequestRetrieve
    strong = deliberate(_bundle(s_max=0.9))
    assert _speak_intent(strong) == "summary"
    assert "RequestRetrieve" not in _kind_list(strong)

    # Weak evidence → assertion (labels present); typically includes EditGraph
    weak = deliberate(_bundle(s_max=0.5))
    assert _speak_intent(weak) == "assertion"
    # EditGraph allowed but optional; if present, cap respected
    kinds = _kind_list(weak)
    assert kinds[0] == "Speak"

    # Low evidence → question + RequestRetrieve
    low = deliberate(_bundle(s_max=0.2))
    assert _speak_intent(low) == "question"
    assert "RequestRetrieve" in _kind_list(low)


def test_label_and_edit_sorting_and_caps():
    # Labels should dedupe and sort: ["alpha", "zeta"] → ["alpha", "zeta"]
    b = _bundle(s_max=0.9, labels=["zeta", "alpha", "alpha"], n_nodes=5, ops_cap=3)
    plan = deliberate(b)
    labels = _speak_labels(plan)
    assert labels == sorted(set(["zeta", "alpha"]))

    # EditGraph edits sorted by id asc; capped by remaining*4 (with ops_cap=3 → remaining=2 → cap_nodes=8)
    edit_ids = _edit_ids(plan)
    assert edit_ids[:3] == ["a", "b", "c"], f"unexpected edit order: {edit_ids}"
    assert len(edit_ids) <= 8


def test_request_retrieve_owner_and_k_when_low_evidence():
    b = _bundle(s_max=0.1, owner_scope="world", n_nodes=2)
    plan = deliberate(b)
    # find RR op
    rr = None
    for op in plan.ops:
        if getattr(op, "kind", None) == "RequestRetrieve":
            rr = op
            break
    assert rr is not None, "RequestRetrieve should be present for low evidence"
    assert getattr(rr, "owner", None) == "world"
    assert getattr(rr, "k", 0) >= 1 and getattr(rr, "k", 0) <= 3  # half of k_retrieval=6 → 3


# --- New facade tests for select_policy/run_policy ---


def _policy_cfg():
    return {
        "t3": {"tokens": 128, "max_ops_per_turn": 3},
        "t2": {"k_retrieval": 4, "sim_threshold": 0.3, "owner_scope": "any"},
    }


def _policy_bundle():
    return {
        "cfg": _policy_cfg(),
        "agent": {"caps": {"ops": 3}},
        "slice_caps": {},
        "t2": {"metrics": {"sim_stats": {"max": 0.2}}, "retrieved": []},
        "t1": {"touched_nodes": []},
        "text": {"input": "hi", "labels_from_t1": []},
    }


def test_select_policy_defaults_to_rulebased():
    handle = policy.select_policy({}, ctx=None)
    assert handle == {"name": "rulebased", "meta": {}}


def test_select_policy_honors_backend_llm():
    cfg_root = {"t3": {"backend": "llm"}}
    handle = policy.select_policy(cfg_root, ctx=None)
    assert handle["name"] == "llm"


def test_run_policy_rulebased_emits_expected_ops():
    bundle = _policy_bundle()
    handle = {"name": "rulebased", "meta": {}}
    result = policy.run_policy(handle, bundle, bundle["cfg"], ctx=object())
    ops = result["plan"]
    assert getattr(ops[0], "kind", None) == "Speak"
    assert getattr(ops[1], "kind", None) == "RequestRetrieve"
    assert result["rationale"] == ""


def test_run_policy_llm_delegates(monkeypatch):
    called = {}

    def fake_plan_with_llm(ctx, state, cfg):
        called["args"] = (ctx, state, cfg)
        return {"plan": ["step"], "rationale": "ok"}

    monkeypatch.setattr(policy, "plan_with_llm", fake_plan_with_llm)

    handle = {"name": "llm", "meta": {}}
    state = types.SimpleNamespace(logs=[])
    cfg_root = _policy_cfg()
    out = policy.run_policy(handle, _policy_bundle(), cfg_root, ctx="ctx", state=state)

    assert out == {"plan": ["step"], "rationale": "ok"}
    assert called["args"] == ("ctx", state, cfg_root)

    with pytest.raises(ValueError):
        policy.run_policy(handle, {}, cfg_root, ctx="ctx")
