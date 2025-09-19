

import pytest
from dataclasses import asdict
from datetime import datetime, timezone

from clematis.engine.stages.t3 import rag_once
from clematis.engine.types import Plan


def _bundle(s_max=0.2, labels=None, ops_cap=3, tokens=256, owner_scope="any"):
    labels = labels if labels is not None else ["beta", "alpha", "alpha"]
    # T1 nodes with deltas above epsilon_edit=0.10 so an EditGraph can be added when allowed
    nodes = []
    for i, nid in enumerate(["b", "a", "c", "e", "d", "f"]):
        nodes.append({"id": nid, "label": nid.upper(), "delta": 0.2 + 0.01 * i})

    now = datetime(2025, 9, 19, 0, 0, 0, tzinfo=timezone.utc).isoformat()
    return {
        "version": "t3-bundle-v1",
        "now": now,
        "agent": {"id": "agentA", "style_prefix": "", "caps": {"tokens": tokens, "ops": ops_cap}},
        "world": {"hot_labels": [], "k": 0},
        "t1": {"touched_nodes": nodes, "metrics": {"pops": 0, "iters": 0, "propagations": 0, "radius_cap_hits": 0, "layer_cap_hits": 0, "node_budget_hits": 0}},
        "t2": {"retrieved": [], "metrics": {"tier_sequence": [], "k_returned": 0, "sim_stats": {"mean": s_max/2, "max": s_max}, "cache_used": False}},
        "text": {"input": "hello world", "labels_from_t1": labels},
        "cfg": {"t3": {"max_rag_loops": 1, "tokens": tokens, "temp": 0.7,
                          "policy": {"tau_high": 0.8, "tau_low": 0.4, "epsilon_edit": 0.10}},
                 "t2": {"owner_scope": owner_scope, "k_retrieval": 6, "sim_threshold": 0.3}},
    }


def _plan_with_rr(intent="question", owner="any", k=3, tier_pref="cluster_semantic"):
    # Minimal plan dict form is acceptable; Plan requires version/ops fields. Use dataclass API.
    from clematis.engine.types import SpeakOp, RequestRetrieveOp
    speak = SpeakOp(kind="Speak", intent=intent, topic_labels=["alpha", "beta"], max_tokens=256)
    rr = RequestRetrieveOp(kind="RequestRetrieve", query="hello world", owner=owner, k=k, tier_pref=tier_pref, hints={})
    return Plan(version="t3-plan-v1", reflection=False, ops=[speak, rr], request_retrieve=None)


def test_one_shot_blocked_returns_unchanged_plan_and_metrics():
    b = _bundle(s_max=0.2)
    p = _plan_with_rr()

    def fake_retrieve(payload):
        raise AssertionError("retrieve_fn should not be called when already_used=True")

    p2, m = rag_once(b, p, fake_retrieve, already_used=True)
    assert asdict(p2) == asdict(p), "Plan must be unchanged when already_used=True"
    assert m["rag_used"] is False and m["rag_blocked"] is True
    assert m["pre_s_max"] == m["post_s_max"] == 0.2


def test_refinement_improves_intent_and_keeps_ops_capped():
    b = _bundle(s_max=0.2, ops_cap=3)
    p = _plan_with_rr(intent="question")

    seen = {}

    def fake_retrieve(payload):
        # Capture payload for later assertions
        seen.update(payload)
        return {
            "retrieved": [
                {"id": "x2", "score": 0.85, "owner": payload.get("owner", "any"), "quarter": "2025Q3"},
                {"id": "x1", "score": 0.70},
            ],
            "metrics": {"k_returned": 2},
        }

    p2, m = rag_once(b, p, fake_retrieve, already_used=False)
    # Speak intent should flip to summary (>= tau_high)
    speak_intents = [getattr(op, "intent", None) for op in p2.ops if getattr(op, "kind", None) == "Speak"]
    assert speak_intents and speak_intents[0] == "summary"
    # Ops should not exceed cap
    assert len(p2.ops) <= b["agent"]["caps"]["ops"]
    # Metrics sanity
    assert m["rag_used"] is True and m["rag_blocked"] is False
    assert m["pre_s_max"] == 0.2 and m["post_s_max"] == 0.85
    assert m["k_retrieved"] == 2
    # Payload sanity
    assert seen.get("query") == "hello world"
    assert seen.get("owner") in ("agent", "world", "any")
    assert seen.get("k") >= 1
    assert isinstance(seen.get("hints", {}), dict)
    assert seen.get("hints", {}).get("now") == b["now"]


def test_no_improvement_intent_stays_question():
    b = _bundle(s_max=0.2)
    p = _plan_with_rr(intent="question")

    def fake_retrieve(_):
        return {"retrieved": [{"id": "x", "score": 0.25}]}

    p2, m = rag_once(b, p, fake_retrieve)
    speak_intents = [getattr(op, "intent", None) for op in p2.ops if getattr(op, "kind", None) == "Speak"]
    assert speak_intents[0] == "question"
    assert m["post_s_max"] == pytest.approx(0.25)


def test_optional_editgraph_added_when_evidence_sufficient_and_absent():
    # Pre evidence weak but ≥ tau_low after RAG; ensure one EditGraph added if within cap
    b = _bundle(s_max=0.45)  # weak evidence already ≥ tau_low
    # Build a plan that has only Speak + RR, no EditGraph yet
    p = _plan_with_rr(intent="assertion")

    def fake_retrieve(_):
        # Keep evidence around same level
        return {"retrieved": [{"id": "y", "score": 0.5}]}

    p2, _ = rag_once(b, p, fake_retrieve)
    kinds = [getattr(op, "kind", None) for op in p2.ops]
    assert "EditGraph" in kinds, "Expected an EditGraph op to be added"


def test_determinism_same_inputs_same_outputs():
    b = _bundle(s_max=0.2)
    p = _plan_with_rr()

    def fake_retrieve(payload):
        return {"retrieved": [{"id": "x2", "score": 0.85}, {"id": "x1", "score": 0.70}]}

    p2a, m1 = rag_once(b, p, fake_retrieve)
    p2b, m2 = rag_once(b, p, fake_retrieve)
    assert asdict(p2a) == asdict(p2b)
    assert m1 == m2


def test_noop_when_no_rr_in_plan():
    # Plan without RequestRetrieve should be returned unchanged
    from clematis.engine.types import SpeakOp
    b = _bundle(s_max=0.2)
    speak = SpeakOp(kind="Speak", intent="question", topic_labels=["alpha"], max_tokens=256)
    plan = Plan(version="t3-plan-v1", reflection=False, ops=[speak], request_retrieve=None)

    def fake_retrieve(_):
        raise AssertionError("retrieve_fn should not be called when no RR op present")

    p2, m = rag_once(b, plan, fake_retrieve)
    assert asdict(p2) == asdict(plan)
    assert m["rag_used"] is False and m["rag_blocked"] is False and m["k_retrieved"] == 0