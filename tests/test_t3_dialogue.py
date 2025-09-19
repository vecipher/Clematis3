import pytest
from dataclasses import asdict
from datetime import datetime, timezone

from clematis.engine.stages.t3 import make_dialog_bundle, speak
from clematis.engine.types import Plan, SpeakOp


class Ctx:
    def __init__(self, template=None, include_k=2, k_retrieval=3, style_prefix="calm"):
        self.now = datetime(2025, 9, 19, 0, 0, 0, tzinfo=timezone.utc)
        self.agent_id = "agentA"
        self.agent = {"style_prefix": style_prefix}
        self.cfg = {
            "t3": {
                "tokens": 256,
                "max_ops_per_turn": 3,
                "max_rag_loops": 1,
                "temp": 0.7,
                "dialogue": {
                    "template": template or "style:{style_prefix} labels:{labels} intent:{intent} snippets:{snippets}",
                    "include_top_k_snippets": include_k,
                },
            },
            "t2": {"owner_scope": "any", "k_retrieval": k_retrieval, "sim_threshold": 0.3},
        }
        self.input_text = "hello world"


class Obj:
    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)


def _t1(labels=("l2", "l1")):
    # Provide graph_deltas with labels so bundle can extract labels_from_t1
    deltas = [
        {"id": "n1", "label": labels[0], "delta": 0.2},
        {"id": "n2", "label": labels[1], "delta": 0.3},
    ]
    return Obj(graph_deltas=deltas, metrics={})


def _t2_hits():
    # Unordered hits; expect sorting by score desc then id asc, and trimming to k_retrieval
    return Obj(
        retrieved=[
            {"id": "e2", "score": 0.80, "owner": "any", "quarter": "2025Q3"},
            {"id": "e1", "score": 0.90, "owner": "any", "quarter": "2025Q3"},
            {"id": "e3", "score": 0.70},  # missing owner/quarter OK
            {"id": "e4", "score": 0.60},
        ],
        metrics={"sim_stats": {"mean": 0.75, "max": 0.90}, "tier_sequence": ["exact_semantic"], "k_returned": 4, "cache_used": False},
    )


def _plan(intent="summary", labels=("z", "a", "a"), max_tokens=None):
    lab = list(labels)
    speak = SpeakOp(kind="Speak", intent=intent, topic_labels=lab, max_tokens=(max_tokens or 256))
    return Plan(version="t3-plan-v1", reflection=False, ops=[speak], request_retrieve=None)


def test_dialog_bundle_determinism_and_shape():
    ctx = Ctx(k_retrieval=3, include_k=2)
    t1 = _t1()
    t2 = _t2_hits()
    state = {}

    b1 = make_dialog_bundle(ctx, state, t1, t2, plan=_plan())
    b2 = make_dialog_bundle(ctx, state, t1, t2, plan=_plan())

    assert b1 == b2, "Dialog bundle must be deterministic"
    # Basic shape
    assert b1["version"] == "t3-dialog-bundle-v1"
    assert isinstance(b1["retrieved"], list) and len(b1["retrieved"]) == 3  # trimmed to k_retrieval
    # Sorted by score desc then id asc → e1 (0.90), e2 (0.80), e3 (0.70)
    assert [h["id"] for h in b1["retrieved"]] == ["e1", "e2", "e3"]


def test_speak_determinism_and_template_fields():
    ctx = Ctx(include_k=2)
    t1 = _t1(labels=("alpha", "zeta"))
    t2 = _t2_hits()
    state = {}
    plan = _plan(intent="summary", labels=("zeta", "alpha", "alpha"))

    dialog_bundle = make_dialog_bundle(ctx, state, t1, t2, plan)
    u1, m1 = speak(dialog_bundle, plan)
    u2, m2 = speak(dialog_bundle, plan)

    # Deterministic utterance and metrics
    assert u1 == u2
    assert m1 == m2

    # Template fields present
    assert "style:calm" in u1
    assert "labels:alpha, zeta" in u1  # dedupe + sort
    assert "intent:summary" in u1
    # Snippets capped by include_top_k_snippets=2 → e1, e2
    assert ("snippets:e1, e2" in u1) or ("snippets: e1, e2" in u1)


def test_snippets_cap_respected():
    ctx = Ctx(include_k=1)
    dialog_bundle = make_dialog_bundle(ctx, {}, _t1(), _t2_hits(), _plan())
    u, _ = speak(dialog_bundle, _plan())
    assert "e1" in u and "e2" not in u, "Only top-1 snippet id should appear"


def test_token_budget_truncation_via_speakop_cap():
    # Force a very small token budget via SpeakOp.max_tokens
    ctx = Ctx(include_k=2)
    dialog_bundle = make_dialog_bundle(ctx, {}, _t1(), _t2_hits(), _plan(max_tokens=5))
    u, metrics = speak(dialog_bundle, _plan(max_tokens=5))
    assert metrics["truncated"] is True
    assert metrics["tokens"] == 5
    assert len(u.split()) == 5


def test_style_prefix_auto_prefix_when_missing_in_template():
    # Use a template without {style_prefix} to ensure auto prefixing with "calm| "
    ctx = Ctx(template="labels:{labels} intent:{intent} snippets:{snippets}")
    dialog_bundle = make_dialog_bundle(ctx, {}, _t1(), _t2_hits(), _plan())
    u, metrics = speak(dialog_bundle, _plan())
    assert u.startswith("calm| "), f"Unexpected utterance prefix: {u[:12]}"
    assert metrics["style_prefix_used"] is True


def test_fallback_labels_from_bundle_when_plan_labels_missing():
    ctx = Ctx()
    plan = _plan(labels=())  # empty labels on SpeakOp
    dialog_bundle = make_dialog_bundle(ctx, {}, _t1(labels=("l2", "l1")), _t2_hits(), plan)
    u, _ = speak(dialog_bundle, plan)
    # labels_from_t1 are ["l1", "l2"] sorted alphabetically
    assert "labels:l1, l2" in u
