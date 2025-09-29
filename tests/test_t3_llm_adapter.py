import pytest
from dataclasses import asdict
from datetime import datetime, timezone

from clematis.engine.stages.t3 import make_dialog_bundle, build_llm_prompt, llm_speak
from clematis.engine.types import Plan, SpeakOp
from clematis.adapters.llm import DeterministicLLMAdapter


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
                    "template": template
                    or "style:{style_prefix} labels:{labels} intent:{intent} snippets:{snippets}",
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
    deltas = [
        {"id": "n1", "label": labels[0], "delta": 0.2},
        {"id": "n2", "label": labels[1], "delta": 0.3},
    ]
    return Obj(graph_deltas=deltas, metrics={})


def _t2_hits():
    return Obj(
        retrieved=[
            {"id": "e2", "score": 0.80, "owner": "any", "quarter": "2025Q3"},
            {"id": "e1", "score": 0.90, "owner": "any", "quarter": "2025Q3"},
            {"id": "e3", "score": 0.70},
            {"id": "e4", "score": 0.60},
        ],
        metrics={
            "sim_stats": {"mean": 0.75, "max": 0.90},
            "tier_sequence": ["exact_semantic"],
            "k_returned": 4,
            "cache_used": False,
        },
    )


def _plan(intent="summary", labels=("z", "a", "a"), max_tokens=None):
    lab = list(labels)
    speak = SpeakOp(kind="Speak", intent=intent, topic_labels=lab, max_tokens=(max_tokens or 256))
    return Plan(version="t3-plan-v1", reflection=False, ops=[speak], request_retrieve=None)


def test_build_llm_prompt_contains_sorted_fields():
    ctx = Ctx(include_k=2)
    t1 = _t1(labels=("alpha", "zeta"))
    t2 = _t2_hits()
    plan = _plan(intent="summary", labels=("zeta", "alpha", "alpha"))
    db = make_dialog_bundle(ctx, {}, t1, t2, plan)

    prompt = build_llm_prompt(db, plan)
    # style, intent, labels (sorted), snippets top-2 (e1, e2), and input present
    assert "style_prefix: calm" in prompt
    assert "intent: summary" in prompt
    assert "labels: alpha, zeta" in prompt
    assert "snippets: e1, e2" in prompt
    assert "input: hello world" in prompt


def test_llm_speak_with_deterministic_adapter_is_stable_and_capped():
    ctx = Ctx(include_k=2)
    t1 = _t1(labels=("alpha", "zeta"))
    t2 = _t2_hits()
    plan = _plan(intent="summary", labels=("zeta", "alpha", "alpha"), max_tokens=8)

    db = make_dialog_bundle(ctx, {}, t1, t2, plan)
    adapter = DeterministicLLMAdapter()

    u1, m1 = llm_speak(db, plan, adapter)
    u2, m2 = llm_speak(db, plan, adapter)

    # Deterministic utterance and metrics
    assert u1 == u2
    assert m1 == m2

    # Budget respected (SpeakOp.max_tokens = 8)
    assert m1["tokens"] == 8
    assert len(u1.split()) == 8
    assert m1.get("backend") == "llm" and m1.get("adapter") in (
        "DeterministicLLMAdapter",
        adapter.name,
    )


def test_llm_speak_prefixes_style_if_missing():
    # Use a template without {style_prefix}; llm_speak should still prefix with "calm| "
    ctx = Ctx(template="labels:{labels} intent:{intent} snippets:{snippets}")
    plan = _plan()
    db = make_dialog_bundle(ctx, {}, _t1(), _t2_hits(), plan)
    adapter = DeterministicLLMAdapter()
    u, m = llm_speak(db, plan, adapter)
    assert u.startswith("calm| ")
    assert m["style_prefix_used"] is True
