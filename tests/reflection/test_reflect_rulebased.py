import copy
from clematis.engine.stages.t3 import reflect, ReflectionBundle

class _Ctx:
    agent_id = "AgentA"
    now_iso = "2025-01-01T00:00:00Z"

def _cfg(summary_tokens=8, topk=2, embed=True, ops=1):
    return {
        "t3": {"reflection": {"summary_tokens": summary_tokens, "topk_snippets": topk, "embed": embed}},
        "scheduler": {"budgets": {"ops_reflection": ops}},
    }

def test_reflect_deterministic_basic():
    b = ReflectionBundle(
        ctx=_Ctx(),
        state_view=None,
        plan=object(),
        utter="Hello, World! This is a Test.",
        snippets=["Alpha--beta", "Gamma... delta"],
    )
    out1 = reflect(b, _cfg(), embedder=None)
    out2 = reflect(b, _cfg(), embedder=None)
    assert out1.summary == out2.summary
    assert out1.memory_entries == out2.memory_entries
    assert out1.metrics == out2.metrics
    entry = out1.memory_entries[0]
    assert "vec_full" in entry and isinstance(entry["vec_full"], list) and len(entry["vec_full"]) == 32

def test_reflect_respects_token_cap_and_topk():
    b = ReflectionBundle(
        ctx=_Ctx(),
        state_view=None,
        plan=object(),
        utter="A B C D E F G H I J",
        snippets=["K L M N O", "P Q R S T", "U V W X Y Z"],
    )
    out = reflect(b, _cfg(summary_tokens=5, topk=1), embedder=None)
    assert len(out.summary.split()) == 5
    assert out.metrics["used_topk"] == 1

def test_reflect_ops_cap_zero_yields_no_entries():
    b = ReflectionBundle(ctx=_Ctx(), state_view=None, plan=object(), utter="x", snippets=[])
    out = reflect(b, _cfg(ops=0), embedder=None)
    assert out.memory_entries == []
