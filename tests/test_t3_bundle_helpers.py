from datetime import datetime, timezone

from clematis.engine.stages.t3 import bundle


class _Ctx:
    def __init__(self):
        self.now = datetime(2026, 1, 2, 3, 4, 5, tzinfo=timezone.utc)
        self.agent_id = "agent-X"
        self.agent = {"style_prefix": "calm"}
        self.cfg = {
            "t3": {"tokens": 512, "max_ops_per_turn": 4, "max_rag_loops": 2},
            "t2": {"owner_scope": "world", "k_retrieval": 3, "sim_threshold": 0.25},
        }
        self.slice_budgets = {"t3_ops": 2}
        self.input_text = "hi"


class _T1:
    graph_deltas = None
    deltas = [
        {"id": "b", "label": "B", "delta": 0.05},
        {"id": "a", "label": "A", "delta": 0.25},
        {"id": "c", "label": "C", "delta": 0.10},
    ]
    metrics = None


class _T2:
    metrics = None
    retrieved = [
        {"id": "doc-2", "_score": 0.3, "owner": "world"},
        {"id": "doc-1", "_score": 0.8, "owner": "agent"},
    ]


def test_bundle_sorts_nodes_and_scores() -> None:
    ctx = _Ctx()
    state = {"world_hot_labels": ["z", "a"]}
    t1 = _T1()
    t2 = _T2()

    out = bundle.assemble_bundle(ctx, state, t1, t2)

    # slice budgets preserved
    assert out["slice_caps"] == {"t3_ops": 2}
    # cfg snapshot uses ctx cfg
    assert out["cfg"]["t3"]["tokens"] == 512
    assert out["cfg"]["t2"]["k_retrieval"] == 3

    nodes = out["t1"]["touched_nodes"]
    assert [n["id"] for n in nodes] == ["a", "b", "c"]

    metrics = out["t1"]["metrics"]
    assert metrics == {
        "pops": 0,
        "iters": 0,
        "propagations": 0,
        "radius_cap_hits": 0,
        "layer_cap_hits": 0,
        "node_budget_hits": 0,
    }

    retrieved = out["t2"]["retrieved"]
    assert [r["id"] for r in retrieved] == ["doc-1", "doc-2"]
    assert out["t2"]["metrics"]["k_returned"] == 2

    labels = out["text"]["labels_from_t1"]
    assert labels == ["A", "B", "C"]


def test_bundle_defaults_when_metrics_present() -> None:
    ctx = _Ctx()
    ctx.slice_budgets = {}

    class _T2Metrics:
        metrics = {"k_returned": 5, "sim_stats": {"mean": 0.1, "max": 0.42}}
        retrieved = []

    out = bundle.assemble_bundle(ctx, state={}, t1=_T1(), t2=_T2Metrics())
    assert out["slice_caps"] == {}
    assert out["t2"]["metrics"]["k_returned"] == 5
    assert out["t2"]["metrics"]["sim_stats"] == {"mean": 0.1, "max": 0.42}
