from datetime import datetime, timezone

from clematis.engine.stages.t3 import make_plan_bundle


class _Ctx:
    def __init__(self) -> None:
        self.now = datetime(2025, 9, 19, 0, 0, 0, tzinfo=timezone.utc)
        self.agent_id = "agent-A"
        self.agent = {"style_prefix": "calm"}
        self.cfg = {
            "t3": {"tokens": 256, "max_ops_per_turn": 3, "max_rag_loops": 1, "temp": 0.7},
            "t2": {"owner_scope": "any", "k_retrieval": 3, "sim_threshold": 0.3},
        }
        self.input_text = "hello"
        self.slice_budgets = {"t3_ops": 2}


class _T1:
    graph_deltas = []
    metrics = {}


class _T2:
    metrics = {"sim_stats": {"max": 0.0, "mean": 0.0}, "k_returned": 0}
    retrieved = []


def test_bundle_facade_minimal() -> None:
    ctx = _Ctx()
    state = {"world_hot_labels": ["X"]}
    bundle = make_plan_bundle(ctx, state, _T1(), _T2())

    assert bundle["version"] == "t3-bundle-v1"
    assert bundle["agent"]["id"] == "agent-A"
    assert bundle["slice_caps"] == {"t3_ops": 2}
    assert "t1" in bundle and "t2" in bundle and "cfg" in bundle
