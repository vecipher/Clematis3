import importlib
from types import SimpleNamespace as SNS
import os

def _cfg():
    return {
        "t3": {
            "allow_reflection": True,
            "reflection": {"topk_snippets": 0, "summary_tokens": 16, "embed": False},
        },
        "scheduler": {"budgets": {"time_ms_reflection": 5, "ops_reflection": 1}},
    }

def test_reflection_flag_off_no_run(monkeypatch):
    monkeypatch.setenv("CI", "true")
    monkeypatch.setenv("CLEMATIS_NETWORK_BAN", "1")

    core = importlib.import_module("clematis.engine.orchestrator.core")

    cfg = _cfg()
    ctx = SNS(turn_id=1, agent_id="A", now_ms=12345, cfg=cfg, _dry_run_until_t4=False)
    state = {"cfg": cfg}
    plan = SNS(ops=[], reflection=False)
    t2 = SNS(retrieved=[])

    res = core._run_reflection_if_enabled(ctx, state, plan, "hi", t2)
    assert res is None  # gate closed

def test_reflection_flag_from_state_runs(monkeypatch):
    monkeypatch.setenv("CI", "true")
    monkeypatch.setenv("CLEMATIS_NETWORK_BAN", "1")

    core = importlib.import_module("clematis.engine.orchestrator.core")

    cfg = _cfg()
    ctx = SNS(turn_id=1, agent_id="A", now_ms=12345, cfg=cfg, _dry_run_until_t4=False)
    state = {"cfg": cfg}
    plan = SNS(ops=[], reflection=False)  # planner didn't set Plan.reflection
    t2 = SNS(retrieved=[])

    # LLM planner path stashes the flag here (policy.py does this)
    state["_planner_reflection_flag"] = True

    res = core._run_reflection_if_enabled(ctx, state, plan, "hi", t2)
    assert res is not None  # gate opened via stashed flag
    assert getattr(res, "memory_entries", []) == [] or isinstance(res.memory_entries, list)
