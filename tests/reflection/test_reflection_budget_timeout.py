# -*- coding: utf-8 -*-
import importlib
from types import SimpleNamespace as SNS
import pytest

class _DetIndex:
    def __init__(self):
        self.kind = "inmemory"
        self.rows = []
    def add(self, ep):
        self.rows.append(ep)

def _mk_ctx():
    return SNS(
        turn_id=1, agent_id="AgentA",
        now_ms=12_345, now_iso="1970-01-01T00:00:12.345Z",
        _dry_run_until_t4=False,
    )

def _mk_cfg():
    return {
        "t3": {"allow_reflection": True, "reflection": {"embed": True, "summary_tokens": 128, "topk_snippets": 3}},
        "scheduler": {"budgets": {"time_ms_reflection": 1, "ops_reflection": 5}},
    }

def _mk_state(idx, cfg):
    return {"cfg": cfg, "memory_index": idx}

def test_reflection_budget_timeout(monkeypatch):
    monkeypatch.setenv("CI", "true")
    monkeypatch.setenv("CLEMATIS_NETWORK_BAN", "1")

    core = importlib.import_module("clematis.engine.orchestrator.core")
    writer = importlib.import_module("clematis.engine.orchestrator.reflection")

    # Monkeypatch perf_counter to simulate elapsed > budget inside the reflection window
    import time as _time
    calls = {"n": 0}
    def fake_perf():
        calls["n"] += 1
        # first call: t0, second call (post-reflect): jump by 0.01s (10ms)
        return 0.0 if calls["n"] == 1 else 0.01
    monkeypatch.setattr(core.time, "perf_counter", fake_perf, raising=True)

    idx = _DetIndex()
    cfg = _mk_cfg()
    ctx = _mk_ctx()
    state = _mk_state(idx, cfg)

    plan = SNS(reflection=True)
    utter = "Budget test"
    t2 = SNS(snippets=["x", "y"])

    res = core._run_reflection_if_enabled(ctx, state, plan, utter, t2)
    assert res is not None
    # On timeout: metrics.reason set, and memory_entries should be empty (per PR79 contract)
    assert res.metrics.get("reason") == "reflection_timeout"
    assert not res.memory_entries

    rep = writer.write_reflection_entries(ctx, state, cfg, res)
    # Writer should write nothing
    assert rep.ops_written == 0
    assert idx.rows == []
