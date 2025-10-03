# -*- coding: utf-8 -*-
import importlib
from types import SimpleNamespace as SNS
import copy
import pytest

class _DetIndex:
    def __init__(self):
        self.kind = "inmemory"
        self.rows = []
    def add(self, ep):
        self.rows.append(copy.deepcopy(ep))

def _mk_ctx():
    return SNS(
        turn_id=1, agent_id="AgentA",
        now_ms=12_345, now_iso="1970-01-01T00:00:12.345Z",
        _dry_run_until_t4=False,
    )

def _mk_cfg():
    # Minimal validated-like structure
    return {
        "t3": {"allow_reflection": True, "reflection": {"embed": True, "summary_tokens": 128, "topk_snippets": 3}},
        "scheduler": {"budgets": {"time_ms_reflection": 6000, "ops_reflection": 5}},
    }

def _mk_state(idx, cfg):
    return {"cfg": cfg, "memory_index": idx}

def _run_once(core, writer, ctx, state, cfg):
    plan = SNS(reflection=True)
    utter = "Hello THERE!"
    t2 = SNS(snippets=["alpha", "beta", "gamma"])
    res = core._run_reflection_if_enabled(ctx, state, plan, utter, t2)  # PR79
    assert res is not None
    rep = writer.write_reflection_entries(ctx, state, cfg, res)         # PR80
    return res, rep

def test_reflection_deterministic(monkeypatch):
    monkeypatch.setenv("CI", "true")
    monkeypatch.setenv("CLEMATIS_NETWORK_BAN", "1")

    core = importlib.import_module("clematis.engine.orchestrator.core")
    writer = importlib.import_module("clematis.engine.orchestrator.reflection")

    idx1 = _DetIndex()
    cfg = _mk_cfg()

    # Run 1
    ctx1 = _mk_ctx()
    state1 = _mk_state(idx1, cfg)
    res1, rep1 = _run_once(core, writer, ctx1, state1, cfg)

    # Run 2 (fresh index; same inputs)
    idx2 = _DetIndex()
    ctx2 = _mk_ctx()
    state2 = _mk_state(idx2, cfg)
    res2, rep2 = _run_once(core, writer, ctx2, state2, cfg)

    # Reflection results equality (summary & metrics stable)
    assert res1.summary == res2.summary
    assert res1.metrics == res2.metrics

    # Deterministic memory writes: identical IDs/ts/content
    assert len(idx1.rows) == len(idx2.rows) > 0
    for a, b in zip(idx1.rows, idx2.rows):
        assert a == b

    # Deterministic write reports
    assert rep1.ops_attempted == rep2.ops_attempted
    assert rep1.ops_written == rep2.ops_written
    assert rep1.ts_iso == rep2.ts_iso
    assert rep1.errors == rep2.errors
