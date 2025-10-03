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

def _mk_cfg(cap=1):
    return {
        "t3": {"allow_reflection": True, "reflection": {"embed": False, "summary_tokens": 128, "topk_snippets": 3}},
        "scheduler": {"budgets": {"time_ms_reflection": 6000, "ops_reflection": cap}},
    }

def _mk_state(idx, cfg):
    return {"cfg": cfg, "memory_index": idx}

def test_reflection_ops_cap(monkeypatch):
    monkeypatch.setenv("CI", "true")
    monkeypatch.setenv("CLEMATIS_NETWORK_BAN", "1")

    core = importlib.import_module("clematis.engine.orchestrator.core")
    writer = importlib.import_module("clematis.engine.orchestrator.reflection")

    # Monkeypatch reflect(...) to return 3 entries (simulate verbose reflection)
    rmod = importlib.import_module("clematis.engine.stages.t3.reflect")
    class _FakeRes:
        def __init__(self):
            self.summary = "s"
            self.metrics = {"ms": 0.0}
            self.memory_entries = [
                {"text": "e1", "tags": ["reflection"], "kind": "summary"},
                {"text": "e2", "tags": ["reflection"], "kind": "summary"},
                {"text": "e3", "tags": ["reflection"], "kind": "summary"},
            ]
    def fake_reflect(bundle, cfg, embedder=None):
        return _FakeRes()
    monkeypatch.setattr(rmod, "reflect", fake_reflect, raising=True)

    idx = _DetIndex()
    cfg = _mk_cfg(cap=1)
    ctx = _mk_ctx()
    state = _mk_state(idx, cfg)

    plan = SNS(reflection=True)
    utter = "Ops cap test"
    t2 = SNS(snippets=["x"])

    res = core._run_reflection_if_enabled(ctx, state, plan, utter, t2)
    assert res is not None
    assert len(res.memory_entries) == 3  # produced by fake reflect

    rep = writer.write_reflection_entries(ctx, state, cfg, res)
    # Writer must cap to 1
    assert rep.ops_attempted == 3
    assert rep.ops_written == 1
    assert len(idx.rows) == 1
