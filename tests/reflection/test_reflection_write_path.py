# -*- coding: utf-8 -*-
import copy
import types

from clematis.engine.orchestrator.reflection import write_reflection_entries, WriteReport

class _Ctx:
    now_ms = 12345
    now_iso = "1970-01-01T00:00:12.345Z"
    agent_id = "AgentA"
    turn_id = 1

class _State:
    def __init__(self, index):
        self.memory_index = index

class _DetIndex:
    def __init__(self):
        self.kind = "inmemory"
        self.rows = []

    def add(self, ep):
        # Deterministic append (no reordering)
        self.rows.append(copy.deepcopy(ep))

class _BadIndex(_DetIndex):
    def add(self, ep):
        raise RuntimeError("boom")

class _Result:
    def __init__(self, entries):
        self.memory_entries = entries

CFG = {
    "scheduler": {"budgets": {"ops_reflection": 2}},
    "t3": {"reflection": {"embed": True}}
}

def test_write_is_deterministic_and_capped():
    ctx = _Ctx()
    idx = _DetIndex()
    state = _State(idx)
    entries = [
        {"text": "alpha", "tags": ["reflection"], "kind": "summary", "vec_full": [0.1] * 32},
        {"text": "beta",  "tags": ["reflection"], "kind": "summary", "vec_full": [0.2] * 32},
        {"text": "gamma", "tags": ["reflection"], "kind": "summary", "vec_full": [0.3] * 32},
    ]
    res = _Result(entries)

    r1 = write_reflection_entries(ctx, state, CFG, res)
    r2 = write_reflection_entries(ctx, state, CFG, res)

    # ops cap = 2, entries attempted=3 → write 2 twice (deterministic)
    assert r1.ops_attempted == 3 and r1.ops_written == 2 and not r1.errors
    assert r2.ops_attempted == 3 and r2.ops_written == 2 and not r2.errors

    # Deterministic contents (IDs and ts identical across runs)
    assert len(idx.rows) == 4  # two writes * two runs
    ids = [row["id"] for row in idx.rows]
    assert ids[0] == ids[2] and ids[1] == ids[3]
    for row in idx.rows:
        assert row["ts"] == "1970-01-01T00:00:12.345Z"
        assert row["tags"] == ["reflection"]
        assert row["kind"] == "summary"
        assert "vec_full" in row and isinstance(row["vec_full"], list) and len(row["vec_full"]) == 32

def test_write_fails_soft_when_index_missing():
    ctx = _Ctx()
    state = _State(index=None)  # no index on state
    res = _Result([{"text": "alpha"}])
    rep = write_reflection_entries(ctx, state, CFG, res)
    assert rep.ops_attempted == 1
    assert rep.ops_written == 0
    assert rep.reason in ("index_missing", "index_select_error")

def test_write_fails_soft_on_add_error():
    ctx = _Ctx()
    bad = _BadIndex()
    state = _State(bad)
    res = _Result([{"text": "alpha"}, {"text": "beta"}])
    rep = write_reflection_entries(ctx, state, CFG, res)
    assert rep.ops_attempted == 2
    assert rep.ops_written == 0
    assert rep.reason == "partial_failure"
    assert rep.errors and all(e.startswith("add_error[") for e in rep.errors)
