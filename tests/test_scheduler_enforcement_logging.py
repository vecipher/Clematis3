

# PR27 — Enforcement logging shape (+ ready-set sanity)
# Goal: Verify scheduler.jsonl records include 'enforced': True on yields,
# and that turn.jsonl slices carry the expected yield fields when enabled.
# We also sanity-check the ready-set hook default.

import json
import pytest


def test_scheduler_record_includes_enforced_boolean(monkeypatch, tmp_path):
    # Route logs_dir() to a temporary directory
    import clematis.io.paths as paths_module
    monkeypatch.setattr(paths_module, "logs_dir", lambda: str(tmp_path), raising=True)

    # Use the central writer
    from clematis.io.log import append_jsonl

    record = {
        "turn": 42,
        "slice": 1,
        "agent": "Ambrose",
        "policy": "round_robin",
        "reason": "BUDGET_T1_ITERS",
        "enforced": True,             # <- PR27 addition we care about
        "stage_end": "T1",
        "quantum_ms": 20,
        "wall_ms": 200,
        "budgets": {"t1_iters": 3, "t2_k": 2, "t3_ops": 1},
        "consumed": {"ms": 7, "t1_iters": 3},
        "queued": [],
        "ms": 0,
    }
    append_jsonl("scheduler.jsonl", record)

    p = tmp_path / "scheduler.jsonl"
    assert p.exists()
    line = p.read_text(encoding="utf-8").strip()
    parsed = json.loads(line)
    assert parsed.get("enforced") is True
    # Minimal required shape for downstream tooling
    required = {"turn", "slice", "agent", "policy", "reason", "stage_end", "quantum_ms", "wall_ms", "budgets", "consumed"}
    assert required.issubset(parsed.keys())


def test_turn_slice_carries_yield_fields(monkeypatch, tmp_path):
    # Route logs_dir() to a temporary directory
    import clematis.io.paths as paths_module
    monkeypatch.setattr(paths_module, "logs_dir", lambda: str(tmp_path), raising=True)

    from clematis.io.log import append_jsonl

    # Simulate a partial turn slice logged at a boundary with enforcement
    turn_slice = {
        "turn": 7,
        "agent": "Kafka",
        "durations_ms": {"t1": 5.0, "t2": 0.0, "t4": 0.0, "apply": 0.0, "total": 5.0},
        "t1": {"pops": 2, "iters": 1, "graphs_touched": 1},
        "t2": {},
        "t4": {},
        "slice_idx": 1,
        "yielded": True,             # <- PR27 expected
        "yield_reason": "QUANTUM_EXCEEDED",
    }
    append_jsonl("turn.jsonl", turn_slice)

    p = tmp_path / "turn.jsonl"
    assert p.exists()
    parsed = json.loads(p.read_text(encoding="utf-8").splitlines()[-1])
    assert parsed.get("yielded") is True
    assert "yield_reason" in parsed
    assert parsed.get("slice_idx") == 1


def test_readyset_hook_default_true():
    """The ready-set hook exists and defaults to (True, 'DEFAULT_TRUE')."""
    orchestrator = pytest.importorskip("clematis.engine.orchestrator")
    fn = getattr(orchestrator, "agent_ready", None)
    assert callable(fn), "agent_ready hook must be defined"
    ok, reason = fn(ctx=object(), state=None, agent_id="any")
    assert ok is True and isinstance(reason, str) and reason