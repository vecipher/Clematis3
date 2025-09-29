# PR26 — Scheduler logging & yield-decision smoke tests
# These tests avoid full orchestrator wiring and instead validate:
# 1) _should_yield precedence logic (WALL_MS > BUDGET_* > QUANTUM_EXCEEDED)
# 2) JSONL logging writes a valid scheduler record to the logs directory

import json
from typing import Dict

import pytest


def _mk_slice_ctx(budgets: Dict[str, int]) -> Dict:
    return {
        "slice_idx": 1,
        "started_ms": 0,
        "budgets": budgets,
        "agent_id": "AgentA",
    }


def test_should_yield_precedence_wall_budget_quantum():
    # Import the internal helper from orchestrator (exposed by our PR26 scaffolding)
    from clematis.engine.orchestrator import _should_yield  # type: ignore

    budgets = {
        "t1_iters": 3,
        "t1_pops": 10,
        "t2_k": 5,
        "t3_ops": 2,
        "wall_ms": 50,
        "quantum_ms": 20,
    }
    sc = _mk_slice_ctx(budgets)

    # WALL_MS has highest precedence
    reason = _should_yield(sc, {"ms": 60})
    assert reason == "WALL_MS"

    # Budget reasons take precedence over quantum
    reason = _should_yield(sc, {"ms": 10, "t1_iters": 3})
    assert reason == "BUDGET_T1_ITERS"

    # Quantum triggers only when no wall/budget reasons apply
    reason = _should_yield(sc, {"ms": 25})
    assert reason == "QUANTUM_EXCEEDED"

    # No yield if none of the thresholds are met
    reason = _should_yield(sc, {"ms": 5})
    assert reason is None


def test_scheduler_jsonl_written(monkeypatch, tmp_path):
    # Route logs_dir() to a temporary directory
    import clematis.io.paths as paths_module

    monkeypatch.setattr(paths_module, "logs_dir", lambda: str(tmp_path), raising=True)

    # Use the central JSONL writer
    from clematis.io.log import append_jsonl

    record = {
        "turn": 123,
        "slice": 1,
        "agent": "AgentA",
        "policy": "round_robin",
        "reason": "BUDGET_T2_K",
        "stage_end": "T2",
        "quantum_ms": 20,
        "wall_ms": 200,
        "budgets": {"t1_iters": 50, "t2_k": 64, "t3_ops": 3},
        "consumed": {"ms": 19, "t1_iters": 12, "t2_k": 64, "t3_ops": 0},
        "queued": ["AgentB", "AgentC"],
        "ms": 0,
    }

    append_jsonl("scheduler.jsonl", record)

    p = tmp_path / "scheduler.jsonl"
    assert p.exists(), "scheduler.jsonl should be created in logs_dir()"
    lines = p.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1, "expected exactly one record written"

    parsed = json.loads(lines[0])
    # Minimal required keys to consider the record valid
    required = {
        "turn",
        "slice",
        "agent",
        "policy",
        "reason",
        "stage_end",
        "quantum_ms",
        "wall_ms",
        "budgets",
        "consumed",
    }
    assert required.issubset(parsed.keys()), f"missing keys: {required - set(parsed.keys())}"
    assert parsed["agent"] == "AgentA"
    assert isinstance(parsed["budgets"], dict) and isinstance(parsed["consumed"], dict)
