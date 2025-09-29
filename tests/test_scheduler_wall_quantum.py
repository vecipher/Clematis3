# PR26 — WALL vs QUANTUM precedence tests (and edge cases)

import pytest


def _mk_slice_ctx(budgets):
    return {
        "slice_idx": 1,
        "started_ms": 0,
        "budgets": budgets,
        "agent_id": "AgentA",
    }


def test_wall_ms_beats_budget_and_quantum():
    from clematis.engine.orchestrator import _should_yield  # type: ignore

    budgets = {
        "t1_iters": 3,
        "quantum_ms": 20,
        "wall_ms": 50,
    }
    sc = _mk_slice_ctx(budgets)
    # All three conditions could be satisfied if we choose ms=50 and t1_iters==3.
    # Precedence requires WALL_MS to win.
    reason = _should_yield(sc, {"ms": 50, "t1_iters": 3})
    assert reason == "WALL_MS"


def test_quantum_triggers_when_no_wall_or_budget():
    from clematis.engine.orchestrator import _should_yield  # type: ignore

    budgets = {
        # No per-stage budgets present; wall is high so it won't trigger.
        "quantum_ms": 30,
        "wall_ms": 1000,
    }
    sc = _mk_slice_ctx(budgets)
    # Hit quantum exactly.
    reason = _should_yield(sc, {"ms": 30})
    assert reason == "QUANTUM_EXCEEDED"


def test_no_yield_when_under_all_thresholds():
    from clematis.engine.orchestrator import _should_yield  # type: ignore

    budgets = {
        "t1_iters": 10,
        "t1_pops": 10,
        "t2_k": 10,
        "t3_ops": 10,
        "quantum_ms": 50,
        "wall_ms": 200,
    }
    sc = _mk_slice_ctx(budgets)
    reason = _should_yield(sc, {"ms": 5, "t1_iters": 1, "t1_pops": 1, "t2_k": 1, "t3_ops": 1})
    assert reason is None


def test_wall_equality_triggers():
    from clematis.engine.orchestrator import _should_yield  # type: ignore

    budgets = {
        "quantum_ms": 10,
        "wall_ms": 10,  # equality should trigger wall
    }
    sc = _mk_slice_ctx(budgets)
    reason = _should_yield(sc, {"ms": 10})
    assert reason == "WALL_MS"


def test_quantum_only_when_no_budget_hit():
    from clematis.engine.orchestrator import _should_yield  # type: ignore

    budgets = {
        "t3_ops": 2,
        "quantum_ms": 20,
        "wall_ms": 999999,
    }
    sc = _mk_slice_ctx(budgets)
    # If we hit quantum but not the budget (ops < budget), quantum should fire.
    reason = _should_yield(sc, {"ms": 25, "t3_ops": 1})
    assert reason == "QUANTUM_EXCEEDED"
