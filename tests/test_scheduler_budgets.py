

# PR26 — Scheduler budget reason tests
# We validate that each budget cap (t1_iters, t1_pops, t2_k, t3_ops) triggers
# the correct yield reason at a stage boundary, and that budgets take precedence
# over quantum (but below wall).

import pytest


def _mk_slice_ctx(budgets):
    return {
        "slice_idx": 1,
        "started_ms": 0,
        "budgets": budgets,
        "agent_id": "AgentA",
    }


@pytest.mark.parametrize(
    "budget_key,consumed_key,reason_name,value",
    [
        ("t1_iters", "t1_iters", "BUDGET_T1_ITERS", 3),
        ("t1_pops",  "t1_pops",  "BUDGET_T1_POPS",  10),
        ("t2_k",     "t2_k",     "BUDGET_T2_K",     5),
        ("t3_ops",   "t3_ops",   "BUDGET_T3_OPS",   2),
    ],
)
def test_each_budget_triggers_reason(budget_key, consumed_key, reason_name, value):
    # Import the internal helper (added in PR26 scaffolding).
    from clematis.engine.orchestrator import _should_yield  # type: ignore

    budgets = {
        budget_key: value,
        "quantum_ms": 999999,  # ensure quantum won't fire
        "wall_ms": 9999999,    # ensure wall won't fire
    }
    sc = _mk_slice_ctx(budgets)
    # Keep elapsed small to avoid quantum; exactly hit the budget.
    consumed = {"ms": 1, consumed_key: value}
    reason = _should_yield(sc, consumed)
    assert reason == reason_name


def test_budget_precedes_quantum_when_both_hit():
    from clematis.engine.orchestrator import _should_yield  # type: ignore

    budgets = {
        "t3_ops": 2,
        "quantum_ms": 20,
        "wall_ms": 999999,
    }
    sc = _mk_slice_ctx(budgets)
    # Both exceeded: ms >= quantum and t3_ops == budget -> expect budget reason
    reason = _should_yield(sc, {"ms": 25, "t3_ops": 2})
    assert reason == "BUDGET_T3_OPS"


def test_no_yield_when_under_all_thresholds():
    from clematis.engine.orchestrator import _should_yield  # type: ignore

    budgets = {
        "t1_iters": 10,
        "t1_pops":  10,
        "t2_k":     10,
        "t3_ops":   10,
        "quantum_ms": 50,
        "wall_ms": 200,
    }
    sc = _mk_slice_ctx(budgets)
    reason = _should_yield(sc, {"ms": 5, "t1_iters": 1, "t1_pops": 1, "t2_k": 1, "t3_ops": 1})
    assert reason is None


def test_missing_budget_key_does_not_trigger():
    from clematis.engine.orchestrator import _should_yield  # type: ignore

    budgets = {
        # No t1_iters key present → should not trigger even if consumed has t1_iters
        "quantum_ms": 100,
        "wall_ms": 1000,
    }
    sc = _mk_slice_ctx(budgets)
    reason = _should_yield(sc, {"ms": 10, "t1_iters": 5})
    # Quantum not hit; no budget present; expect None
    assert reason is None