# Clematis3 — M5 Scheduler Core tests (PR25)
# Focus: deterministic selection & bookkeeping, no orchestrator wiring.

import pytest

from clematis.scheduler import (
    init_scheduler_state,
    next_turn,
    on_yield,
)


class FixedCtx:
    """Tiny fixed clock for deterministic tests."""

    def __init__(self, t=0):
        self._t = int(t)

    def now_ms(self):
        return self._t

    def set(self, t):
        self._t = int(t)


def test_round_robin_determinism_and_bookkeeping():
    # Unsorted input; init must canonicalize to lex order.
    ids = ["Ringer", "Ambrose", "Kafka"]
    state = init_scheduler_state(ids, now_ms=0)
    assert state["queue"] == ["Ambrose", "Kafka", "Ringer"]

    ctx = FixedCtx(0)
    agent, budgets, reason = next_turn(
        ctx, state, policy="round_robin", fairness_cfg={"aging_ms": 200}
    )
    assert agent == "Ambrose"
    assert reason == "ROUND_ROBIN"
    assert budgets == {}

    # Bookkeeping on yield: last_ran updated, consec incremented; queue unchanged.
    on_yield(
        ctx, state, agent, consumed={}, reason="EXPLICIT_YIELD", fairness_cfg={"aging_ms": 200}
    )
    assert state["last_ran_ms"]["Ambrose"] == 0
    assert state["consec_turns"]["Ambrose"] == 1
    assert state["queue"] == ["Ambrose", "Kafka", "Ringer"]


def test_fair_queue_aging_tiers_and_tie_break():
    state = init_scheduler_state(["Ambrose", "Kafka", "Ringer"], now_ms=0)
    ctx = FixedCtx(500)

    # All idle = 500; with aging_ms=200 -> tier=2 for all; tie-break lex => Ambrose.
    agent, _, reason = next_turn(ctx, state, policy="fair_queue", fairness_cfg={"aging_ms": 200})
    assert agent == "Ambrose"
    assert reason == "AGING_BOOST"

    # Yield updates last_ran for Ambrose only.
    on_yield(
        ctx, state, agent, consumed={}, reason="EXPLICIT_YIELD", fairness_cfg={"aging_ms": 200}
    )
    assert state["last_ran_ms"]["Ambrose"] == 500

    # Advance time: A idle=200 (tier=1), K/R idle=700 (tier=3) => pick Kafka (lex among K/R).
    ctx.set(700)
    agent2, _, _ = next_turn(ctx, state, policy="fair_queue", fairness_cfg={"aging_ms": 200})
    assert agent2 == "Kafka"


def test_fair_queue_degenerate_aging_falls_back_to_rr():
    state = init_scheduler_state(["Kafka", "Ambrose"], now_ms=100)
    ctx = FixedCtx(150)

    # aging_ms <= 0 triggers RR behavior; reason remains AGING_BOOST by design.
    agent, _, reason = next_turn(ctx, state, policy="fair_queue", fairness_cfg={"aging_ms": 0})
    assert agent == "Ambrose"
    assert reason == "AGING_BOOST"

    agent_rr, _, _ = next_turn(ctx, state, policy="round_robin", fairness_cfg={"aging_ms": 0})
    assert agent_rr == agent


def test_empty_queue_behavior_is_stable():
    state = init_scheduler_state([], now_ms=0)
    ctx = FixedCtx(0)

    a1, _, r1 = next_turn(ctx, state, policy="round_robin", fairness_cfg={"aging_ms": 200})
    assert a1 == ""
    assert r1 == "ROUND_ROBIN"

    a2, _, r2 = next_turn(ctx, state, policy="fair_queue", fairness_cfg={"aging_ms": 200})
    assert a2 == ""
    assert r2 == "AGING_BOOST"


def test_negative_idle_guard_chooses_other_agent():
    # Simulate clock skew: Ambrose "ran in the future".
    state = init_scheduler_state(["Ambrose", "Kafka"], now_ms=1000)
    state["last_ran_ms"]["Ambrose"] = 2000  # future run
    ctx = FixedCtx(1500)

    # idle(A)=max(0,1500-2000)=0 => tier=0; idle(K)=500 => tier=2 -> choose Kafka.
    agent, _, _ = next_turn(ctx, state, policy="fair_queue", fairness_cfg={"aging_ms": 200})
    assert agent == "Kafka"
