# PR27 — Fairness: max_consecutive_turns tests (core scheduler only)
# We exercise next_turn/on_yield with mct=1 to ensure deterministic alternation
# and correct RESET_CONSEC behavior when all agents are saturated.

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


def test_round_robin_mct1_alternates_and_resets_two_agents():
    # Unsorted input; init must canonicalize to lex order ["Ambrose","Kafka"].
    state = init_scheduler_state(["Kafka", "Ambrose"], now_ms=0)
    ctx = FixedCtx(0)
    fairness = {"max_consecutive_turns": 1, "aging_ms": 200}

    # 1st pick → Ambrose
    a1, _, r1 = next_turn(ctx, state, policy="round_robin", fairness_cfg=fairness)
    assert a1 == "Ambrose" and r1 == "ROUND_ROBIN"
    on_yield(ctx, state, a1, consumed={}, reason="SLICE", fairness_cfg=fairness)

    # 2nd pick → Kafka (A is saturated at mct=1)
    a2, _, r2 = next_turn(ctx, state, policy="round_robin", fairness_cfg=fairness)
    assert a2 == "Kafka" and r2 == "ROUND_ROBIN"
    on_yield(ctx, state, a2, consumed={}, reason="SLICE", fairness_cfg=fairness)

    # 3rd pick → all saturated; expect RESET_CONSEC selecting lex-min ("Ambrose")
    a3, _, r3 = next_turn(ctx, state, policy="round_robin", fairness_cfg=fairness)
    assert r3 == "RESET_CONSEC" and a3 == "Ambrose"

    # After a RESET_CONSEC pick, reset counters at slice end
    on_yield(ctx, state, a3, consumed={}, reason="SLICE", fairness_cfg=fairness, reset=True)

    # 4th pick → Ambrose again (fresh counters), then alternation continues
    a4, _, r4 = next_turn(ctx, state, policy="round_robin", fairness_cfg=fairness)
    assert a4 == "Ambrose" and r4 == "ROUND_ROBIN"
    on_yield(ctx, state, a4, consumed={}, reason="SLICE", fairness_cfg=fairness)
    a5, _, _ = next_turn(ctx, state, policy="round_robin", fairness_cfg=fairness)
    assert a5 == "Kafka"


def test_round_robin_mct1_three_agents_then_reset():
    state = init_scheduler_state(["Ringer", "Kafka", "Ambrose"], now_ms=0)
    ctx = FixedCtx(0)
    fairness = {"max_consecutive_turns": 1, "aging_ms": 200}

    # Picks proceed in lex order until all hit 1
    picks = []
    for _ in range(3):
        a, _, r = next_turn(ctx, state, policy="round_robin", fairness_cfg=fairness)
        picks.append((a, r))
        on_yield(ctx, state, a, consumed={}, reason="SLICE", fairness_cfg=fairness)
    assert [p[0] for p in picks] == ["Ambrose", "Kafka", "Ringer"]
    assert all(r == "ROUND_ROBIN" for _, r in picks)

    # Now saturated: expect RESET_CONSEC → "Ambrose"
    a, _, r = next_turn(ctx, state, policy="round_robin", fairness_cfg=fairness)
    assert r == "RESET_CONSEC" and a == "Ambrose"
    on_yield(ctx, state, a, consumed={}, reason="SLICE", fairness_cfg=fairness, reset=True)

    # After reset, cycle restarts from lex-min
    a2, _, r2 = next_turn(ctx, state, policy="round_robin", fairness_cfg=fairness)
    assert a2 == "Ambrose" and r2 == "ROUND_ROBIN"


def test_fair_queue_respects_mct_even_with_highest_tier():
    # Make Ambrose appear most idle (highest tier), but mct=1 should exclude it after it runs once.
    state = init_scheduler_state(["Ambrose", "Kafka", "Ringer"], now_ms=0)
    # Simulate that Kafka and Ringer ran recently (lower idle)
    state["last_ran_ms"]["Kafka"] = 900
    state["last_ran_ms"]["Ringer"] = 900
    ctx = FixedCtx(1000)
    fairness = {"max_consecutive_turns": 1, "aging_ms": 200}

    # First pick (fair_queue): Ambrose (highest tier)
    a1, _, r1 = next_turn(ctx, state, policy="fair_queue", fairness_cfg=fairness)
    assert a1 == "Ambrose" and r1 == "AGING_BOOST"
    on_yield(ctx, state, a1, consumed={}, reason="SLICE", fairness_cfg=fairness)

    # Second pick: Ambrose is saturated; choose among eligible {Kafka,Ringer}
    a2, _, r2 = next_turn(ctx, state, policy="fair_queue", fairness_cfg=fairness)
    # Kafka and Ringer have equal tiers; tie-break lex → Kafka
    assert a2 == "Kafka" and r2 == "AGING_BOOST"
