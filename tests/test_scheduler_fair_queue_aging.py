# PR27 — Fairness: fair_queue aging tests (core scheduler only)
# We validate deterministic aging tiers and lexicographic tie-breaks.

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


def test_picks_highest_tier_then_updates_idle():
    # Initialize three agents in lex order: Ambrose < Kafka < Ringer
    state = init_scheduler_state(["Kafka", "Ambrose", "Ringer"], now_ms=0)

    # Set last_ran so Ambrose is most idle (highest tier), Kafka next, Ringer least
    # now=1000; aging_ms=200 -> tiers: A=900//200=4, K=400//200=2, R=100//200=0
    state["last_ran_ms"]["Ambrose"] = 100
    state["last_ran_ms"]["Kafka"] = 600
    state["last_ran_ms"]["Ringer"] = 900
    ctx = FixedCtx(1000)
    fairness = {"aging_ms": 200, "max_consecutive_turns": 10}

    a1, _, r1 = next_turn(ctx, state, policy="fair_queue", fairness_cfg=fairness)
    assert a1 == "Ambrose" and r1 == "AGING_BOOST"

    # on_yield updates Ambrose's last_ran to now (1000). With the same now, Ambrose idle=0,
    # so the next highest tier between Kafka (idle=400 -> tier=2) and Ringer (idle=100 -> tier=0) is Kafka.
    on_yield(ctx, state, a1, consumed={}, reason="SLICE", fairness_cfg=fairness)
    a2, _, r2 = next_turn(ctx, state, policy="fair_queue", fairness_cfg=fairness)
    assert a2 == "Kafka" and r2 == "AGING_BOOST"


def test_lex_tiebreak_when_tiers_equal():
    state = init_scheduler_state(["Ringer", "Kafka", "Ambrose"], now_ms=0)
    # now=600; set all last_ran so idle=400 -> tier=2 for all (aging_ms=200)
    state["last_ran_ms"]["Ambrose"] = 200
    state["last_ran_ms"]["Kafka"] = 200
    state["last_ran_ms"]["Ringer"] = 200
    ctx = FixedCtx(600)
    fairness = {"aging_ms": 200, "max_consecutive_turns": 10}

    a, _, r = next_turn(ctx, state, policy="fair_queue", fairness_cfg=fairness)
    # All tiers equal; tie-break lex → Ambrose
    assert a == "Ambrose" and r == "AGING_BOOST"


def test_aging_ms_zero_falls_back_to_lex_pick():
    state = init_scheduler_state(["Kafka", "Ambrose"], now_ms=0)
    # Make Kafka appear older, but with aging_ms=0 tiers collapse to 0 so it's lex pick.
    state["last_ran_ms"]["Kafka"] = 0
    state["last_ran_ms"]["Ambrose"] = 500
    ctx = FixedCtx(1000)
    fairness = {"aging_ms": 0, "max_consecutive_turns": 10}

    a, _, r = next_turn(ctx, state, policy="fair_queue", fairness_cfg=fairness)
    assert a == "Ambrose" and r == "AGING_BOOST"


def test_future_last_ran_is_clamped_to_zero_idle():
    # If an agent's last_ran is in the future, its idle must not go negative.
    state = init_scheduler_state(["Ambrose", "Kafka"], now_ms=0)
    state["last_ran_ms"]["Ambrose"] = 2000  # future
    state["last_ran_ms"]["Kafka"] = 500  # past
    ctx = FixedCtx(1500)
    fairness = {"aging_ms": 200, "max_consecutive_turns": 10}

    # idle(Ambrose)=max(0,1500-2000)=0 -> tier 0
    # idle(Kafka)=1000 -> tier 5 -> pick Kafka
    a, _, r = next_turn(ctx, state, policy="fair_queue", fairness_cfg=fairness)
    assert a == "Kafka" and r == "AGING_BOOST"
