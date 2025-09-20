# Copyright (c) 2025.
# Clematis3 — M5 Scheduler Core (PR25)
# -----------------------------------------------------------------------------
# This module implements the pure, engine-level scheduler core used by M5.
# Scope for PR25:
#   • Pure selection + state bookkeeping; NO orchestrator wiring here.
#   • Deterministic policies: round-robin and fair-queue with aging tiers.
#   • No RNG; tie-breaks by lexicographic agent_id.
#   • No queue rotation or max_consecutive_turns enforcement yet (PR26/PR27).
#   • Clock is provided by ctx.now_ms() if available; otherwise 0 (tests stub).
#
# Safe to import in tests; does not touch logs or runtime behavior by itself.
# -----------------------------------------------------------------------------

from __future__ import annotations

from typing import Dict, List, Tuple, TypedDict, Literal


# --------------------------- Types & Public API ------------------------------

ReasonPick = Literal["ROUND_ROBIN", "AGING_BOOST"]


class SchedulerState(TypedDict):
    """
    Pure scheduler state. The orchestrator owns lifecycle; this state is
    created once and updated via `on_yield()`. No rotations happen here.
    """
    queue: List[str]              # canonical lex order at init; no mutation here
    last_ran_ms: Dict[str, int]   # last slice-end time per agent (for aging)
    consec_turns: Dict[str, int]  # tracked only; enforcement deferred to PR27


class FairnessCfg(TypedDict, total=False):
    """
    Fairness-related parameters. Only 'aging_ms' is used in PR25.
    """
    aging_ms: int  # integer bucket size for fair-queue priority tiers


__all__ = [
    "SchedulerState",
    "FairnessCfg",
    "ReasonPick",
    "init_scheduler_state",
    "next_turn",
    "on_yield",
]


# ------------------------------- Helpers ------------------------------------

def _now_ms(ctx) -> int:
    """
    Deterministic clock accessor.
    Returns ctx.now_ms() if present, else 0 (tests should stub ctx.now_ms()).
    """
    try:
        fn = getattr(ctx, "now_ms", None)
        if callable(fn):
            return int(fn())
    except Exception:
        # Never raise from the core; fallback to 0 for determinism in tests.
        pass
    return 0


def init_scheduler_state(agent_ids: List[str], now_ms: int = 0) -> SchedulerState:
    """
    Initialize scheduler state with a canonical lexicographic queue and
    zeroed counters. No side-effects, no orchestrator dependencies.
    """
    q = sorted(agent_ids)
    return {
        "queue": q,
        "last_ran_ms": {a: int(now_ms) for a in q},
        "consec_turns": {a: 0 for a in q},
    }


def _pick_round_robin(sched: SchedulerState) -> str:
    """
    Pure round-robin pick: head of the queue.
    Rotation is the orchestrator's job in PR26; do not mutate here.
    """
    return sched["queue"][0] if sched["queue"] else ""


def _pick_fair_queue(sched: SchedulerState, now_ms: int, aging_ms: int) -> str:
    """
    Deterministic fair-queue pick using integer aging tiers.

        tier(agent) = max(0, now_ms - last_ran_ms[agent]) // aging_ms

    Choose highest tier; tie-break lex(agent_id). If aging_ms <= 0, fall back
    to round-robin deterministically.
    """
    if not sched["queue"]:
        return ""
    if aging_ms <= 0:
        return _pick_round_robin(sched)

    best_agent = None
    best_tier = -1
    for a in sched["queue"]:
        last = sched["last_ran_ms"].get(a, 0)
        idle = now_ms - last
        if idle < 0:
            # Guard against clock skew; never let it go negative.
            idle = 0
        tier = idle // aging_ms
        if tier > best_tier or (tier == best_tier and (best_agent is None or a < best_agent)):
            best_tier = tier
            best_agent = a
    return best_agent or ""


# ------------------------------- Core API -----------------------------------

def next_turn(
    ctx,
    sched: SchedulerState,
    policy: str,
    fairness_cfg: FairnessCfg,
) -> Tuple[str, Dict[str, int], ReasonPick]:
    """
    Pure selection. Returns (agent_id, slice_budgets, reason).

    - In PR25, `slice_budgets` is an empty dict to keep the core decoupled.
      The orchestrator (PR26) will derive budgets from config and inject them.

    - Determinism:
        * "round_robin": pick head of queue.
        * "fair_queue" : pick by aging tiers (idle_ms // aging_ms), tie-break lex.

    - No queue mutations here; the orchestrator handles rotations in later PRs.
    """
    now = _now_ms(ctx)
    if policy == "fair_queue":
        agent = _pick_fair_queue(sched, now, int(fairness_cfg.get("aging_ms", 200)))
        return agent, {}, "AGING_BOOST"
    # Default / unknown policy falls back to round-robin deterministically
    agent = _pick_round_robin(sched)
    return agent, {}, "ROUND_ROBIN"


def on_yield(
    ctx,
    sched: SchedulerState,
    agent_id: str,
    consumed: Dict[str, int],
    reason: str,
    fairness_cfg: FairnessCfg,
) -> None:
    """
    Post-slice bookkeeping (no ordering changes in PR25):

    - Update last_ran_ms[agent_id] = ctx.now_ms()
    - Increment consec_turns[agent_id]

    Parameters:
        consumed: forward-compat placeholder for recording per-slice counters.
        reason  : yield reason string (enum will be formalized in PR26).
    """
    now = _now_ms(ctx)
    if agent_id in sched["last_ran_ms"]:
        sched["last_ran_ms"][agent_id] = now
    if agent_id in sched["consec_turns"]:
        sched["consec_turns"][agent_id] += 1