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

ReasonPick = Literal["ROUND_ROBIN", "AGING_BOOST", "RESET_CONSEC"]


class SchedulerState(TypedDict):
    """
    Pure scheduler state. The orchestrator owns lifecycle; this state is
    created once and updated via `on_yield()`. No rotations happen here.
    """

    queue: List[str]  # canonical lex order at init; no mutation here
    last_ran_ms: Dict[str, int]  # last slice-end time per agent (for aging)
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


# Max value used when fairness_cfg does not specify max_consecutive_turns
_MAX_INT = 10**9


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

    - slice_budgets is empty in PR25/27 core; orchestrator derives actual budgets.
    - Determinism:
        * "round_robin": pick first ELIGIBLE agent by queue order.
        * "fair_queue" : among ELIGIBLE agents, pick highest aging tier (idle_ms // aging_ms), tie-break lex.
    - Eligibility uses max_consecutive_turns (mct). If all agents are saturated (no eligible),
      select lexicographic-min agent and return reason="RESET_CONSEC" to signal a counter reset.
    - No queue mutations here; the orchestrator handles rotations (RR) in later stages.
    """
    now = _now_ms(ctx)
    aging_ms = int(fairness_cfg.get("aging_ms", 200))
    mct = int(getattr(fairness_cfg, "get", lambda *_: _MAX_INT)("max_consecutive_turns", _MAX_INT))  # type: ignore

    # Build eligible set under mct
    q = sched["queue"]
    if not q:
        return "", {}, "ROUND_ROBIN" if policy != "fair_queue" else "AGING_BOOST"

    eligible = [a for a in q if sched["consec_turns"].get(a, 0) < mct]

    # If no eligible agents, pick lex-min and signal RESET_CONSEC
    if not eligible:
        agent = min(q)  # deterministic
        return agent, {}, "RESET_CONSEC"

    if policy == "fair_queue":
        # Among eligible, choose highest tier; tie-break lex
        best_agent = None
        best_tier = -1
        for a in eligible:
            last = sched["last_ran_ms"].get(a, 0)
            idle = now - last
            if idle < 0:
                idle = 0
            tier = (idle // aging_ms) if aging_ms > 0 else 0
            if tier > best_tier or (tier == best_tier and (best_agent is None or a < best_agent)):
                best_tier = tier
                best_agent = a
        return (best_agent or eligible[0]), {}, "AGING_BOOST"
    else:
        # Round-robin: take first eligible according to queue order (no rotation here)
        return eligible[0], {}, "ROUND_ROBIN"


def on_yield(
    ctx,
    sched: SchedulerState,
    agent_id: str,
    consumed: Dict[str, int],
    reason: str,
    fairness_cfg: FairnessCfg,
    reset: bool = False,
) -> None:
    """
    Post-slice bookkeeping:

    - Always update last_ran_ms[agent_id] = ctx.now_ms()
    - Increment consec_turns[agent_id] by 1, unless `reset=True`, in which case
      all consec_turns are zeroed (deterministically) after the slice.

    Notes:
      * We do NOT rotate the queue here; RR rotation is the orchestrator's job.
      * `reset=True` is used when `next_turn` returned reason="RESET_CONSEC" (all agents saturated).
    """
    now = _now_ms(ctx)
    if agent_id in sched["last_ran_ms"]:
        sched["last_ran_ms"][agent_id] = now

    if reset:
        # Zero all counters deterministically
        for a in sched["consec_turns"].keys():
            sched["consec_turns"][a] = 0
        return

    if agent_id in sched["consec_turns"]:
        sched["consec_turns"][agent_id] += 1
