"""
Compatibility shim (PR25 → PR26):
The scheduler core moved to clematis.engine.scheduler.
This shim re-exports the public API without side effects.
"""

from clematis.engine.scheduler import (
    SchedulerState,
    FairnessCfg,
    ReasonPick,
    init_scheduler_state,
    next_turn,
    on_yield,
)

__all__ = [
    "SchedulerState",
    "FairnessCfg",
    "ReasonPick",
    "init_scheduler_state",
    "next_turn",
    "on_yield",
]
