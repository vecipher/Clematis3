from __future__ import annotations
from typing import Any, Dict
import time
from .types import TurnCtx, TurnResult
from .stages.t1 import t1_propagate
from .stages.t2 import t2_semantic
from .stages.t3 import make_plan_bundle, make_dialog_bundle  # placeholders for future PRs
from .stages.t4 import t4_filter
from .stages.apply_stage import apply_changes
from ..io.log import append_jsonl


class Orchestrator:
    """Single canonical turn loop with first-class observability.

    Stages: T1 (propagate) → T2 (semantic) → T3 (placeholder) → T4 (meta-filter) → Apply → Health
    This adds precise per-stage durations and a turn summary while keeping behavior unchanged.
    """

    def run_turn(self, ctx: TurnCtx, state: Dict[str, Any], input_text: str) -> TurnResult:
        turn_id = getattr(ctx, "turn_id", "-")
        agent_id = getattr(ctx, "agent_id", "-")
        now = getattr(ctx, "now", None)

        total_t0 = time.perf_counter()

        # --- T1 ---
        t0 = time.perf_counter()
        t1 = t1_propagate(ctx, state, input_text)
        t1_ms = round((time.perf_counter() - t0) * 1000.0, 3)
        append_jsonl(
            "t1.jsonl",
            {
                "turn": turn_id,
                "agent": agent_id,
                **t1.metrics,
                "ms": t1_ms,
                **({"now": now} if now else {}),
            },
        )

        # --- T2 ---
        t0 = time.perf_counter()
        t2 = t2_semantic(ctx, state, input_text, t1)
        t2_ms = round((time.perf_counter() - t0) * 1000.0, 3)
        append_jsonl(
            "t2.jsonl",
            {
                "turn": turn_id,
                "agent": agent_id,
                **t2.metrics,
                "ms": t2_ms,
                **({"now": now} if now else {}),
            },
        )

        # --- T3 placeholders (deliberation/dialogue) ---
        # These are stubs until PR3; keep minimal but observable.
        plan = {
            "version": "t3-plan-v1",
            "ops": [{"kind": "Speak", "payload": {"style": "neutral"}}],
            "request_retrieve": None,
            "reflection": False,
        }
        append_jsonl(
            "t3_plan.jsonl",
            {
                "turn": turn_id,
                "agent": agent_id,
                "ops_counts": {"Speak": 1},
                "requested_retrieve": False,
                "reflection": False,
                **({"now": now} if now else {}),
            },
        )

        utter = "Hello (demo)."  # t3 dialogue stub
        append_jsonl(
            "t3_dialogue.jsonl",
            {
                "turn": turn_id,
                "agent": agent_id,
                "tokens_in": 0,
                "tokens_out": 3,
                **({"now": now} if now else {}),
            },
        )

        # --- T4 ---
        t0 = time.perf_counter()
        t4 = t4_filter(ctx, state, t1, t2, plan, utter)
        t4_ms = round((time.perf_counter() - t0) * 1000.0, 3)
        append_jsonl(
            "t4.jsonl",
            {
                "turn": turn_id,
                "agent": agent_id,
                **t4.metrics,
                "approved": len(getattr(t4, "approved_deltas", [])),
                "rejected": len(getattr(t4, "rejected_ops", [])),
                "ms": t4_ms,
                **({"now": now} if now else {}),
            },
        )

        # --- Apply & persist ---
        t0 = time.perf_counter()
        apply = apply_changes(ctx, state, t4)
        apply_ms = round((time.perf_counter() - t0) * 1000.0, 3)
        append_jsonl(
            "apply.jsonl",
            {
                "turn": turn_id,
                "agent": agent_id,
                **apply.applied,
                "snapshot_id": "demo",
                "ms": apply_ms,
                **({"now": now} if now else {}),
            },
        )

        # --- Health summary + per-turn rollup ---
        from . import health
        health.check_and_log(ctx, state, t1, t2, t4, apply, append_jsonl)

        total_ms = round((time.perf_counter() - total_t0) * 1000.0, 3)
        append_jsonl(
            "turn.jsonl",
            {
                "turn": turn_id,
                "agent": agent_id,
                "durations_ms": {"t1": t1_ms, "t2": t2_ms, "t4": t4_ms, "apply": apply_ms, "total": total_ms},
                "t1": {
                    "pops": t1.metrics.get("pops"),
                    "iters": t1.metrics.get("iters"),
                    "graphs_touched": t1.metrics.get("graphs_touched"),
                },
                "t2": {
                    "k_returned": t2.metrics.get("k_returned"),
                    "k_used": t2.metrics.get("k_used"),
                    "cache_used": t2.metrics.get("cache_used"),
                },
                "t4": {
                    "approved": len(getattr(t4, "approved_deltas", [])),
                    "rejected": len(getattr(t4, "rejected_ops", [])),
                },
                **({"now": now} if now else {}),
            },
        )

        return TurnResult(line=apply.line, events=[])
