from __future__ import annotations
from typing import Any, Dict
from dataclasses import asdict
from .types import TurnCtx, TurnResult
from .stages.t1 import t1_propagate
from .stages.t2 import t2_semantic
from .stages.t3 import make_plan_bundle, make_dialog_bundle
from .stages.t4 import t4_filter
from .stages.apply_stage import apply_changes
from ..io.log import append_jsonl

class Orchestrator:
    def run_turn(self, ctx: TurnCtx, state: Dict[str, Any], input_text: str) -> TurnResult:
        t1 = t1_propagate(ctx, state, input_text)
        append_jsonl("t1.jsonl", {"turn":ctx.turn_id, "agent":ctx.agent_id, **t1.metrics})

        t2 = t2_semantic(ctx, state, input_text, t1)
        append_jsonl("t2.jsonl", {"turn":ctx.turn_id, "agent":ctx.agent_id, **t2.metrics})

        plan = {"version":"t3-plan-v1","ops":[{"kind":"Speak","payload":{"style":"neutral"}}], "request_retrieve": None, "reflection": False}
        append_jsonl("t3_plan.jsonl", {"turn":ctx.turn_id, "agent":ctx.agent_id, "ops_counts":{"Speak":1}, "requested_retrieve": False, "reflection": False})

        utter = "Hello (demo)."
        append_jsonl("t3_dialogue.jsonl", {"turn":ctx.turn_id, "agent":ctx.agent_id, "tokens_in":0, "tokens_out":3})

        t4 = t4_filter(ctx, state, t1, t2, plan)
        append_jsonl("t4.jsonl", {"turn":ctx.turn_id, "agent":ctx.agent_id, **t4.metrics, "approved":len(t4.approved_deltas)})

        apply = apply_changes(ctx, state, t4)
        append_jsonl("apply.jsonl", {"turn":ctx.turn_id, "agent":ctx.agent_id, **apply.applied, "snapshot_id":"demo"})

        # health
        from . import health
        health.check_and_log(ctx, state, t1, t2, t4, apply, append_jsonl)

        return TurnResult(line=apply.line, events=[])
