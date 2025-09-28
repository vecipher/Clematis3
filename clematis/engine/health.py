from __future__ import annotations
from typing import Dict, Any


def check_and_log(ctx, state, t1, t2, t4, apply, log_fn):
    # Minimal health checks for demo
    evt = {"turn": ctx.turn_id, "agent": ctx.agent_id, "code": "OK", "message": "demo"}
    log_fn("health.jsonl", evt)
