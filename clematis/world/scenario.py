from __future__ import annotations
from datetime import datetime, timezone
from ..engine.types import TurnCtx, Config
from ..engine.orchestrator import Orchestrator

def run_one_turn(agent_id: str, state: dict, text: str, cfg: Config) -> str:
    ctx = TurnCtx(turn_id="demo-1", agent_id=agent_id, scene_tags=["demo"], now=datetime.now(timezone.utc).isoformat(), cfg=cfg)
    orch = Orchestrator()
    result = orch.run_turn(ctx, state, text)
    return result.line
