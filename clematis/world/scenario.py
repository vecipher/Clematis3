from __future__ import annotations
from datetime import datetime, timezone
from ..engine.types import TurnCtx, Config
from ..engine.orchestrator import Orchestrator
from ..graph.store import InMemoryGraphStore, Node, Edge

def run_one_turn(
    agent_id: str,
    state: dict,
    text: str,
    cfg: Config,
    *,
    pick_reason: str | None = None,
) -> str:
    ctx = TurnCtx(turn_id="demo-1", agent_id=agent_id, scene_tags=["demo"], now=datetime.now(timezone.utc).isoformat(), cfg=cfg)
    if pick_reason is not None:
        # PR28: allow driver/demo to propagate selection rationale into scheduler logs
        setattr(ctx, "_sched_pick_reason", str(pick_reason))
    # Ensure a graph store & a tiny surface graph for the demo
    store = state.setdefault("store", InMemoryGraphStore())
    g = store.ensure("g:surface")
    if not g.nodes:  # first run: seed nodes/edges
        store.upsert_nodes("g:surface", [Node(id="n:hello", label="hello"), Node(id="n:world", label="world"), Node(id="n:reply", label="reply")])
        store.upsert_edges("g:surface", [
            Edge(id="e:h->w", src="n:hello", dst="n:world", weight=0.8, rel="supports"),
            Edge(id="e:w->r", src="n:world", dst="n:reply", weight=0.5, rel="associates"),
        ])
    state["active_graphs"] = ["g:surface"]
    orch = Orchestrator()
    result = orch.run_turn(ctx, state, text)
    return result.line
