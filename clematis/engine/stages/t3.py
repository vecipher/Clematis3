from __future__ import annotations
from typing import Dict, Any
from ..types import Plan

def make_plan_bundle(ctx, state, t1, t2) -> Dict[str, Any]:
    return {"turn":{"id":ctx.turn_id,"agent":ctx.agent_id,"scene":ctx.scene_tags},
            "surface": {"nodes": {}, "edges": []},
            "t1":{"deltas": t1.graph_deltas},
            "t2":{"retrieval":{"snippets": []}, "residual": t2.graph_deltas_residual},
            "menu": ["CreateGraph","EditGraph","SetMetaFilter","RequestRetrieve","Speak"],
            "limits":{"max_rag_loops": ctx.cfg.t3.get("max_rag_loops",1)}}

def make_dialog_bundle(ctx, state, t1, t2, plan) -> Dict[str, Any]:
    return {"persona": {"style": "neutral"},
            "context": {"last_partner_line": "", "scene_tags": ctx.scene_tags},
            "surface": {"top_nodes": [], "vectors": {} },
            "retrieval_snippets": [],
            "constraints": {"max_tokens": ctx.cfg.t3.get("tokens",512)}}
