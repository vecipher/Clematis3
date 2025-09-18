from __future__ import annotations
from typing import Dict, Any, List
from ..types import T2Result, EpisodeRef

def t2_semantic(ctx, state, text: str, t1) -> T2Result:
    retrieved: List[EpisodeRef] = []
    residual = []
    metrics = {"tier_sequence": ["exact_semantic"], "k_returned": 0, "sim_stats": {}, "residual_count": 0, "cap_hits": 0}
    return T2Result(retrieved=retrieved, graph_deltas_residual=residual, metrics=metrics)
