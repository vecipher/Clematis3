from __future__ import annotations
from typing import Dict, Any
from ..types import T1Result

def t1_propagate(ctx, state, text: str) -> T1Result:
    # No-op deltas + minimal metrics for demo
    deltas = []
    metrics = {"pops": 0, "radius_cap_hits": 0, "node_budget_hits": 0, "max_delta": 0.0, "graphs_touched": 0}
    return T1Result(graph_deltas=deltas, metrics=metrics)
