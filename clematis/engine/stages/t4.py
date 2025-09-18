from __future__ import annotations
from typing import Dict, Any, List
from ..types import T4Result

def t4_filter(ctx, state, t1, t2, plan) -> T4Result:
    # Approve nothing (demo). In real impl, gate deltas/ops.
    return T4Result(approved_deltas=[], rejected_ops=plan.ops, reasons=["demo-noop"], metrics={"delta_norm":0.0,"novelty":0.0,"churn":0})
