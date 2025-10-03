from __future__ import annotations
from typing import Any, Dict, Tuple

from .bundle import assemble_bundle
from .metrics import finalize as finalize_metrics
from .policy import run_policy, select_policy


def t3_pipeline(ctx: Any, state: Any, t1: Any, t2: Any) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    bundle = assemble_bundle(ctx, state, t1, t2)
    cfg_root = bundle.get("cfg", {}) if isinstance(bundle, dict) else {}
    handle = select_policy(cfg_root, ctx)
    policy_output = run_policy(handle, bundle, cfg_root, ctx, state=state)
    steps = len(policy_output.get("plan", [])) if isinstance(policy_output, dict) else 0
    base_metrics = {
        "bundle": {
            "t1": bool(bundle.get("t1")),
            "t2": bool(bundle.get("t2")),
        }
    }
    metrics = finalize_metrics(
        cfg_root,
        base_metrics,
        policy_name=handle.get("name", "unknown"),
        steps_count=steps,
    )
    return bundle, metrics
