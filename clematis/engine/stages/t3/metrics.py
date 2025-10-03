from __future__ import annotations
from typing import Any, Dict


def finalize(
    cfg_root: Dict[str, Any],
    base_metrics: Dict[str, Any] | None = None,
    *,
    policy_name: str,
    steps_count: int,
    extra: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    metrics: Dict[str, Any] = {}
    if base_metrics:
        metrics.update(base_metrics)
    metrics["policy"] = str(policy_name)
    metrics["steps"] = int(steps_count)
    try:
        t3_cfg = (cfg_root.get("t3") if isinstance(cfg_root, dict) else None) or {}
        metrics["cfg_echo"] = {
            "tokens": int(t3_cfg.get("tokens", 256)),
            "max_ops": int(t3_cfg.get("max_ops_per_turn", 3)),
        }
    except Exception:
        metrics["cfg_echo"] = {}
    if extra:
        metrics.update(extra)
    return metrics


__all__ = ["finalize"]
