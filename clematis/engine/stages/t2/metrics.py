from __future__ import annotations

from typing import Any, Dict, List, Optional

from .config import metrics_gate_on


def emit_t2_metrics(
    event: Dict[str, Any],
    cfg: Any,
    *,
    reader_engaged: bool,
    shard_count: Optional[int] = None,
    partition_layout: Optional[str] = None,
    tier_sequence: Optional[List[str]] = None,
) -> None:
    if metrics_gate_on(cfg):
        metrics = event.setdefault("metrics", {})
        if reader_engaged:
            metrics["tier_sequence"] = tier_sequence or ["embed_store"]
            if shard_count is not None:
                metrics["reader_shards"] = int(shard_count)
            if partition_layout:
                metrics["partition_layout"] = str(partition_layout)
    else:
        if "metrics" in event and not event["metrics"]:
            del event["metrics"]


def estimate_result_cost(retrieved: List[Any], metrics: Dict[str, Any]) -> int:
    try:
        texts_sum = sum(len(getattr(ep, "text", "") or "") for ep in (retrieved or []))
        ids_sum = sum(len(getattr(ep, "id", "") or "") for ep in (retrieved or []))
        base = 16 * len(retrieved or [])
        metric_overhead = 64
        k_used = int((metrics or {}).get("k_used", 0)) if isinstance(metrics, dict) else 0
        k_ret = int((metrics or {}).get("k_returned", 0)) if isinstance(metrics, dict) else 0
        return texts_sum + ids_sum + base + metric_overhead + 8 * (k_used + k_ret)
    except Exception:
        return 1024
