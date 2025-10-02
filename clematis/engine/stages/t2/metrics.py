from __future__ import annotations

from typing import Any, Dict, List, Optional

from .config import metrics_gate_on, cfg_get as _cfg_get


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


def assemble_metrics(
    *,
    cfg: Any,
    base_metrics: Dict[str, Any],
    tier_sequence: Optional[List[str]] = None,
    tiers: Optional[List[str]] = None,
    gate_on: Optional[bool] = None,
    use_reader: bool = False,
    perf_enabled: Optional[bool] = None,
    metrics_enabled: Optional[bool] = None,
    partitions_cfg: Optional[Dict[str, Any]] = None,
    pr33_store_dtype: Optional[str] = None,
    pr33_layout: Optional[str] = None,
    pr33_shards: Optional[int] = None,
    reader_mode: Optional[str] = None,
    t2_task_count: int = 0,
    t2_parallel_workers: int = 0,
    t2_partition_count: int = 0,
    backend_fallback_reason: Optional[str] = None,
    hybrid_used: bool = False,
    hybrid_info: Optional[Dict[str, Any]] = None,
    q_fusion_used: bool = False,
    q_fusion_meta: Optional[Dict[str, Any]] = None,
    q_mmr_used: bool = False,
    q_mmr_selected_n: int = 0,
) -> Dict[str, Any]:
    """Pure assembly of the T2 metrics dict matching core.py shape.

    This helper mirrors the existing structure produced in core.py when the
    perf/metrics gate is ON. It does **not** emit traces and does not compute
    diversity (left to core.py since it needs access to `retrieved` texts).
    """
    out: Dict[str, Any] = dict(base_metrics or {})

    if gate_on is None:
        gate_on = metrics_gate_on(cfg)
    if not gate_on:
        return out

    # Reader diagnostics only when the reader actually engaged under perf gate
    if use_reader and perf_enabled:
        partitions_list: List[str] = []
        if isinstance(partitions_cfg, dict):
            by = partitions_cfg.get("by") or partitions_cfg.get("partitions")
            if isinstance(by, (list, tuple)):
                partitions_list = [str(x) for x in by]
        reader_meta = {
            "embed_store_dtype": pr33_store_dtype,
            "precompute_norms": bool(_cfg_get(cfg, ["perf", "t2", "precompute_norms"], False)),
            "layout": pr33_layout,
            "shards": int(pr33_shards or 0),
        }
        if partitions_list:
            reader_meta["partitions"] = list(partitions_list)
        out["reader"] = reader_meta

    # Selected reader mode for observability
    if reader_mode:
        out["t2.reader_mode"] = str(reader_mode)

    if backend_fallback_reason:
        out["backend_fallback_reason"] = str(backend_fallback_reason)

    # Quality (fusion/MMR) metrics (compact) — match existing key names
    if bool(_cfg_get(cfg, ["t2", "quality", "enabled"], False)) and q_fusion_used:
        out["t2q.fusion_mode"] = "score_interp"
        out["t2q.alpha_semantic"] = float(_cfg_get(cfg, ["t2", "quality", "fusion", "alpha_semantic"], 0.6))
        if isinstance(q_fusion_meta, dict) and "t2q.lex_hits" in q_fusion_meta:
            try:
                out["t2q.lex_hits"] = int(q_fusion_meta["t2q.lex_hits"])  # type: ignore[index]
            except Exception:
                pass

    if bool(_cfg_get(cfg, ["t2", "quality", "enabled"], False)):
        try:
            lam = float(_cfg_get(cfg, ["t2", "quality", "mmr", "lambda"], 0.5))
            out["t2q.mmr.lambda"] = lam
            selected_n = int(q_mmr_selected_n or 0) if q_mmr_used else 0
            out["t2q.mmr.selected"] = selected_n
        except Exception:
            pass

    # PR33: gated perf counters (namespaced)
    out["t2.embed_dtype"] = "fp32"
    if pr33_store_dtype:
        out["t2.embed_store_dtype"] = str(pr33_store_dtype)
    out["t2.precompute_norms"] = bool(_cfg_get(cfg, ["perf", "t2", "precompute_norms"], False))
    if pr33_shards is not None:
        out["t2.reader_shards"] = int(pr33_shards)
    if pr33_layout is not None:
        out["t2.partition_layout"] = str(pr33_layout)
    # Parallel stats
    if int(t2_task_count or 0) > 0:
        out["t2.task_count"] = int(t2_task_count)
        out["t2.parallel_workers"] = int(t2_parallel_workers or 0)
    if int(t2_partition_count or 0) > 0:
        out["t2.partition_count"] = int(t2_partition_count)

    return out

def finalize(
    *,
    cfg: Any,
    base_metrics: Dict[str, Any],
    tier_sequence: Optional[List[str]] = None,
    tiers: Optional[List[str]] = None,
    gate_on: Optional[bool] = None,
    use_reader: bool = False,
    perf_enabled: Optional[bool] = None,
    metrics_enabled: Optional[bool] = None,
    partitions_cfg: Optional[Dict[str, Any]] = None,
    pr33_store_dtype: Optional[str] = None,
    pr33_layout: Optional[str] = None,
    pr33_shards: Optional[int] = None,
    reader_mode: Optional[str] = None,
    t2_task_count: int = 0,
    t2_parallel_workers: int = 0,
    t2_partition_count: int = 0,
    backend_fallback_reason: Optional[str] = None,
    hybrid_used: bool = False,
    hybrid_info: Optional[Dict[str, Any]] = None,
    q_fusion_used: bool = False,
    q_fusion_meta: Optional[Dict[str, Any]] = None,
    q_mmr_used: bool = False,
    q_mmr_selected_n: int = 0,
) -> Dict[str, Any]:
    """Phase‑1 finalize: assemble metrics only (no emits).
    This keeps behavior identical to the current code path while letting
    core.py shrink. In a later phase we can optionally add `_emit_t2_metrics`
    here once tests are green.
    """
    return assemble_metrics(
        cfg=cfg,
        base_metrics=base_metrics,
        tier_sequence=tier_sequence,
        tiers=tiers,
        gate_on=gate_on,
        use_reader=use_reader,
        perf_enabled=perf_enabled,
        metrics_enabled=metrics_enabled,
        partitions_cfg=partitions_cfg,
        pr33_store_dtype=pr33_store_dtype,
        pr33_layout=pr33_layout,
        pr33_shards=pr33_shards,
        reader_mode=reader_mode,
        t2_task_count=t2_task_count,
        t2_parallel_workers=t2_parallel_workers,
        t2_partition_count=t2_partition_count,
        backend_fallback_reason=backend_fallback_reason,
        hybrid_used=hybrid_used,
        hybrid_info=hybrid_info,
        q_fusion_used=q_fusion_used,
        q_fusion_meta=q_fusion_meta,
        q_mmr_used=q_mmr_used,
        q_mmr_selected_n=q_mmr_selected_n,
    )
