from __future__ import annotations

from typing import Any, Dict, List, Tuple, Optional

from .config import cfg_get, ensure_dict, quality_cfg_snapshot as _quality_cfg_snapshot
from .helpers import items_for_fusion
from ..hybrid import rerank_with_gel

try:
    from .quality_trace import emit_trace as _emit_quality_trace  # type: ignore
except Exception:
    _emit_quality_trace = None  # shadow tracing is optional and must never break retrieval


QualityResult = Tuple[
    List[Any],
    bool,
    Dict[str, Any],
    bool,
    Dict[str, Any],
    bool,
    int,
]


def apply_quality(
    ctx,
    state,
    retrieved: List[Any],
    q_text: str,
    cfg_root: Dict[str, Any],
    cfg_t2: Dict[str, Any],
) -> QualityResult:
    hybrid_used = False
    hybrid_info: Dict[str, Any] = {}
    q_fusion_used = False
    q_fusion_meta: Dict[str, Any] = {}
    q_mmr_used = False
    q_mmr_selected_n = 0

    # Hybrid rerank
    hybrid_cfg = ensure_dict(cfg_t2.get("hybrid", {}))
    if bool(hybrid_cfg.get("enabled", False)):
        try:
            new_items, hmetrics = rerank_with_gel(ctx, state, retrieved)
            retrieved = new_items
            hybrid_used = bool(hmetrics.get("hybrid_used", False))
            hybrid_info = {
                key: hmetrics.get(key)
                for key in (
                    "anchor_top_m",
                    "walk_hops",
                    "edge_threshold",
                    "lambda_graph",
                    "damping",
                    "degree_norm",
                    "k_max",
                    "k_considered",
                    "k_reordered",
                )
                if key in hmetrics
            }
        except Exception:
            hybrid_used = False
            hybrid_info = {}

    # Fusion/MMR
    try:
        if bool(cfg_get(cfg_root, ["t2", "quality", "enabled"], False)):
            try:
                from .quality_ops import fuse as quality_fuse  # type: ignore
            except Exception:
                quality_fuse = None
            if quality_fuse is not None:
                items = items_for_fusion(retrieved)
                fused_out = quality_fuse(q_text, items, cfg=cfg_root)
                if isinstance(fused_out, tuple):
                    fused_items, q_fusion_meta = fused_out
                else:
                    fused_items, q_fusion_meta = fused_out, {}
                try:
                    sem_rank_map = {entry["id"]: idx + 1 for idx, entry in enumerate(items)}
                    score_fused_map = {
                        entry.get("id"): float(entry.get("score_fused", 0.0))
                        for entry in (fused_items or [])
                    }
                    if isinstance(q_fusion_meta, dict):
                        q_fusion_meta.setdefault("sem_rank_map", sem_rank_map)
                        q_fusion_meta.setdefault("score_fused_map", score_fused_map)
                except Exception:
                    pass
                id_to_ref = {ref.id: ref for ref in retrieved}
                new_order = [id_to_ref[it.get("id")] for it in fused_items if it.get("id") in id_to_ref]
                if new_order:
                    retrieved = new_order
                    q_fusion_used = True
                try:
                    from .quality_ops import maybe_apply_mmr as quality_mmr  # type: ignore
                except Exception:
                    quality_mmr = None
                mmr_enabled = bool(cfg_get(cfg_root, ["t2", "quality", "mmr", "enabled"], False))
                if quality_mmr is not None and mmr_enabled:
                    qcfg = cfg_get(cfg_root, ["t2", "quality"], {}) or {}
                    mmr_items = quality_mmr(fused_items, qcfg)
                    q_mmr_used = True
                    id_to_ref = {ref.id: ref for ref in retrieved}
                    mmr_order = [id_to_ref[it.get("id")] for it in (mmr_items or []) if it.get("id") in id_to_ref]
                    q_mmr_selected_n = len(mmr_order)
                    if mmr_order:
                        retrieved = mmr_order
    except Exception:
        q_fusion_used = False
        q_fusion_meta = {}

    if not q_mmr_used:
        try:
            try:
                from .quality_ops import maybe_apply_mmr as quality_mmr_fallback  # type: ignore
            except Exception:
                quality_mmr_fallback = None
            if quality_mmr_fallback is not None and bool(
                cfg_get(cfg_root, ["t2", "quality", "mmr", "enabled"], False)
            ):
                items_fb = items_for_fusion(retrieved)
                qcfg_fb = cfg_get(cfg_root, ["t2", "quality"], {}) or {}
                mmr_items_fb = quality_mmr_fallback(items_fb, qcfg_fb)
                q_mmr_used = True
                id_to_ref_fb = {ref.id: ref for ref in retrieved}
                mmr_order_fb = [
                    id_to_ref_fb[item.get("id")]
                    for item in (mmr_items_fb or [])
                    if item.get("id") in id_to_ref_fb
                ]
                q_mmr_selected_n = len(mmr_order_fb)
                if mmr_order_fb:
                    retrieved = mmr_order_fb
        except Exception:
            pass

    # --- PR36: Shadow tracing (no-op) — triple-gated; does not mutate rankings or metrics ---
    try:
        perf_enabled = bool(cfg_get(cfg_root, ["perf", "enabled"], False))
        metrics_enabled = bool(cfg_get(cfg_root, ["perf", "metrics", "report_memory"], False))
        q_enabled = bool(cfg_get(cfg_root, ["t2", "quality", "enabled"], False))
        q_shadow = bool(cfg_get(cfg_root, ["t2", "quality", "shadow"], False))
        if (
            _emit_quality_trace is not None
            and _quality_cfg_snapshot is not None
            and perf_enabled
            and metrics_enabled
            and q_shadow
            and not q_enabled
        ):
            # Best-effort reason: prefer ctx, then fallback to cfg.perf.metrics.trace_reason
            reason = None
            try:
                reason = getattr(ctx, "trace_reason", None)
                if reason is None and isinstance(ctx, dict):
                    reason = ctx.get("trace_reason")
            except Exception:
                reason = None
            if reason is None:
                try:
                    reason = cfg_get(cfg_root, ["perf", "metrics", "trace_reason"], None)
                except Exception:
                    reason = None

            cfg_snap = _quality_cfg_snapshot(cfg_root)
            # Trace the current ordering; rankings are unchanged in shadow mode
            trace_items = [
                {"id": getattr(ref, "id", None), "score": float(getattr(ref, "score", 0.0))}
                for ref in retrieved
            ]
            meta = {
                "k": len(retrieved),
                "reason": str(reason) if reason else "shadow",
                "note": "PR36 shadow trace; rankings unchanged",
            }
            _emit_quality_trace(cfg_snap, q_text, trace_items, meta)
    except Exception:
        # Never fail the request due to tracing issues
        pass

    return (
        retrieved,
        hybrid_used,
        hybrid_info,
        q_fusion_used,
        q_fusion_meta,
        q_mmr_used,
        q_mmr_selected_n,
    )
