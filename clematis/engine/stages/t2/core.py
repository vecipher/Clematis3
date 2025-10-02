from __future__ import annotations

import datetime as dt
from typing import Any, Dict, List

import numpy as np

from ...cache import stable_key
from ...types import EpisodeRef, T2Result
from ...util.embed_store import open_reader
from ...util.parallel import run_parallel
from ....adapters.embeddings import BGEAdapter
from ....memory.index import InMemoryIndex
from .cache import get_cache as _get_cache
from .config import (
    cfg_get as _cfg_get,
    ensure_dict as _ensure_dict,
    metrics_gate_on as _metrics_gate_on,
    quality_cfg_snapshot as _quality_cfg_snapshot,
)
from .helpers import (
    EpRefShim as _EpRefShim,
    build_label_map as _build_label_map,
    gather_changed_labels as _gather_changed_labels,
    items_for_fusion as _items_for_fusion,
    owner_for_query as _owner_for_query,
    parse_iso as _parse_iso,
    quality_digest as _quality_digest,
)
from ..hybrid import rerank_with_gel
from .metrics import (
    emit_t2_metrics as _emit_t2_metrics,
    estimate_result_cost as _estimate_result_cost,
)
from .parallel import (
    collect_shard_hits as _collect_shard_hits,
    t2_parallel_enabled as _t2_parallel_enabled,
)
from ..t2_quality_mmr import MMRItem, avg_pairwise_distance
from ..t2_quality_trace import emit_trace as emit_quality_trace
from ..t2_shard import merge_tier_hits_across_shards_dict as _merge_tier_hits_across_shards, _qscore

# PR40: Safe import for optional Lance partitioned reader
try:
    from ..t2_lance_reader import LancePartitionSpec, PartitionedReader  # PR40 optional
except Exception:
    LancePartitionSpec = None  # type: ignore
    PartitionedReader = None  # type: ignore


# Helper: build a deterministic items list for quality fusion based on current ordering
def _init_index_from_cfg(state: dict, cfg_t2: dict):
    """Instantiate and cache the memory index based on t2.backend with safe fallback.
    Returns: (index, backend_selected:str, fallback_reason:str|None)
    """
    idx = state.get("mem_index")
    if idx is not None:
        return (
            idx,
            state.get("mem_backend", str(cfg_t2.get("backend", "inmemory")).lower()),
            state.get("mem_backend_fallback_reason"),
        )

    backend = str(cfg_t2.get("backend", "inmemory")).lower()
    fallback_reason = None

    if backend == "lancedb":
        try:
            from clematis.memory.lance_index import (
                LanceIndex,
            )  # deferred import; adapter defers lancedb

            lcfg = cfg_t2.get("lancedb", {}) or {}
            idx = LanceIndex(
                uri=str(lcfg.get("uri", "./.data/lancedb")),
                table=str(lcfg.get("table", "episodes")),
                meta_table=str(lcfg.get("meta_table", "meta")),
            )
        except Exception as e:
            # Fallback to in-memory index, record reason
            idx = InMemoryIndex()
            fallback_reason = f"{type(e).__name__}: {e}"
            backend = "inmemory"
    else:
        idx = InMemoryIndex()
        backend = "inmemory"

    state["mem_index"] = idx
    state["mem_backend"] = backend
    if fallback_reason:
        state["mem_backend_fallback_reason"] = fallback_reason
    return idx, backend, fallback_reason


import hashlib
import json


def t2_semantic(ctx, state, text: str, t1) -> T2Result:
    cfg_root_raw = _cfg_get(ctx, ["cfg"], getattr(ctx, "cfg", {}))
    if cfg_root_raw is None:
        cfg_root_raw = {}
    cfg = cfg_root_raw
    cfg_root = _ensure_dict(cfg_root_raw)
    cfg_t2 = _ensure_dict(cfg_root.get("t2", {}))
    # PR33 config (identity-safe; metrics only unless explicitly wired)
    embed_store_dtype_cfg = str(
        _cfg_get(ctx, ["cfg", "perf", "t2", "embed_store_dtype"], "fp32") or "fp32"
    ).lower()
    precompute_norms_cfg = bool(_cfg_get(ctx, ["cfg", "perf", "t2", "precompute_norms"], False))
    partitions_cfg = _cfg_get(ctx, ["cfg", "perf", "t2", "reader", "partitions"], {}) or {}
    embed_root = str(cfg_t2.get("embed_root", "./.data/t2"))
    # PR40: determine requested reader mode (flat|partition|auto), default flat
    reader_mode_req = str(_cfg_get(ctx, ["cfg", "t2", "reader", "mode"], "flat") or "flat").lower()
    reader_mode_sel = "flat"
    # Check availability of a partitioned fixture if requested/auto
    _p_avail = False
    if (
        reader_mode_req in ("partition", "auto")
        and PartitionedReader is not None
        and LancePartitionSpec is not None
    ):
        try:
            # Use embed_root as a conservative root; partition keys (if any) come from perf.t2.reader.partitions
            by = None
            if isinstance(partitions_cfg, dict):
                _by = partitions_cfg.get("by") or partitions_cfg.get("partitions")
                if isinstance(_by, (list, tuple)):
                    by = tuple(str(x) for x in _by)
            spec = LancePartitionSpec(root=str(embed_root), by=tuple(by or ()))
            preader = PartitionedReader(spec)  # flat_iter not required for availability check
            _p_avail = bool(preader.available())
        except Exception:
            _p_avail = False
    if reader_mode_req == "partition":
        reader_mode_sel = "partition" if _p_avail else "flat"
    elif reader_mode_req == "auto":
        reader_mode_sel = "partition" if _p_avail else "flat"
    else:
        reader_mode_sel = "flat"
    # Ensure a memory index exists in state, honoring backend selection with safe fallback
    index, backend_selected, backend_fallback_reason = _init_index_from_cfg(state, cfg_t2)
    # Query text = user text + labels changed by T1
    t1_labels = _gather_changed_labels(state, t1)
    q_text = (text or "").strip()
    if t1_labels:
        q_text = (q_text + " " + " ".join(sorted(t1_labels))).strip()
    # Deterministic embedding
    enc = BGEAdapter(dim=int(cfg_root.get("k_surface", 32)))
    enc_obj = getattr(ctx, "enc", None) or enc
    q_vec = enc_obj.encode([q_text])[0]
    # PR33: discover shards (for metrics only; retrieval path remains unchanged)
    pr33_reader = None
    pr33_layout = "none"
    pr33_shards = 0
    pr33_store_dtype = embed_store_dtype_cfg
    perf_on_tmp = bool(_cfg_get(ctx, ["cfg", "perf", "enabled"], False))
    if perf_on_tmp and isinstance(partitions_cfg, dict) and partitions_cfg.get("enabled", False):
        try:
            pr33_reader = open_reader(embed_root, partitions=partitions_cfg)
            pr33_layout = pr33_reader.meta.get("partition_layout", "none")
            pr33_shards = int(pr33_reader.meta.get("shards", 0))
            pr33_store_dtype = str(
                pr33_reader.meta.get("embed_store_dtype", pr33_store_dtype)
            ).lower()
        except Exception:
            pr33_reader = None
    # PR33: gate for reader-backed retrieval
    use_reader = bool(
        perf_on_tmp
        and pr33_reader is not None
        and isinstance(partitions_cfg, dict)
        and partitions_cfg.get("enabled", False)
    )
    # Tiered retrieval
    tiers_all: List[str] = list(
        cfg_t2.get("tiers", ["exact_semantic", "cluster_semantic", "archive"])
    )
    tiers: List[str] = ["embed_store"] if use_reader else tiers_all
    k_retrieval: int = int(cfg_t2.get("k_retrieval", 64))
    exact_recent_days: int = int(cfg_t2.get("exact_recent_days", 30))
    sim_threshold: float = float(cfg_t2.get("sim_threshold", 0.3))
    clusters_top_m: int = int(cfg_t2.get("clusters_top_m", 3))
    now_str = getattr(ctx, "now", None)
    # PR41: floor "now" to midnight UTC so exact_recent_days is inclusive by calendar day
    if not now_str:
        _now_dt = dt.datetime.now(dt.timezone.utc)
        _now_floor = dt.datetime(_now_dt.year, _now_dt.month, _now_dt.day, tzinfo=dt.timezone.utc)
        # Use Z suffix for downstream parsers expecting ISO8601 with Z
        now_str = _now_floor.isoformat().replace("+00:00", "Z")
    # Cache setup
    cache, cache_kind = _get_cache(ctx, cfg_t2)
    try:
        index_ver = int(index.index_version())
    except Exception:
        index_ver = len(getattr(index, "_eps", []))
    ckey_payload = {
        "q": q_text,
        "exact_recent_days": exact_recent_days,
        "sim_threshold": sim_threshold,
        "clusters_top_m": clusters_top_m,
    }
    qcfg = _ensure_dict(cfg_t2.get("quality", {}))
    if bool(qcfg.get("enabled", False)):
        ckey_payload["q_digest"] = _quality_digest(qcfg)
    if use_reader:
        ckey_payload["embed_store"] = {
            "dtype": pr33_store_dtype,
            "layout": pr33_layout,
            "shards": pr33_shards,
        }
    ckey = (
        "t2",
        tuple(tiers),
        stable_key(ckey_payload),
        index_ver,
    )
    perf_enabled = bool(_cfg_get(ctx, ["cfg", "perf", "enabled"], False))
    metrics_enabled = bool(_cfg_get(ctx, ["cfg", "perf", "metrics", "report_memory"], False))
    gate_on = bool(perf_enabled and metrics_enabled)
    cache_evicted_n = 0
    cache_evicted_b = 0
    cache_used = False
    cache_hits = 0
    cache_misses = 0
    if cache is not None:
        hit = cache.get(ckey)
        if hit is not None:
            # Mark cache hit only when metrics gate is ON
            if _metrics_gate_on(cfg_root):
                try:
                    hit.metrics["cache_used"] = True
                    hit.metrics["cache_hits"] = hit.metrics.get("cache_hits", 0) + 1
                    hit.metrics["cache_misses"] = hit.metrics.get("cache_misses", 0)
                except Exception:
                    pass
            return hit
        else:
            cache_misses += 1
    raw_hits_by_id: Dict[str, Any] = {}
    # PR68: counters for optional parallel T2
    t2_task_count = 0
    t2_parallel_workers = 0
    t2_partition_count = 0
    # Execute tiers with dedupe
    retrieved = []  # List[EpisodeRef]
    seen_ids = set()
    # If PR33 reader is enabled, perform retrieval directly from the embed store (cosine over shards)
    if use_reader:
        tier_sequence: List[str] = ["embed_store"]
        # Build a local id->episode map to hydrate texts where possible
        _eps_local = getattr(index, "_eps", [])
        _ep_map = {str(e.get("id")): e for e in _eps_local} if isinstance(_eps_local, list) else {}
        retrieved = []
        seen_ids = set()
        # Deterministic cosine scores in fp32
        q = np.asarray(q_vec, dtype=np.float32)
        qn = float(np.linalg.norm(q, ord=2))
        if qn == 0.0:
            qn = 1.0
        items = []
        batch_sz = int(cfg_t2.get("reader_batch", 8192))
        for ids_b, vecs_b, norms_b in pr33_reader.iter_blocks(batch=batch_sz):
            denom = norms_b * qn
            denom = np.where(denom == 0.0, 1.0, denom)
            scores_b = (vecs_b @ q) / denom
            # Accumulate tuples for deterministic final sort
            for i, _id in enumerate(ids_b):
                items.append((-float(scores_b[i]), _id, float(scores_b[i])))
        # Stable ordering: (-score, lex(id))
        items.sort(key=lambda t: (t[0], t[1]))
        for _, _id, sc in items[:k_retrieval]:
            src = _ep_map.get(_id, {"id": _id, "text": "", "score": sc})
            ref = _EpRefShim(src if isinstance(src, dict) else {"id": _id, "text": "", "score": sc})
            # Ensure cosine score is set
            try:
                ref.score = float(sc)
            except Exception:
                pass
            if ref.id in seen_ids:
                continue
            retrieved.append(ref)
            seen_ids.add(ref.id)
        # Skip legacy tiered retrieval
        pass
    else:
        owner_query = _owner_for_query(ctx, cfg_t2)
        tier_sequence: List[str] = []
        # PR68: parallel T2 across in-memory shards, gated
        if _t2_parallel_enabled(cfg_root, backend_selected, index):
            try:
                suggested = int(_cfg_get(ctx, ["cfg", "perf", "parallel", "max_workers"], 1) or 1)
            except Exception:
                suggested = 1
            # Enumerate shard views (private API on in-memory index). Fall back gracefully.
            try:
                shards_iter = index._iter_shards_for_t2("exact_semantic", suggested=suggested)  # type: ignore[attr-defined]
                shards = list(shards_iter)
            except TypeError:
                shards = list(index._iter_shards_for_t2("exact_semantic"))  # type: ignore[attr-defined]
            except Exception:
                shards = []
            if len(shards) > 1:
                t2_task_count = len(shards)
                t2_parallel_workers = min(len(shards), suggested)
                # PR69: expose partition count for LanceDB path (gated metrics only)
                if str(backend_selected).lower() == "lancedb":
                    t2_partition_count = len(shards)

                def _task(sh):
                    return _collect_shard_hits(
                        sh,
                        tiers,
                        owner_query,
                        q_vec,
                        k_retrieval,
                        now_str,
                        float(sim_threshold),
                        int(clusters_top_m),
                    )

                tasks = [(i, (lambda S=sh: _task(S))) for i, sh in enumerate(shards)]
                shard_hits = run_parallel(
                    tasks,
                    max_workers=t2_parallel_workers,
                    merge_fn=None,
                    order_key=None,  # key is the integer i; ordering is stable by submit index
                )
                merged, used_tiers = _merge_tier_hits_across_shards(shard_hits, tiers, k_retrieval)
                tier_sequence = used_tiers
                retrieved = []
                seen_ids = set()
                for d in merged:
                    hid = str(d.get("id"))
                    raw_hits_by_id[hid] = d
                    ref = _EpRefShim(d)
                    if ref.id in seen_ids:
                        continue
                    retrieved.append(ref)
                    seen_ids.add(ref.id)
            else:
                # Fallback to sequential when no shards are available
                tier_sequence = []
                for tier in tiers:
                    tier_sequence.append(tier)
                    hints: Dict[str, Any] = {"sim_threshold": sim_threshold}
                    if tier == "exact_semantic":
                        hints.update({"recent_days": exact_recent_days})
                    elif tier == "cluster_semantic":
                        hints.update({"clusters_top_m": clusters_top_m})
                    elif tier == "archive":
                        # no extra hints by default
                        pass
                    else:
                        continue
                    if now_str:
                        hints["now"] = now_str
                    hits = index.search_tiered(
                        owner=owner_query, q_vec=q_vec, k=k_retrieval, tier=tier, hints=hints
                    )
                    for h in hits:
                        if isinstance(h, dict):
                            hid = str(h.get("id"))
                            raw_hits_by_id[hid] = h
                            ref = _EpRefShim(h)
                        else:
                            ref = h
                        if ref.id in seen_ids:
                            continue
                        retrieved.append(ref)
                        seen_ids.add(ref.id)
                        if len(retrieved) >= k_retrieval:
                            break
                    if len(retrieved) >= k_retrieval:
                        break
        else:
            # Original sequential path (identity)
            tier_sequence = []
            for tier in tiers:
                tier_sequence.append(tier)
                hints: Dict[str, Any] = {"sim_threshold": sim_threshold}
                if tier == "exact_semantic":
                    hints.update({"recent_days": exact_recent_days})
                elif tier == "cluster_semantic":
                    hints.update({"clusters_top_m": clusters_top_m})
                elif tier == "archive":
                    # no extra hints by default
                    pass
                else:
                    continue
                if now_str:
                    hints["now"] = now_str
                hits = index.search_tiered(
                    owner=owner_query, q_vec=q_vec, k=k_retrieval, tier=tier, hints=hints
                )
                for h in hits:
                    if isinstance(h, dict):
                        hid = str(h.get("id"))
                        raw_hits_by_id[hid] = h
                        ref = _EpRefShim(h)
                    else:
                        ref = h
                    if ref.id in seen_ids:
                        continue
                    retrieved.append(ref)
                    seen_ids.add(ref.id)
                    if len(retrieved) >= k_retrieval:
                        break
                if len(retrieved) >= k_retrieval:
                    break
    # --- Combined scoring (alpha * cosine + beta * recency + gamma * importance) ---
    ranking = cfg_t2.get("ranking", {})
    alpha = float(ranking.get("alpha_sim", 0.75))
    beta = float(ranking.get("beta_recency", 0.2))
    gamma = float(ranking.get("gamma_importance", 0.05))

    # Build episode lookup (id -> full dict) for recency/importance
    eps = getattr(index, "_eps", [])
    ep_by_id = {str(e.get("id")): e for e in eps} if isinstance(eps, list) else {}
    if not ep_by_id and raw_hits_by_id:
        ep_by_id = dict(raw_hits_by_id)

    # Reference 'now' from ctx if available
    now_utc = _parse_iso(now_str) if now_str else dt.datetime.now(dt.timezone.utc)

    HORIZON_DAYS = 365.0  # normalization horizon for recency
    rescored: List[tuple[EpisodeRef, float, float]] = []  # (ref, combined, cos)

    for ref in retrieved:
        cos = float(ref.score)  # from index
        cos_norm = (cos + 1.0) / 2.0  # [-1,1] -> [0,1]

        ep = ep_by_id.get(ref.id, {})
        ts = ep.get("ts")
        if ts:
            age_days = max(0.0, (now_utc - _parse_iso(ts)).total_seconds() / 86400.0)
        else:
            age_days = HORIZON_DAYS  # treat unknown as old
        recency = max(0.0, min(1.0, 1.0 - (age_days / HORIZON_DAYS)))

        importance = float((ep.get("aux") or {}).get("importance", 0.5))
        importance = max(0.0, min(1.0, importance))

        combined = alpha * cos_norm + beta * recency + gamma * importance
        rescored.append((ref, combined, cos))

    # Deterministic ordering by combined score desc, then id asc
    rescored.sort(key=lambda t: (-t[1], t[0].id))
    retrieved = [r for (r, _, _) in rescored]

    # --- Optional GEL-based hybrid re-ranking (feature-flagged) ---
    # Prepare hybrid placeholders; we will attach them to `metrics` later
    hcfg = _ensure_dict(cfg_t2.get("hybrid", {}))
    if bool(hcfg.get("enabled", False)):
        try:
            new_items, hmetrics = rerank_with_gel(ctx, state, retrieved)
            retrieved = new_items
            hybrid_used = bool(hmetrics.get("hybrid_used", False))
            hybrid_info = {
                k: hmetrics.get(k)
                for k in (
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
                if k in hmetrics
            }
        except Exception:
            hybrid_used = False
            hybrid_info = {}
    else:
        hybrid_used = False
        hybrid_info = {}

    # PR37 fusion bookkeeping
    q_fusion_used = False
    q_fusion_meta = {}
    # PR38 fusion bookkeeping
    q_mmr_used = False
    q_mmr_selected_n = 0

    # --- PR37: Optional lexical BM25 + fusion (alpha), behind t2.quality.enabled ---
    try:
        q_enabled_cfg = bool(_cfg_get(cfg_root, ["t2", "quality", "enabled"], False))
        if q_enabled_cfg:
            # Import lazily to avoid hard dependency before PR37 lands
            try:
                from .t2_quality import fuse as quality_fuse  # type: ignore
            except Exception:
                quality_fuse = None  # fallback: skip fusion if module not present
            if quality_fuse is not None:
                # Build fusion input from the *current* ordering
                items_for_fusion = _items_for_fusion(retrieved)
                fused_out = quality_fuse(q_text, items_for_fusion, cfg=cfg_root)
                # Allow either a list or (list, meta) return
                if isinstance(fused_out, tuple):
                    fused_items, q_fusion_meta = fused_out
                else:
                    fused_items, q_fusion_meta = fused_out, {}
                # Build optional maps for enabled-path traces
                try:
                    sem_rank_map = {d["id"]: i + 1 for i, d in enumerate(items_for_fusion)}
                    score_fused_map = {
                        d.get("id"): float(d.get("score_fused", 0.0)) for d in (fused_items or [])
                    }
                    if isinstance(q_fusion_meta, dict):
                        q_fusion_meta.setdefault("sem_rank_map", sem_rank_map)
                        q_fusion_meta.setdefault("score_fused_map", score_fused_map)
                except Exception:
                    pass
                # Reorder `retrieved` according to fused id ordering; keep only known ids
                id_to_ref = {r.id: r for r in retrieved}
                new_order = []
                for it in fused_items:
                    iid = it.get("id")
                    if iid in id_to_ref:
                        new_order.append(id_to_ref[iid])
                if new_order:
                    retrieved = new_order
                    q_fusion_used = True
                # --- PR38: Optional MMR diversification over fused list ---
                try:
                    from .t2_quality import maybe_apply_mmr as quality_mmr  # type: ignore
                except Exception:
                    quality_mmr = None
                try:
                    mmr_enabled = bool(
                        _cfg_get(cfg_root, ["t2", "quality", "mmr", "enabled"], False)
                    )
                except Exception:
                    mmr_enabled = False
                if quality_mmr is not None and mmr_enabled:
                    qcfg = _cfg_get(cfg_root, ["t2", "quality"], {}) or {}
                    mmr_items = quality_mmr(fused_items, qcfg)
                    # Mark that MMR executed, even if it yields an empty/no-op selection
                    q_mmr_used = True
                    # Reorder EpisodeRefs according to MMR output (when any ids are returned)
                    id_to_ref = {r.id: r for r in retrieved}
                    mmr_order = []
                    for it in mmr_items or []:
                        iid = it.get("id")
                        if iid in id_to_ref:
                            mmr_order.append(id_to_ref[iid])
                    q_mmr_selected_n = len(mmr_order)
                    if mmr_order:
                        retrieved = mmr_order
    except Exception:
        # Fusion must never break retrieval; if anything goes wrong, keep baseline ordering
        q_fusion_used = False
        q_fusion_meta = {}

    # PR38 fallback: if fusion step failed/was skipped, still attempt MMR over current ordering
    if not q_mmr_used:
        try:
            try:
                from .t2_quality import maybe_apply_mmr as _quality_mmr_fallback  # type: ignore
            except Exception:
                _quality_mmr_fallback = None
            _mmr_enabled_fb = bool(_cfg_get(cfg_root, ["t2", "quality", "mmr", "enabled"], False))
            if _quality_mmr_fallback is not None and _mmr_enabled_fb:
                _items_fb = _items_for_fusion(retrieved)
                _qcfg_fb = _cfg_get(cfg_root, ["t2", "quality"], {}) or {}
                _mmr_items_fb = _quality_mmr_fallback(_items_fb, _qcfg_fb)
                q_mmr_used = True
                id_to_ref_fb = {r.id: r for r in retrieved}
                _mmr_order_fb = []
                for it in _mmr_items_fb or []:
                    _iid = it.get("id")
                    if _iid in id_to_ref_fb:
                        _mmr_order_fb.append(id_to_ref_fb[_iid])
                q_mmr_selected_n = len(_mmr_order_fb)
                if _mmr_order_fb:
                    retrieved = _mmr_order_fb
        except Exception:
            pass

    # --- M5 scheduler slice caps (use-only clamp) ---
    # We cap how many retrieval hits are **used** downstream in this slice (not how many are fetched).
    # The full `retrieved` list is kept for metrics/logging and possible caching.
    caps = getattr(ctx, "slice_budgets", None) or {}
    _t2_cap = caps.get("t2_k")
    if _t2_cap is None:
        used_hits = retrieved
    else:
        try:
            _cap_val = int(_t2_cap)
            if _cap_val < 0:
                _cap_val = 0
        except Exception:
            _cap_val = 0
        used_hits = retrieved[:_cap_val]

    # Residual propagation: map episode texts back to node labels
    residual_cap = int(cfg_t2.get("residual_cap_per_turn", 32))
    label_map = _build_label_map(state)
    chosen_nodes: List[str] = []
    seen_nodes = set()
    for ep in used_hits:
        t_low = (ep.text or "").lower()
        # match any label substring inside episode text
        for lb, nid in label_map.items():
            if not lb:
                continue
            if lb in t_low and nid not in seen_nodes:
                chosen_nodes.append(nid)
                seen_nodes.add(nid)
                if len(chosen_nodes) >= residual_cap:
                    break
        if len(chosen_nodes) >= residual_cap:
            break
    # Convert to deltas (monotonic nudges; never undo T1)
    graph_deltas_residual = [{"op": "upsert_node", "id": nid} for nid in sorted(set(chosen_nodes))]
    # Metrics
    scores = [float(h.score) for h in retrieved]
    sim_stats = {
        "mean": float(np.mean(scores)) if scores else 0.0,
        "max": float(np.max(scores)) if scores else 0.0,
    }
    combined_scores = [s for (_, s, _) in rescored] if 'rescored' in locals() and rescored else []
    score_stats = {
        "mean": float(np.mean(combined_scores)) if combined_scores else 0.0,
        "max": float(np.max(combined_scores)) if combined_scores else 0.0,
    }

    # --- Legacy baseline cache semantics for identity (PR29) ---
    tiers_for_baseline = tier_sequence if 'tier_sequence' in locals() and tier_sequence else tiers
    _legacy_cacheable = {"exact_semantic", "cluster_semantic"}
    legacy_miss_baseline = sum(1 for t in tiers_for_baseline if str(t) in _legacy_cacheable)

    cache_misses_out = int(cache_misses)
    if not gate_on:
        # In disabled path, PR29 logged a miss per cacheable tier even with empty hits
        cache_misses_out = max(cache_misses_out, legacy_miss_baseline)

    # cache_used should be True if the cache subsystem was consulted at all; in PR29
    # this was True even on all-miss first turns.
    cache_used_out = (cache is not None) and ((cache_hits + cache_misses_out) > 0 or not gate_on)

    # --- Identity baseline metrics (always present) ---
    metrics = {
        "tier_sequence": tier_sequence if 'tier_sequence' in locals() and tier_sequence else tiers,
        "k_returned": int(len(retrieved)),
        "k_used": int(len(used_hits)),
        "k_residual": int(len(graph_deltas_residual)),
        "sim_stats": sim_stats,
        "score_stats": score_stats,
        "owner_scope": str(cfg_t2.get("owner_scope", "any")).lower(),
        "caps": {"residual_cap": int(residual_cap)},
        "cache_enabled": cache is not None,
        "cache_used": cache_used_out,
        "cache_hits": int(cache_hits),
        "cache_misses": cache_misses_out,
        "backend": backend_selected,
        "backend_fallback": bool(backend_fallback_reason),
        "hybrid_used": bool(hybrid_used),
    }
    if hybrid_info:
        metrics["hybrid"] = hybrid_info

    # --- Additional gated metrics (only when perf.enabled && perf.metrics.report_memory) ---
    if gate_on:
        # Emit reader diagnostics only when the reader actually engaged under perf gate
        if use_reader and perf_enabled:
            partitions_list = []
            if isinstance(partitions_cfg, dict):
                by = partitions_cfg.get("by") or partitions_cfg.get("partitions")
                if isinstance(by, (list, tuple)):
                    partitions_list = [str(x) for x in by]
            reader_meta = {
                "embed_store_dtype": pr33_store_dtype,
                "precompute_norms": bool(precompute_norms_cfg),
                "layout": pr33_layout,
                "shards": int(pr33_shards or 0),
            }
            if partitions_list:
                reader_meta["partitions"] = list(partitions_list)
            metrics["reader"] = reader_meta
        # PR40: emit selected reader mode for observability
        try:
            metrics["t2.reader_mode"] = str(reader_mode_sel)
        except Exception:
            pass
        if backend_fallback_reason:
            metrics["backend_fallback_reason"] = str(backend_fallback_reason)
        # PR37: emit quality fusion metrics only when enabled and fusion executed
        if bool(_cfg_get(cfg, ["t2", "quality", "enabled"], False)) and q_fusion_used:
            metrics["t2q.fusion_mode"] = "score_interp"
            metrics["t2q.alpha_semantic"] = float(
                _cfg_get(cfg, ["t2", "quality", "fusion", "alpha_semantic"], 0.6)
            )
            # If the fuser provided additional meta (e.g., lex_hits), forward selected fields
            if isinstance(q_fusion_meta, dict) and "t2q.lex_hits" in q_fusion_meta:
                try:
                    metrics["t2q.lex_hits"] = int(q_fusion_meta["t2q.lex_hits"])
                except Exception:
                    pass
        # PR37: enabled-path tracing (triple-gated). Writes fused ordering details.
        if bool(_cfg_get(cfg, ["t2", "quality", "enabled"], False)) and q_fusion_used:
            try:
                # Prefer a caller-supplied reason (e.g., "examples_smoke", "eval"); fallback to "enabled"
                _reason_en = None
                try:
                    _reason_en = getattr(ctx, "trace_reason", None)
                    if _reason_en is None and isinstance(ctx, dict):  # tolerate dict-style ctx
                        _reason_en = ctx.get("trace_reason")
                except Exception:
                    _reason_en = None
                if _reason_en is None:
                    try:
                        _reason_en = _cfg_get(cfg, ["perf", "metrics", "trace_reason"], None)
                    except Exception:
                        _reason_en = None
                cfg_snap = _quality_cfg_snapshot(cfg)
                sem_rank_map = {}
                score_fused_map = {}
                if isinstance(q_fusion_meta, dict):
                    sem_rank_map = q_fusion_meta.get("sem_rank_map", {}) or {}
                    score_fused_map = q_fusion_meta.get("score_fused_map", {}) or {}
                trace_items = []
                for idx, r in enumerate(retrieved, start=1):
                    sid = getattr(r, "id", None)
                    trace_items.append(
                        {
                            "id": sid,
                            "rank_sem": int(sem_rank_map.get(sid, 0)),
                            "rank_fused": int(idx),
                            "score_fused": float(score_fused_map.get(sid, 0.0)),
                        }
                    )
                meta = {
                    "k": len(retrieved),
                    "reason": str(_reason_en) if _reason_en else "enabled",
                    "note": "PR37 enabled trace; fused ordering active",
                    "alpha": float(
                        _cfg_get(cfg, ["t2", "quality", "fusion", "alpha_semantic"], 0.6)
                    ),
                    "lex_hits": int(q_fusion_meta.get("t2q.lex_hits", 0))
                    if isinstance(q_fusion_meta, dict)
                    else 0,
                }
                emit_quality_trace(cfg_snap, q_text, trace_items, meta)
            except Exception:
                # Never fail the request due to tracing issues
                pass
        # PR38: emit MMR metrics only when quality+MMR are enabled and MMR ran
        if bool(_cfg_get(cfg, ["t2", "quality", "enabled"], False)):
            try:
                lam = float(_cfg_get(cfg, ["t2", "quality", "mmr", "lambda"], 0.5))
                metrics["t2q.mmr.lambda"] = lam
                selected_n = int(q_mmr_selected_n or 0) if q_mmr_used else 0
                metrics["t2q.mmr.selected"] = selected_n
                # compute diversity over head if any; else 0.0
                head_refs = retrieved[:selected_n] if selected_n > 0 else []
                mmr_items_head = []
                import unicodedata
                import re

                for r in head_refs:
                    s = unicodedata.normalize("NFKC", getattr(r, "text", "") or "").lower()
                    toks = [t for t in re.split(r"[^0-9a-zA-Z]+", s) if t]
                    from clematis.engine.stages.t2_quality_mmr import MMRItem, avg_pairwise_distance

                    mmr_items_head.append(
                        MMRItem(id=str(getattr(r, "id", "")), rel=0.0, toks=frozenset(toks))
                    )
                metrics["t2q.diversity_avg_pairwise"] = (
                    float(avg_pairwise_distance(mmr_items_head)) if mmr_items_head else 0.0
                )
            except Exception:
                pass
        # PR33: gated perf counters (namespaced)
        metrics["t2.embed_dtype"] = "fp32"
        metrics["t2.embed_store_dtype"] = pr33_store_dtype
        metrics["t2.precompute_norms"] = bool(precompute_norms_cfg)
        if pr33_reader is not None:
            metrics["t2.reader_shards"] = pr33_shards
            metrics["t2.partition_layout"] = pr33_layout
        # PR68: minimal parallel T2 observability (only when metrics gate is on)
        try:
            if int(t2_task_count or 0) > 0:
                metrics["t2.task_count"] = int(t2_task_count)
                metrics["t2.parallel_workers"] = int(t2_parallel_workers or 0)
        except Exception:
            pass
        try:
            if int(t2_partition_count or 0) > 0:
                metrics["t2.partition_count"] = int(t2_partition_count)
        except Exception:
            pass
    result = T2Result(
        retrieved=retrieved, graph_deltas_residual=graph_deltas_residual, metrics=metrics
    )
    if cache is not None:
        if cache_kind == "bytes":
            cost = _estimate_result_cost(retrieved, metrics)
            ev_n, ev_b = cache.put(ckey, result, cost)
            cache_evicted_n += int(ev_n or 0)
            cache_evicted_b += int(ev_b or 0)
        else:
            # legacy LRU (no size accounting)
            cache.put(ckey, result)
        # sync simple hit/miss counters only when metrics gate is ON
        if gate_on:
            result.metrics["cache_used"] = True
            result.metrics["cache_misses"] = result.metrics.get("cache_misses", 0) + 1
            if cache_kind == "bytes":
                result.metrics["t2.cache_evictions"] = (
                    result.metrics.get("t2.cache_evictions", 0) + cache_evicted_n
                )
                result.metrics["t2.cache_bytes"] = (
                    result.metrics.get("t2.cache_bytes", 0) + cache_evicted_b
                )

    # --- PR36: Shadow tracing (no-op) — triple-gated; does not mutate rankings or metrics ---
    try:
        q_enabled = bool(_cfg_get(cfg, ["t2", "quality", "enabled"], False))
        q_shadow = bool(_cfg_get(cfg, ["t2", "quality", "shadow"], False))
        if perf_enabled and metrics_enabled and q_shadow and not q_enabled:
            # Prefer a caller-supplied reason; fallback to "shadow"
            _reason_sh = None
            try:
                _reason_sh = getattr(ctx, "trace_reason", None)
                if _reason_sh is None and isinstance(ctx, dict):
                    _reason_sh = ctx.get("trace_reason")
            except Exception:
                _reason_sh = None
            if _reason_sh is None:
                try:
                    _reason_sh = _cfg_get(cfg, ["perf", "metrics", "trace_reason"], None)
                except Exception:
                    _reason_sh = None
            cfg_snap = _quality_cfg_snapshot(cfg)
            # serialize only id+score for trace; keep ordering identical to returned list
            trace_items = [
                {"id": h.id, "score": float(getattr(h, "score", 0.0))} for h in retrieved
            ]
            meta = {
                "k": len(retrieved),
                "reason": str(_reason_sh) if _reason_sh else "shadow",
                "note": "PR36 shadow trace; rankings unchanged",
            }
            emit_quality_trace(cfg_snap, q_text, trace_items, meta)
    except Exception:
        # Never fail the request due to tracing issues
        pass

    return result


# --- Compatibility entrypoint for tests & CI (Gate C) ---
def run_t2(cfg, corpus_dir: str | None = None, query: str = "", **kwargs):
    """
    Stable adapter used by tests. It constructs a minimal ctx/state and calls t2_semantic.
    - `cfg`: loaded configuration object (must expose `t2` and related fields).
    - `corpus_dir`: accepted for signature compatibility; unused here (reader is gated elsewhere).
    - `query`: text query forwarded to T2.
    Returns a T2Result.
    """

    class _Ctx:
        def __init__(self, cfg_obj):
            self.cfg = cfg_obj
            # Allow tests to set a deterministic 'now' via cfg or kwargs; default None (t2_semantic handles it)
            self.now = kwargs.get("now", None)
            # Embedding adapter may be provided via kwargs; else BGEAdapter will be constructed inside t2_semantic
            self.enc = kwargs.get("enc", None)
            # Optional agent id for owner_scope="agent"
            self.agent_id = kwargs.get("agent_id", None)

    class _T1:
        # Minimal T1 result stub: no deltas → no residual labels mixed in
        graph_deltas: list = []

    state = {}  # let t2_semantic lazily initialize the memory index and backend tags

    q = str(query or "").strip()
    return t2_semantic(_Ctx(cfg), state, q, _T1())


def t2_pipeline(cfg, query: str, emit_metric=None, ctx=None):
    # Propagate trace_reason into cfg so deeper stages (which see engine ctx) can still read it deterministically
    _reason = None
    try:
        _reason = getattr(ctx, "trace_reason", None)
        if _reason is None and isinstance(ctx, dict):
            _reason = ctx.get("trace_reason")
    except Exception:
        _reason = None
    if _reason:
        try:
            if "perf" not in cfg or not isinstance(cfg["perf"], dict):
                cfg["perf"] = {}
            if "metrics" not in cfg["perf"] or not isinstance(cfg["perf"]["metrics"], dict):
                cfg["perf"]["metrics"] = {}
            cfg["perf"]["metrics"]["trace_reason"] = str(_reason)
        except Exception:
            pass

    res = run_t2(cfg, query=query, ctx=ctx)
    if emit_metric and hasattr(res, "metrics"):
        for k, v in res.metrics.items():
            try:
                emit_metric(k, v)
            except Exception:
                pass
    return res
