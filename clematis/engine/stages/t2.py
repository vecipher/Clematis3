from __future__ import annotations
from typing import Dict, Any, List
from ..types import T2Result, EpisodeRef
from ..cache import LRUCache, stable_key
from ...adapters.embeddings import BGEAdapter
from ...memory.index import InMemoryIndex
import numpy as np
import datetime as dt
from .hybrid import rerank_with_gel

# Config-driven cache (constructed lazily per cfg)
_T2_CACHE = None
_T2_CACHE_CFG = None

def _get_cache(cfg_t2: dict):
    c = cfg_t2.get("cache", {})
    enabled = bool(c.get("enabled", True))
    if not enabled:
        return None
    max_entries = int(c.get("max_entries", 512))
    ttl_s = int(c.get("ttl_s", 300))
    global _T2_CACHE, _T2_CACHE_CFG
    if _T2_CACHE is None or _T2_CACHE_CFG != (max_entries, ttl_s):
        _T2_CACHE = LRUCache(max_entries=max_entries, ttl_s=ttl_s)
        _T2_CACHE_CFG = (max_entries, ttl_s)
    return _T2_CACHE

def _gather_changed_labels(state: dict, t1) -> List[str]:
    """Collect labels for nodes touched by T1 across active graphs, deterministically sorted."""
    store = state.get("store")
    active_graphs = state.get("active_graphs", [])
    if store is None:
        return []
    # Node ids touched by T1
    nid_set = {d.get("id") for d in getattr(t1, "graph_deltas", []) if d.get("op") == "upsert_node"}
    labels: List[str] = []
    for gid in active_graphs:
        g = store.get_graph(gid)
        for nid in sorted(nid_set):
            n = g.nodes.get(nid)
            if n and n.label:
                labels.append(n.label)
    # de-dup while stable
    seen = set()
    out: List[str] = []
    for lb in labels:
        if lb not in seen:
            seen.add(lb)
            out.append(lb)
    return out

def _build_label_map(state: dict) -> Dict[str, str]:
    """label(lower) -> node_id across active graphs (last one wins but we sort anyway later)."""
    store = state.get("store")
    active_graphs = state.get("active_graphs", [])
    m: Dict[str, str] = {}
    if not store:
        return m
    for gid in active_graphs:
        g = store.get_graph(gid)
        for n in g.nodes.values():
            if n.label:
                m[n.label.lower()] = n.id
    return m

def _parse_iso(ts: str) -> dt.datetime:
  try:
      return dt.datetime.fromisoformat((ts or "").replace("Z", "+00:00")).astimezone(dt.timezone.utc)
  except Exception:
      return dt.datetime.now(dt.timezone.utc)

def _owner_for_query(ctx, cfg_t2: dict):
  scope = str(cfg_t2.get("owner_scope", "any")).lower()
  if scope == "agent":
      return getattr(ctx, "agent_id", None)
  if scope == "world":
      return "world"
  return None  # any

class _EpRefShim:
    __slots__ = ("id", "text", "score")
    def __init__(self, d: Dict[str, Any]):
        self.id = str(d.get("id"))
        self.text = d.get("text", "")
        s = d.get("score", d.get("_score", 0.0))
        try:
            self.score = float(s)
        except Exception:
            self.score = 0.0


def _init_index_from_cfg(state: dict, cfg_t2: dict):
    """Instantiate and cache the memory index based on t2.backend with safe fallback.
    Returns: (index, backend_selected:str, fallback_reason:str|None)
    """
    idx = state.get("mem_index")
    if idx is not None:
        return idx, state.get("mem_backend", str(cfg_t2.get("backend", "inmemory")).lower()), state.get("mem_backend_fallback_reason")

    backend = str(cfg_t2.get("backend", "inmemory")).lower()
    fallback_reason = None

    if backend == "lancedb":
        try:
            from ...memory.lance_index import LanceIndex  # deferred import; adapter defers lancedb
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

def t2_semantic(ctx, state, text: str, t1) -> T2Result:
    cfg = ctx.cfg
    cfg_t2 = cfg.t2
    # Ensure a memory index exists in state, honoring backend selection with safe fallback
    index, backend_selected, backend_fallback_reason = _init_index_from_cfg(state, cfg_t2)
    # Query text = user text + labels changed by T1
    t1_labels = _gather_changed_labels(state, t1)
    q_text = (text or "").strip()
    if t1_labels:
        q_text = (q_text + " " + " ".join(sorted(t1_labels))).strip()
    # Deterministic embedding
    enc = BGEAdapter(dim=int(cfg.k_surface) if hasattr(cfg, 'k_surface') else 32)
    q_vec = enc.encode([q_text])[0]
    # Tiered retrieval
    tiers: List[str] = list(cfg_t2.get("tiers", ["exact_semantic", "cluster_semantic", "archive"]))
    k_retrieval: int = int(cfg_t2.get("k_retrieval", 64))
    exact_recent_days: int = int(cfg_t2.get("exact_recent_days", 30))
    sim_threshold: float = float(cfg_t2.get("sim_threshold", 0.3))
    clusters_top_m: int = int(cfg_t2.get("clusters_top_m", 3))
    now_str = getattr(ctx, "now", None)
    # Cache setup
    cache = _get_cache(cfg_t2)
    try:
        index_ver = int(index.index_version())
    except Exception:
        index_ver = len(getattr(index, "_eps", []))
    ckey = (
        "t2",
        tuple(tiers),
        stable_key({
            "q": q_text,
            "exact_recent_days": exact_recent_days,
            "sim_threshold": sim_threshold,
            "clusters_top_m": clusters_top_m,
        }),
        index_ver,
    )
    cache_used = False
    cache_hits = 0
    cache_misses = 0
    if cache is not None:
        hit = cache.get(ckey)
        if hit is not None:
            # Mark this as a cache hit on the returned metrics (without recomputing)
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
    # Execute tiers with dedupe
    retrieved = []  # List[EpisodeRef]
    seen_ids = set()
    tier_sequence: List[str] = []
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
        owner_query = _owner_for_query(ctx, cfg_t2)
        if now_str:
            hints["now"] = now_str
        hits = index.search_tiered(owner=owner_query, q_vec=q_vec, k=k_retrieval, tier=tier, hints=hints)
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
    now_str = getattr(ctx, "now", None)
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
    hybrid_used = False
    hybrid_info: Dict[str, Any] = {}
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

    # Residual propagation: map episode texts back to node labels
    residual_cap = int(cfg_t2.get("residual_cap_per_turn", 32))
    label_map = _build_label_map(state)
    chosen_nodes: List[str] = []
    seen_nodes = set()
    for ep in retrieved:
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
    metrics = {
        "tier_sequence": tiers,
        "k_returned": len(retrieved),
        "k_used": len(graph_deltas_residual),
        "sim_stats": sim_stats,
        "score_stats": score_stats,
        "owner_scope": str(cfg_t2.get("owner_scope", "any")).lower(),
        "caps": {"residual_cap": residual_cap},
        "cache_enabled": cache is not None,
        "cache_used": cache_used,
        "cache_hits": cache_hits,
        "cache_misses": cache_misses,
        "backend": backend_selected,
        "backend_fallback": bool(backend_fallback_reason),
    }
    if backend_fallback_reason:
        metrics["backend_fallback_reason"] = str(backend_fallback_reason)
    metrics["hybrid_used"] = hybrid_used
    if hybrid_info:
        metrics["hybrid"] = hybrid_info
    result = T2Result(retrieved=retrieved, graph_deltas_residual=graph_deltas_residual, metrics=metrics)
    if cache is not None:
        cache.put(ckey, result)
        result.metrics["cache_used"] = True
        result.metrics["cache_misses"] = result.metrics.get("cache_misses", 0) + 1
    return result
