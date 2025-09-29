from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Tuple

# Public API
#   rerank_with_gel(ctx, state, items) -> (new_items, metrics)
#
# Items contract (flexible, read-only): each item must at minimum expose an ID and a
# base similarity score. We adapt the following shapes deterministically:
#   - tuple:          (id, score, *rest)
#   - dict:           {"id"|"episode_id"|"node_id"|"target_id": str, "score"|"similarity"|"sim": float, ...}
#   - object attrs:   .id/.episode_id/.node_id/.target_id and .score/.similarity/.sim
# The function returns a new list (no in-place mutation) and small metrics.


# ---------------------------
# Helpers to read config & state
# ---------------------------


def _get_cfg(ctx: Any):
    """Return ctx.config or ctx.cfg; may be a dict-like or attribute object."""
    c = getattr(ctx, "config", None)
    if c is not None:
        return c
    c = getattr(ctx, "cfg", None)
    if c is not None:
        return c
    return {}


def _maybe_get(o: Any, key: str, default: Any = None):
    if isinstance(o, dict):
        return o.get(key, default)
    return getattr(o, key, default)


def _hybrid_cfg(ctx: Any) -> Dict[str, Any]:
    cfg = _get_cfg(ctx)
    t2 = _maybe_get(cfg, "t2", {}) or {}
    h = _maybe_get(t2, "hybrid", {}) or {}
    # If `h` is an attribute object, copy known fields into a dict
    if not isinstance(h, dict):
        attrs = (
            "enabled",
            "use_graph",
            "anchor_top_m",
            "walk_hops",
            "edge_threshold",
            "lambda_graph",
            "damping",
            "degree_norm",
            "max_bonus",
            "k_max",
        )
        h = {k: _maybe_get(_maybe_get(t2, "hybrid", {}), k, None) for k in attrs}
        h = {k: v for k, v in h.items() if v is not None}
    # Safe defaults (must mirror configs/validate.py)
    h.setdefault("enabled", False)
    h.setdefault("use_graph", True)
    h.setdefault("anchor_top_m", 8)
    h.setdefault("walk_hops", 1)
    h.setdefault("edge_threshold", 0.10)
    h.setdefault("lambda_graph", 0.25)
    h.setdefault("damping", 0.50)
    h.setdefault("degree_norm", "none")
    h.setdefault("max_bonus", 0.50)
    h.setdefault("k_max", 128)
    return h


def _graph_store(state: Any) -> Dict[str, Any]:
    # Prefer attribute, else dict key; ensure minimal structure
    g = getattr(state, "graph", None)
    if g is None and isinstance(state, dict):
        g = state.get("graph")
    if g is None:
        g = {"nodes": {}, "edges": {}}
    g.setdefault("edges", {})
    return g


# ---------------------------
# Item adapters (ID and score)
# ---------------------------


def _get_id(item: Any) -> str:
    if isinstance(item, tuple) and item:
        return str(item[0])
    if isinstance(item, dict):
        for k in ("id", "episode_id", "node_id", "target_id"):
            if k in item:
                return str(item[k])
    for k in ("id", "episode_id", "node_id", "target_id"):
        if hasattr(item, k):
            return str(getattr(item, k))
    # stable fallback
    return repr(item)


def _get_sim(item: Any) -> float:
    if isinstance(item, tuple) and len(item) >= 2:
        try:
            return float(item[1])
        except Exception:
            return 0.0
    if isinstance(item, dict):
        for k in ("score", "similarity", "sim", "weight"):
            if k in item:
                try:
                    return float(item[k])
                except Exception:
                    return 0.0
    for k in ("score", "similarity", "sim", "weight"):
        if hasattr(item, k):
            try:
                return float(getattr(item, k))
            except Exception:
                return 0.0
    return 0.0


# ---------------------------
# Graph helpers
# ---------------------------

_ARROW = "→"  # canonical undirected edge key uses unicode arrow


def _edge_key(a: str, b: str) -> str:
    a = str(a)
    b = str(b)
    return f"{a}{_ARROW}{b}" if a <= b else f"{b}{_ARROW}{a}"


def _edge_weight(edges: Dict[str, Any], a: str, b: str) -> float:
    rec = edges.get(_edge_key(a, b))
    if not isinstance(rec, dict):
        return 0.0
    try:
        return float(rec.get("weight", 0.0))
    except Exception:
        return 0.0


def _degree(
    edges: Dict[str, Any], vid: str, threshold: float, considered: Optional[set] = None
) -> int:
    """Count neighbors with |w| >= threshold. If `considered` set provided, restrict to it."""
    cnt = 0
    if not isinstance(edges, dict) or not edges:
        return 0
    for key, rec in edges.items():
        if not isinstance(rec, dict):
            continue
        src = rec.get("src")
        dst = rec.get("dst")
        if src is None or dst is None:
            # fall back to parse key
            if _ARROW in key:
                src, dst = key.split(_ARROW, 1)
            else:
                continue
        if src != vid and dst != vid:
            continue
        other = dst if src == vid else src
        if considered is not None and other not in considered:
            continue
        try:
            w = abs(float(rec.get("weight", 0.0)))
        except Exception:
            w = 0.0
        if w >= threshold:
            cnt += 1
    return cnt


# ---------------------------
# Core reranker
# ---------------------------


def rerank_with_gel(ctx: Any, state: Any, items: List[Any]) -> Tuple[List[Any], Dict[str, Any]]:
    """
    Deterministic hybrid reranker that blends dense similarity with graph evidence.

    Returns: (new_items, metrics)
      - new_items: reordered copy of `items` (only within the top-k slice), rest preserved.
      - metrics: minimal dict for logging (no side effects here).
    """
    cfg = _hybrid_cfg(ctx)
    if not cfg.get("enabled", False):
        return list(items), {"hybrid_used": False}

    # Extract graph and early-outs
    g = _graph_store(state)
    edges: Dict[str, Any] = g.get("edges", {}) or {}
    if not cfg.get("use_graph", True) or not edges or not items:
        return list(items), {"hybrid_used": False}

    # Build working slice and ID/sim pairs (assume caller already sorted by (-sim, id))
    k_considered = min(len(items), int(cfg.get("k_max", 128)))
    if k_considered <= 1:
        return list(items), {"hybrid_used": False, "k_considered": k_considered}

    work = list(items[:k_considered])
    tail = list(items[k_considered:])

    ids: List[str] = [_get_id(x) for x in work]
    sims: List[float] = [_get_sim(x) for x in work]
    considered = set(ids)

    anchor_top_m = max(1, min(int(cfg.get("anchor_top_m", 8)), k_considered))
    anchors = ids[:anchor_top_m]

    thresh = float(cfg.get("edge_threshold", 0.10))
    lam = float(cfg.get("lambda_graph", 0.25))
    hops = int(cfg.get("walk_hops", 1))
    damping = float(cfg.get("damping", 0.50)) if hops == 2 else 0.0
    degree_norm = str(cfg.get("degree_norm", "none"))
    max_bonus = float(cfg.get("max_bonus", 0.50))

    disable_one_hop = (hops == 2) or (anchor_top_m < 2 and hops == 1)

    # 1-hop contributions: sum over anchors
    one_hop: Dict[str, float] = {vid: 0.0 for vid in ids}
    any_contrib = False
    for v in ids:
        acc = 0.0
        if not disable_one_hop:
            for u in anchors:
                if u == v:
                    continue
                w = _edge_weight(edges, u, v)
                if abs(w) >= thresh:
                    acc += w
        if acc != 0.0:
            one_hop[v] = acc
            any_contrib = True

    # 2-hop contributions (best-path via intermediate inside the slice)
    two_hop_best: Dict[str, float] = {vid: 0.0 for vid in ids}
    if hops == 2:
        # Precompute best anchor->w link for each w (subject to threshold)
        best_aw: Dict[str, float] = {}
        for w_id in ids:
            best = 0.0
            for u in anchors:
                if u == w_id:
                    continue
                w1 = _edge_weight(edges, u, w_id)
                if abs(w1) >= thresh and abs(w1) > abs(best):
                    best = w1
            best_aw[w_id] = best
        for v in ids:
            best_path = 0.0
            for w_id in ids:
                if w_id == v:
                    continue
                bw = best_aw.get(w_id, 0.0)
                if bw == 0.0:
                    continue
                w2 = _edge_weight(edges, w_id, v)
                if abs(w2) >= thresh:
                    val = bw * w2
                    if abs(val) > abs(best_path):
                        best_path = val
            if best_path != 0.0:
                two_hop_best[v] = damping * best_path
                any_contrib = True

    # Degree normalization if requested
    deg_cache: Dict[str, int] = {}
    if degree_norm == "invdeg":
        for vid in ids:
            deg_cache[vid] = _degree(edges, vid, thresh, considered)

    # Compute bonuses and hybrid scores
    hybrid_scores: List[Tuple[float, str, int]] = []  # (score, id, original_index)
    for idx, vid in enumerate(ids):
        bonus = one_hop.get(vid, 0.0) + two_hop_best.get(vid, 0.0)
        if degree_norm == "invdeg":
            d = deg_cache.get(vid, 0)
            if d > 0:
                bonus = bonus / float(d)
        # clamp bonus
        if bonus > max_bonus:
            bonus = max_bonus
        elif bonus < -max_bonus:
            bonus = -max_bonus
        # final score
        s = float(sims[idx]) + lam * float(bonus)
        hybrid_scores.append((s, vid, idx))

    # If no graph contribution at all, keep order unchanged
    if not any_contrib:
        return list(items), {
            "hybrid_used": False,
            "k_considered": k_considered,
            "k_reordered": 0,
            "anchor_top_m": anchor_top_m,
            "walk_hops": hops,
            "edge_threshold": thresh,
            "lambda_graph": lam,
            "damping": damping,
            "degree_norm": degree_norm,
            "k_max": int(cfg.get("k_max", 128)),
        }

    # Keep the original top-1 fixed; reorder only the remainder deterministically
    if k_considered >= 2:
        rest = list(range(1, k_considered))
        rest.sort(key=lambda i: (-hybrid_scores[i][0], hybrid_scores[i][1]))
        order = [0] + rest
    else:
        order = [0]

    # Count reorders within the slice
    k_reordered = sum(1 for i, j in enumerate(order) if i != j)

    # Rebuild items: reorder top slice, then append untouched tail
    reordered = [work[i] for i in order] + tail

    metrics = {
        "hybrid_used": True,
        "k_considered": k_considered,
        "k_reordered": int(k_reordered),
        "anchor_top_m": anchor_top_m,
        "walk_hops": hops,
        "edge_threshold": thresh,
        "lambda_graph": lam,
        "damping": damping,
        "degree_norm": degree_norm,
        "k_max": int(cfg.get("k_max", 128)),
    }
    return reordered, metrics


__all__ = ["rerank_with_gel"]
