# Clematis
# GEL (Graph Evolution Layer) — PR22: Edge Update + Decay Policy
#
# This module is self‑contained and deterministic. It mutates only the in‑memory
# state graph store (attached to `state.graph`) and returns metrics so callers
# can log them. All behavior is gated by config under `graph.*` and callers
# should check `graph.enabled` before invoking.
#
# Public surface (kept tiny, stable):
#   observe_retrieval(ctx, state, items, turn=None, agent=None) -> dict
#   tick(ctx, state, decay_dt=1, turn=None, agent=None) -> dict
#
# State layout (in‑memory, persisted by snapshot.py):
#   state.graph = {
#       "nodes": {<id>: {"id": str, "label": str|None, "attrs": dict}},  # optional
#       "edges": {<key>: {
#           "id": str, "src": str, "dst": str,
#           "weight": float, "rel": "coact",
#           "updated_at": None|str,  # reserved; timestamps optional
#           "attrs": {"coact": int, "last_seen_turn": int|None},
#       }},
#       "meta": {"schema": "v1"},
#   }

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Tuple


# Helper: gate all behavior when GEL is disabled
def _graph_enabled(ctx: Any) -> bool:
    try:
        return bool((_graph_cfg(ctx) or {}).get("enabled", False))
    except Exception:
        return False


# ---------------------------
# Config handling (with defaults)
# ---------------------------


def _graph_cfg(ctx: Any) -> Dict[str, Any]:
    # Accept both dict contexts and objects with .config/.cfg
    if isinstance(ctx, dict):
        root = ctx
    else:
        root = getattr(ctx, "config", None) or getattr(ctx, "cfg", None) or {}

    # Start from any provided graph sub-config
    base_graph: Dict[str, Any] = {}
    if isinstance(root, dict):
        base_graph = dict(root.get("graph") or {})

    # Defaults are conservative and deterministic
    g: Dict[str, Any] = dict(base_graph)
    g.setdefault("enabled", False)
    g.setdefault("coactivation_threshold", 0.20)
    g.setdefault("observe_top_k", 64)
    g.setdefault("pair_cap_per_obs", 2048)

    upd = dict(g.get("update") or {})
    upd.setdefault("mode", "additive")  # additive | proportional
    upd.setdefault("alpha", 0.02)
    upd.setdefault("clamp_min", -1.0)
    upd.setdefault("clamp_max", 1.0)
    g["update"] = upd

    dec = dict(g.get("decay") or {})
    dec.setdefault("half_life_turns", 200)
    dec.setdefault("floor", 0.0)
    g["decay"] = dec

    # PR24: merge/split/promotion defaults (feature-flagged)
    mg = dict(g.get("merge") or {})
    mg.setdefault("enabled", False)
    mg.setdefault("min_size", 3)
    mg.setdefault("min_avg_w", 0.20)
    mg.setdefault("max_diameter", 2)
    mg.setdefault("cap_per_turn", 4)
    g["merge"] = mg

    sp = dict(g.get("split") or {})
    sp.setdefault("enabled", False)
    sp.setdefault("weak_edge_thresh", 0.05)
    sp.setdefault("min_component_size", 2)
    sp.setdefault("cap_per_turn", 4)
    g["split"] = sp

    pr = dict(g.get("promotion") or {})
    pr.setdefault("enabled", False)
    pr.setdefault("label_mode", "lexmin")  # or "concat_k"
    pr.setdefault("topk_label_ids", 3)
    pr.setdefault("attach_weight", 0.5)
    pr.setdefault("cap_per_turn", 2)
    g["promotion"] = pr

    return g


# ---------------------------
# State access helpers
# ---------------------------


def _ensure_graph_store(state: Any) -> Dict[str, Any]:
    """Return the mutable graph store attached to state, creating if absent."""
    # Prefer attribute `graph`, else dict key, else attach attribute.
    store = None
    if hasattr(state, "graph"):
        store = getattr(state, "graph")
    elif isinstance(state, dict):
        store = state.get("graph")
    if store is None:
        store = {"nodes": {}, "edges": {}, "meta": {"schema": "v1"}}
        try:
            setattr(state, "graph", store)
        except Exception:
            if isinstance(state, dict):
                state["graph"] = store
            else:
                # last resort: return store without attaching (still usable by caller in this run)
                pass
    # Ensure sub‑maps exist
    store.setdefault("nodes", {})
    store.setdefault("edges", {})
    store.setdefault("meta", {"schema": "v1"})
    meta = store.setdefault("meta", {"schema": "v1"})
    meta.setdefault("merges", [])
    meta.setdefault("splits", [])
    meta.setdefault("promotions", [])
    meta.setdefault("concept_nodes_count", 0)
    return store


def _edge_key(a: str, b: str) -> Tuple[str, str, str]:
    """Canonical undirected key: (key, src, dst) with src < dst lexicographically."""
    sa, sb = str(a), str(b)
    if sa <= sb:
        src, dst = sa, sb
    else:
        src, dst = sb, sa
    return f"{src}→{dst}", src, dst


def _clamp(x: float, lo: float, hi: float) -> float:
    return hi if x > hi else lo if x < lo else x


def _as_id_score(item: Any) -> Tuple[str, float]:
    """Best‑effort adapter: support (id,score), dicts, or objects with fields."""
    if isinstance(item, tuple) and len(item) >= 2:
        return str(item[0]), float(item[1])
    if isinstance(item, dict):
        # common keys
        if "id" in item and "score" in item:
            return str(item["id"]), float(item["score"])
        for k in ("episode_id", "node_id", "target_id"):
            if k in item and ("score" in item or "similarity" in item):
                return str(item[k]), float(item.get("score", item.get("similarity", 0.0)))
    # object attributes
    for idk in ("id", "episode_id", "node_id", "target_id"):
        if hasattr(item, idk):
            idv = getattr(item, idk)
            score = 0.0
            if hasattr(item, "score"):
                score = getattr(item, "score")
            elif hasattr(item, "similarity"):
                score = getattr(item, "similarity")
            return str(idv), float(score)
    # fallback: stable repr, score 0
    return repr(item), 0.0


# ---------------------------
# Graph utilities for PR24 (deterministic)
# ---------------------------


def _build_adj(
    edges: Dict[str, Any], *, nodes: Optional[List[str]] = None, min_w: float = 0.0
) -> Dict[str, List[str]]:
    """Undirected adjacency filtered by |weight| >= min_w; neighbors sorted.
    If `nodes` provided, restrict to that induced set.
    """
    allow = set(nodes) if nodes is not None else None
    adj: Dict[str, List[str]] = {}
    if not isinstance(edges, dict):
        return adj
    for key, rec in edges.items():
        if not isinstance(rec, dict):
            continue
        try:
            w = abs(float(rec.get("weight", 0.0)))
        except Exception:
            w = 0.0
        if w < float(min_w):
            continue
        a = str(rec.get("src"))
        b = str(rec.get("dst"))
        if allow is not None and (a not in allow or b not in allow):
            continue
        adj.setdefault(a, [])
        adj.setdefault(b, [])
        adj[a].append(b)
        adj[b].append(a)
    for k in list(adj.keys()):
        adj[k] = sorted(set(adj[k]))
    return adj


def _connected_components(adj: Dict[str, List[str]]) -> List[List[str]]:
    """Lexicographically deterministic connected components from adjacency."""
    seen: set[str] = set()
    comps: List[List[str]] = []
    for start in sorted(adj.keys()):
        if start in seen:
            continue
        stack = [start]
        seen.add(start)
        comp: List[str] = []
        while stack:
            v = stack.pop()
            comp.append(v)
            for u in adj.get(v, []):
                if u not in seen:
                    seen.add(u)
                    stack.append(u)
        comps.append(sorted(comp))
    return comps


def _component_edges(edges: Dict[str, Any], nodes: List[str]) -> List[float]:
    """Collect absolute weights for edges whose endpoints are both in `nodes`."""
    node_set = set(nodes)
    ws: List[float] = []
    for key, rec in edges.items():
        if not isinstance(rec, dict):
            continue
        a = str(rec.get("src"))
        b = str(rec.get("dst"))
        if a in node_set and b in node_set:
            try:
                ws.append(abs(float(rec.get("weight", 0.0))))
            except Exception:
                pass
    return ws


def _diameter_unweighted(adj: Dict[str, List[str]], nodes: List[str]) -> int:
    """Compute unweighted diameter (max shortest-path length) within `nodes`.
    Uses BFS from each node; bounded by |nodes| and deterministic ordering.
    Returns 0 for singletons.
    """
    if len(nodes) <= 1:
        return 0
    node_set = set(nodes)
    import collections

    diam = 0
    for s in nodes:  # nodes already sorted
        # BFS from s restricted to the component
        q = collections.deque([s])
        dist = {s: 0}
        while q:
            v = q.popleft()
            for u in adj.get(v, []):
                if u in node_set and u not in dist:
                    dist[u] = dist[v] + 1
                    q.append(u)
        if len(dist) < len(nodes):
            # disconnected in provided adj; treat as infinite => filter elsewhere
            return 10**9
        ecc = max(dist.values())
        if ecc > diam:
            diam = ecc
    return diam


# ---------------------------
# PR24: Merge / Split / Promotion (feature-flagged, deterministic)
# ---------------------------


def merge_candidates(ctx: Any, state: Any) -> List[Dict[str, Any]]:
    if not _graph_enabled(ctx):
        return []
    cfg = _graph_cfg(ctx)
    mg = cfg.get("merge", {})
    edges = _ensure_graph_store(state).get("edges", {})
    min_w = float(mg.get("min_avg_w", 0.20))
    min_size = int(mg.get("min_size", 3))
    max_d = int(mg.get("max_diameter", 2))

    # Build strong-edge graph and components
    adj = _build_adj(edges, min_w=min_w)
    comps = _connected_components(adj)

    out: List[Dict[str, Any]] = []
    for nodes in comps:
        if len(nodes) < min_size:
            continue
        # diameter on the same strong-edge graph
        d = _diameter_unweighted(adj, nodes)
        if d > max_d:
            continue
        ws = _component_edges(edges, nodes)
        avg_w = (sum(ws) / len(ws)) if ws else 0.0
        sig = "|".join(nodes)
        out.append(
            {
                "type": "merge_candidate",
                "nodes": nodes,
                "size": len(nodes),
                "avg_w": avg_w,
                "diameter": d,
                "signature": sig,
            }
        )

    # Deterministic ordering
    out.sort(key=lambda c: (-float(c["avg_w"]), -int(c["size"]), tuple(c["nodes"])))
    return out


def apply_merge(ctx: Any, state: Any, cluster: Dict[str, Any]) -> Dict[str, Any]:
    if not _graph_enabled(ctx):
        return {"event": "merge_applied", "size": 0, "avg_w": 0.0, "diameter": 0}
    g = _ensure_graph_store(state)
    meta = g.setdefault("meta", {})
    merges = meta.setdefault("merges", [])
    rec = {
        "nodes": list(cluster.get("nodes", [])),
        "size": int(cluster.get("size", len(cluster.get("nodes", [])))),
        "avg_w": float(cluster.get("avg_w", 0.0)),
        "diameter": int(cluster.get("diameter", 0)),
        "signature": str(cluster.get("signature", "")),
    }
    merges.append(rec)
    return {
        "event": "merge_applied",
        "size": rec["size"],
        "avg_w": rec["avg_w"],
        "diameter": rec["diameter"],
    }


def split_candidates(ctx: Any, state: Any) -> List[Dict[str, Any]]:
    if not _graph_enabled(ctx):
        return []
    cfg = _graph_cfg(ctx)
    sp = cfg.get("split", {})
    edges = _ensure_graph_store(state).get("edges", {})
    weak = float(sp.get("weak_edge_thresh", 0.05))
    min_comp = int(sp.get("min_component_size", 2))

    # Components on the current nonzero-edge graph
    adj_all = _build_adj(edges, min_w=0.0)
    comps = _connected_components(adj_all)

    out: List[Dict[str, Any]] = []
    for nodes in comps:
        if len(nodes) < 2:
            continue
        # Count edges within this component and those that would be removed
        node_set = set(nodes)
        total_in = 0
        removed = 0
        for key, rec in edges.items():
            if not isinstance(rec, dict):
                continue
            a = str(rec.get("src"))
            b = str(rec.get("dst"))
            if a in node_set and b in node_set:
                total_in += 1
                try:
                    if abs(float(rec.get("weight", 0.0))) < weak:
                        removed += 1
                except Exception:
                    pass
        # New components after removing weak edges
        adj_strong = _build_adj(edges, nodes=nodes, min_w=weak)
        subcomps = _connected_components(adj_strong)
        if len(subcomps) <= 1:
            continue
        # ensure each new component meets size threshold
        if any(len(c) < min_comp for c in subcomps):
            continue
        sig = "|".join(nodes)
        out.append(
            {
                "type": "split_candidate",
                "original": nodes,
                "parts": subcomps,
                "removed_edges": int(removed),
                "orig_edges": int(total_in),
                "signature": sig,
            }
        )

    out.sort(key=lambda s: (-int(s["removed_edges"]), tuple(s["original"])))
    return out


def apply_split(ctx: Any, state: Any, split: Dict[str, Any]) -> Dict[str, Any]:
    if not _graph_enabled(ctx):
        return {"event": "split_applied", "removed_edges": 0, "parts": 0}
    g = _ensure_graph_store(state)
    meta = g.setdefault("meta", {})
    splits = meta.setdefault("splits", [])
    rec = {
        "original": list(split.get("original", [])),
        "parts": [list(p) for p in split.get("parts", [])],
        "removed_edges": int(split.get("removed_edges", 0)),
        "orig_edges": int(split.get("orig_edges", 0)),
        "signature": str(split.get("signature", "")),
    }
    splits.append(rec)
    return {
        "event": "split_applied",
        "removed_edges": rec["removed_edges"],
        "parts": len(rec["parts"]),
    }


def promote_clusters(ctx: Any, state: Any, clusters: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not _graph_enabled(ctx):
        return []
    cfg = _graph_cfg(ctx)
    pr = cfg.get("promotion", {})
    mode = str(pr.get("label_mode", "lexmin"))
    topk = int(pr.get("topk_label_ids", 3))
    attach_w = float(pr.get("attach_weight", 0.5))
    # clamp
    if attach_w > 1.0:
        attach_w = 1.0
    elif attach_w < -1.0:
        attach_w = -1.0

    promos: List[Dict[str, Any]] = []
    for c in clusters or []:
        nodes = list(c.get("nodes", []))
        if not nodes:
            continue
        nodes_sorted = sorted(nodes)
        cid = f"c::{nodes_sorted[0]}"  # deterministic id by lexmin member
        if mode == "concat_k":
            label = "+".join(nodes_sorted[: max(1, topk)])
        else:
            label = nodes_sorted[0]
        promos.append(
            {
                "concept_id": cid,
                "label": label,
                "members": nodes_sorted,
                "attach_weight": attach_w,
            }
        )
    # deterministic order by concept id
    promos.sort(key=lambda p: p["concept_id"])
    return promos


def apply_promotion(ctx: Any, state: Any, promo: Dict[str, Any]) -> Dict[str, Any]:
    if not _graph_enabled(ctx):
        return {"event": "promotion_applied", "concept": "", "members": 0}
    g = _ensure_graph_store(state)
    nodes = g.setdefault("nodes", {})
    edges = g.setdefault("edges", {})
    meta = g.setdefault("meta", {})

    cid = str(promo.get("concept_id"))
    label = str(promo.get("label", cid))
    members = [str(x) for x in promo.get("members", [])]
    w = float(promo.get("attach_weight", 0.5))

    # Upsert concept node
    if cid not in nodes:
        nodes[cid] = {"id": cid, "label": label, "attrs": {"kind": "concept"}}
        meta["concept_nodes_count"] = int(meta.get("concept_nodes_count", 0)) + 1
    else:
        # do not change existing labels deterministically
        pass

    # Attach edges concept<->member deterministically
    for m in members:
        key, src, dst = _edge_key(cid, m)
        rec = edges.get(key)
        if rec is None:
            rec = {
                "id": key,
                "src": src,
                "dst": dst,
                "weight": w,
                "rel": "concept",
                "updated_at": None,
                "attrs": {},
            }
            edges[key] = rec
        else:
            # deterministic overwrite to the configured weight
            rec["rel"] = "concept"
            rec["weight"] = w

    # keep meta edges_count consistent for inspector/health
    meta["edges_count"] = len(edges)
    return {"event": "promotion_applied", "concept": cid, "members": len(members)}


# ---------------------------
# Public API
# ---------------------------


def observe_retrieval(
    ctx: Any,
    state: Any,
    items: Iterable[Any],
    *,
    turn: Optional[int] = None,
    agent: Optional[str] = None,
) -> Dict[str, Any]:
    """Observe a retrieval list and update co‑activation edges deterministically.

    Args:
        ctx/state: engine context and state; only `state.graph` is mutated.
        items: iterable of retrieved items; each item should carry (id, score).
        turn: optional turn index for `last_seen_turn` bookkeeping.
        agent: optional agent name, included in returned metrics only.

    Returns metrics suitable for logging.
    """
    cfg = _graph_cfg(ctx)
    if not bool(cfg.get("enabled", False)):
        # Fast no-op: do not touch state; return minimal, deterministic metrics.
        upd = cfg.get("update", {}) or {}
        return {
            "event": "observe_retrieval",
            "agent": agent,
            "k_in": 0,
            "k_used": 0,
            "pairs_updated": 0,
            "threshold": float(cfg.get("coactivation_threshold", 0.20)),
            "mode": str(upd.get("mode", "additive")),
            "alpha": float(upd.get("alpha", 0.02)),
        }

    gstore = _ensure_graph_store(state)

    threshold = float(cfg["coactivation_threshold"])
    top_k = int(cfg["observe_top_k"])
    pair_cap = int(cfg["pair_cap_per_obs"])
    upd = cfg["update"]
    mode = str(upd.get("mode", "additive"))
    alpha = float(upd.get("alpha", 0.02))
    clamp_min = float(upd.get("clamp_min", -1.0))
    clamp_max = float(upd.get("clamp_max", 1.0))

    # Normalize, filter, and sort input deterministically: (-score, id)
    norm: List[Tuple[str, float]] = [_as_id_score(it) for it in (items or [])]
    k_in = len(norm)
    norm = [(i, s) for (i, s) in norm if s >= threshold]
    norm.sort(key=lambda t: (-t[1], t[0]))
    used = norm[:top_k]

    # Generate undirected pairs in lexicographic id order, cap total pairs
    pairs_updated = 0
    edges = gstore["edges"]

    # Early exit fast path
    if not used or pair_cap == 0:
        return {
            "event": "observe_retrieval",
            "agent": agent,
            "k_in": k_in,
            "k_used": len(used),
            "pairs_updated": 0,
            "threshold": threshold,
            "mode": mode,
            "alpha": alpha,
        }

    # Deterministic nested loops with prefix truncation
    cap_left = pair_cap
    for i in range(len(used)):
        if cap_left <= 0:
            break
        ida, sa = used[i]
        for j in range(i + 1, len(used)):
            if cap_left <= 0:
                break
            idb, sb = used[j]
            key, src, dst = _edge_key(ida, idb)
            rec = edges.get(key)
            if rec is None:
                rec = {
                    "id": key,
                    "src": src,
                    "dst": dst,
                    "weight": 0.0,
                    "rel": "coact",
                    "updated_at": None,
                    "attrs": {"coact": 0, "last_seen_turn": None},
                }
                edges[key] = rec
            # Update weight with selected mode
            w = float(rec.get("weight", 0.0))
            if mode == "proportional":
                inc = alpha * (1.0 - min(abs(w), 1.0))
                w = _clamp(w + inc, clamp_min, clamp_max)
            else:  # additive
                w = _clamp(w + alpha, clamp_min, clamp_max)
            rec["weight"] = w
            # Bump attrs
            attrs = rec.setdefault("attrs", {})
            attrs["coact"] = int(attrs.get("coact", 0)) + 1
            if turn is not None:
                attrs["last_seen_turn"] = int(turn)
            pairs_updated += 1
            cap_left -= 1

    return {
        "event": "observe_retrieval",
        "agent": agent,
        "k_in": k_in,
        "k_used": len(used),
        "pairs_updated": pairs_updated,
        "threshold": threshold,
        "mode": mode,
        "alpha": alpha,
    }


def tick(
    ctx: Any,
    state: Any,
    *,
    decay_dt: int = 1,
    turn: Optional[int] = None,
    agent: Optional[str] = None,
) -> Dict[str, Any]:
    """Apply exponential decay to all edges; drop those below the floor.

    Args:
        decay_dt: logical time delta in "turns" to apply in this tick (default 1).
        turn/agent: optional, echoed in metrics; `updated_at` bookkeeping can use turn.

    Returns metrics including counts of decayed and dropped edges.
    """
    cfg = _graph_cfg(ctx)
    if not bool(cfg.get("enabled", False)):
        return {
            "event": "edge_decay",
            "agent": agent,
            "decayed_edges": 0,
            "dropped_edges": 0,
            "half_life_turns": float(cfg.get("decay", {}).get("half_life_turns", 200)),
            "floor": float(cfg.get("decay", {}).get("floor", 0.0)),
        }

    gstore = _ensure_graph_store(state)
    edges = gstore.get("edges", {})

    half_life = float(cfg["decay"].get("half_life_turns", 200))
    floor = float(cfg["decay"].get("floor", 0.0))

    if not edges:
        return {
            "event": "edge_decay",
            "agent": agent,
            "decayed_edges": 0,
            "dropped_edges": 0,
            "half_life_turns": half_life,
            "floor": floor,
        }

    # Compute decay factor once; if half_life is huge, this approaches 1.
    dt = max(0, int(decay_dt))
    if half_life <= 0:
        decay_factor = 0.0
    else:
        decay_factor = 0.5 ** (float(dt) / half_life)

    decayed = 0
    dropped = 0
    to_delete: List[str] = []

    for key, rec in edges.items():
        w = float(rec.get("weight", 0.0))
        w2 = w * decay_factor
        if abs(w2) < floor:
            to_delete.append(key)
            dropped += 1
        else:
            if w2 != w:
                rec["weight"] = w2
                decayed += 1
            # update bookkeeping
            if turn is not None:
                rec["updated_at"] = None  # reserved; keep None unless real timestamping is added
            attrs = rec.setdefault("attrs", {})
            if attrs.get("last_seen_turn") is None and turn is not None:
                attrs["last_seen_turn"] = int(turn)

    for key in to_delete:
        edges.pop(key, None)

    # Maintain an easy counter in meta for inspector/health
    meta = gstore.setdefault("meta", {})
    meta["edges_count"] = len(edges)

    return {
        "event": "edge_decay",
        "agent": agent,
        "decayed_edges": decayed,
        "dropped_edges": dropped,
        "half_life_turns": half_life,
        "floor": floor,
    }


__all__ = [
    "observe_retrieval",
    "tick",
    # PR24 APIs
    "merge_candidates",
    "apply_merge",
    "split_candidates",
    "apply_split",
    "promote_clusters",
    "apply_promotion",
]
