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


# ---------------------------
# Config handling (with defaults)
# ---------------------------

def _graph_cfg(ctx: Any) -> Dict[str, Any]:
    base = getattr(ctx, "config", None) or getattr(ctx, "cfg", None) or {}
    g = dict((base.get("graph") or {}))
    # Defaults are conservative and deterministic
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
# Public API
# ---------------------------

def observe_retrieval(ctx: Any, state: Any, items: Iterable[Any], *, turn: Optional[int] = None, agent: Optional[str] = None) -> Dict[str, Any]:
    """Observe a retrieval list and update co‑activation edges deterministically.

    Args:
        ctx/state: engine context and state; only `state.graph` is mutated.
        items: iterable of retrieved items; each item should carry (id, score).
        turn: optional turn index for `last_seen_turn` bookkeeping.
        agent: optional agent name, included in returned metrics only.

    Returns metrics suitable for logging.
    """
    cfg = _graph_cfg(ctx)
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
    norm: List[Tuple[str, float]] = [
        _as_id_score(it) for it in (items or [])
    ]
    k_in = len(norm)
    norm = [(i, s) for (i, s) in norm if s >= threshold]
    norm.sort(key=lambda t: (-t[1], t[0]))
    used = norm[: top_k]

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


def tick(ctx: Any, state: Any, *, decay_dt: int = 1, turn: Optional[int] = None, agent: Optional[str] = None) -> Dict[str, Any]:
    """Apply exponential decay to all edges; drop those below the floor.

    Args:
        decay_dt: logical time delta in "turns" to apply in this tick (default 1).
        turn/agent: optional, echoed in metrics; `updated_at` bookkeeping can use turn.

    Returns metrics including counts of decayed and dropped edges.
    """
    cfg = _graph_cfg(ctx)
    gstore = _ensure_graph_store(state)
    edges = gstore.get("edges", {})

    half_life = float(cfg["decay"].get("half_life_turns", 200))
    floor = float(cfg["decay"].get("floor", 0.0))

    if not edges:
        return {"event": "edge_decay", "agent": agent, "decayed_edges": 0, "dropped_edges": 0, "half_life_turns": half_life, "floor": floor}

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


__all__ = ["observe_retrieval", "tick"]
