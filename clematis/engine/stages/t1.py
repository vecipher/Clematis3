from __future__ import annotations
from collections import defaultdict
import heapq
from typing import Dict, Any, List, Tuple
from ..types import T1Result
from ..cache import LRUCache, stable_key

# module-level (OK for pure stage since it only caches)
_T1_CACHE = LRUCache(max_entries=512, ttl_s=300)



EPS = 1e-6

def _match_keywords(text: str, labels: List[Tuple[str, str]]) -> Dict[str, float]:
    """Very simple seed matcher: if label appears in text → seed weight 1.0."""
    t = text.lower()
    seeds: Dict[str, float] = {}
    for node_id, label in labels:
        if label and label.lower() in t:
            seeds[node_id] = max(seeds.get(node_id, 0.0), 1.0)
    return seeds

def _compute_decay(distance: int, cfg_t1: dict) -> float:
    mode = cfg_t1.get("decay", {}).get("mode", "exp_floor")
    if mode == "attn_quad":
        alpha = float(cfg_t1["decay"].get("alpha", 0.8))
        return 1.0 / (1.0 + alpha * (distance ** 2))
    # default exp_floor
    rate = float(cfg_t1["decay"].get("rate", 0.6))
    floor = float(cfg_t1["decay"].get("floor", 0.05))
    return max(rate ** distance, floor)

def t1_propagate(ctx, state, text: str) -> T1Result:
    cfg_t1 = ctx.cfg.t1
    edge_mult = cfg_t1.get("edge_type_mult", {"supports": 1.0, "associates": 0.6, "contradicts": 0.8})
    queue_budget = int(cfg_t1.get("queue_budget", 10_000))
    node_budget = float(cfg_t1.get("node_budget", 1.5))
    radius_cap  = int(cfg_t1.get("radius_cap", 4))
    iter_cap    = int(cfg_t1.get("iter_cap", 50))

    store = state.get("store")
    active_graphs = state.get("active_graphs", [])
    total_pops = 0
    radius_hits = 0
    node_hits = 0
    max_delta = 0.0

    all_deltas: List[Dict[str, Any]] = []

    for gid in active_graphs:
        g = store.get_graph(gid)
        # Build a tiny label list for seeding
        labels = [(n.id, n.label) for n in g.nodes.values()]
        seeds = _match_keywords(text, labels)
        if not seeds:
            continue
        # ---- T1 cache pre-check ----
        etag = store.version_etag(gid)
        decay_cfg = cfg_t1.get("decay", {})
        # edge_mult is already computed above; include its shape in the key
        seed_ids = sorted(seeds.keys())
        ckey = ("t1", gid, etag, stable_key(decay_cfg), stable_key(edge_mult), tuple(seed_ids))
        hit = _T1_CACHE.get(ckey)
        if hit is not None:
            all_deltas.extend(hit["deltas"])
            total_pops += hit["metrics"]["pops"]
            continue

        csr = store.csr(gid)  # dict[src] -> list[(dst, Edge)]
        acc = defaultdict(float)     # accumulated contribution per node
        dist = {}                    # hop distance per node
        pops = 0

        # max-heap by remaining magnitude (use negative for heapq)
        pq: List[Tuple[float, str, float]] = []
        for nid, w in seeds.items():
            heapq.heappush(pq, (-abs(w), nid, float(w)))
            acc[nid] += w
            dist[nid] = 0
            max_delta = max(max_delta, abs(w))

        while pq and pops < queue_budget and pops < iter_cap:
            _, u, w = heapq.heappop(pq)
            pops += 1
            if abs(acc[u]) >= node_budget:
                node_hits += 1
                continue
            if u not in csr:
                continue
            for (v, e) in csr[u]:
                d = dist[u] + 1
                if d > radius_cap:
                    radius_hits += 1
                    continue
                decay = _compute_decay(d, cfg_t1)
                contrib = w * float(e.weight) * float(edge_mult.get(e.rel, 0.6)) * decay
                if abs(contrib) < EPS:
                    continue
                acc[v] += contrib
                max_delta = max(max_delta, abs(contrib))
                if (v not in dist) or (d < dist[v]):
                    dist[v] = d
                if abs(acc[v]) < node_budget:
                    heapq.heappush(pq, (-abs(contrib), v, contrib))

        # Convert accumulators to per-graph deltas and cache the result
        deltas_for_gid: List[Dict[str, Any]] = []
        for nid, val in acc.items():
            if abs(val) < EPS:
                continue
            deltas_for_gid.append({"op": "upsert_node", "id": nid})
        result = {"deltas": deltas_for_gid, "metrics": {"pops": pops}}
        _T1_CACHE.put(ckey, result)
        all_deltas.extend(deltas_for_gid)
        total_pops += pops

    metrics = {
        "pops": total_pops,
        "radius_cap_hits": radius_hits,
        "node_budget_hits": node_hits,
        "max_delta": max_delta,
        "graphs_touched": len(active_graphs),
    }
    return T1Result(graph_deltas=all_deltas, metrics=metrics)