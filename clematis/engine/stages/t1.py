from __future__ import annotations
from collections import defaultdict
import heapq
from typing import Dict, Any, List, Tuple
from ..types import T1Result
from ..cache import LRUCache, stable_key, ThreadSafeCache, ThreadSafeBytesCache
from ..util.ring import DedupeRing
from ..util.lru_det import DeterministicLRUSet
from ..util.lru_bytes import LRUBytes
from ..util.parallel import run_parallel

# module-level cache holder configured via Config.t1.cache
_T1_CACHE = None
_T1_CACHE_CFG = None
_T1_CACHE_KIND = None


# Small gate helper for T1 parallelism
def _t1_parallel_enabled(cfg) -> bool:
    """
    Flag gate for PR66: enable parallel T1 when perf.parallel is on, t1 gate is true,
    and max_workers > 1. Defaults keep parallelism OFF.
    """
    enabled = bool(_cfg_get(cfg, ["perf", "parallel", "enabled"], False))
    t1_gate = bool(_cfg_get(cfg, ["perf", "parallel", "t1"], False))
    workers = int(_cfg_get(cfg, ["perf", "parallel", "max_workers"], 0) or 0)
    return bool(enabled and t1_gate and workers > 1)


def _get_cache(ctx, cfg_t1: dict):
    """
    PR32-aware cache selection for T1 results.
    Returns: (cache, kind_str) where kind_str in {"bytes", "lru"} or (None, None).
    """
    perf_on = bool(_cfg_get(ctx, ["cfg", "perf", "enabled"], False))
    max_e = int(_cfg_get(ctx, ["cfg", "perf", "t1", "cache", "max_entries"], 0) or 0)
    max_b = int(_cfg_get(ctx, ["cfg", "perf", "t1", "cache", "max_bytes"], 0) or 0)
    global _T1_CACHE, _T1_CACHE_CFG, _T1_CACHE_KIND
    # PR32 path: size-aware cache behind perf gate
    if perf_on and (max_e > 0 or max_b > 0):
        cfg_tuple = ("bytes", max_e, max_b)
        if _T1_CACHE is None or _T1_CACHE_CFG != cfg_tuple:
            _T1_CACHE = ThreadSafeBytesCache(LRUBytes(max_entries=max_e, max_bytes=max_b))  # type: ignore[arg-type]
            _T1_CACHE_CFG = cfg_tuple
            _T1_CACHE_KIND = "bytes"
        return _T1_CACHE, _T1_CACHE_KIND
    # Legacy fallback (pre-PR32 semantics) using t1.cache (LRU with TTL)
    c = cfg_t1.get("cache", {}) or {}
    enabled = bool(c.get("enabled", True))
    if not enabled:
        return None, None
    max_entries = int(c.get("max_entries", 512))
    ttl_s = int(c.get("ttl_s", 300))
    cfg_tuple = ("lru", max_entries, ttl_s)
    if _T1_CACHE is None or _T1_CACHE_CFG != cfg_tuple:
        # PR65: wrap legacy LRU with a thin lock to make shared access safe under optional parallel paths.
        _T1_CACHE = ThreadSafeCache(LRUCache(max_entries=max_entries, ttl_s=ttl_s))  # type: ignore[arg-type]
        _T1_CACHE_CFG = cfg_tuple
        _T1_CACHE_KIND = "lru"
    return _T1_CACHE, _T1_CACHE_KIND


EPS = 1e-6


def _cfg_get(obj, path, default=None):
    """
    Safe nested config getter that works with both dict-like and attribute-like cfg objects.
    """
    cur = obj
    for i, key in enumerate(path):
        try:
            if isinstance(cur, dict):
                cur = cur.get(key, {} if i < len(path) - 1 else default)
            else:
                cur = getattr(cur, key)
        except Exception:
            return default
    return cur


def _estimate_t1_cost(deltas, metrics_dict):
    """Deterministic, conservative byte estimate for caching a T1 result."""
    try:
        n = len(deltas or [])
        id_bytes = 0
        for d in deltas or []:
            if isinstance(d, dict):
                v = d.get("id")
                if v is not None:
                    id_bytes += len(str(v))
        # Base overhead per entry + metrics terms
        base = 24 * n
        metric_overhead = 64
        pops = int((metrics_dict or {}).get("pops", 0)) if isinstance(metrics_dict, dict) else 0
        props = (
            int((metrics_dict or {}).get("propagations", 0))
            if isinstance(metrics_dict, dict)
            else 0
        )
        return id_bytes + base + metric_overhead + 8 * (pops + props)
    except Exception:
        return 1024


def _match_keywords(text: str, labels: List[Tuple[str, str]]) -> Dict[str, float]:
    """
    Seed matcher: case-insensitive, deterministic.
    Iterates labels sorted by their lowercase form to ensure stable seeding order.
    """
    t = text.lower()
    seeds: Dict[str, float] = {}
    for node_id, label in sorted(labels, key=lambda x: (x[1] or "").lower()):
        if label and label.lower() in t:
            # Use max to keep idempotent accumulation if the same node matches multiple phrases
            seeds[node_id] = max(seeds.get(node_id, 0.0), 1.0)
    return seeds


def _compute_decay(distance: int, cfg_t1: dict) -> float:
    mode = cfg_t1.get("decay", {}).get("mode", "exp_floor")
    if mode == "attn_quad":
        alpha = float(cfg_t1["decay"].get("alpha", 0.8))
        return 1.0 / (1.0 + alpha * (distance**2))
    # default exp_floor
    rate = float(cfg_t1["decay"].get("rate", 0.6))
    floor = float(cfg_t1["decay"].get("floor", 0.05))
    return max(rate**distance, floor)


def t1_propagate(ctx, state, text: str) -> T1Result:
    cfg_t1 = ctx.cfg.t1
    cache, cache_kind = _get_cache(ctx, cfg_t1)
    edge_mult = cfg_t1.get(
        "edge_type_mult", {"supports": 1.0, "associates": 0.6, "contradicts": 0.8}
    )
    queue_budget = int(cfg_t1.get("queue_budget", 10_000))
    node_budget = float(cfg_t1.get("node_budget", 1.5))
    radius_cap = int(cfg_t1.get("radius_cap", 4))
    iter_cap = int(cfg_t1.get("iter_cap", 50))

    # Effective depth cap: min(iter_cap_layers, iter_cap) so legacy iter_cap still works
    iter_cap_layers_cfg = int(cfg_t1.get("iter_cap_layers", 50))
    base_iter_cap_layers = min(iter_cap_layers_cfg, iter_cap)
    # Optional hard cap on number of relaxations (edge traversals)
    relax_cap = cfg_t1.get("relax_cap", None)
    if relax_cap is not None:
        relax_cap = int(relax_cap)

    # --- M5 scheduler slice caps (optional, read-only clamps) ---
    caps = getattr(ctx, "slice_budgets", None) or {}
    slice_t1_iters = caps.get("t1_iters")
    slice_t1_pops = caps.get("t1_pops")
    effective_iter_cap_layers = (
        base_iter_cap_layers
        if slice_t1_iters is None
        else min(base_iter_cap_layers, int(slice_t1_iters))
    )
    effective_queue_budget = (
        queue_budget if slice_t1_pops is None else min(queue_budget, int(slice_t1_pops))
    )

    # ---- PR31 perf caps & dedupe (disabled-path default OFF) ----
    perf_enabled = bool(_cfg_get(ctx.cfg, ["perf", "enabled"], False))
    metrics_enabled = bool(_cfg_get(ctx.cfg, ["perf", "metrics", "report_memory"], False))
    frontier_cap_cfg = int(_cfg_get(ctx.cfg, ["perf", "t1", "caps", "frontier"], 0) or 0)
    visited_cap_cfg = int(_cfg_get(ctx.cfg, ["perf", "t1", "caps", "visited"], 0) or 0)
    dedupe_window_cfg = int(_cfg_get(ctx.cfg, ["perf", "t1", "dedupe_window"], 0) or 0)

    # Frontier cap acts in addition to effective_queue_budget; take the stricter bound
    effective_frontier_cap = None
    if perf_enabled and frontier_cap_cfg > 0:
        effective_frontier_cap = min(frontier_cap_cfg, effective_queue_budget)

    # PR31 counters (only emitted when both gates are ON)
    t1_frontier_evicted = 0
    t1_dedup_hits = 0
    t1_visited_evicted = 0
    t1_cache_evicted = 0
    t1_cache_bytes = 0

    store = state.get("store")
    active_graphs = state.get("active_graphs", [])
    total_pops = 0
    total_iters = 0
    total_propagations = 0
    radius_hits = 0
    layer_hits = 0  # times we skipped due to iter_cap_layers
    node_hits = 0
    max_delta = 0.0
    cache_hits = 0
    cache_misses = 0

    all_deltas: List[Dict[str, Any]] = []

    def _t1_one_graph(gid: str):
        # Returns (deltas_for_gid, per_graph_metrics)
        g = store.get_graph(gid)
        labels = [(n.id, n.label) for n in g.nodes.values()]
        seeds = _match_keywords(text, labels)
        if not seeds:
            return [], {
                "pops": 0,
                "iters": 0,
                "propagations": 0,
                "radius_cap_hits": 0,
                "layer_cap_hits": 0,
                "node_budget_hits": 0,
                "_max_delta_local": 0.0,
                "_cache_hit": 0,
                "_cache_miss": 0,
                "_t1_frontier_evicted": 0,
                "_t1_dedup_hits": 0,
                "_t1_visited_evicted": 0,
                "_t1_cache_evicted": 0,
                "_t1_cache_bytes": 0,
            }

        # ---- T1 cache pre-check ----
        etag = store.version_etag(gid)
        decay_cfg = cfg_t1.get("decay", {})
        seed_ids = sorted(seeds.keys())
        policy_caps = {
            "radius_cap": radius_cap,
            "iter_cap": iter_cap,
            "iter_cap_layers": effective_iter_cap_layers,
            "relax_cap": relax_cap,
            "queue_budget": effective_queue_budget,
            "node_budget": node_budget,
        }
        policy_caps.update(
            {
                "frontier_cap": int(frontier_cap_cfg or 0),
                "visited_cap": int(visited_cap_cfg or 0),
                "dedupe_window": int(dedupe_window_cfg or 0),
            }
        )
        ckey = (
            "t1",
            gid,
            etag,
            stable_key(decay_cfg),
            stable_key(edge_mult),
            stable_key(policy_caps),
            tuple(seed_ids),
        )
        if cache is not None:
            hit = cache.get(ckey)
            if hit is not None:
                return hit["deltas"], {
                    "pops": hit["metrics"]["pops"],
                    "iters": hit["metrics"].get("iters", 0),
                    "propagations": hit["metrics"].get("propagations", 0),
                    "radius_cap_hits": hit["metrics"].get("radius_cap_hits", 0),
                    "layer_cap_hits": hit["metrics"].get("layer_cap_hits", 0),
                    "node_budget_hits": hit["metrics"].get("node_budget_hits", 0),
                    "_max_delta_local": 0.0,  # cached path does not affect max_delta in current design
                    "_cache_hit": 1,
                    "_cache_miss": 0,
                    "_t1_frontier_evicted": 0,
                    "_t1_dedup_hits": 0,
                    "_t1_visited_evicted": 0,
                    "_t1_cache_evicted": 0,
                    "_t1_cache_bytes": 0,
                }

        # --- Compute fresh ---
        csr = store.csr(gid)  # dict[src] -> list[(dst, Edge)]
        acc = defaultdict(float)  # accumulated contribution per node
        dist = {}  # hop distance per node
        pops = 0

        layers_processed = 0  # unique layers beyond seeds we actually explored
        propagations = 0
        layer_hits_local = 0
        radius_cap_hits_local = 0
        node_budget_hits_local = 0

        # PR31 structures (only used when enabled)
        ring = DedupeRing(dedupe_window_cfg) if (perf_enabled and dedupe_window_cfg > 0) else None
        visited_lru = (
            DeterministicLRUSet(visited_cap_cfg) if (perf_enabled and visited_cap_cfg > 0) else None
        )

        # max-heap by remaining magnitude (use negative for heapq)
        pq: List[Tuple[float, str, str, float]] = []
        local_max_delta = 0.0
        for nid, w in seeds.items():
            item = (-abs(w), nid, nid, float(w))
            if ring and ring.contains(nid):
                # Count hits locally; merged deterministically later
                local_t1_dedup_hits = 1
            else:
                heapq.heappush(pq, item)
                if effective_frontier_cap is not None and len(pq) > effective_frontier_cap:
                    ev = len(pq) - effective_frontier_cap
                    pq = heapq.nsmallest(effective_frontier_cap, pq)
                    heapq.heapify(pq)
                    local_t1_frontier_evicted = ev
                else:
                    local_t1_frontier_evicted = 0
                if ring:
                    ring.add(nid)
            acc[nid] += w
            dist[nid] = 0
            local_max_delta = max(local_max_delta, abs(w))

        # Visit PQ
        local_t1_dedup_hits_total = 0
        local_t1_frontier_evicted_total = 0
        local_t1_visited_evicted_total = 0

        # Initialize counters from seeding step
        if 'local_t1_dedup_hits' in locals():
            local_t1_dedup_hits_total += locals()['local_t1_dedup_hits']
        if 'local_t1_frontier_evicted' in locals():
            local_t1_frontier_evicted_total += locals()['local_t1_frontier_evicted']

        while pq and pops < effective_queue_budget:
            _, node_key, u, w = heapq.heappop(pq)
            pops += 1
            if visited_lru and visited_lru.contains(u):
                continue
            if visited_lru:
                if visited_lru.add(u):
                    local_t1_visited_evicted_total += 1
            layer = dist.get(u, 0)  # 0 for seeds, 1 for first neighbors, etc.

            if layer > 0 and layer > layers_processed:
                layers_processed = layer
                if layers_processed > effective_iter_cap_layers:
                    # Do not expand this or deeper layers; skip expansion but keep popping remaining PQ
                    continue

            if abs(acc[u]) >= node_budget:
                node_budget_hits_local += 1
                continue
            if u not in csr:
                continue

            stop_relax = False
            for v, e in csr[u]:
                d = dist[u] + 1
                if d > radius_cap:
                    radius_cap_hits_local += 1
                    continue
                if d > effective_iter_cap_layers:
                    layer_hits_local += 1
                    continue

                decay = _compute_decay(d, cfg_t1)
                contrib = w * float(e.weight) * float(edge_mult.get(e.rel, 0.6)) * decay
                if abs(contrib) < EPS:
                    continue

                acc[v] += contrib
                propagations += 1
                local_max_delta = max(local_max_delta, abs(contrib))
                if (v not in dist) or (d < dist[v]):
                    dist[v] = d
                if abs(acc[v]) < node_budget:
                    item2 = (-abs(contrib), v, v, contrib)
                    if ring and ring.contains(v):
                        local_t1_dedup_hits_total += 1
                    else:
                        heapq.heappush(pq, item2)
                        if effective_frontier_cap is not None and len(pq) > effective_frontier_cap:
                            ev = len(pq) - effective_frontier_cap
                            pq = heapq.nsmallest(effective_frontier_cap, pq)
                            heapq.heapify(pq)
                            local_t1_frontier_evicted_total += ev
                        if ring:
                            ring.add(v)
                else:
                    node_budget_hits_local += 1

                if relax_cap is not None and propagations >= relax_cap:
                    stop_relax = True
                    break

            if stop_relax:
                break

        deltas_for_gid: List[Dict[str, Any]] = []
        for nid, val in sorted(acc.items(), key=lambda kv: kv[0]):
            if abs(val) < EPS:
                continue
            deltas_for_gid.append({"op": "upsert_node", "id": nid})
        result_metrics = {
            "pops": pops,
            "iters": min(layers_processed, effective_iter_cap_layers),
            "propagations": propagations,
            "radius_cap_hits": radius_cap_hits_local,
            "layer_cap_hits": layer_hits_local,
            "node_budget_hits": node_budget_hits_local,
        }
        # Cache write (if any), collect eviction stats
        local_cache_evicted = 0
        local_cache_bytes = 0
        if cache is not None:
            result = {"deltas": deltas_for_gid, "metrics": result_metrics}
            if cache_kind == "bytes":
                cost = _estimate_t1_cost(deltas_for_gid, result_metrics)
                ev_n, ev_b = cache.put(ckey, result, cost)
                local_cache_evicted += int(ev_n or 0)
                local_cache_bytes += int(ev_b or 0)
            else:
                cache.put(ckey, result)
        return deltas_for_gid, {
            **result_metrics,
            "_max_delta_local": local_max_delta,
            "_cache_hit": 0,
            "_cache_miss": 1 if cache is not None else 0,
            "_t1_frontier_evicted": local_t1_frontier_evicted_total,
            "_t1_dedup_hits": local_t1_dedup_hits_total,
            "_t1_visited_evicted": local_t1_visited_evicted_total,
            "_t1_cache_evicted": local_cache_evicted,
            "_t1_cache_bytes": local_cache_bytes,
        }

    # Choose path: sequential (unchanged) vs parallel fanout via run_parallel
    if not _t1_parallel_enabled(ctx.cfg):
        for gid in active_graphs:
            deltas_for_gid, m = _t1_one_graph(gid)
            if deltas_for_gid:
                all_deltas.extend(deltas_for_gid)
            total_pops += m["pops"]
            total_iters += m["iters"]
            total_propagations += m["propagations"]
            layer_hits += m["layer_cap_hits"]
            radius_hits += m.get("radius_cap_hits", 0)
            node_hits += m.get("node_budget_hits", 0)
            max_delta = max(max_delta, m["_max_delta_local"])
            cache_hits += m["_cache_hit"]
            cache_misses += m["_cache_miss"]
            if perf_enabled and metrics_enabled:
                t1_frontier_evicted += m["_t1_frontier_evicted"]
                t1_dedup_hits += m["_t1_dedup_hits"]
                t1_visited_evicted += m["_t1_visited_evicted"]
                t1_cache_evicted += m["_t1_cache_evicted"]
                t1_cache_bytes += m["_t1_cache_bytes"]
    else:
        # Build (key, thunk) with a stable order key that preserves the original active_graphs order.
        tasks: List[Tuple[Tuple[int, str], Any]] = []
        for idx, gid in enumerate(active_graphs):
            def make_thunk(_gid=gid):
                def _():
                    return _t1_one_graph(_gid)
                return _
            tasks.append(((idx, str(gid)), make_thunk()))
        def merge_fn(pairs):
            # pairs sorted by (index, gid)
            agg_deltas = []
            agg_pops = agg_iters = agg_props = 0
            agg_radius_hits = 0
            agg_layer_hits = 0
            agg_node_hits = 0
            agg_cache_hits = agg_cache_miss = 0
            agg_max_delta = 0.0
            agg_frontier_ev = agg_dedup = agg_visited_ev = agg_cache_ev = agg_cache_b = 0
            for _, (deltas_for_gid, m) in pairs:
                if deltas_for_gid:
                    agg_deltas.extend(deltas_for_gid)
                agg_pops += m["pops"]
                agg_iters += m["iters"]
                agg_props += m["propagations"]
                agg_radius_hits += m.get("radius_cap_hits", 0)
                agg_layer_hits += m.get("layer_cap_hits", 0)
                agg_node_hits += m.get("node_budget_hits", 0)
                agg_max_delta = max(agg_max_delta, m["_max_delta_local"])
                agg_cache_hits += m["_cache_hit"]
                agg_cache_miss += m["_cache_miss"]
                if perf_enabled and metrics_enabled:
                    agg_frontier_ev += m["_t1_frontier_evicted"]
                    agg_dedup += m["_t1_dedup_hits"]
                    agg_visited_ev += m["_t1_visited_evicted"]
                    agg_cache_ev += m["_t1_cache_evicted"]
                    agg_cache_b += m["_t1_cache_bytes"]
            return (agg_deltas, agg_pops, agg_iters, agg_props, agg_radius_hits, agg_layer_hits, agg_node_hits,
                    agg_max_delta, agg_cache_hits, agg_cache_miss, agg_frontier_ev, agg_dedup, agg_visited_ev,
                    agg_cache_ev, agg_cache_b)
        (all_deltas, total_pops, total_iters, total_propagations, radius_hits, layer_hits, node_hits,
         max_delta, cache_hits, cache_misses, t1_frontier_evicted, t1_dedup_hits, t1_visited_evicted,
         t1_cache_evicted, t1_cache_bytes) = run_parallel(
            tasks,
            max_workers=int(_cfg_get(ctx.cfg, ["perf", "parallel", "max_workers"], 0) or 0),
            merge_fn=merge_fn,
            order_key=lambda k: (k[0], k[1]),
        )

    metrics = {
        "pops": total_pops,
        "iters": total_iters,
        "propagations": total_propagations,
        "radius_cap_hits": radius_hits,
        "layer_cap_hits": layer_hits,
        "node_budget_hits": node_hits,
        "max_delta": max_delta,
        "graphs_touched": len(active_graphs),
        "cache_hits": cache_hits,
        "cache_misses": cache_misses,
        "cache_used": cache_hits > 0,
        "cache_enabled": cache is not None,
    }
    if perf_enabled and metrics_enabled:
        metrics.update(
            {
                "t1_frontier_evicted": t1_frontier_evicted,
                "t1_dedup_hits": t1_dedup_hits,
                "t1_visited_evicted": t1_visited_evicted,
                "t1.cache_evictions": t1_cache_evicted,
                "t1.cache_bytes": t1_cache_bytes,
            }
        )
    return T1Result(graph_deltas=all_deltas, metrics=metrics)
