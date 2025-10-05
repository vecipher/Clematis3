from __future__ import annotations
from collections import defaultdict
import heapq
from typing import Dict, Any, List, Tuple
import numpy as np
from ..types import T1Result
from ..cache import LRUCache, stable_key, ThreadSafeCache, ThreadSafeBytesCache
from ..util.ring import DedupeRing
from ..util.lru_det import DeterministicLRUSet
from ..util.lru_bytes import LRUBytes

from ..util.parallel import run_parallel
from ..util.metrics import metrics_gate_on, metrics_update_gated

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


EPS = 1e-8


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


def _canon_nodes_vals(d_nodes: np.ndarray, d_vals: np.ndarray):
    """Sort by node id and enforce ABI dtypes for stable comparisons."""
    o = np.argsort(d_nodes, kind="stable")
    return d_nodes[o].astype(np.int32, copy=False), d_vals[o].astype(np.float32, copy=False)

# === PR98: Native parity harness scaffolding (no behavior change by default) ===
from typing import Sequence, Optional


def _t1_one_graph_python_inner(
    *,
    indptr: np.ndarray,
    indices: np.ndarray,
    weights: np.ndarray,
    rel_mult: np.ndarray,
    seeds: np.ndarray,
    params: Dict[str, Any],
    key_rank: np.ndarray,
    seed_weights: Optional[np.ndarray] = None,
    rel_code: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Factored single-graph propagation kernel (Python reference).

    This is the semantics-preserving inner loop used for strict parity.
    Inputs are CSR arrays with per-edge relation multipliers, a stable key_rank
    (lexicographic rank of node ids), and seed node ids (indices in key_rank space).

    Returns (d_nodes[int32], d_vals[float32], metrics[dict]).

    Notes:
    • Only "perf-OFF" semantics are implemented here (no dedupe ring, no visited LRU,
      no explicit frontier caps). That matches the PR98 goal.
    • "rel_code" is accepted but unused by the Python path; retained for forward-compat.
    • If seed_weights is omitted, unit weights are assumed for seeds.
    """
    # Canonical dtypes
    indptr = indptr.astype(np.int32, copy=False)
    indices = indices.astype(np.int32, copy=False)
    weights = weights.astype(np.float32, copy=False)
    rel_mult = rel_mult.astype(np.float32, copy=False)
    seeds = seeds.astype(np.int32, copy=False)
    key_rank = key_rank.astype(np.int32, copy=False)
    if seed_weights is None:
        seed_weights = np.ones_like(seeds, dtype=np.float32)
    else:
        seed_weights = seed_weights.astype(np.float32, copy=False)

    # Params
    decay_p = params.get("decay", {}) if isinstance(params, dict) else {}
    rate = np.float32(decay_p.get("rate", 0.6))
    floor = np.float32(decay_p.get("floor", 0.05))
    radius_cap = int(params.get("radius_cap", 4))
    iter_cap_layers = int(params.get("iter_cap_layers", 50))
    node_budget = np.float32(params.get("node_budget", 1.5))

    n_nodes = int(key_rank.max()) + 1 if key_rank.size > 0 else 0
    # If key_rank is empty, infer from CSR
    if n_nodes == 0 and indptr.size > 0:
        n_nodes = indptr.size - 1

    # Accumulator and distances
    acc = np.zeros(n_nodes, dtype=np.float32)
    dist = np.full(n_nodes, -1, dtype=np.int32)

    # Priority queue uses a min-heap with composite key mirroring Rust:
    # key = (-priority_f32, key_rank, node)
    import heapq as _heapq
    def _pq_key(priority_f32, kr, node):
        return (-float(np.float32(priority_f32)), int(kr), int(node))
    pq: List[Tuple[Tuple[float, int, int], int, float]] = []

    local_max_delta = 0.0
    for s, sw in zip(seeds.tolist(), seed_weights.tolist()):
        sw_f32 = float(np.float32(sw))
        acc[s] += sw_f32
        dist[s] = 0
        _heapq.heappush(pq, (_pq_key(sw_f32, key_rank[s], s), int(s), sw_f32))
        local_max_delta = max(local_max_delta, abs(sw_f32))

    pops = 0
    propagations = 0
    layers_processed = 0
    radius_cap_hits_local = 0
    layer_cap_hits_local = 0
    node_budget_hits_local = 0

    def _decay(d: int) -> float:
        return max(rate ** d, floor)

    while pq:
        _, u, w = _heapq.heappop(pq)
        pops += 1
        layer = int(dist[u]) if dist[u] >= 0 else 0
        if layer > 0 and layer > layers_processed:
            layers_processed = layer
            if layers_processed > iter_cap_layers:
                # Do not expand beyond iter cap; continue draining queue
                continue

        if abs(acc[u]) >= node_budget:
            node_budget_hits_local += 1
            continue

        # CSR row for u
        row_start = int(indptr[u])
        row_end = int(indptr[u + 1])
        if row_end <= row_start:
            continue

        for e_idx in range(row_start, row_end):
            v = int(indices[e_idx])
            d = layer + 1
            if d > radius_cap:
                radius_cap_hits_local += 1
                continue
            if d > iter_cap_layers:
                layer_cap_hits_local += 1
                continue

            contrib_f32 = float(
                np.float32(w) * np.float32(weights[e_idx]) * np.float32(rel_mult[e_idx]) * np.float32(_decay(d))
            )
            if abs(contrib_f32) < EPS:
                continue

            acc[v] += contrib_f32
            propagations += 1
            local_max_delta = max(local_max_delta, abs(contrib_f32))
            if dist[v] < 0 or d < dist[v]:
                dist[v] = d
            if abs(acc[v]) < float(node_budget):
                _heapq.heappush(pq, (_pq_key(contrib_f32, key_rank[v], v), v, contrib_f32))
            else:
                node_budget_hits_local += 1

    # Build deterministic outputs: sort by node id
    nz = np.nonzero(np.abs(acc) >= EPS)[0]
    if nz.size:
        nz_sorted = np.sort(nz.astype(np.int32, copy=False))
        d_nodes = nz_sorted.astype(np.int32, copy=False)
        d_vals = acc[nz_sorted].astype(np.float32, copy=False)
    else:
        d_nodes = np.asarray([], dtype=np.int32)
        d_vals = np.asarray([], dtype=np.float32)

    metrics = {
        "pops": int(pops),
        "iters": int(min(layers_processed, iter_cap_layers)),
        "propagations": int(propagations),
        "radius_cap_hits": int(radius_cap_hits_local),
        "layer_cap_hits": int(layer_cap_hits_local),
        "node_budget_hits": int(node_budget_hits_local),
        "_max_delta_local": float(local_max_delta),
    }
    return d_nodes, d_vals, metrics


def _t1_one_graph_native(
    *,
    indptr: np.ndarray,
    indices: np.ndarray,
    weights: np.ndarray,
    rel_code: np.ndarray,
    rel_mult: np.ndarray,
    key_rank: np.ndarray,
    seeds: np.ndarray,
    params: Dict[str, Any],
    seed_weights: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Calls the native kernel (Python stub or Rust ext) with Rust-compatible signature."""
    from clematis.native import t1 as native_t1  # lazy import to avoid cycles
    if seed_weights is None:
        seed_weights = np.ones_like(seeds, dtype=np.float32)

    # Expand param dict into positional call expected by the stub
    decay = params.get("decay", {}) if isinstance(params, dict) else {}
    rate = float(decay.get("rate", 0.6))
    floor = float(decay.get("floor", 0.05))
    radius_cap = int(params.get("radius_cap", 4))
    iter_cap_layers = int(params.get("iter_cap_layers", 50))
    node_budget = float(params.get("node_budget", 1.5))

    return native_t1.propagate_one_graph_rs(
        indptr=indptr,
        indices=indices,
        weights=weights,
        rel_code=rel_code,
        rel_mult=rel_mult,
        seed_nodes=seeds,
        seed_weights=seed_weights,
        key_rank=key_rank,
        rate=rate,
        floor=floor,
        radius_cap=radius_cap,
        iter_cap_layers=iter_cap_layers,
        node_budget=node_budget,
    )


def _native_t1_allowed(cfg) -> bool:
    """PR99 gate: allow native only if backend is valid **and** perf caps/dedupe are OFF.

    We disallow the native path when perf.t1.caps.* or perf.t1.dedupe_window are enabled,
    because PR99 implements only perf-OFF semantics in the kernel.
    """
    try:
        block = (cfg.get("perf") or {}).get("native", {}).get("t1", {})
        backend_ok = block.get("backend", "rust") in {"rust", "python"}
        perf_t1 = (cfg.get("perf") or {}).get("t1", {})
        caps = perf_t1.get("caps", {}) or {}
        dedupe_window = int(perf_t1.get("dedupe_window", 0) or 0)
        caps_on = any(bool(v) for v in caps.values())
        return bool(backend_ok and not caps_on and dedupe_window == 0)
    except Exception:
        return False


def _t1_one_graph_dispatch(
    *,
    cfg,
    indptr: np.ndarray,
    indices: np.ndarray,
    weights: np.ndarray,
    rel_code: np.ndarray,
    rel_mult: np.ndarray,
    key_rank: np.ndarray,
    seeds: np.ndarray,
    params: Dict[str, Any],
    seed_weights: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Test-visible dispatcher for strict parity.

    If cfg.perf.native.t1.enabled and native available + allowed:
      - strict_parity=True → compute both paths, assert identical, return python.
      - else → return native result.
    Otherwise returns python result.
    """
    block = (((cfg or {}).get("perf") or {}).get("native") or {}).get("t1") or {}
    strict = bool(block.get("strict_parity", False))
    enabled = bool(block.get("enabled", False))

    if enabled and _native_t1_allowed(cfg):
        from clematis.native import t1 as native_t1  # lazy
        if native_t1.available():
            n_nodes, n_vals, n_met = _t1_one_graph_native(
                indptr=indptr,
                indices=indices,
                weights=weights,
                rel_code=rel_code,
                rel_mult=rel_mult,
                key_rank=key_rank,
                seeds=seeds,
                params=params,
                seed_weights=seed_weights,
            )
            if strict:
                p_nodes, p_vals, p_met = _t1_one_graph_python_inner(
                    indptr=indptr,
                    indices=indices,
                    weights=weights,
                    rel_mult=rel_mult,
                    seeds=seeds,
                    params=params,
                    key_rank=key_rank,
                    seed_weights=seed_weights,
                    rel_code=rel_code,
                )
                n_nodes_c, n_vals_c = _canon_nodes_vals(n_nodes, n_vals)
                p_nodes_c, p_vals_c = _canon_nodes_vals(p_nodes, p_vals)
                if not (np.array_equal(n_nodes_c, p_nodes_c) and np.array_equal(n_vals_c, p_vals_c) and n_met == p_met):
                    raise AssertionError("strict_parity: native != python")
                return p_nodes, p_vals, p_met
            return n_nodes, n_vals, n_met

    # Fallback: python path
    return _t1_one_graph_python_inner(
        indptr=indptr,
        indices=indices,
        weights=weights,
        rel_mult=rel_mult,
        seeds=seeds,
        params=params,
        key_rank=key_rank,
        seed_weights=seed_weights,
        rel_code=rel_code,
    )
# === End PR98 scaffolding ===


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
    # PR67: compute task/worker counts (used only when metrics gate + parallel gate are ON)
    task_count = len(active_graphs)
    requested_workers = int(_cfg_get(ctx.cfg, ["perf", "parallel", "max_workers"], 0) or 0)
    effective_workers = min(requested_workers, max(1, task_count))
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
                return (
                    hit["deltas"],
                    {
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
                    },
                )

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
            return (
                agg_deltas,
                agg_pops,
                agg_iters,
                agg_props,
                agg_radius_hits,
                agg_layer_hits,
                agg_node_hits,
                agg_max_delta,
                agg_cache_hits,
                agg_cache_miss,
                agg_frontier_ev,
                agg_dedup,
                agg_visited_ev,
                agg_cache_ev,
                agg_cache_b,
            )

        (
            all_deltas,
            total_pops,
            total_iters,
            total_propagations,
            radius_hits,
            layer_hits,
            node_hits,
            max_delta,
            cache_hits,
            cache_misses,
            t1_frontier_evicted,
            t1_dedup_hits,
            t1_visited_evicted,
            t1_cache_evicted,
            t1_cache_bytes,
        ) = run_parallel(
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
    # PR67: emit minimal parallel metrics at the very end so they persist in all paths
    if metrics_gate_on(ctx.cfg):
        metrics_update_gated(
            metrics,
            {"parallel_workers": int(effective_workers), "task_count": int(task_count)},
            ctx.cfg,
        )

    return T1Result(graph_deltas=all_deltas, metrics=metrics)


# --- test hook: used by tests/perf/test_bench_t1_smoke.py (Option B) ---
def _run_kernel_for_bench(graph_dict):
    """
    Accepts {"num_nodes": int, "edges": [(u,v,w,rel), ...]}.
    Uses CLEMATIS_NATIVE_T1=1 to choose backend via cfg.perf.native.t1.enabled.
    Runs exactly one single-graph T1 propagation through the same dispatcher
    used in production so gates/semantics are preserved. Return value is
    (d_nodes[int32], d_vals[float32], metrics[dict]); the perf test ignores it.
    """
    import os
    import numpy as _np

    num_nodes = int((graph_dict or {}).get("num_nodes", 0) or 0)
    edges = list((graph_dict or {}).get("edges") or [])

    # Build CSR from edge list
    deg = [0] * num_nodes
    for u, v, *_ in edges:
        u_i = int(u)
        v_i = int(v)
        if 0 <= u_i < num_nodes and 0 <= v_i < num_nodes:
            deg[u_i] += 1

    indptr = _np.zeros(num_nodes + 1, dtype=_np.int32)
    for i in range(num_nodes):
        indptr[i + 1] = indptr[i] + deg[i]
    m = int(indptr[-1])

    indices = _np.zeros(m, dtype=_np.int32)
    weights = _np.zeros(m, dtype=_np.float32)
    rel_mult = _np.zeros(m, dtype=_np.float32)
    rel_code = _np.zeros(m, dtype=_np.int32)

    mult_map = {"supports": 1.0, "associates": 0.6, "contradicts": 0.8}
    code_map = {"supports": 0, "associates": 1, "contradicts": 2}

    cursor = indptr[:-1].copy()
    for (u, v, w, rel) in edges:
        u_i = int(u)
        v_i = int(v)
        # assume edges were counted; skip out-of-range defensively
        if not (0 <= u_i < num_nodes and 0 <= v_i < num_nodes):
            continue
        k = int(cursor[u_i])
        cursor[u_i] = k + 1
        indices[k] = v_i
        weights[k] = float(w)
        rel_mult[k] = float(mult_map.get(rel, 0.6))
        rel_code[k] = int(code_map.get(rel, 1))

    key_rank = _np.arange(num_nodes, dtype=_np.int32)

    # Deterministic small seed set
    seeds = _np.arange(min(64, num_nodes), dtype=_np.int32)
    seed_weights = _np.ones_like(seeds, dtype=_np.float32)

    params = {
        "decay": {"rate": 0.6, "floor": 0.05},
        "radius_cap": 4,
        "iter_cap_layers": 50,
        "node_budget": 1.5,
    }

    # Minimal cfg: enable/disable native via env; keep perf caps OFF so native is allowed
    cfg = {
        "perf": {
            "native": {"t1": {"enabled": os.getenv("CLEMATIS_NATIVE_T1", "0") == "1", "strict_parity": False, "backend": "rust"}},
            "t1": {"caps": {}, "dedupe_window": 0},
            "parallel": {"max_workers": 0},
        }
    }

    return _t1_one_graph_dispatch(
        cfg=cfg,
        indptr=indptr,
        indices=indices,
        weights=weights,
        rel_code=rel_code,
        rel_mult=rel_mult,
        key_rank=key_rank,
        seeds=seeds,
        params=params,
        seed_weights=seed_weights,
    )
