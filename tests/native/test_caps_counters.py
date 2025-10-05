import numpy as np
import pytest

import clematis.engine.stages.t1 as t1


def _params(frontier=0, visited=0, dedupe_window=0):
    params = {
        "decay": {"mode": "exp_floor", "rate": 0.6, "floor": 0.05},
        "radius_cap": 3,
        "iter_cap_layers": 10,
        "node_budget": 1.5,
    }
    caps = {}
    if frontier:
        caps["frontier"] = int(frontier)
    if visited:
        caps["visited"] = int(visited)
    if caps:
        params["caps"] = caps
    if dedupe_window:
        params["dedupe_window"] = int(dedupe_window)
    return params


def _cfg(frontier=0, visited=0, dedupe_window=0, enable_native=True):
    return {
        "perf": {
            "native": {"t1": {"enabled": bool(enable_native), "backend": "rust"}},
            "t1": {
                "caps": {
                    "frontier": int(frontier or 0),
                    "visited": int(visited or 0),
                },
                "dedupe_window": int(dedupe_window or 0),
            },
        }
    }


def _rel(n_edges):
    rel_code = np.zeros(n_edges, dtype=np.int32)
    rel_mult = np.ones(8, dtype=np.float32)
    rel_mult[:3] = np.asarray([1.0, 0.6, 0.8], dtype=np.float32)
    return rel_code, rel_mult


@pytest.mark.skipif("not hasattr(__import__('clematis.native.t1', fromlist=['t1']), 'available') or not __import__('clematis.native.t1', fromlist=['t1']).available()")
@pytest.mark.timeout(15)
def test_queue_cap_dropped_increments():
    # Graph: node 0 -> {1,2} ensures second enqueue hits frontier cap when cap==1
    indptr = np.asarray([0, 2, 2, 2], dtype=np.int32)  # 3 nodes, 2 edges out of 0
    indices = np.asarray([1, 2], dtype=np.int32)
    weights = np.asarray([1.0, 1.0], dtype=np.float32)
    rel_code, rel_mult = _rel(indices.size)

    seeds = np.asarray([0], dtype=np.int32)
    seed_weights = np.asarray([1.0], dtype=np.float32)
    key_rank = np.arange(3, dtype=np.int32)

    cfg = _cfg(frontier=1, visited=0, dedupe_window=0, enable_native=True)
    params = _params(frontier=1, visited=0, dedupe_window=0)

    n_nodes, n_vals, met = t1._t1_one_graph_dispatch(
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
    nt = dict(met)
    nt.update(met.get("native_t1", {}))
    assert nt.get("used_native", 0) >= 1
    assert nt.get("queue_cap_dropped", 0) >= 1


@pytest.mark.skipif("not hasattr(__import__('clematis.native.t1', fromlist=['t1']), 'available') or not __import__('clematis.native.t1', fromlist=['t1']).available()")
@pytest.mark.timeout(15)
def test_dedupe_skips_increments():
    # Graph: node 0 -> {1,1} duplicate edges, dedupe_window=1 should skip the second
    indptr = np.asarray([0, 2, 2, 2], dtype=np.int32)
    indices = np.asarray([1, 1], dtype=np.int32)
    weights = np.asarray([1.0, 1.0], dtype=np.float32)
    rel_code, rel_mult = _rel(indices.size)

    seeds = np.asarray([0], dtype=np.int32)
    seed_weights = np.asarray([1.0], dtype=np.float32)
    key_rank = np.arange(3, dtype=np.int32)

    cfg = _cfg(frontier=0, visited=0, dedupe_window=1, enable_native=True)
    params = _params(frontier=0, visited=0, dedupe_window=1)

    n_nodes, n_vals, met = t1._t1_one_graph_dispatch(
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
    nt = dict(met)
    nt.update(met.get("native_t1", {}))
    assert nt.get("used_native", 0) >= 1
    assert nt.get("dedupe_skips", 0) >= 1


@pytest.mark.skipif("not hasattr(__import__('clematis.native.t1', fromlist=['t1']), 'available') or not __import__('clematis.native.t1', fromlist=['t1']).available()")
@pytest.mark.timeout(15)
def test_visited_cap_hits_increments():
    # Graph: small chain 0->1, 1->2; visited cap=1 should stop after first expansion
    indptr = np.asarray([0, 1, 2, 2], dtype=np.int32)
    indices = np.asarray([1, 2], dtype=np.int32)
    weights = np.asarray([1.0, 1.0], dtype=np.float32)
    rel_code, rel_mult = _rel(indices.size)

    seeds = np.asarray([0], dtype=np.int32)
    seed_weights = np.asarray([1.0], dtype=np.float32)
    key_rank = np.arange(3, dtype=np.int32)

    cfg = _cfg(frontier=0, visited=1, dedupe_window=0, enable_native=True)
    params = _params(frontier=0, visited=1, dedupe_window=0)

    n_nodes, n_vals, met = t1._t1_one_graph_dispatch(
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
    nt = dict(met)
    nt.update(met.get("native_t1", {}))
    assert nt.get("used_native", 0) >= 1
    assert nt.get("visited_cap_hits", 0) >= 1
