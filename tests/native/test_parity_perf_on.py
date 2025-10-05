import os
from typing import Dict, Any, Tuple

import numpy as np
import pytest
import time

import clematis.engine.stages.t1 as t1


def _make_csr_graph(n_nodes: int = 5000, deg: int = 10, seed: int = 123) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Deterministic synthetic CSR graph with ~ n_nodes*deg edges."""
    rng = np.random.default_rng(seed)
    indptr = np.zeros(n_nodes + 1, dtype=np.int32)
    indices_list = []
    weights_list = []
    for u in range(n_nodes):
        # fixed out-degree for stability; allow self-loops
        nbrs = rng.integers(0, n_nodes, size=deg, endpoint=False)
        indices_list.extend(int(v) for v in nbrs)
        # strictly positive weights
        w = rng.random(deg, dtype=np.float32) * 0.5 + 0.5
        weights_list.extend(float(x) for x in w)
        indptr[u + 1] = indptr[u] + deg
    indices = np.asarray(indices_list, dtype=np.int32)
    weights = np.asarray(weights_list, dtype=np.float32)
    return indptr, indices, weights


def _params_with_caps(queue_cap: int, dedupe_window: int, visited_cap: int) -> Dict[str, Any]:
    # Baseline params aligned with defaults used elsewhere
    params: Dict[str, Any] = {
        "decay": {"mode": "exp_floor", "rate": 0.6, "floor": 0.05},
        "radius_cap": 3,
        "iter_cap_layers": 10,
        "node_budget": 1.5,
    }
    caps: Dict[str, Any] = {}
    if queue_cap > 0:
        caps["frontier"] = int(queue_cap)
    if visited_cap > 0:
        caps["visited"] = int(visited_cap)
    if caps:
        params["caps"] = caps
    if dedupe_window > 0:
        params["dedupe_window"] = int(dedupe_window)
    return params


def _cfg_native_enabled(queue_cap: int, dedupe_window: int, visited_cap: int) -> Dict[str, Any]:
    return {
        "perf": {
            "native": {"t1": {"enabled": True, "backend": "rust"}},
            "t1": {
                "caps": {"frontier": int(queue_cap or 0), "visited": int(visited_cap or 0)},
                "dedupe_window": int(dedupe_window or 0),
            },
        }
    }


def _cfg_native_disabled(queue_cap: int, dedupe_window: int, visited_cap: int) -> Dict[str, Any]:
    c = _cfg_native_enabled(queue_cap, dedupe_window, visited_cap)
    c["perf"]["native"]["t1"]["enabled"] = False
    return c


@pytest.mark.parametrize(
    "queue_cap, dedupe_window, visited_cap",
    [
        (0, 0, 0),
        (128, 0, 0),
        (0, 64, 0),
        (0, 0, 256),
        (128, 64, 256),
    ],
)
@pytest.mark.filterwarnings("ignore::UserWarning")
@pytest.mark.filterwarnings("ignore::RuntimeWarning")
@pytest.mark.timeout(30)
def test_native_equals_python_under_caps(queue_cap, dedupe_window, visited_cap, monkeypatch):
    # Ensure strict parity is off for this test, we compare results explicitly
    monkeypatch.delenv("CLEMATIS_STRICT_PARITY", raising=False)
    monkeypatch.setenv("CLEMATIS_NATIVE_T1", "1")  # allow native fast-path if cfg enables it

    # Build deterministic graph (~30k edges)
    n = 1000
    deg = 8
    indptr, indices, weights = _make_csr_graph(n_nodes=n, deg=deg, seed=123)

    # Relation: per-edge codes with 3-type multiplier table (supports/associates/contradicts)
    # Large lookup table to tolerate any internal code remapping
    rel_mult = np.ones(131072, dtype=np.float32)
    rel_mult[:3] = np.asarray([1.0, 0.6, 0.8], dtype=np.float32)
    rel_code = np.zeros(indices.shape[0], dtype=np.int32)  # all edges type 0

    # Seeds and ranks
    seeds = np.asarray([0, 17, 512], dtype=np.int32)
    seed_weights = np.ones_like(seeds, dtype=np.float32)
    key_rank = np.arange(n, dtype=np.int32)

    # Shared params (with caps for python inner)
    params = _params_with_caps(queue_cap, dedupe_window, visited_cap)

    # 1) Python reference
    t0 = time.perf_counter(); print("[parity_caps] python inner start…", flush=True)
    p_nodes, p_vals, p_met = t1._t1_one_graph_python_inner(
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
    print(f"[parity_caps] python inner done in {time.perf_counter()-t0:.3f}s", flush=True)

    # 2) Native call via dispatcher (cfg enables native & forwards caps)
    t1_ = time.perf_counter(); print("[parity_caps] dispatcher start…", flush=True)
    cfg_native = _cfg_native_enabled(queue_cap, dedupe_window, visited_cap)
    n_nodes, n_vals, n_met = t1._t1_one_graph_dispatch(
        cfg=cfg_native,
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
    print(f"[parity_caps] dispatcher done in {time.perf_counter()-t1_:.3f}s", flush=True)

    # Canonicalize (sort by node id)
    def canon(nodes: np.ndarray, vals: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        idx = np.argsort(nodes)  # nodes are int32
        return nodes[idx], vals[idx]

    p_nodes_c, p_vals_c = canon(p_nodes, p_vals)
    n_nodes_c, n_vals_c = canon(n_nodes, n_vals)

    # Exact node id equality and tight float parity
    np.testing.assert_array_equal(p_nodes_c, n_nodes_c)
    np.testing.assert_allclose(p_vals_c, n_vals_c, rtol=1e-6, atol=1e-7)

    # Native should be used when available; else verify import fallback was recorded
    try:
        import clematis.native.t1 as nt1  # type: ignore
        have_native = bool(getattr(nt1, "available", lambda: False)())
    except Exception:
        have_native = False

    nt = (n_met.get("native_t1", {}) or {})
    if have_native:
        assert nt.get("used_native", 0) >= 1
    else:
        assert nt.get("fallback_import_failed", 0) >= 1


@pytest.mark.timeout(30)
@pytest.mark.parametrize("queue_cap, dedupe_window, visited_cap", [(256, 0, 0)])
def test_dispatch_disabled_matches_python(queue_cap, dedupe_window, visited_cap, monkeypatch):
    monkeypatch.delenv("CLEMATIS_NATIVE_T1", raising=False)
    monkeypatch.delenv("CLEMATIS_STRICT_PARITY", raising=False)

    n = 600
    deg = 8
    indptr, indices, weights = _make_csr_graph(n_nodes=n, deg=deg, seed=321)

    # Use a generously sized lookup to avoid OOB if kernel maps codes > 2
    rel_mult = np.ones(131072, dtype=np.float32)
    rel_mult[:3] = np.asarray([1.0, 0.6, 0.8], dtype=np.float32)
    rel_code = np.zeros(indices.shape[0], dtype=np.int32)

    seeds = np.asarray([1, 8, 99], dtype=np.int32)
    seed_weights = np.ones_like(seeds, dtype=np.float32)
    key_rank = np.arange(n, dtype=np.int32)

    params = _params_with_caps(queue_cap, dedupe_window, visited_cap)

    # Python baseline
    t0 = time.perf_counter(); print("[disabled_native] python inner start…", flush=True)
    p_nodes, p_vals, p_met = t1._t1_one_graph_python_inner(
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
    print(f"[disabled_native] python inner done in {time.perf_counter()-t0:.3f}s", flush=True)

    # Dispatcher with native disabled in cfg
    t1_ = time.perf_counter(); print("[disabled_native] dispatcher start…", flush=True)
    cfg_py = _cfg_native_disabled(queue_cap, dedupe_window, visited_cap)
    d_nodes, d_vals, d_met = t1._t1_one_graph_dispatch(
        cfg=cfg_py,
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
    print(f"[disabled_native] dispatcher done in {time.perf_counter()-t1_:.3f}s", flush=True)

    # Must match exactly
    np.testing.assert_array_equal(np.sort(p_nodes), np.sort(d_nodes))
    np.testing.assert_allclose(np.sort(p_vals), np.sort(d_vals), rtol=1e-6, atol=1e-7)

    # And native should not be marked as used
    assert (d_met.get("native_t1", {}) or {}).get("used_native", 0) == 0
