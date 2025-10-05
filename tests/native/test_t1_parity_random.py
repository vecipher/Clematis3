import numpy as np
import pytest

hypothesis = pytest.importorskip("hypothesis")
from hypothesis import given, strategies as st, settings
from unittest.mock import patch

def _cfg_native():
    return {"perf": {"native": {"t1": {"enabled": True, "backend": "rust", "strict_parity": True}}}}

def _tiny_graph(max_nodes=5, max_edges=8):
    # Build a small, connected-ish random CSR with non-negative weights
    n = np.random.randint(1, max_nodes+1)
    nodes = list(range(n))
    edges = []
    for u in range(n):
        deg = np.random.randint(0, max(1, max_edges // max(1,n)))
        for _ in range(deg):
            v = np.random.randint(0, n)
            w = np.random.rand() * 1.5
            edges.append((u, v, w))
    edges.sort()
    indptr = [0]
    indices = []
    weights = []
    rel_code= []
    rel_mult= []
    i = 0
    for u in range(n):
        while i < len(edges) and edges[i][0] == u:
            _, v, w = edges[i]
            indices.append(int(v))
            weights.append(float(w))
            rel_code.append(0)
            rel_mult.append(1.0)
            i += 1
        indptr.append(len(indices))
    return (np.asarray(indptr, np.int32),
            np.asarray(indices, np.int32),
            np.asarray(weights, np.float32),
            np.asarray(rel_code, np.int32),
            np.asarray(rel_mult, np.float32))

@pytest.mark.skipif(__import__("importlib").util.find_spec("clematis.native._t1_rs") is None, reason="native ext not built")
@settings(max_examples=10, deadline=None)
@given(seed=st.integers(min_value=1, max_value=5))
def test_random_parity(seed):
    import clematis.native.t1 as nt1
    import clematis.engine.stages.t1 as st1

    np.random.seed(seed)
    # Ensure native path is considered available for the duration of this example
    with patch("clematis.native.t1.available", return_value=True):
        indptr, indices, weights, rel_code, rel_mult = _tiny_graph()
        n_nodes = len(indptr)-1
        key_rank = np.arange(n_nodes, dtype=np.int32)
        seeds = np.asarray([0], np.int32)
        params = {"decay":{"mode":"exp_floor","rate":0.6,"floor":0.05},
                  "radius_cap":4,"iter_cap_layers":50,"node_budget":1.5}

        d_nodes, d_vals, metrics = st1._t1_one_graph_dispatch(
            cfg=_cfg_native(), indptr=indptr, indices=indices, weights=weights,
            rel_code=rel_code, rel_mult=rel_mult, key_rank=key_rank, seeds=seeds, params=params
        )
        assert d_nodes.dtype == np.int32
        assert d_vals.dtype == np.float32
        assert isinstance(metrics, dict)
