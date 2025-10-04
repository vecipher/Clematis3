

import numpy as np


def test_contract_shapes(monkeypatch):
    # Import the stub entrypoint and the python inner we will fake.
    import clematis.native.t1 as nt1
    import clematis.engine.stages.t1 as st1

    def fake_inner(**kwargs):
        # Deterministic tiny output with canonical dtypes
        d_nodes = np.asarray([2, 5], dtype=np.int32)
        d_vals = np.asarray([0.25, -0.5], dtype=np.float32)
        metrics = {"iters": 3}
        # Basic sanity: inner received the expected keys
        assert "indptr" in kwargs and "indices" in kwargs and "weights" in kwargs
        assert "rel_mult" in kwargs and "seeds" in kwargs and "params" in kwargs
        assert "key_rank" in kwargs
        # Optional keys may be present, ensure they are accepted
        _ = kwargs.get("seed_weights", None)
        _ = kwargs.get("rel_code", None)
        return d_nodes, d_vals, metrics

    # Swap in our fake inner to decouple this test from the real implementation
    monkeypatch.setattr(st1, "_t1_one_graph_python_inner", fake_inner, raising=True)

    # Minimal but well-formed CSR + metadata
    indptr = np.asarray([0, 1, 2], dtype=np.int32)       # 2 rows
    indices = np.asarray([1, 0], dtype=np.int32)
    weights = np.asarray([1.0, 0.5], dtype=np.float32)
    rel_code = np.asarray([0, 0], dtype=np.int32)
    rel_mult = np.asarray([1.0, 1.0], dtype=np.float32)
    seed_nodes = np.asarray([0], dtype=np.int32)
    seed_weights = np.asarray([1.0], dtype=np.float32)
    key_rank = np.asarray([0, 1], dtype=np.int32)

    # Kernel params (FFI-shape positional args)
    rate = 0.6
    floor = 0.05
    radius_cap = 4
    iter_cap_layers = 50
    node_budget = 1.5

    d_nodes, d_vals, metrics = nt1.propagate_one_graph_rs(
        indptr,
        indices,
        weights,
        rel_code,
        rel_mult,
        seed_nodes,
        seed_weights,
        key_rank,
        rate,
        floor,
        radius_cap,
        iter_cap_layers,
        node_budget,
    )

    # Contract assertions
    assert isinstance(metrics, dict) and metrics.get("iters") == 3
    assert d_nodes.dtype == np.int32
    assert d_vals.dtype == np.float32
    assert d_nodes.shape == (2,)
    assert d_vals.shape == (2,)
