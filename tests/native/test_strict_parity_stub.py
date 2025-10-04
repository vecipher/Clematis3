

import numpy as np


def _cfg_native_strict():
    return {
        "perf": {
            "native": {
                "t1": {"enabled": True, "backend": "python", "strict_parity": True}
            }
        }
    }


def test_strict_parity_vs_python(monkeypatch):
    import clematis.native.t1 as nt1
    import clematis.engine.stages.t1 as st1

    # Force the native branch to be available so the dispatcher takes it.
    monkeypatch.setattr(nt1, "available", lambda: True, raising=True)

    # Tiny but non-trivial CSR graph with 2 nodes (0,1)
    indptr = np.asarray([0, 2, 3], dtype=np.int32)
    indices = np.asarray([1, 0, 1], dtype=np.int32)
    weights = np.asarray([1.0, 0.5, 0.2], dtype=np.float32)
    rel_code = np.asarray([0, 0, 0], dtype=np.int32)
    rel_mult = np.asarray([1.0, 0.8, 1.2], dtype=np.float32)
    key_rank = np.asarray([0, 1], dtype=np.int32)
    seeds = np.asarray([0], dtype=np.int32)

    params = {
        "decay": {"mode": "exp_floor", "rate": 0.6, "floor": 0.05},
        "radius_cap": 4,
        "iter_cap_layers": 50,
        "node_budget": 1.5,
    }

    # Call the dispatcher; in strict parity it will compute both and assert equality.
    d_nodes, d_vals, metrics = st1._t1_one_graph_dispatch(
        cfg=_cfg_native_strict(),
        indptr=indptr,
        indices=indices,
        weights=weights,
        rel_code=rel_code,
        rel_mult=rel_mult,
        key_rank=key_rank,
        seeds=seeds,
        params=params,
    )

    # If mismatch existed, the dispatcher would have raised. Validate outputs.
    assert d_nodes.dtype == np.int32
    assert d_vals.dtype == np.float32
    assert isinstance(metrics, dict)
    # Should be stable across runs
    assert d_nodes.ndim == 1 and d_vals.ndim == 1
