import numpy as np
import pytest

def _cfg_native():
    return {"perf": {"native": {"t1": {"enabled": True, "backend": "rust", "strict_parity": True}}}}

@pytest.mark.skipif(__import__("importlib").util.find_spec("clematis.native._t1_rs") is None, reason="native ext not built")
def test_small_parity(monkeypatch):
    import clematis.native.t1 as nt1
    import clematis.engine.stages.t1 as st1

    # Force available -> True (should already be if extension loaded)
    monkeypatch.setattr(nt1, "available", lambda: True, raising=True)

    indptr  = np.asarray([0, 2, 3], dtype=np.int32)
    indices = np.asarray([1, 0, 1], dtype=np.int32)
    weights = np.asarray([1.0, 0.5, 0.2], dtype=np.float32)
    rel_code= np.asarray([0, 0, 0], dtype=np.int32)
    rel_mult= np.asarray([1.0, 0.8, 1.2], dtype=np.float32)  # per-edge
    key_rank= np.asarray([0, 1], dtype=np.int32)
    seeds   = np.asarray([0], dtype=np.int32)
    params  = {"decay": {"mode":"exp_floor","rate":0.6,"floor":0.05},
               "radius_cap":4,"iter_cap_layers":50,"node_budget":1.5}

    d_nodes, d_vals, metrics = st1._t1_one_graph_dispatch(
        cfg=_cfg_native(), indptr=indptr, indices=indices, weights=weights,
        rel_code=rel_code, rel_mult=rel_mult, key_rank=key_rank, seeds=seeds, params=params
    )
    assert d_nodes.dtype == np.int32
    assert d_vals.dtype == np.float32
    assert isinstance(metrics, dict)
