import numpy as np
import pytest

def _cfg_caps_on():
    return {
        "perf": {
            "native": {"t1": {"enabled": True, "backend": "rust", "strict_parity": False}},
            "t1": {"caps": {"frontier": 100}, "dedupe_window": 8}
        }
    }

def test_native_used_when_supported_caps_on(monkeypatch):
    import clematis.native.t1 as nt1
    import clematis.engine.stages.t1 as st1

    # Pretend extension is available, but ensure we error if native is actually called.
    monkeypatch.setattr(nt1, "available", lambda: True, raising=True)
    # Return a simple deterministic native result to simulate success
    def native_ok(**kwargs):
        return np.asarray([0], dtype=np.int32), np.asarray([1.0], dtype=np.float32), {}
    monkeypatch.setattr(nt1, "propagate_one_graph_rs", native_ok, raising=True)

    indptr  = np.asarray([0, 1], dtype=np.int32)
    indices = np.asarray([0], dtype=np.int32)
    weights = np.asarray([1.0], dtype=np.float32)
    rel_code= np.asarray([0], dtype=np.int32)
    rel_mult= np.asarray([1.0], dtype=np.float32)
    key_rank= np.asarray([0], dtype=np.int32)
    seeds   = np.asarray([0], dtype=np.int32)
    params  = {"decay":{"mode":"exp_floor","rate":0.6,"floor":0.05},"radius_cap":1,"iter_cap_layers":1,"node_budget":1.5}

    # Should NOT call native; should compute via python without raising
    d_nodes, d_vals, metrics = st1._t1_one_graph_dispatch(
        cfg=_cfg_caps_on(), indptr=indptr, indices=indices, weights=weights,
        rel_code=rel_code, rel_mult=rel_mult, key_rank=key_rank, seeds=seeds, params=params
    )
    assert d_nodes.dtype == np.int32
    nt = metrics.get("native_t1", {})
    # Native path should have been used (no gating on supported caps)
    assert nt.get("used_native", 0) >= 1
    # And no gating counters should be incremented
    assert nt.get("fallback_gated_caps", 0) == 0
    assert nt.get("fallback_gated_dedupe", 0) == 0
