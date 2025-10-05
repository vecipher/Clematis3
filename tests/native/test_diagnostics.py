import os
import logging
import importlib
from typing import Tuple

import numpy as np
import pytest

import clematis.engine.stages.t1 as t1


# --- helpers -----------------------------------------------------------------

def _mini_inputs(n: int = 8, edges=None):
    if edges is None:
        edges = [(i, (i + 1) % n, 1.0, "associates") for i in range(n)]

    # CSR build
    deg = [0] * n
    for u, v, *_ in edges:
        deg[int(u)] += 1

    indptr = np.zeros(n + 1, dtype=np.int32)
    for i in range(n):
        indptr[i + 1] = indptr[i] + deg[i]
    m = int(indptr[-1])

    indices = np.zeros(m, dtype=np.int32)
    weights = np.zeros(m, dtype=np.float32)
    rel_mult = np.zeros(m, dtype=np.float32)
    rel_code = np.zeros(m, dtype=np.int32)

    mult_map = {"supports": 1.0, "associates": 0.6, "contradicts": 0.8}
    code_map = {"supports": 0, "associates": 1, "contradicts": 2}

    cursor = indptr[:-1].copy()
    for (u, v, w, rel) in edges:
        u_i = int(u)
        k = int(cursor[u_i])
        cursor[u_i] = k + 1
        indices[k] = int(v)
        weights[k] = float(w)
        rel_mult[k] = float(mult_map.get(rel, 0.6))
        rel_code[k] = int(code_map.get(rel, 1))

    key_rank = np.arange(n, dtype=np.int32)
    seeds = np.arange(min(4, n), dtype=np.int32)
    seed_weights = np.ones_like(seeds, dtype=np.float32)

    params = {
        "decay": {"rate": 0.6, "floor": 0.05},
        "radius_cap": 3,
        "iter_cap_layers": 10,
        "node_budget": 1.0,
    }

    cfg = {
        "perf": {
            "native": {"t1": {"enabled": True, "strict_parity": False}},
            "t1": {"caps": {}, "dedupe_window": 0},
        }
    }

    return cfg, indptr, indices, weights, rel_code, rel_mult, key_rank, seeds, params, seed_weights


def _call_dispatch(cfg, indptr, indices, weights, rel_code, rel_mult, key_rank, seeds, params, seed_weights):
    return t1._t1_one_graph_dispatch(
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


# --- tests -------------------------------------------------------------------

def test_import_failure_falls_back_and_logs_once(monkeypatch, caplog):
    caplog.set_level(logging.INFO)
    # reset once-only guard per test
    monkeypatch.setattr(t1, "_LOG_ONCE_KEYS", set(), raising=False)

    # Force native import path to report unavailable
    import clematis.native.t1 as nt1
    monkeypatch.setattr(nt1, "available", lambda: False, raising=True)

    # Stub python inner to keep this hermetic and fast
    def py_inner(**kwargs):
        return np.array([0], dtype=np.int32), np.array([0.1], dtype=np.float32), {}

    monkeypatch.setattr(t1, "_t1_one_graph_python_inner", py_inner, raising=True)

    cfg, indptr, indices, weights, rel_code, rel_mult, key_rank, seeds, params, seed_weights = _mini_inputs()
    nodes, vals, metrics = _call_dispatch(cfg, indptr, indices, weights, rel_code, rel_mult, key_rank, seeds, params, seed_weights)

    nt = metrics.get("native_t1", {})
    assert nt.get("fallback_import_failed", 0) == 1
    assert any("import/available() failed" in rec.getMessage() for rec in caplog.records)


def test_gated_caps_and_dedupe_increments_counters_no_native(monkeypatch):
    # reset once-only guard per test
    monkeypatch.setattr(t1, "_LOG_ONCE_KEYS", set(), raising=False)

    # If native gets called here, fail the test immediately
    def native_should_not_be_called(**kwargs):
        raise AssertionError("native path should not be called when gated")

    monkeypatch.setattr(t1, "_t1_one_graph_native", native_should_not_be_called, raising=True)

    def py_inner(**kwargs):
        return np.array([1], dtype=np.int32), np.array([0.2], dtype=np.float32), {}

    monkeypatch.setattr(t1, "_t1_one_graph_python_inner", py_inner, raising=True)

    cfg, indptr, indices, weights, rel_code, rel_mult, key_rank, seeds, params, seed_weights = _mini_inputs()
    cfg["perf"]["t1"]["caps"] = {"budget": True}
    cfg["perf"]["t1"]["dedupe_window"] = 2

    nodes, vals, metrics = _call_dispatch(cfg, indptr, indices, weights, rel_code, rel_mult, key_rank, seeds, params, seed_weights)

    nt = metrics.get("native_t1", {})
    assert nt.get("fallback_gated_caps", 0) == 1
    assert nt.get("fallback_gated_dedupe", 0) == 1


def test_runtime_exception_fallback_increments_counter_and_logs(monkeypatch, caplog):
    caplog.set_level(logging.ERROR)
    monkeypatch.setattr(t1, "_LOG_ONCE_KEYS", set(), raising=False)

    # Make native "available"
    import clematis.native.t1 as nt1
    monkeypatch.setattr(nt1, "available", lambda: True, raising=True)

    # Force the native inner to raise MemoryError
    def native_raises(**kwargs):
        raise MemoryError("OOM")

    monkeypatch.setattr(t1, "_t1_one_graph_native", native_raises, raising=True)

    # Python inner returns a trivial result
    def py_inner(**kwargs):
        return np.array([2], dtype=np.int32), np.array([0.3], dtype=np.float32), {}

    monkeypatch.setattr(t1, "_t1_one_graph_python_inner", py_inner, raising=True)

    cfg, indptr, indices, weights, rel_code, rel_mult, key_rank, seeds, params, seed_weights = _mini_inputs()

    nodes, vals, metrics = _call_dispatch(cfg, indptr, indices, weights, rel_code, rel_mult, key_rank, seeds, params, seed_weights)

    nt = metrics.get("native_t1", {})
    assert nt.get("fallback_runtime_exc", 0) == 1
    assert any("raised MemoryError" in rec.getMessage() for rec in caplog.records)


def test_strict_parity_mismatch_raises_with_diff(monkeypatch, caplog):
    caplog.set_level(logging.ERROR)
    monkeypatch.setattr(t1, "_LOG_ONCE_KEYS", set(), raising=False)

    # Make native "available"
    import clematis.native.t1 as nt1
    monkeypatch.setattr(nt1, "available", lambda: True, raising=True)

    # Python/native mismatch on a single value beyond tolerances
    def py_inner(**kwargs):
        return np.array([0, 1], dtype=np.int32), np.array([0.50, 0.70], dtype=np.float32), {}

    def native_inner(**kwargs):
        return np.array([0, 1], dtype=np.int32), np.array([0.50, 0.701], dtype=np.float32), {}

    monkeypatch.setattr(t1, "_t1_one_graph_python_inner", py_inner, raising=True)
    monkeypatch.setattr(t1, "_t1_one_graph_native", native_inner, raising=True)

    cfg, indptr, indices, weights, rel_code, rel_mult, key_rank, seeds, params, seed_weights = _mini_inputs()
    cfg["perf"]["native"]["t1"]["strict_parity"] = True

    with pytest.raises(AssertionError) as ei:
        _call_dispatch(cfg, indptr, indices, weights, rel_code, rel_mult, key_rank, seeds, params, seed_weights)

    msg = str(ei.value)
    assert "strict parity mismatch" in msg
    assert any("strict parity mismatch" in rec.getMessage() for rec in caplog.records)


def test_log_once_semantics_import_failed(monkeypatch, caplog):
    caplog.set_level(logging.WARNING)
    monkeypatch.setattr(t1, "_LOG_ONCE_KEYS", set(), raising=False)

    import clematis.native.t1 as nt1
    monkeypatch.setattr(nt1, "available", lambda: False, raising=True)

    def py_inner(**kwargs):
        return np.array([0], dtype=np.int32), np.array([0.1], dtype=np.float32), {}

    monkeypatch.setattr(t1, "_t1_one_graph_python_inner", py_inner, raising=True)

    cfg, indptr, indices, weights, rel_code, rel_mult, key_rank, seeds, params, seed_weights = _mini_inputs()

    for _ in range(2):
        _call_dispatch(cfg, indptr, indices, weights, rel_code, rel_mult, key_rank, seeds, params, seed_weights)

    msgs = [rec.getMessage() for rec in caplog.records if "import/available() failed" in rec.getMessage()]
    assert len(msgs) == 1


def test_env_override_enables_native_even_if_cfg_disabled(monkeypatch):
    import clematis.native.t1 as nt1
    # reset once-only guard
    monkeypatch.setattr(t1, "_LOG_ONCE_KEYS", set(), raising=False)
    # Make native report available
    monkeypatch.setattr(nt1, "available", lambda: True, raising=True)

    # Stub native inner to confirm it was used
    def native_ok(**kwargs):
        return np.array([3], dtype=np.int32), np.array([0.4], dtype=np.float32), {}

    monkeypatch.setattr(t1, "_t1_one_graph_native", native_ok, raising=True)

    # cfg disables native, but env should override
    cfg, indptr, indices, weights, rel_code, rel_mult, key_rank, seeds, params, seed_weights = _mini_inputs()
    cfg["perf"]["native"]["t1"]["enabled"] = False

    monkeypatch.setenv("CLEMATIS_NATIVE_T1", "1")

    nodes, vals, metrics = _call_dispatch(cfg, indptr, indices, weights, rel_code, rel_mult, key_rank, seeds, params, seed_weights)
    nt = metrics.get("native_t1", {})
    assert nt.get("used_native", 0) == 1, f"expected used_native=1, got {nt}"


def test_pyo3_log_once_is_once_even_across_multiple_calls(monkeypatch, caplog):
    import clematis.native.t1 as nt1
    caplog.set_level(logging.ERROR)
    # reset once-only guards
    monkeypatch.setattr(nt1, "_LOG_ONCE_KEYS", set(), raising=False)

    # Provide a dummy _rs backend that always raises from the PyO3 boundary
    class _DummyRS:
        def t1_propagate_one_graph(self, *args, **kwargs):
            raise MemoryError("pyO3 OOM")

    monkeypatch.setattr(nt1, "_rs", _DummyRS(), raising=False)
    # Ensure the wrapper path is taken
    monkeypatch.setattr(nt1, "_HAVE_RS", True, raising=False)

    # Minimal valid shapes
    indptr = np.array([0, 0], dtype=np.int32)
    indices = np.array([], dtype=np.int32)
    weights = np.array([], dtype=np.float32)
    rel_code = np.array([], dtype=np.int32)
    rel_mult = np.array([], dtype=np.float32)
    seed_nodes = np.array([0], dtype=np.int32)
    seed_weights = np.array([1.0], dtype=np.float32)
    key_rank = np.array([0], dtype=np.int32)

    for _ in range(2):
        with pytest.raises(MemoryError):
            nt1.propagate_one_graph_rs(
                indptr,
                indices,
                weights,
                rel_code,
                rel_mult,
                seed_nodes,
                seed_weights,
                key_rank,
                rate=0.6,
                floor=0.05,
                radius_cap=3,
                iter_cap_layers=10,
                node_budget=1.0,
            )

    msgs = [rec.getMessage() for rec in caplog.records
            if getattr(rec, "name", "") == "clematis.native.t1" and "PyO3 raised" in rec.getMessage()]
    assert len(msgs) == 1, f"expected one PyO3 error log, got {len(msgs)}: {msgs}"
