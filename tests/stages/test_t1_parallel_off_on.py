from __future__ import annotations

"""
PR66 staging tests for T1 parallelization.

These focus on the **gate semantics** to ensure we only attempt a parallel fanout
when explicitly enabled and with `max_workers > 1`. Full functional parity tests
(OFF vs ON equivalence) are added separately once the wiring stabilizes across
environments. This file intentionally avoids constructing real graphs.
"""

from types import SimpleNamespace

import pytest
from clematis.engine.types import Config

# Import the T1 stage module (skip if not present in this environment)
t1 = pytest.importorskip("clematis.engine.stages.t1")


# --- Gate semantics -----------------------------------------------------------

def _ns(obj: dict) -> SimpleNamespace:
    """Recursively wrap dicts as SimpleNamespace for cfg-style access."""
    def wrap(x):
        if isinstance(x, dict):
            return SimpleNamespace(**{k: wrap(v) for k, v in x.items()})
        return x
    return wrap(obj)  # type: ignore[return-value]


def test_t1_parallel_gate_defaults_off():
    # No perf/parallel -> gate must be False
    cfg = _ns({})
    assert t1._t1_parallel_enabled(cfg) is False

    # perf present but parallel missing
    cfg = _ns({"perf": {}})
    assert t1._t1_parallel_enabled(cfg) is False

    # parallel present but disabled
    cfg = _ns({"perf": {"parallel": {"enabled": False, "t1": True, "max_workers": 8}}})
    assert t1._t1_parallel_enabled(cfg) is False

    # enabled but t1 flag is False
    cfg = _ns({"perf": {"parallel": {"enabled": True, "t1": False, "max_workers": 8}}})
    assert t1._t1_parallel_enabled(cfg) is False

    # enabled and t1 True but workers <= 1 (sequential path) → gate False
    for workers in (0, 1, None):
        cfg = _ns({"perf": {"parallel": {"enabled": True, "t1": True, "max_workers": workers}}})
        assert t1._t1_parallel_enabled(cfg) is False


def test_t1_parallel_gate_true_only_when_expected():
    # Minimal config that should enable the gate
    cfg = _ns({"perf": {"parallel": {"enabled": True, "t1": True, "max_workers": 2}}})
    assert t1._t1_parallel_enabled(cfg) is True

    # Dict-shaped config is also accepted
    cfg_dict = {"perf": {"parallel": {"enabled": True, "t1": True, "max_workers": 4}}}
    assert t1._t1_parallel_enabled(cfg_dict) is True


# --- Integration: functional equivalence test for parallel ON vs sequential ---

def _mk_cfg(workers: int, enabled: bool = True, t1_gate: bool = True) -> Config:
    cfg = Config()
    # Ensure perf exists and is a dict
    if not hasattr(cfg, "perf") or cfg.perf is None:
        setattr(cfg, "perf", {})
    if not isinstance(cfg.perf, dict):
        # Best-effort conversion to dict; fallback to fresh dict
        try:
            cfg.perf = dict(cfg.perf)  # type: ignore[arg-type]
        except Exception:
            cfg.perf = {}
    # Ensure nested parallel dict
    if "parallel" not in cfg.perf or not isinstance(cfg.perf["parallel"], dict):
        cfg.perf["parallel"] = {}
    cfg.perf["parallel"].update({
        "enabled": enabled,
        "t1": t1_gate,
        "t2": False,
        "agents": False,
        "max_workers": workers,
    })
    return cfg


def _mk_ctx_with_log_buffer(cfg):
    buf = []
    return SimpleNamespace(cfg=cfg, append_jsonl=buf.append, _logbuf=buf)


def _build_tiny_store():
    # Import minimal graph primitives from the public API the rest of the tests use
    from clematis.graph.store import InMemoryGraphStore, Node, Edge
    store = InMemoryGraphStore()
    # Three small graphs A,B,C with simple single-edge structure
    for gid in ("A", "B", "C"):
        store.ensure(gid)
        store.upsert_nodes(gid, [Node(id=f"{gid}:seed", label=f"{gid}-seed"),
                                 Node(id=f"{gid}:n1", label=f"{gid}-n1")])
        store.upsert_edges(gid, [Edge(id=f"{gid}:e1", src=f"{gid}:seed", dst=f"{gid}:n1", weight=1.0, rel="supports")])
    return store


def test_t1_parallel_on_equals_sequential_when_enabled(monkeypatch):
    # Build store and state with deterministic active_graphs order
    store = _build_tiny_store()
    state = {"store": store, "active_graphs": ["A", "B", "C"]}

    # Disable cache for clean equivalence (avoid cache-hit/vs-miss log drift)
    monkeypatch.setattr(t1, "_get_cache", lambda *a, **kw: (None, "lru"), raising=True)

    # Baseline sequential
    cfg_seq = _mk_cfg(workers=1, enabled=True, t1_gate=True)
    ctx_seq = _mk_ctx_with_log_buffer(cfg_seq)
    r_seq = t1.t1_propagate(ctx_seq, state, text="seed")
    deltas_seq = list(r_seq.graph_deltas)
    logs_seq = list(ctx_seq._logbuf)
    metrics_seq = dict(r_seq.metrics)

    # Parallel ON
    cfg_par = _mk_cfg(workers=4, enabled=True, t1_gate=True)
    ctx_par = _mk_ctx_with_log_buffer(cfg_par)
    r_par = t1.t1_propagate(ctx_par, state, text="seed")
    deltas_par = list(r_par.graph_deltas)
    logs_par = list(ctx_par._logbuf)
    metrics_par = dict(r_par.metrics)

    # Equality checks
    assert deltas_par == deltas_seq
    assert logs_par == logs_seq
    # Metrics equality: ignore any perf-only counters that may be absent if perf is disabled;
    # here, perf is effectively minimal, so direct dict equality should hold.
    assert metrics_par == metrics_seq
