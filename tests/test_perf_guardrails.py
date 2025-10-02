import math
import os
import random
import time
import pytest
from clematis.engine.types import Config

# Import T4 entrypoint; skip the entire module if unavailable
_t4 = pytest.importorskip("clematis.engine.stages.t4")

# Import T1 entrypoint; skip if unavailable
_t1 = pytest.importorskip("clematis.engine.stages.t1")
t1_propagate = _t1.t1_propagate

t4_filter = _t4.t4_filter

# --------------------------
# Minimal helpers (no engine deps)
# --------------------------


class _Ctx:
    def __init__(self, cfg):
        self.config = cfg
        self.cfg = cfg
        self._logbuf = []

    def append_jsonl(self, rec):
        # Minimal sink for T1 logging; keeps deterministic order
        self._logbuf.append(rec)


class _State:
    pass


class _Op:
    __slots__ = ("target_kind", "target_id", "attr", "delta", "kind", "op_idx", "idx")

    def __init__(
        self,
        target_id: str,
        delta: float,
        kind: str = "EditGraph",
        target_kind: str = "Node",
        attr: str = "weight",
        op_idx: int | None = None,
        idx: int | None = None,
    ):
        self.target_kind = target_kind
        self.target_id = target_id
        self.attr = attr
        self.delta = float(delta)
        self.kind = kind
        self.op_idx = op_idx
        self.idx = idx


def _mk_cfg(churn=64, l2=1.5, novelty=0.3):
    return {
        "t4": {
            "enabled": True,
            "delta_norm_cap_l2": float(l2),
            "novelty_cap_per_node": float(novelty),
            "churn_cap_edges": int(churn),
            "cooldowns": {"EditGraph": 2, "CreateGraph": 10},
            "weight_min": -1.0,
            "weight_max": 1.0,
            "cache": {
                "enabled": True,
                "namespaces": ["t2:semantic"],
                "max_entries": 512,
                "ttl_sec": 600,
            },
        }
    }


def _gen_ops(N: int, seed: int = 1337):
    rng = random.Random(seed)
    ids = [f"n{i:04d}" for i in range(256)]  # enough collisions to stress combine
    ops = []
    for i in range(N):
        tgt = rng.choice(ids)
        amt = rng.uniform(-2.0, 2.0)
        ops.append(_Op(target_id=tgt, delta=amt, op_idx=i, idx=i))
    plan = {"proposed_deltas": ops, "ops": ops, "deltas": ops}
    return plan


# --------------------------
# Tiny graph store for T1 gate tests
# --------------------------
try:
    from clematis.graph.store import InMemoryGraphStore, Node, Edge
except (
    Exception
):  # pragma: no cover — if the module layout differs, the entire module is already importorskip'ed
    InMemoryGraphStore = None  # type: ignore
    Node = None  # type: ignore
    Edge = None  # type: ignore


def _store_three():
    store = InMemoryGraphStore()
    for gid in ("A", "B", "C"):
        store.ensure(gid)
        store.upsert_nodes(
            gid,
            [
                Node(id=f"{gid}:seed", label="seed"),
                Node(id=f"{gid}:n1", label="n1"),
            ],
        )
        store.upsert_edges(
            gid,
            [
                Edge(
                    id=f"{gid}:e1", src=f"{gid}:seed", dst=f"{gid}:n1", weight=1.0, rel="supports"
                ),
            ],
        )
    return store, ["A", "B", "C"]


def _ensure_perf(cfg: Config) -> Config:
    if not hasattr(cfg, "perf") or cfg.perf is None:
        setattr(cfg, "perf", {})
    elif not isinstance(cfg.perf, dict):
        try:
            cfg.perf = dict(cfg.perf)  # type: ignore[arg-type]
        except Exception:
            cfg.perf = {}
    return cfg


perf = pytest.mark.perf


@perf
def test_t4_churn_bound_large_is_respected():
    """Large-N sanity: approved deltas must not exceed churn cap."""
    if not os.getenv("RUN_PERF"):
        pytest.skip("opt-in perf tests: set RUN_PERF=1")
    ctx = _Ctx(_mk_cfg(churn=64))
    state = _State()

    plan = _gen_ops(10_000, seed=42)
    res = t4_filter(ctx, state, {}, {}, plan, {})

    approved = getattr(res, "approved_deltas", []) or getattr(res, "approved", []) or []
    assert len(approved) <= 64


@perf
def test_t4_time_budget_smoke():
    """Very loose time budget smoke test; skipped unless RUN_PERF=1.
    Adjust threshold if your CI is significantly slower.
    """
    if not os.getenv("RUN_PERF"):
        pytest.skip("opt-in perf tests: set RUN_PERF=1")

    ctx = _Ctx(_mk_cfg(churn=64))
    state = _State()

    plan = _gen_ops(8_000, seed=123)

    t0 = time.perf_counter()
    _ = t4_filter(ctx, state, {}, {}, plan, {})
    dt_ms = (time.perf_counter() - t0) * 1000.0

    # Generous threshold; intended to catch egregious O(N^2) slips.
    assert dt_ms < 1000.0, f"t4_filter took {dt_ms:.1f} ms for N=8000 (budget 1000ms)"


@perf
def test_t4_relative_scaling_doubling_does_not_quadruple():
    """Relative scaling: doubling N should not 4x the time (O(N^2)).
    Skipped unless RUN_PERF=1 to avoid flakiness.
    """
    if not os.getenv("RUN_PERF"):
        pytest.skip("opt-in perf tests: set RUN_PERF=1")

    ctx = _Ctx(_mk_cfg(churn=64))
    state = _State()

    plan1 = _gen_ops(2_000, seed=7)
    t1 = time.perf_counter()
    _ = t4_filter(ctx, state, {}, {}, plan1, {})
    dt1 = time.perf_counter() - t1

    plan2 = _gen_ops(4_000, seed=8)
    t2 = time.perf_counter()
    _ = t4_filter(ctx, state, {}, {}, plan2, {})
    dt2 = time.perf_counter() - t2

    ratio = dt2 / max(dt1, 1e-6)
    # Allow up to ~3x when doubling input (quite generous); flags egregious superlinear regressions.
    assert ratio < 3.0, f"Doubling N caused {ratio:.2f}x time; expected <3x"


def test_t1_parallel_metrics_gated_off_by_default():
    """Without perf.enabled/report_memory, extra parallel metrics must not appear."""
    cfg = _ensure_perf(Config())
    # Parallel gate ON but metrics gate OFF by default
    cfg.perf.setdefault("parallel", {}).update({"enabled": True, "t1": True, "max_workers": 8})
    store, ag = _store_three()
    ctx = _Ctx(cfg)
    res = t1_propagate(ctx, {"store": store, "active_graphs": ag}, "seed")
    assert "parallel_workers" not in res.metrics
    assert "task_count" not in res.metrics


def test_t1_parallel_metrics_require_report_memory_and_gate_on():
    """When both gates are ON, metrics should include workers/task_count with correct values."""
    cfg = _ensure_perf(Config())
    cfg.perf.setdefault("enabled", True)
    cfg.perf.setdefault("metrics", {})["report_memory"] = True
    cfg.perf.setdefault("parallel", {}).update({"enabled": True, "t1": True, "max_workers": 8})
    store, ag = _store_three()
    ctx = _Ctx(cfg)
    res = t1_propagate(ctx, {"store": store, "active_graphs": ag}, "seed")
    assert res.metrics.get("task_count") == 3
    assert res.metrics.get("parallel_workers") == min(8, 3)


@pytest.mark.skipif(os.environ.get("RUN_PERF") != "1", reason="perf bench is opt-in")
def test_bench_t1_script_smoke(capsys):
    """Smoke: bench prints JSON when requested; no assertions on perf."""
    try:
        mod = __import__(
            "clematis.scripts.bench_t1", fromlist=["main"]
        )  # prefer package import under clematis
    except ModuleNotFoundError:
        import importlib.util
        import sys
        from pathlib import Path
        import clematis as _c

        base = Path(_c.__file__).resolve().parents[0]  # clematis package directory
        bench_path = base / "scripts" / "bench_t1.py"
        if not bench_path.exists():
            pytest.skip("bench_t1.py not found under clematis/scripts/")
        spec = importlib.util.spec_from_file_location("clematis.scripts.bench_t1", str(bench_path))
        mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod
        assert spec.loader is not None
        spec.loader.exec_module(mod)
    import sys

    argv_bak = sys.argv[:]
    try:
        sys.argv = [
            "bench_t1.py",
            "--graphs",
            "3",
            "--iters",
            "2",
            "--workers",
            "4",
            "--parallel",
            "--json",
        ]
        mod.main()
        out = capsys.readouterr().out.strip()
        assert out.startswith("{") and out.endswith("}")
    finally:
        sys.argv = argv_bak
