import math
import os
import random
import time
import pytest

# Import T4 entrypoint; skip the entire module if unavailable
_t4 = pytest.importorskip("clematis.engine.stages.t4")

t4_filter = _t4.t4_filter

# --------------------------
# Minimal helpers (no engine deps)
# --------------------------


class _Ctx:
    def __init__(self, cfg):
        self.config = cfg
        self.cfg = cfg


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
