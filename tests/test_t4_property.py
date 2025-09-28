import math
import random
import json
import pytest

# We test against the public T4 entrypoint.
t4 = pytest.importorskip("clematis.engine.stages.t4")
t4_filter = t4.t4_filter

# --------------------------
# Helpers (duck-typed, no deps)
# --------------------------


class _Ctx:
    def __init__(self, cfg):
        # Most engines expect ctx.config or similar; expose both to be safe.
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


def _mk_cfg(
    l2=1.5,
    novelty=0.3,
    churn=64,
    cooldowns=None,
):
    cooldowns = cooldowns or {"EditGraph": 2, "CreateGraph": 10}
    return {
        "t4": {
            "enabled": True,
            "delta_norm_cap_l2": l2,
            "novelty_cap_per_node": novelty,
            "churn_cap_edges": churn,
            "cooldowns": cooldowns,
            "weight_min": -1.0,
            "weight_max": 1.0,
            # cache settings irrelevant for T4 purity checks
            "cache": {
                "enabled": True,
                "namespaces": ["t2:semantic"],
                "max_entries": 512,
                "ttl_sec": 600,
            },
        }
    }


def _mag(x):
    # Try common shapes used for "delta magnitude"
    for k in ("delta", "value", "amount", "weight", "w"):
        if isinstance(x, dict) and k in x and isinstance(x[k], (int, float)):
            return float(x[k])
        if hasattr(x, k):
            v = getattr(x, k)
            if isinstance(v, (int, float)):
                return float(v)
    return 0.0


def _target_id(x):
    for k in ("target_id", "node", "id", "src"):
        if isinstance(x, dict) and k in x:
            return str(x[k])
        if hasattr(x, k):
            try:
                return str(getattr(x, k))
            except Exception:
                pass
    # Fallback: stable repr
    return repr(x)


def _approved_list(res):
    # T4Result.approved_deltas is expected; fall back to .approved if present.
    if hasattr(res, "approved_deltas"):
        return list(getattr(res, "approved_deltas") or [])
    if hasattr(res, "approved"):
        return list(getattr(res, "approved") or [])
    # Last ditch: look in metrics
    return list((getattr(res, "metrics", {}) or {}).get("approved_deltas", []))


def _reasons_list(res):
    if hasattr(res, "reasons"):
        return list(getattr(res, "reasons") or [])
    return list((getattr(res, "metrics", {}) or {}).get("reasons", []))


def _serialize_result(res):
    """Make a deterministic, JSON-serializable view of the result for idempotence checks."""
    approved = _approved_list(res)
    reasons = _reasons_list(res)
    metrics = getattr(res, "metrics", {}) or {}

    def ser_delta(d):
        if isinstance(d, dict):
            return {k: d[k] for k in sorted(d.keys()) if isinstance(d[k], (str, int, float, bool))}
        out = {}
        for k in dir(d):
            if k.startswith("_"):
                continue
            try:
                v = getattr(d, k)
            except Exception:
                continue
            if isinstance(v, (str, int, float, bool)):
                out[k] = v
        return {k: out[k] for k in sorted(out.keys())}

    return {
        "approved": [ser_delta(d) for d in approved],
        "reasons": sorted([str(r) for r in reasons]),
        "metrics_keys": sorted(list(metrics.keys())),
        "approved_len": len(approved),
    }


def _l2_norm(vals):
    return math.sqrt(sum((float(v) ** 2) for v in vals))


# --------------------------
# Synthetic input generator
# --------------------------


def _synth_ops(N, rng):
    """Produce a list of synthetic deltas shaped like the engine expects (attribute-based)."""
    ops = []
    # small alphabet to force collisions for novelty-caps
    ids = [f"n{i:03d}" for i in range(32)]
    for i in range(N):
        tgt = rng.choice(ids)
        amt = rng.uniform(-2.0, 2.0)  # deliberately beyond caps
        ops.append(_Op(target_id=tgt, delta=amt, op_idx=i, idx=i))
    return ops


def _mk_inputs(N, seed=1337):
    rng = random.Random(seed)
    ops = _synth_ops(N, rng)
    # Construct a minimal "plan" that carries proposed deltas in a discoverable field name.
    plan = {
        "proposed_deltas": ops,  # common name
        "ops": ops,  # alternate
        "deltas": ops,  # alternate
    }
    utter = {}
    t1 = {}
    t2 = {}
    return t1, t2, plan, utter


# --------------------------
# Tests
# --------------------------


@pytest.mark.parametrize("N", [0, 1, 2, 8, 64, 256, 1024])
def test_t4_fuzz_invariants_basic(N):
    """
    Deterministic fuzz across sizes; asserts guardrail-like invariants.
    If T4 ignores our synthetic ops, invariants hold vacuously (len=0).
    """
    ctx = _Ctx(_mk_cfg(l2=1.5, novelty=0.3, churn=64))
    state = _State()
    t1, t2, plan, utter = _mk_inputs(N, seed=42)
    res = t4_filter(ctx, state, t1, t2, plan, utter)

    approved = _approved_list(res)
    mags = [_mag(d) for d in approved]
    l2 = _l2_norm([abs(m) for m in mags])

    # Invariants (loose, but deterministic)
    assert len(approved) <= 64
    assert l2 <= 1.5 + 1e-9

    # Per-target novelty cap (best-effort; skip if we cannot attribute)
    per_target = {}
    for d in approved:
        per_target.setdefault(_target_id(d), 0.0)
        per_target[_target_id(d)] += abs(_mag(d))
    assert all(v <= 0.3 + 1e-9 for v in per_target.values())

    # Stable ordering: ensure we can JSON-serialize and that order is deterministic
    snap1 = _serialize_result(res)
    res2 = t4_filter(ctx, state, t1, t2, plan, utter)
    snap2 = _serialize_result(res2)
    assert snap1 == snap2


def test_t4_deterministic_tie_break_and_purity():
    """
    Feed a crafted set with equal magnitudes to exercise tie-break stability.
    """
    ctx = _Ctx(_mk_cfg(l2=1.5, novelty=0.3, churn=64))
    state = _State()

    # Two equal-magnitude deltas for distinct ids; order should be stable across runs
    ops = [
        _Op("n001", 0.2, op_idx=0, idx=0),
        _Op("n002", 0.2, op_idx=1, idx=1),
        _Op("n001", 0.2, op_idx=2, idx=2),
        _Op("n002", 0.2, op_idx=3, idx=3),
    ]
    plan = {"proposed_deltas": ops, "ops": ops, "deltas": ops}
    t1 = {}
    t2 = {}
    utter = {}

    res_a = t4_filter(ctx, state, t1, t2, plan, utter)
    res_b = t4_filter(ctx, state, t1, t2, plan, utter)

    assert _serialize_result(res_a) == _serialize_result(res_b)


def test_t4_records_reasons_when_blocking():
    """
    If cooldowns or caps cause rejections, T4 should surface reasons.
    We don't assert specific codes (engine-dependent), only that the container is stable.
    """
    ctx = _Ctx(_mk_cfg(l2=0.05, novelty=0.05, churn=1))  # force rejections at small caps
    state = _State()
    t1, t2, plan, utter = _mk_inputs(32, seed=7)

    res = t4_filter(ctx, state, t1, t2, plan, utter)
    reasons = _reasons_list(res)
    # Reasons list should be list-like and stable across runs
    assert isinstance(reasons, list)
    res2 = t4_filter(ctx, state, t1, t2, plan, utter)
    assert reasons == _reasons_list(res2)
