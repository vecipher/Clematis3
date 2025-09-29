import math
from types import SimpleNamespace

import pytest

from clematis.engine.stages import t4_filter
from clematis.engine.types import ProposedDelta


# -----------------
# Helper builders
# -----------------


def mk_ctx(t4_overrides=None, turn=0):
    cfg = {
        "enabled": True,
        "delta_norm_cap_l2": 1.5,
        "novelty_cap_per_node": 0.3,
        "churn_cap_edges": 64,
        "cooldowns": {"EditGraph": 2, "CreateGraph": 10},
        # forward-looking keys harmless for PR12
        "weight_min": -1.0,
        "weight_max": 1.0,
        "snapshot_every_n_turns": 1,
        "snapshot_dir": "./.data/snapshots",
        "cache_bust_mode": "on-apply",
    }
    if t4_overrides:
        cfg.update(t4_overrides)
    return SimpleNamespace(turn_id=turn, config=SimpleNamespace(t4=cfg))


def mk_state(cooldowns=None):
    meta = SimpleNamespace(cooldowns=cooldowns or {})
    return SimpleNamespace(meta=meta)


def mk_delta(key: str, val: float, op_idx=None, attr="weight"):
    # key like "n:a" or "e:s|rel|d"
    if key.startswith("n:"):
        kind = "node"
        tid = key.split("n:", 1)[1]
        target_id = f"n:{tid}"
    elif key.startswith("e:"):
        kind = "edge"
        tid = key.split("e:", 1)[1]
        target_id = f"e:{tid}"
    else:
        raise ValueError("key must start with 'n:' or 'e:'")
    return ProposedDelta(target_kind=kind, target_id=target_id, attr=attr, delta=val, op_idx=op_idx)


def mk_plan(ops=None, deltas=None):
    # Plan can be a dict; t4_filter supports dict-based access
    return {"ops": ops or [], "deltas": deltas or []}


# ---------------
# Tests
# ---------------


def test_determinism_same_inputs_same_result():
    ctx = mk_ctx()
    state = mk_state()
    ops = [{"kind": "EditGraph"}]
    deltas1 = [
        mk_delta("n:a", 0.10, op_idx=0),
        mk_delta("n:b", 0.20, op_idx=0),
        mk_delta("n:c", -0.05, op_idx=0),
    ]
    deltas2 = list(reversed(deltas1))  # shuffled order

    out1 = t4_filter(ctx, state, None, None, mk_plan(ops=ops, deltas=deltas1), utter=None)
    out2 = t4_filter(ctx, state, None, None, mk_plan(ops=ops, deltas=deltas2), utter=None)

    # Approved deltas must be identical (keys + values)
    def as_map(out):
        return {
            f"{d.target_kind}:{d.target_id}:{d.attr}": pytest.approx(d.delta)
            for d in out.approved_deltas
        }

    assert as_map(out1) == as_map(out2)
    assert out1.reasons == out2.reasons
    assert out1.metrics["counts"]["approved"] == out2.metrics["counts"]["approved"]


def test_cooldown_blocks_ops_and_deltas_removed():
    # cooldown for EditGraph is 2; last_turn=9, current=10 -> blocked
    ctx = mk_ctx(turn=10)
    state = mk_state(cooldowns={"EditGraph": 9})
    ops = [{"kind": "EditGraph"}]
    deltas = [mk_delta("n:a", 0.25, op_idx=0)]
    out = t4_filter(ctx, state, None, None, mk_plan(ops=ops, deltas=deltas), utter=None)

    assert "COOLDOWN_BLOCKED" in out.reasons
    assert out.metrics["cooldowns"]["blocked_ops"] == 1
    assert out.metrics["counts"]["approved"] == 0
    assert len(out.rejected_ops) == 1
    assert out.rejected_ops[0].kind == "EditGraph"
    assert out.rejected_ops[0].idx == 0


def test_novelty_cap_clamps_per_target():
    ctx = mk_ctx({"novelty_cap_per_node": 0.1})
    state = mk_state()
    ops = [{"kind": "EditGraph"}]
    deltas = [mk_delta("n:a", 0.5, op_idx=0)]
    out = t4_filter(ctx, state, None, None, mk_plan(ops=ops, deltas=deltas), utter=None)

    assert "NOVELTY_SPIKE" in out.reasons
    assert len(out.approved_deltas) == 1
    assert out.approved_deltas[0].delta == pytest.approx(0.1)


def test_l2_cap_scales_preserving_ratios_and_sign():
    # Build three deltas; set tiny L2 cap so scaling is applied.
    ctx = mk_ctx({"delta_norm_cap_l2": 0.5, "novelty_cap_per_node": 10.0})
    state = mk_state()
    ops = [{"kind": "EditGraph"}]
    base = {
        "n:a": 1.0,
        "n:b": -2.0,
        "n:c": 3.0,
    }
    deltas = [mk_delta(k, v, op_idx=0) for k, v in base.items()]
    out = t4_filter(ctx, state, None, None, mk_plan(ops=ops, deltas=deltas), utter=None)

    assert "DELTA_NORM_HIGH" in out.reasons
    scale = out.metrics["clamps"]["l2_scale"]
    assert scale < 1.0

    # Map by target to verify scaled values
    got = {f"{d.target_id}": d.delta for d in out.approved_deltas}
    for k, v in base.items():
        tid = k  # already like "n:a"
        assert got[tid] == pytest.approx(v * scale)
        # sign preserved
        assert (got[tid] >= 0) == (v >= 0)


def test_churn_cap_keeps_top_k_by_abs_value_with_tie_break():
    ctx = mk_ctx({"churn_cap_edges": 3, "novelty_cap_per_node": 10.0})
    state = mk_state()
    ops = [{"kind": "EditGraph"}]
    mags = [0.5, 0.4, 0.3, 0.2, 0.1]
    deltas = [mk_delta(f"n:{chr(ord('a') + i)}", m, op_idx=0) for i, m in enumerate(mags)]
    out = t4_filter(ctx, state, None, None, mk_plan(ops=ops, deltas=deltas), utter=None)

    assert "CHURN_CAP_HIT" in out.reasons
    assert out.metrics["counts"]["approved"] == 3
    kept = sorted([abs(d.delta) for d in out.approved_deltas], reverse=True)
    assert kept == mags[:3]


def test_duplicate_targets_are_combined_before_caps():
    ctx = mk_ctx({"novelty_cap_per_node": 10.0})
    state = mk_state()
    ops = [{"kind": "EditGraph"}, {"kind": "EditGraph"}]
    deltas = [
        mk_delta("n:x", 0.2, op_idx=0),
        mk_delta("n:x", 0.3, op_idx=1),
    ]
    out = t4_filter(ctx, state, None, None, mk_plan(ops=ops, deltas=deltas), utter=None)

    assert out.metrics["counts"]["approved"] == 1
    only = out.approved_deltas[0]
    assert only.target_id == "n:x"
    assert only.delta == pytest.approx(0.5)


def test_zero_and_empty_inputs():
    # zero delta survives unless churn trims
    ctx = mk_ctx({"churn_cap_edges": 10})
    state = mk_state()
    ops = [{"kind": "EditGraph"}]
    out1 = t4_filter(
        ctx,
        state,
        None,
        None,
        mk_plan(ops=ops, deltas=[mk_delta("n:a", 0.0, op_idx=0)]),
        utter=None,
    )
    assert out1.metrics["clamps"]["l2_scale"] == pytest.approx(1.0)
    assert len(out1.approved_deltas) == 1
    assert out1.approved_deltas[0].delta == pytest.approx(0.0)
    assert out1.reasons == []  # no caps hit

    # empty input
    out2 = t4_filter(ctx, state, None, None, mk_plan(), utter=None)
    assert out2.approved_deltas == []
    assert out2.rejected_ops == []
    assert out2.reasons == []
    assert out2.metrics["counts"]["approved"] == 0
