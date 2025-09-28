from __future__ import annotations

from typing import Dict, Any, List, Tuple, Optional
from math import sqrt

from ..types import T4Result, ProposedDelta, OpRef


def t4_filter(ctx, state, t1, t2, plan, utter) -> T4Result:
    """
    Pure, deterministic meta-filter.
    Enforces cooldowns (op-level), per-target novelty clamp, global L2 cap (uniform scaling),
    and churn cap (keep top-K by |Δ| with stable tie-breaks).
    No side-effects, no I/O.
    """
    cfg = _get_cfg(ctx)

    # Extract ops & deltas from plan (allow dataclass or dict)
    ops = _get_plan_ops(plan)
    deltas_in = _get_plan_deltas(plan)

    # 0) Canonicalize & combine duplicates (by ckey)
    combined = _combine_by_ckey(deltas_in)

    # 1) Cooldown blocking at op-level
    blocked_ops = _collect_blocked_ops(ops, state, ctx, cfg.get("cooldowns", {}))
    after_cd = [d for d in combined if (d.op_idx is None) or (d.op_idx not in blocked_ops)]
    rejected_ops = [OpRef(kind=_get_op_kind(ops, i), idx=i) for i in sorted(blocked_ops)]

    # 2) Novelty clamp (per-target magnitude clamp)
    clamped, novelty_count = _novelty_clamp(after_cd, cfg["novelty_cap_per_node"])

    # 3) Global L2 scaling (uniform) if needed
    scaled, scale = _l2_scale(clamped, cfg["delta_norm_cap_l2"])

    # 4) Churn cap: keep top-K by |Δ|, tie-break by canonical key
    approved, dropped_tail = _churn_cap(scaled, cfg["churn_cap_edges"])

    # Reasons (ordered by pipeline stage)
    reasons: List[str] = []
    if blocked_ops:
        reasons.append("COOLDOWN_BLOCKED")
    if novelty_count > 0:
        reasons.append("NOVELTY_SPIKE")
    if scale < 0.999999:
        reasons.append("DELTA_NORM_HIGH")
    if dropped_tail > 0:
        reasons.append("CHURN_CAP_HIT")

    metrics: Dict[str, Any] = {
        "counts": {
            "input": len(deltas_in),
            "after_cooldown": len(after_cd),
            "after_novelty": len(clamped),
            "after_l2": len(scaled),
            "approved": len(approved),
            "dropped_tail": dropped_tail,
        },
        "clamps": {
            "novelty_clamped": novelty_count,
            "l2_scale": scale,
        },
        "cooldowns": {
            "blocked_ops": len(blocked_ops),
        },
        "caps": {
            "delta_norm_cap_l2": cfg["delta_norm_cap_l2"],
            "novelty_cap_per_node": cfg["novelty_cap_per_node"],
            "churn_cap_edges": cfg["churn_cap_edges"],
        },
    }

    # Deterministic output order: sort by canonical key
    approved_sorted = sorted(approved, key=_canonical_key)

    return T4Result(
        approved_deltas=approved_sorted,
        rejected_ops=rejected_ops,
        reasons=reasons,
        metrics=metrics,
    )


# ------------------------
# Helpers (pure functions)
# ------------------------


def _get_cfg(ctx) -> Dict[str, Any]:
    # Safe config accessor with deterministic defaults
    t4_default = {
        "delta_norm_cap_l2": 1.5,
        "novelty_cap_per_node": 0.3,
        "churn_cap_edges": 64,
        "cooldowns": {},
    }
    cfg = getattr(getattr(ctx, "config", object()), "t4", None)
    if isinstance(cfg, dict):
        # merge shallowly with defaults to allow missing keys without crashing
        out = dict(t4_default)
        out.update(cfg)
        # Coerce expected numeric types (defensive)
        out["delta_norm_cap_l2"] = float(out.get("delta_norm_cap_l2", 1.5))
        out["novelty_cap_per_node"] = float(out.get("novelty_cap_per_node", 0.3))
        out["churn_cap_edges"] = int(out.get("churn_cap_edges", 64))
        if "cooldowns" not in out or not isinstance(out["cooldowns"], dict):
            out["cooldowns"] = {}
        return out
    return t4_default


def _get_plan_ops(plan: Any) -> List[Any]:
    if plan is None:
        return []
    if hasattr(plan, "ops"):
        return getattr(plan, "ops") or []
    if isinstance(plan, dict):
        return plan.get("ops", []) or []
    return []


def _get_plan_deltas(plan: Any) -> List[ProposedDelta]:
    if plan is None:
        return []
    if hasattr(plan, "deltas"):
        return list(getattr(plan, "deltas") or [])
    if isinstance(plan, dict):
        return list(plan.get("deltas", []) or [])
    return []


def _canonical_key(d: ProposedDelta) -> str:
    # Stable identity across all steps
    return f"{d.target_kind}:{d.target_id}:{d.attr}"


def _combine_by_ckey(deltas: List[ProposedDelta]) -> List[ProposedDelta]:
    """
    Combine duplicate targets by summing additive deltas.
    Keep the smallest op_idx and idx for provenance determinism.
    Output sorted by canonical key for stability.
    """
    if not deltas:
        return []

    accum: Dict[str, Tuple[float, Optional[int], Optional[int], ProposedDelta]] = {}
    for d in deltas:
        ckey = _canonical_key(d)
        if ckey in accum:
            cur_delta, cur_op_idx, cur_idx, exemplar = accum[ckey]
            new_delta = cur_delta + float(d.delta)
            new_op_idx = _min_optional_int(cur_op_idx, d.op_idx)
            new_idx = _min_optional_int(cur_idx, d.idx)
            accum[ckey] = (new_delta, new_op_idx, new_idx, exemplar)
        else:
            accum[ckey] = (float(d.delta), d.op_idx, d.idx, d)

    combined: List[ProposedDelta] = []
    for ckey in sorted(accum.keys()):
        val, op_idx, idx, exemplar = accum[ckey]
        combined.append(
            ProposedDelta(
                target_kind=exemplar.target_kind,
                target_id=exemplar.target_id,
                attr=exemplar.attr,
                delta=val,
                op_idx=op_idx,
                idx=idx,
            )
        )
    return combined


def _min_optional_int(a: Optional[int], b: Optional[int]) -> Optional[int]:
    if a is None:
        return b
    if b is None:
        return a
    return a if a <= b else b


def _collect_blocked_ops(
    ops: List[Any], state: Any, ctx: Any, cooldowns: Dict[str, int]
) -> set[int]:
    """
    Return set of op indices that are within cooldown.
    Expects cooldowns: {kind: turns}.
    Looks up last_turn per kind from state.meta.cooldowns (if present).
    """
    if not ops or not cooldowns:
        return set()

    turn = _get_turn(ctx)
    last_turn_map = _get_last_turn_map(state)

    blocked: set[int] = set()
    for i, op in enumerate(ops):
        kind = _get_op_kind(ops, i)
        if not kind:
            continue
        cd = cooldowns.get(kind)
        if not cd:
            continue
        last_t = _map_get(last_turn_map, kind)
        if isinstance(last_t, int):
            if (turn - last_t) < int(cd):
                blocked.add(i)
    return blocked


def _get_turn(ctx: Any) -> int:
    for name in ("turn_id", "turn", "current_turn"):
        if hasattr(ctx, name):
            try:
                return int(getattr(ctx, name))
            except Exception:
                pass
    return 0


def _get_last_turn_map(state: Any) -> Dict[str, int]:
    # Try object.attr then dict-style fallbacks
    meta = getattr(state, "meta", None)
    if meta is None and isinstance(state, dict):
        meta = state.get("meta")
    cooldowns = None
    if meta is not None:
        cooldowns = getattr(meta, "cooldowns", None)
        if cooldowns is None and isinstance(meta, dict):
            cooldowns = meta.get("cooldowns")
    return cooldowns if isinstance(cooldowns, dict) else {}


def _map_get(m: Dict[str, Any], k: str, default: Any = None) -> Any:
    try:
        return m.get(k, default)
    except Exception:
        return default


def _get_op_kind(ops: List[Any], idx: int) -> str:
    try:
        op = ops[idx]
    except Exception:
        return ""
    # Try attribute access first, then dict
    if hasattr(op, "kind"):
        val = getattr(op, "kind")
        return str(val) if val is not None else ""
    if isinstance(op, dict):
        return str(op.get("kind", ""))
    return ""


def _novelty_clamp(deltas: List[ProposedDelta], cap: float) -> Tuple[List[ProposedDelta], int]:
    if not deltas:
        return [], 0
    cap = abs(float(cap))
    out: List[ProposedDelta] = []
    clamped_count = 0
    for d in deltas:
        mag = abs(d.delta)
        if mag > cap:
            new_delta = cap if d.delta > 0 else -cap
            clamped_count += 1
        else:
            new_delta = d.delta
        out.append(
            ProposedDelta(
                target_kind=d.target_kind,
                target_id=d.target_id,
                attr=d.attr,
                delta=new_delta,
                op_idx=d.op_idx,
                idx=d.idx,
            )
        )
    return out, clamped_count


def _l2_scale(deltas: List[ProposedDelta], cap_l2: float) -> Tuple[List[ProposedDelta], float]:
    if not deltas:
        return [], 1.0
    cap = float(cap_l2)
    # Compute L2 norm of the vector of deltas
    s = 0.0
    for d in deltas:
        s += float(d.delta) * float(d.delta)
    norm = sqrt(s)
    if norm <= cap or norm == 0.0:
        return deltas, 1.0
    scale = cap / norm
    scaled: List[ProposedDelta] = [
        ProposedDelta(
            target_kind=d.target_kind,
            target_id=d.target_id,
            attr=d.attr,
            delta=d.delta * scale,
            op_idx=d.op_idx,
            idx=d.idx,
        )
        for d in deltas
    ]
    return scaled, float(scale)


def _churn_cap(deltas: List[ProposedDelta], k: int) -> Tuple[List[ProposedDelta], int]:
    k = int(k)
    n = len(deltas)
    if n <= k:
        return deltas, 0
    # Rank by |Δ| desc, tie-break by canonical key asc
    ranked = sorted(deltas, key=lambda d: (-abs(float(d.delta)), _canonical_key(d)))
    kept = ranked[:k]
    dropped = n - k
    return kept, dropped
