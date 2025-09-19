from __future__ import annotations
from typing import Dict, Any, Optional, List
from dataclasses import asdict
import os
import json
import time

from .types import ApplyResult, T4Result, ProposedDelta


# -------- helpers: state access (dict or attr style) --------

def _state_get(obj: Any, key: str, default=None):
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _state_set(obj: Any, key: str, value: Any):
    if isinstance(obj, dict):
        obj[key] = value
    else:
        setattr(obj, key, value)


def _get_cfg(ctx) -> Dict[str, Any]:
    t4_default = {
        "weight_min": -1.0,
        "weight_max": 1.0,
        "snapshot_every_n_turns": 1,
        "snapshot_dir": "./.data/snapshots",
    }
    cfg = getattr(getattr(ctx, "config", object()), "t4", None)
    if isinstance(cfg, dict):
        out = dict(t4_default)
        out.update(cfg)
        out["weight_min"] = float(out.get("weight_min", -1.0))
        out["weight_max"] = float(out.get("weight_max", 1.0))
        out["snapshot_every_n_turns"] = int(out.get("snapshot_every_n_turns", 1))
        out["snapshot_dir"] = str(out.get("snapshot_dir", "./.data/snapshots"))
        return out
    return t4_default


def _now_ms() -> int:
    # Monotonic-ish timestamp for metrics; not used in ids.
    return int(time.time() * 1000)


def _bump_version_etag(state: Any) -> str:
    """
    Monotonic, deterministic bump stored on state. Prefer an integer counter
    we control to avoid relying on store internals.
    """
    current = _state_get(state, "version_etag", None)
    if current is None:
        new_val = "1"
    else:
        try:
            new_val = str(int(current) + 1)
        except Exception:
            # If prior value wasn't numeric, start anew with "1"
            new_val = "1"
    _state_set(state, "version_etag", new_val)
    return new_val


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _should_snapshot(ctx, cfg) -> bool:
    try:
        turn = int(getattr(ctx, "turn_id", 0))
    except Exception:
        turn = 0
    every = max(1, int(cfg.get("snapshot_every_n_turns", 1)))
    # Snapshot on every Nth turn. Define turn 0 as a snapshot turn for simplicity.
    return (turn % every) == 0


def _snapshot_path(cfg: Dict[str, Any], ctx) -> str:
    agent = getattr(ctx, "agent_id", "agent")
    turn = getattr(ctx, "turn_id", 0)
    dir_ = cfg["snapshot_dir"]
    _ensure_dir(dir_)
    fname = f"state_{agent}.json"
    # Turn IDs may be non-numeric (e.g., "demo-1"); store as string
    return os.path.join(dir_, fname), str(turn)


def _serialize_deltas(deltas: List[ProposedDelta]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for d in deltas:
        # dataclass or dataclass-like; use safe dict
        try:
            out.append({
                "target_kind": d.target_kind,
                "target_id": d.target_id,
                "attr": d.attr,
                "delta": float(d.delta),
                "op_idx": d.op_idx,
                "idx": d.idx,
            })
        except Exception:
            # Fall back to asdict if available
            try:
                out.append(asdict(d))
            except Exception:
                pass
    return out


# -------- main API --------

def apply_changes(ctx, state, t4: T4Result) -> ApplyResult:
    """
    Apply approved deltas to the store, clamp values into [weight_min, weight_max]
    if the store reports clamping, bump version_etag, optionally write a snapshot.
    Returns ApplyResult with counts and paths; resilient to dict/attr state layouts.
    """
    started = _now_ms()
    cfg = _get_cfg(ctx)

    store = _state_get(state, "store", None)
    if store is None:
        # Graceful failure mode: nothing to apply
        version_etag = _bump_version_etag(state)
        should_snap = _should_snapshot(ctx, cfg)
        snap_path = None
        if should_snap:
            snap_path, turn = _snapshot_path(cfg, ctx)
            with open(snap_path, "w", encoding="utf-8") as f:
                json.dump(
                    {"turn": turn, "agent": getattr(ctx, "agent_id", "agent"),
                     "version_etag": version_etag, "applied": 0, "deltas": []},
                    f,
                )
        return ApplyResult(
            applied=0,
            clamps=0,
            version_etag=version_etag,
            snapshot_path=snap_path,
            metrics={"ms": _now_ms() - started, "notes": "no-store"},
        )

    # Apply deltas one batch; if store supports batching, use it; else per-delta.
    deltas = list(t4.approved_deltas or [])
    applied_count = 0
    clamp_count = 0

    # Prefer a batch API if present
    apply_fn = getattr(store, "apply_deltas", None)
    if callable(apply_fn):
        try:
            res = apply_fn("g:surface", deltas)
            # Defensive extraction of counts
            applied_count += int(_safe_get(res, "edits", 0))
            clamp_count += int(_safe_get(res, "clamps", _safe_get(res, "clamped", 0)))
        except Exception:
            # Fallback to per-delta loop
            for d in deltas:
                try:
                    r = apply_fn("g:surface", [d])
                    applied_count += int(_safe_get(r, "edits", 0))
                    clamp_count += int(_safe_get(r, "clamps", _safe_get(r, "clamped", 0)))
                except Exception:
                    # continue applying others
                    continue
    else:
        # No known API; cannot apply. Return zero but still bump version/snapshot.
        version_etag = _bump_version_etag(state)
        should_snap = _should_snapshot(ctx, cfg)
        snap_path = None
        if should_snap:
            snap_path, turn = _snapshot_path(cfg, ctx)
            with open(snap_path, "w", encoding="utf-8") as f:
                json.dump(
                    {"turn": turn, "agent": getattr(ctx, "agent_id", "agent"),
                     "version_etag": version_etag, "applied": 0, "deltas": _serialize_deltas(deltas)},
                    f,
                )
        return ApplyResult(
            applied=0,
            clamps=0,
            version_etag=version_etag,
            snapshot_path=snap_path,
            metrics={"ms": _now_ms() - started, "notes": "no-apply-fn"},
        )

    # Bump version etag after successful apply
    version_etag = _bump_version_etag(state)

    # Snapshot cadence
    snap_path = None
    if _should_snapshot(ctx, cfg):
        snap_path, turn = _snapshot_path(cfg, ctx)
        with open(snap_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "turn": turn,
                    "agent": getattr(ctx, "agent_id", "agent"),
                    "version_etag": version_etag,
                    "applied": applied_count,
                    "deltas": _serialize_deltas(deltas),
                },
                f,
            )

    metrics = {
        "ms": _now_ms() - started,
        "applied": applied_count,
        "clamps": clamp_count,
    }
    return ApplyResult(
        applied=applied_count,
        clamps=clamp_count,
        version_etag=version_etag,
        snapshot_path=snap_path,
        metrics=metrics,
    )


# -------- small util --------

def _safe_get(maybe_mapping: Any, key: str, default=0):
    try:
        return maybe_mapping.get(key, default)
    except Exception:
        return default