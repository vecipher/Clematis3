"""Agent fan-out and parallel driver helpers for the orchestrator."""

from __future__ import annotations

from types import SimpleNamespace as _SNS
from typing import Any, Dict, List, Tuple, TypedDict
import os as _os
import sys as _sys
import time

from .types import TurnCtx, TurnResult
from .logging import (
    _append_unbuffered,
    _begin_log_capture,
    _end_log_capture,
    _get_logging_callable,
)
from ..stages.state_clone import readonly_snapshot

__all__ = [
    "_clone_ctx_for_agent",
    "_extract_dryrun_artifacts",
    "_make_readonly_snapshot",
    "_resolve_graphs_for_agent",
    "_select_independent_batch",
    "_agents_parallel_enabled",
    "_sort_turn_buffers",
    "_run_turn_compute",
    "_run_agents_parallel_batch",
]


class _TurnBuffer(TypedDict):
    """Envelope capturing compute-phase artifacts for a turn."""

    turn_id: int | str
    slice_idx: int
    agent_id: str
    logs: List[Tuple[str, Dict[str, Any]]]
    deltas: Any
    dialogue: str
    graphs_touched: set[str]
    graph_versions: Dict[str, str]
    t2_info: dict
    plan_reflection: bool


def _core_module():
    """Return the orchestrator core module, importing lazily to avoid cycles."""

    from . import core as _core  # local import to prevent circular dependency

    return _core


def _get_orch_callable(name: str, fallback):
    """Return an orchestrator-level override when patched, else fallback."""

    orch_module = _sys.modules.get("clematis.engine.orchestrator")
    if orch_module is not None:
        attr = getattr(orch_module, name, None)
        if callable(attr):
            return attr
    return fallback


def _clone_ctx_for_agent(ctx: TurnCtx, agent_id: str, turn_id: int | str) -> TurnCtx:
    """Shallow clone of ``TurnCtx`` with agent/turn specialization."""
    base = {}
    for attr in ("cfg", "config", "now", "now_ms", "seed", "slice_idx", "slice_budgets"):
        if hasattr(ctx, attr):
            base[attr] = getattr(ctx, attr)
    base["agent_id"] = str(agent_id)
    base["turn_id"] = turn_id
    base["_dry_run_until_t4"] = True
    return _SNS(**base)


def _extract_dryrun_artifacts(ctx: TurnCtx) -> Tuple[list, str, dict, dict]:
    """Pull artifacts stashed by ``run_turn`` in dry-run mode."""
    t4_obj = getattr(ctx, "_dryrun_t4", None)
    deltas = list(getattr(t4_obj, "approved_deltas", []) or [])
    utter = str(getattr(ctx, "_dryrun_utter", "") or "")
    t1_info = getattr(ctx, "_dryrun_t1", {}) or {}
    t2_info = getattr(ctx, "_dryrun_t2", {}) or {}
    return deltas, utter, t1_info, t2_info


def _make_readonly_snapshot(state: Any) -> Any:
    """Produce a read-only snapshot, honoring orchestrator monkeypatches."""
    orch_module = _sys.modules.get("clematis.engine.orchestrator")
    if orch_module is not None:
        fn = getattr(orch_module, "_make_readonly_snapshot", None)
        if callable(fn) and fn is not _make_readonly_snapshot:
            return fn(state)
    return readonly_snapshot(state)


def _run_turn_compute(ctx: TurnCtx, base_state: Any, agent_id: str, input_text: str) -> _TurnBuffer:
    """Run stages up to T4 on a read-only snapshot, capturing staged logs."""
    turn_id = getattr(ctx, "turn_id", None)
    if turn_id is None:
        turn_id = int(time.time() * 1000) % 10_000_000

    subctx = _clone_ctx_for_agent(ctx, agent_id, turn_id)

    mux, token = _begin_log_capture()
    try:
        ro = _make_readonly_snapshot(base_state)
        core = _core_module()
        core.Orchestrator().run_turn(subctx, ro, input_text)
        deltas, utter, t1_info, t2_info = _extract_dryrun_artifacts(subctx)
        plan_reflection = bool(getattr(subctx, "_dryrun_plan_reflection", False))
        logs = mux.dump()
        buf: _TurnBuffer = {
            "turn_id": subctx.turn_id,
            "slice_idx": int(getattr(subctx, "slice_idx", 0) or 0),
            "agent_id": str(agent_id),
            "logs": logs,
            "deltas": deltas,
            "dialogue": utter,
            "graphs_touched": set(t1_info.get("graphs_touched", []) or []),
            "graph_versions": {},
            "t2_info": t2_info,
            "plan_reflection": plan_reflection,
        }
        return buf
    finally:
        _end_log_capture(token)


def _agents_parallel_enabled(ctx) -> bool:
    """Return True when agent-level parallel execution is allowed."""
    core = _core_module()
    cfg = core._get_cfg(ctx)
    p = (cfg.get("perf") or {}).get("parallel") or {}
    try:
        enabled = core._truthy(p.get("enabled", False))
        agents = core._truthy(p.get("agents", False))
        mw = int(p.get("max_workers", 1) or 1)
        return bool(enabled and agents and mw > 1)
    except Exception:
        return False


def _resolve_graphs_for_agent(state: Any, agent_id: str) -> set[str]:
    """Best-effort resolver for graphs an agent may touch."""
    try:
        agents = state.get("agents") if isinstance(state, dict) else getattr(state, "agents", None)
        if isinstance(agents, dict):
            a = agents.get(agent_id) or {}
            g = a.get("graphs") if isinstance(a, dict) else None
            if g is None and hasattr(a, "graphs"):
                g = getattr(a, "graphs")
            if g is not None:
                return {str(x) for x in g}
        gba = state.get("graphs_by_agent") if isinstance(state, dict) else getattr(state, "graphs_by_agent", None)
        if isinstance(gba, dict) and agent_id in gba:
            return {str(x) for x in (gba.get(agent_id) or [])}
    except Exception:
        pass
    return set()


def _select_independent_batch(agent_ids: List[str], state: Any, max_workers: int) -> List[str]:
    """Greedy selection of agents whose graph sets are pairwise disjoint."""
    picked: List[str] = []
    used: set[str] = set()
    limit = max(1, int(max_workers))
    for aid in agent_ids:
        if len(picked) >= limit:
            break
        gset = _resolve_graphs_for_agent(state, aid)
        if used.isdisjoint(gset):
            picked.append(aid)
            used.update(gset)
    return picked


def _sort_turn_buffers(buffers: List[_TurnBuffer]) -> List[_TurnBuffer]:
    """Sort buffers deterministically by ``(turn_id, slice_idx)``."""

    def _key(buf: _TurnBuffer) -> Tuple[int, str, int]:
        tid = buf.get("turn_id")
        try:
            return (0, int(tid), int(buf.get("slice_idx", 0)))
        except Exception:
            return (1, str(tid), int(buf.get("slice_idx", 0)))

    return sorted(buffers, key=_key)


def _run_agents_parallel_batch(
    ctx: TurnCtx, state: Dict[str, Any], tasks: List[Tuple[str, str]]
) -> List[TurnResult]:
    """Run a batch of ``(agent_id, input_text)`` with compute-then-commit semantics."""

    run_turn_compute = _get_orch_callable("_run_turn_compute", _run_turn_compute)

    if not _agents_parallel_enabled(ctx):
        results: List[TurnResult] = []
        core = _core_module()
        for aid, text in tasks:
            subctx = _clone_ctx_for_agent(ctx, aid, getattr(ctx, "turn_id", 0))
            setattr(subctx, "_dry_run_until_t4", False)
            results.append(core.Orchestrator().run_turn(subctx, state, text))
        return results

    stager = _get_logging_callable("enable_staging")()

    max_workers = int(((_core_module()._get_cfg(ctx).get("perf") or {}).get("parallel") or {}).get("max_workers", 2))
    agent_ids = [aid for aid, _ in tasks]
    picked = _select_independent_batch(agent_ids, state, max_workers)

    base = _make_readonly_snapshot(state)
    buffers: List[_TurnBuffer] = []
    for aid, text in tasks:
        if aid not in picked:
            continue
        buffers.append(run_turn_compute(ctx, base, aid, text))

    for buf in buffers:
        turn_id = buf["turn_id"]
        slice_idx = int(buf.get("slice_idx", 0) or 0)
        for file_path, payload in buf["logs"]:
            key = _get_logging_callable("default_key_for")(
                file_path=file_path, turn_id=turn_id, slice_idx=slice_idx
            )
            try:
                stager.stage(file_path, key, payload)
            except RuntimeError as exc:
                if str(exc) == "LOG_STAGING_BACKPRESSURE":
                    for rec in stager.drain_sorted():
                        _append_unbuffered(rec.file_path, rec.payload)
                    stager.stage(file_path, key, payload)
                else:
                    raise

    results: List[TurnResult] = []
    core = _core_module()
    apply_changes = _get_orch_callable("apply_changes", core.apply_changes)
    iso_from_ms = core._iso_from_ms
    for buf in _sort_turn_buffers(buffers):
        t4_like = _SNS(
            approved_deltas=list(buf["deltas"]),
            rejected_ops=[],
            reasons=["PR70_COMMIT"],
            metrics={"counts": {"approved": len(buf["deltas"])}},
        )
        apply = apply_changes(ctx, state, t4_like)
        apply_payload = {
            "turn": buf["turn_id"],
            "agent": buf["agent_id"],
            "applied": apply.applied,
            "clamps": apply.clamps,
            "version_etag": apply.version_etag,
            "snapshot": apply.snapshot_path,
            "cache_invalidations": int((getattr(apply, "metrics", {}) or {}).get("cache_invalidations", 0)),
            "ms": 0.0,
        }
        key = _get_logging_callable("default_key_for")(
            file_path="apply.jsonl",
            turn_id=buf["turn_id"],
            slice_idx=int(buf.get("slice_idx", 0) or 0),
        )
        payload = dict(apply_payload)
        if "now" in payload:
            if getattr(ctx, "now_ms", None) is not None:
                payload["now"] = iso_from_ms(int(ctx.now_ms))
            else:
                payload.pop("now", None)
        if _os.environ.get("CI", "").lower() == "true":
            payload["ms"] = 0.0
        try:
            stager.stage("apply.jsonl", key, payload)
        except RuntimeError as exc:
            if str(exc) == "LOG_STAGING_BACKPRESSURE":
                for rec in stager.drain_sorted():
                    _append_unbuffered(rec.file_path, rec.payload)
                stager.stage("apply.jsonl", key, payload)
            else:
                raise
        results.append(TurnResult(line=buf["dialogue"], events=[]))

        # PR79: invoke reflection after Apply in parallel path (no logging/writes here)
        try:
            core = _core_module()
            plan_like = _SNS(reflection=bool(buf.get("plan_reflection", False)))
            core._run_reflection_if_enabled(
                ctx=ctx,
                state=state,
                plan=plan_like,
                utter=buf.get("dialogue", ""),
                t2_obj=buf.get("t2_info", {}) or {},
            )
        except Exception:
            # Reflection must never break the turn
            pass

    for rec in stager.drain_sorted():
        _append_unbuffered(rec.file_path, rec.payload)
    _get_logging_callable("disable_staging")()

    return results
