from __future__ import annotations
from typing import Any, Dict
import time
from datetime import datetime, timezone
import os as _os


def _iso_from_ms(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000.0, tz=timezone.utc).isoformat()


from types import SimpleNamespace
from .types import TurnCtx, TurnResult
from .stages.t1 import t1_propagate
from .stages.t2 import t2_semantic
from .stages.t3 import (
    make_plan_bundle,
    make_dialog_bundle,
    deliberate,
    rag_once,
    speak,
    llm_speak,
    build_llm_prompt,
    emit_trace,
)
from .stages.t4 import t4_filter
from .apply import apply_changes
from .snapshot import load_latest_snapshot
from .gel import (
    observe_retrieval as gel_observe,
    tick as gel_tick,
    merge_candidates as gel_merge_candidates,
    apply_merge as gel_apply_merge,
    split_candidates as gel_split_candidates,
    apply_split as gel_apply_split,
    promote_clusters as gel_promote_clusters,
    apply_promotion as gel_apply_promotion,
)
from ..io.log import append_jsonl, _append_jsonl_unbuffered

# --- PR70: parallel driver helpers & readonly snapshot import ---
from .stages.state_clone import readonly_snapshot
from .util.logmux import LogMux, set_mux, reset_mux

from .util.io_logging import enable_staging, disable_staging, default_key_for


from .cache import CacheManager

# --- PR70: minimal compute/commit driver (not wired into run_turn default path) ---
from types import SimpleNamespace as _SNS


def _clone_ctx_for_agent(ctx: TurnCtx, agent_id: str, turn_id: int | str) -> TurnCtx:
    """Shallow clone of TurnCtx with agent/turn specialized; preserves cfg/now/seed.
    We use SimpleNamespace to avoid coupling to concrete ctx classes.
    """
    base = {}
    for attr in ("cfg", "config", "now", "now_ms", "seed", "slice_idx", "slice_budgets"):
        if hasattr(ctx, attr):
            base[attr] = getattr(ctx, attr)
    base["agent_id"] = str(agent_id)
    base["turn_id"] = turn_id
    # PR70: instruct run_turn to stop after T4 and record deltas/metrics
    base["_dry_run_until_t4"] = True
    return _SNS(**base)


def _extract_dryrun_artifacts(ctx: TurnCtx) -> tuple[list, str, dict, dict]:
    """Pull artifacts stashed by run_turn in dry-run mode.
    Returns (approved_deltas, utter, t1_info, t2_info).
    """
    t4_obj = getattr(ctx, "_dryrun_t4", None)
    deltas = list(getattr(t4_obj, "approved_deltas", []) or [])
    utter = str(getattr(ctx, "_dryrun_utter", "") or "")
    t1_info = getattr(ctx, "_dryrun_t1", {}) or {}
    t2_info = getattr(ctx, "_dryrun_t2", {}) or {}
    return deltas, utter, t1_info, t2_info


def _run_turn_compute(
    ctx: TurnCtx, base_state: Any, agent_id: str, input_text: str
) -> "_TurnBuffer":
    """Compute phase: run stages up to T4 on a read-only snapshot, capturing logs.
    The underlying `append_jsonl` is ctx-aware and will buffer to the active LogMux.
    """
    # Assign a deterministic, per-agent turn id if not present
    turn_id = getattr(ctx, "turn_id", None)
    if turn_id is None:
        # Fallback: derive from time monotonic; tests usually set turn_id
        turn_id = int(time.time() * 1000) % 10_000_000

    # Clone a per-agent ctx with dry-run enabled
    subctx = _clone_ctx_for_agent(ctx, agent_id, turn_id)

    # Activate log capture for this compute
    mux, token = _begin_log_capture()
    try:
        # IMPORTANT: read-only snapshot to avoid mutating the live state
        ro = _make_readonly_snapshot(base_state)
        # Run the existing sequential pipeline up to T4 (dry-run will early-return after T4)
        _ = Orchestrator().run_turn(subctx, ro, input_text)
        # Extract artifacts (approved deltas and utter)
        deltas, utter, t1_info, t2_info = _extract_dryrun_artifacts(subctx)
        logs = mux.dump()
        buf: dict = {
            "turn_id": subctx.turn_id,
            "slice_idx": int(getattr(subctx, "slice_idx", 0) or 0),
            "agent_id": str(agent_id),
            "logs": logs,
            "deltas": deltas,
            "dialogue": utter,
            "graphs_touched": set(t1_info.get("graphs_touched", []) or []),
            "graph_versions": {},
        }
        return buf
    finally:
        _end_log_capture(token)


def _run_agents_parallel_batch(
    ctx: TurnCtx, state: dict, tasks: list[tuple[str, str]]
) -> list[TurnResult]:
    """Run a batch of (agent_id, input_text) with compute-then-commit semantics.
    This function is gated by the caller; it does not alter default run_turn behavior.
    """
    # Gate: require agents parallel to be enabled
    if not _agents_parallel_enabled(ctx):
        # Fallback: sequential
        out: list[TurnResult] = []
        for aid, text in tasks:
            subctx = _clone_ctx_for_agent(ctx, aid, getattr(ctx, "turn_id", 0))
            # Disable dry-run for sequential path
            setattr(subctx, "_dry_run_until_t4", False)
            out.append(Orchestrator().run_turn(subctx, state, text))
        return out

    # PR71: centralize log writes via staging for deterministic ordering
    stager = enable_staging()

    # Select independent batch in input order
    max_workers = int(
        ((_get_cfg(ctx).get("perf") or {}).get("parallel") or {}).get("max_workers", 2)
    )
    agent_ids = [aid for aid, _ in tasks]
    picked = _select_independent_batch(agent_ids, state, max_workers)

    # Compute on a single read-only snapshot
    base = _make_readonly_snapshot(state)
    buffers: list = []
    for aid, text in tasks:
        if aid not in picked:
            continue
        buf = _run_turn_compute(ctx, base, aid, text)
        buffers.append(buf)

    # Stage all compute-phase logs first (no disk writes yet)
    for buf in buffers:
        turn_id = buf["turn_id"]
        slice_idx = int(buf.get("slice_idx", 0) or 0)
        for file_path, payload in buf["logs"]:
            key = default_key_for(file_path=file_path, turn_id=turn_id, slice_idx=slice_idx)
            try:
                stager.stage(file_path, key, payload)
            except RuntimeError as e:
                if str(e) == "LOG_STAGING_BACKPRESSURE":
                    # Backpressure is expected; drain deterministically and retry without surfacing stderr noise.
                    for rec in stager.drain_sorted():
                        _append_jsonl_unbuffered(rec.file_path, rec.payload)
                    stager.stage(file_path, key, payload)
                else:
                    raise

    # Apply deltas sequentially in deterministic order and stage apply logs
    results: list[TurnResult] = []
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
            "cache_invalidations": int(
                (getattr(apply, "metrics", {}) or {}).get("cache_invalidations", 0)
            ),
            "ms": 0.0,
        }
        key = default_key_for(
            file_path="apply.jsonl",
            turn_id=buf["turn_id"],
            slice_idx=int(buf.get("slice_idx", 0) or 0),
        )
        # PR76: stabilize identity fields before logging
        _ap = dict(apply_payload)
        # Deterministic timestamp: prefer ctx.now_ms; else drop 'now'
        if "now" in _ap:
            if getattr(ctx, "now_ms", None) is not None:
                _ap["now"] = _iso_from_ms(int(ctx.now_ms))
            else:
                _ap.pop("now", None)
        # Elapsed wall time is noisy; zero it on CI to keep byte identity
        if _os.environ.get("CI", "").lower() == "true":
            _ap["ms"] = 0.0
        try:
            stager.stage("apply.jsonl", key, _ap)
        except RuntimeError as e:
            if str(e) == "LOG_STAGING_BACKPRESSURE":
                # Same fallback as above: flush staged logs and retry, keeping stderr quiet and behavior deterministic.
                for rec in stager.drain_sorted():
                    _append_jsonl_unbuffered(rec.file_path, rec.payload)
                stager.stage("apply.jsonl", key, _ap)
            else:
                raise
        results.append(TurnResult(line=buf["dialogue"], events=[]))

    # Final drain and ordered writes
    for rec in stager.drain_sorted():
        _append_jsonl_unbuffered(rec.file_path, rec.payload)
    disable_staging()

    return results


# --- M5: Scheduler wiring (PR26 — helpers & scaffolding; yields gated) ---
from typing import TypedDict

_sched_next_turn = None
_sched_on_yield = None


def _maybe_load_scheduler():
    """Import engine scheduler lazily to avoid any side-effects in identity path."""
    global _sched_next_turn, _sched_on_yield
    if _sched_next_turn is not None or _sched_on_yield is not None:
        return
    try:
        # PR25 path: engine-local scheduler
        from .scheduler import next_turn as _nt, on_yield as _oy  # type: ignore

        _sched_next_turn, _sched_on_yield = _nt, _oy
    except Exception:
        _sched_next_turn, _sched_on_yield = None, None


class _SliceCtx(TypedDict):
    slice_idx: int
    started_ms: int
    budgets: Dict[str, int]
    agent_id: str


# --- PR70: Turn buffer type for parallel driver scaffolding ---
class _TurnBuffer(TypedDict):
    turn_id: int | str
    slice_idx: int
    agent_id: str
    logs: list[tuple[str, dict]]  # (stream, obj) pairs captured during compute
    deltas: Any  # placeholder for T4-approved deltas (PR70 wiring)
    dialogue: str  # the utterance string (if any)
    graphs_touched: set[str]
    graph_versions: Dict[str, str]


def _m5_enabled(ctx) -> bool:
    cfg = _get_cfg(ctx)
    s = (cfg.get("scheduler") if isinstance(cfg, dict) else {}) or {}
    return bool(s.get("enabled", False))


def _clock(ctx) -> int:
    fn = getattr(ctx, "now_ms", None)
    if callable(fn):
        try:
            return int(fn())
        except Exception:
            return 0
    return 0


def _derive_budgets(ctx) -> Dict[str, int]:
    cfg = _get_cfg(ctx)
    s = (cfg.get("scheduler") if isinstance(cfg, dict) else {}) or {}
    b = s.get("budgets") or {}
    out: Dict[str, int] = {}
    for k in ("t1_pops", "t1_iters", "t2_k", "t3_ops", "wall_ms"):
        v = b.get(k)
        if v is None:
            continue
        out[k] = int(v)
    out["quantum_ms"] = int(s.get("quantum_ms", 20))
    return out


def _should_yield(slice_ctx: _SliceCtx, consumed: Dict[str, int]) -> str | None:
    """
    Decide if a slice should yield based on budgets and elapsed time.
    Precedence: WALL_MS > BUDGET_* > QUANTUM_EXCEEDED.
    NOTE: PR26 scaffolding only logs; enforcement (early return) can be added later.
    """
    budgets = slice_ctx["budgets"]
    # elapsed since slice start
    elapsed_ms = consumed.get("ms", 0)
    # WALL first
    if "wall_ms" in budgets and elapsed_ms >= budgets["wall_ms"]:
        return "WALL_MS"
    # Budgets
    if budgets.get("t1_iters") is not None and consumed.get("t1_iters") == budgets.get("t1_iters"):
        return "BUDGET_T1_ITERS"
    if budgets.get("t1_pops") is not None and consumed.get("t1_pops") == budgets.get("t1_pops"):
        return "BUDGET_T1_POPS"
    if budgets.get("t2_k") is not None and consumed.get("t2_k") == budgets.get("t2_k"):
        return "BUDGET_T2_K"
    if budgets.get("t3_ops") is not None and consumed.get("t3_ops") == budgets.get("t3_ops"):
        return "BUDGET_T3_OPS"
    # Quantum last
    if elapsed_ms >= budgets.get("quantum_ms", 20):
        return "QUANTUM_EXCEEDED"
    return None


# --- PR27: Ready-Set hook (pure, deterministic; default always ready) ---
def agent_ready(ctx, state, agent_id: str) -> tuple[bool, str]:
    """
    Returns (is_ready, reason). Default returns (True, "DEFAULT_TRUE").
    Override or monkeypatch in tests to block specific agents deterministically.
    """
    return True, "DEFAULT_TRUE"


# --- Config accessor for harmonized config usage ---


def _get_cfg(ctx) -> Dict[str, Any]:
    """Normalize config access across ctx.cfg / ctx.config; always returns a dict."""
    cfg = getattr(ctx, "cfg", None)
    if isinstance(cfg, dict):
        return cfg
    if isinstance(cfg, SimpleNamespace):
        return dict(cfg.__dict__)
    cfg2 = getattr(ctx, "config", None)
    if isinstance(cfg2, dict):
        return cfg2
    if isinstance(cfg2, SimpleNamespace):
        return dict(cfg2.__dict__)
    return {}


# --- PR70 minimal driver helpers (scaffolding; not yet wired to run_turn) ---


def _resolve_graphs_for_agent(state: Any, agent_id: str) -> set[str]:
    """Best-effort resolver for graphs an agent may touch.
    Looks for common shapes on state; returns empty set if unknown.
    This keeps the selector safe and deterministic across environments.
    """
    try:
        # Common shapes we tolerate:
        # 1) state["agents"][agent_id]["graphs"] -> iterable of names
        agents = state.get("agents") if isinstance(state, dict) else getattr(state, "agents", None)
        if isinstance(agents, dict):
            a = agents.get(agent_id) or {}
            g = a.get("graphs") if isinstance(a, dict) else None
            if g is None and hasattr(a, "graphs"):
                g = getattr(a, "graphs")
            if g is not None:
                return {str(x) for x in g}
        # 2) state["graphs_by_agent"][agent_id] -> iterable
        gba = (
            state.get("graphs_by_agent")
            if isinstance(state, dict)
            else getattr(state, "graphs_by_agent", None)
        )
        if isinstance(gba, dict) and agent_id in gba:
            return {str(x) for x in (gba.get(agent_id) or [])}
    except Exception:
        pass
    return set()


def _select_independent_batch(agent_ids: list[str], state: Any, max_workers: int) -> list[str]:
    """Greedy selection of agents whose graph sets are pairwise disjoint.
    Deterministic order: preserves the input order; stops at max_workers.
    """
    picked: list[str] = []
    used: set[str] = set()
    for aid in agent_ids:
        if len(picked) >= max(1, int(max_workers)):
            break
        gset = _resolve_graphs_for_agent(state, aid)
        if used.isdisjoint(gset):
            picked.append(aid)
            used.update(gset)
    return picked


def _begin_log_capture() -> tuple[LogMux, object]:
    """Activate a LogMux for the current task and return (mux, token)."""
    mux = LogMux()
    token = set_mux(mux)
    return mux, token


def _end_log_capture(token: object) -> None:
    """Reset the active LogMux using the provided token."""
    try:
        reset_mux(token)
    except Exception:
        pass


def _sort_turn_buffers(buffers: list[_TurnBuffer]) -> list[_TurnBuffer]:
    """Sort buffers deterministically by (turn_id, slice_idx).
    turn_id is compared as int when possible, else as string.
    """

    def _key(b: _TurnBuffer):
        tid = b.get("turn_id")
        try:
            tid_i = int(tid)  # type: ignore[arg-type]
            return (0, tid_i, int(b.get("slice_idx", 0)))
        except Exception:
            return (1, str(tid), int(b.get("slice_idx", 0)))

    return sorted(buffers, key=_key)


def _make_readonly_snapshot(state: Any) -> Any:
    """Create a read-only snapshot facade used by the compute phase."""
    return readonly_snapshot(state)


# --- PR70: agent-level parallel driver gate ---


def _agents_parallel_enabled(ctx) -> bool:
    cfg = _get_cfg(ctx)
    p = (cfg.get("perf") or {}).get("parallel") or {}
    try:
        enabled = bool(p.get("enabled", False))
        agents = bool(p.get("agents", False))
        mw = int(p.get("max_workers", 1) or 1)
        return bool(enabled and agents and mw > 1)
    except Exception:
        return False


# --- Helper for pick reason passthrough to scheduler logs ---
def _get_pick_reason(ctx) -> str | None:
    """Return a driver-supplied pick reason if present on ctx.
    Looks for ctx._sched_pick_reason first, then ctx.pick_reason."""
    pr = getattr(ctx, "_sched_pick_reason", None)
    if pr is None:
        pr = getattr(ctx, "pick_reason", None)
    if pr is None:
        return None
    try:
        return str(pr)
    except Exception:
        return None


# --- Helper to write or capture scheduler events (PR29) ---


def _write_or_capture_scheduler_event(ctx, event: Dict[str, Any]) -> None:
    """If driver logging is enabled on ctx, capture event instead of writing.
    Otherwise, append to scheduler.jsonl.
    """
    try:
        if getattr(ctx, "_driver_writes_scheduler_log", False):
            cap = getattr(ctx, "_sched_capture", None)
            if isinstance(cap, dict):
                cap.clear()
                cap.update(event)
            return
    except Exception:
        # Fall back to writing
        pass
    append_jsonl("scheduler.jsonl", event)


class Orchestrator:
    """Single canonical turn loop with first-class observability.

    Stages: T1 (propagate) → T2 (semantic) → T3 (placeholder) → T4 (meta-filter) → Apply → Health
    This adds precise per-stage durations and a turn summary while keeping behavior unchanged.
    """

    def run_turn(self, ctx: TurnCtx, state: Dict[str, Any], input_text: str) -> TurnResult:
        turn_id = getattr(ctx, "turn_id", "-")
        agent_id = getattr(ctx, "agent_id", "-")
        now = getattr(ctx, "now", None)

        # PR70: allow a dry-run that stops after T4 to enable agent-level compute phase
        _dry_run = bool(getattr(ctx, "_dry_run_until_t4", False))

        total_t0 = time.perf_counter()

        # --- M5 slice init (PR26 scaffolding) ---
        sched_enabled = _m5_enabled(ctx)
        slice_ctx: _SliceCtx | None = None
        if sched_enabled:
            _maybe_load_scheduler()
            budgets = _derive_budgets(ctx)
            # If PR25 core is available, record the pick (no enforcement here).
            if _sched_next_turn is not None:
                try:
                    policy = str(
                        (_get_cfg(ctx).get("scheduler") or {}).get("policy", "round_robin")
                    )
                    fairness = (_get_cfg(ctx).get("scheduler") or {}).get("fairness", {}) or {}
                    # We don't maintain a global sched state here; external loop owns it.
                    # This call is only to keep logs consistent with policy naming.
                    _ = policy, fairness  # placeholders to avoid lints
                except Exception:
                    pass
            # Attach immutable per-slice budgets to ctx for stages to read.
            slice_idx_prev = int(getattr(ctx, "slice_idx", 0) or 0)
            slice_ctx = {
                "slice_idx": slice_idx_prev + 1,
                "started_ms": _clock(ctx),
                "budgets": budgets,
                "agent_id": agent_id,
            }
            setattr(ctx, "slice_idx", slice_ctx["slice_idx"])
            setattr(ctx, "slice_budgets", budgets)
        else:
            # Ensure stages see no scheduler caps when disabled
            if hasattr(ctx, "slice_budgets"):
                try:
                    delattr(ctx, "slice_budgets")
                except Exception:
                    setattr(ctx, "slice_budgets", None)

        # --- Boot hook: load latest snapshot once per process ---
        boot_loaded = (
            state.get("_boot_loaded", False)
            if isinstance(state, dict)
            else getattr(state, "_boot_loaded", False)
        )
        if not boot_loaded:
            try:
                load_latest_snapshot(ctx, state)
            except Exception:
                # Loader must never crash the turn; continue cleanly.
                pass
            finally:
                if isinstance(state, dict):
                    state["_boot_loaded"] = True
                else:
                    setattr(state, "_boot_loaded", True)

        # --- Cache manager bootstrap (version-aware, namespaced) ---
        full_cfg = _get_cfg(ctx)
        t4_cfg_for_cache = (full_cfg.get("t4") if isinstance(full_cfg, dict) else {}) or {}
        cache_cfg = t4_cfg_for_cache.get("cache", {}) if isinstance(t4_cfg_for_cache, dict) else {}
        cache_enabled = (
            bool(cache_cfg.get("enabled", True)) if isinstance(cache_cfg, dict) else False
        )
        cm_existing = (
            state.get("_cache_mgr")
            if isinstance(state, dict)
            else getattr(state, "_cache_mgr", None)
        )
        if cache_enabled and cm_existing is None:
            max_entries = int(cache_cfg.get("max_entries", 512))
            ttl_conf = cache_cfg.get("ttl_sec", cache_cfg.get("ttl_s", 600))
            ttl_sec = int(ttl_conf if ttl_conf is not None else 600)
            cm_new = CacheManager(max_entries=max_entries, ttl_sec=ttl_sec)
            if isinstance(state, dict):
                state["_cache_mgr"] = cm_new
            else:
                setattr(state, "_cache_mgr", cm_new)

        # --- T1 ---
        t0 = time.perf_counter()
        t1 = t1_propagate(ctx, state, input_text)
        t1_ms = round((time.perf_counter() - t0) * 1000.0, 3)
        append_jsonl(
            "t1.jsonl",
            {
                "turn": turn_id,
                "agent": agent_id,
                **t1.metrics,
                "ms": t1_ms,
                **({"now": now} if now else {}),
            },
        )
        # --- M5 liminilas check after T1 ---
        if slice_ctx is not None:
            consumed = {
                "ms": int(round((time.perf_counter() - total_t0) * 1000.0)),
            }
            # Map T1 metrics if there
            try:
                if isinstance(t1.metrics, dict):
                    if t1.metrics.get("iters") is not None:
                        consumed["t1_iters"] = int(t1.metrics.get("iters"))
                    if t1.metrics.get("pops") is not None:
                        consumed["t1_pops"] = int(t1.metrics.get("pops"))
            except Exception:
                pass
            reason = _should_yield(slice_ctx, consumed)
            if reason:
                event = {
                    "turn": turn_id,
                    "slice": slice_ctx["slice_idx"],
                    "agent": agent_id,
                    "policy": (_get_cfg(ctx).get("scheduler") or {}).get("policy", "round_robin"),
                    **({"pick_reason": _get_pick_reason(ctx)} if _get_pick_reason(ctx) else {}),
                    "reason": reason,
                    "enforced": True,
                    "stage_end": "T1",
                    "quantum_ms": slice_ctx["budgets"].get("quantum_ms"),
                    "wall_ms": slice_ctx["budgets"].get("wall_ms"),
                    "budgets": {k: v for k, v in slice_ctx["budgets"].items() if k != "quantum_ms"},
                    "consumed": consumed,
                    "queued": [],  # external loop owns the queue
                    "ms": 0,
                }
                _write_or_capture_scheduler_event(ctx, event)
                total_ms_now = round((time.perf_counter() - total_t0) * 1000.0, 3)
                append_jsonl(
                    "turn.jsonl",
                    {
                        "turn": turn_id,
                        "agent": agent_id,
                        "durations_ms": {
                            "t1": t1_ms,
                            "t2": 0.0,
                            "t4": 0.0,
                            "apply": 0.0,
                            "total": total_ms_now,
                        },
                        "t1": {
                            "pops": t1.metrics.get("pops"),
                            "iters": t1.metrics.get("iters"),
                            "graphs_touched": t1.metrics.get("graphs_touched"),
                        },
                        "t2": {},
                        "t4": {},
                        "slice_idx": slice_ctx["slice_idx"],
                        "yielded": True,
                        "yield_reason": reason,
                        **({"now": now} if now else {}),
                    },
                )
                return TurnResult(line="", events=[])

        # --- T2 (with version-aware cache) ---
        t0 = time.perf_counter()
        # Resolve cache manager
        cm = (
            state.get("_cache_mgr")
            if isinstance(state, dict)
            else getattr(state, "_cache_mgr", None)
        )
        ver = (
            state.get("version_etag")
            if isinstance(state, dict)
            else getattr(state, "version_etag", None)
        ) or "0"
        ns = "t2:semantic"
        key = (ver, str(input_text))

        cache_hit = False
        if cm is not None:
            hit, cached = cm.get(ns, key)
            if hit:
                t2 = cached
                cache_hit = True
            else:
                t2 = t2_semantic(ctx, state, input_text, t1)
                cm.set(ns, key, t2)
        else:
            t2 = t2_semantic(ctx, state, input_text, t1)

        t2_ms = round((time.perf_counter() - t0) * 1000.0, 3)
        append_jsonl(
            "t2.jsonl",
            {
                "turn": turn_id,
                "agent": agent_id,
                **(t2.metrics if isinstance(getattr(t2, "metrics", None), dict) else {}),
                **({"cache_hit": cache_hit} if cm is not None else {}),
                **({"cache_size": (cm.stats.get("size", 0))} if cm is not None else {}),
                "ms": t2_ms,
                **({"now": now} if now else {}),
            },
        )
        # --- M5 boundary check after T2 ---
        if slice_ctx is not None:
            consumed = {
                "ms": int(round((time.perf_counter() - total_t0) * 1000.0)),
            }
            try:
                m = getattr(t2, "metrics", {}) or {}
                if m.get("k_used") is not None:
                    consumed["t2_k"] = int(m.get("k_used"))
            except Exception:
                pass
            reason = _should_yield(slice_ctx, consumed)
            if reason:
                event = {
                    "turn": turn_id,
                    "slice": slice_ctx["slice_idx"],
                    "agent": agent_id,
                    "policy": (_get_cfg(ctx).get("scheduler") or {}).get("policy", "round_robin"),
                    **({"pick_reason": _get_pick_reason(ctx)} if _get_pick_reason(ctx) else {}),
                    "reason": reason,
                    "enforced": True,
                    "stage_end": "T2",
                    "quantum_ms": slice_ctx["budgets"].get("quantum_ms"),
                    "wall_ms": slice_ctx["budgets"].get("wall_ms"),
                    "budgets": {k: v for k, v in slice_ctx["budgets"].items() if k != "quantum_ms"},
                    "consumed": consumed,
                    "queued": [],
                    "ms": 0,
                }
                _write_or_capture_scheduler_event(ctx, event)
                total_ms_now = round((time.perf_counter() - total_t0) * 1000.0, 3)
                append_jsonl(
                    "turn.jsonl",
                    {
                        "turn": turn_id,
                        "agent": agent_id,
                        "durations_ms": {
                            "t1": t1_ms,
                            "t2": t2_ms,
                            "t4": 0.0,
                            "apply": 0.0,
                            "total": total_ms_now,
                        },
                        "t1": {
                            "pops": t1.metrics.get("pops"),
                            "iters": t1.metrics.get("iters"),
                            "graphs_touched": t1.metrics.get("graphs_touched"),
                        },
                        "t2": {
                            "k_returned": t2.metrics.get("k_returned"),
                            "k_used": t2.metrics.get("k_used"),
                            "cache_hit": bool(cache_hit),
                        },
                        "t4": {},
                        "slice_idx": slice_ctx["slice_idx"],
                        "yielded": True,
                        "yield_reason": reason,
                        **({"now": now} if now else {}),
                    },
                )
                return TurnResult(line="", events=[])

        # --- GEL observe (optional; gated by graph.enabled) ---
        graph_cfg_all = (
            _get_cfg(ctx).get("graph") if isinstance(_get_cfg(ctx), dict) else {}
        ) or {}
        graph_enabled = (
            bool(graph_cfg_all.get("enabled", False)) if isinstance(graph_cfg_all, dict) else False
        )
        if graph_enabled and not _dry_run:
            t0_gel = time.perf_counter()
            items = getattr(t2, "retrieved", []) or []  # gel adapts dicts/objs/tuples
            try:
                turn_idx = int(turn_id)
            except Exception:
                turn_idx = None
            gel_metrics = gel_observe(ctx, state, items, turn=turn_idx, agent=agent_id)
            gel_ms = round((time.perf_counter() - t0_gel) * 1000.0, 3)
            append_jsonl(
                "gel.jsonl",
                {
                    "turn": turn_id,
                    "agent": agent_id,
                    **gel_metrics,
                    "ms": gel_ms,
                    **({"now": now} if now else {}),
                },
            )

        # --- T3 (deliberation → optional one-shot RAG → dialogue) ---
        # All pure stage functions; only logging here does I/O.
        t0 = time.perf_counter()
        bundle = make_plan_bundle(ctx, state, t1, t2)
        # Allow tests to monkeypatch a module-level t3_deliberate(ctx, state, bundle)
        delib_fn = globals().get("t3_deliberate")
        if callable(delib_fn):
            plan = delib_fn(ctx, state, bundle)
        else:
            plan = deliberate(bundle)
        plan_ms = round((time.perf_counter() - t0) * 1000.0, 3)
        # --- M5 boundary check after T3 (plan) ---
        if slice_ctx is not None:
            consumed = {
                "ms": int(round((time.perf_counter() - total_t0) * 1000.0)),
            }
            try:
                ops_count = sum(1 for _ in (getattr(plan, "ops", []) or []))
                consumed["t3_ops"] = int(ops_count)
            except Exception:
                pass
            reason = _should_yield(slice_ctx, consumed)
            if reason:
                event = {
                    "turn": turn_id,
                    "slice": slice_ctx["slice_idx"],
                    "agent": agent_id,
                    "policy": (_get_cfg(ctx).get("scheduler") or {}).get("policy", "round_robin"),
                    **({"pick_reason": _get_pick_reason(ctx)} if _get_pick_reason(ctx) else {}),
                    "reason": reason,
                    "enforced": True,
                    "stage_end": "T3",
                    "quantum_ms": slice_ctx["budgets"].get("quantum_ms"),
                    "wall_ms": slice_ctx["budgets"].get("wall_ms"),
                    "budgets": {k: v for k, v in slice_ctx["budgets"].items() if k != "quantum_ms"},
                    "consumed": consumed,
                    "queued": [],
                    "ms": 0,
                }
                _write_or_capture_scheduler_event(ctx, event)
                total_ms_now = round((time.perf_counter() - total_t0) * 1000.0, 3)
                append_jsonl(
                    "turn.jsonl",
                    {
                        "turn": turn_id,
                        "agent": agent_id,
                        "durations_ms": {
                            "t1": t1_ms,
                            "t2": t2_ms,
                            "t4": 0.0,
                            "apply": 0.0,
                            "total": total_ms_now,
                        },
                        "t1": {
                            "pops": t1.metrics.get("pops"),
                            "iters": t1.metrics.get("iters"),
                            "graphs_touched": t1.metrics.get("graphs_touched"),
                        },
                        "t2": {
                            "k_returned": t2.metrics.get("k_returned"),
                            "k_used": t2.metrics.get("k_used"),
                            "cache_hit": bool(cache_hit),
                        },
                        "t4": {},
                        "slice_idx": slice_ctx["slice_idx"],
                        "yielded": True,
                        "yield_reason": reason,
                        **({"now": now} if now else {}),
                    },
                )
                return TurnResult(line="", events=[])

        # RAG: allow at most one refinement if both requested and enabled by config
        cfg = _get_cfg(ctx)
        t3cfg = cfg.get("t3", {}) if isinstance(cfg, dict) else {}
        max_rag_loops = int(t3cfg.get("max_rag_loops", 1)) if isinstance(t3cfg, dict) else 1

        def _retrieve_fn(payload: Dict[str, Any]) -> Dict[str, Any]:
            # Deterministic wrapper around T2: re-run with the provided query; map to the shape rag_once expects.
            q = str(payload.get("query") or input_text)
            t2_alt = t2_semantic(ctx, state, q, t1)
            # Normalize retrieved to list[dict]
            hits = []
            for r in getattr(t2_alt, "retrieved", []) or []:
                if isinstance(r, dict):
                    hits.append(
                        {
                            "id": str(r.get("id")),
                            "score": float(r.get("_score", r.get("score", 0.0)) or 0.0),
                            "owner": str(r.get("owner", "any")),
                            "quarter": str(r.get("quarter", "")),
                        }
                    )
                else:
                    rid = str(getattr(r, "id", ""))
                    if not rid:
                        continue
                    hits.append(
                        {
                            "id": rid,
                            "score": float(getattr(r, "score", 0.0) or 0.0),
                            "owner": str(getattr(r, "owner", "any")),
                            "quarter": str(getattr(r, "quarter", "")),
                        }
                    )
            return {"retrieved": hits, "metrics": getattr(t2_alt, "metrics", {})}

        requested_retrieve = any(
            getattr(op, "kind", None) == "RequestRetrieve" for op in getattr(plan, "ops", []) or []
        )
        rag_metrics = {
            "rag_used": False,
            "rag_blocked": False,
            "pre_s_max": 0.0,
            "post_s_max": 0.0,
            "k_retrieved": 0,
            "owner": None,
            "tier_pref": None,
        }
        if requested_retrieve and max_rag_loops >= 1:
            t0_rag = time.perf_counter()
            plan, rag_metrics = rag_once(bundle, plan, _retrieve_fn, already_used=False)
            rag_ms = round((time.perf_counter() - t0_rag) * 1000.0, 3)
        else:
            rag_ms = 0.0

        # Dialogue synthesis (rule-based vs optional LLM backend)
        t0 = time.perf_counter()
        dialog_bundle = make_dialog_bundle(ctx, state, t1, t2, plan)

        prompt_text = build_llm_prompt(dialog_bundle, plan)

        trace_meta: Dict[str, Any] = {}
        state_logs = None
        if isinstance(state, dict):
            state_logs = state.get("logs")
            if not isinstance(state_logs, list):
                state_logs = []
                state["logs"] = state_logs
        else:
            state_logs = getattr(state, "logs", None)
            if not isinstance(state_logs, list):
                try:
                    state_logs = []
                    setattr(state, "logs", state_logs)
                except Exception:
                    state_logs = None
        if isinstance(state_logs, list):
            trace_meta["state_logs"] = state_logs

        trace_reason = getattr(ctx, "trace_reason", None)
        if trace_reason is None and isinstance(ctx, dict):
            trace_reason = ctx.get("trace_reason")
        if trace_reason is not None:
            trace_meta["trace_reason"] = trace_reason

        emit_trace(dialog_bundle.get("cfg", {}), prompt_text, dialog_bundle, trace_meta)

        # Backend selection
        backend_cfg = (
            str(t3cfg.get("backend", "rulebased")) if isinstance(t3cfg, dict) else "rulebased"
        )
        llm_cfg = t3cfg.get("llm", {}) if isinstance(t3cfg, dict) else {}
        adapter = (
            state.get("llm_adapter", None)
            if isinstance(state, dict)
            else getattr(state, "llm_adapter", None)
        ) or getattr(ctx, "llm_adapter", None)

        backend_used = "rulebased"
        backend_fallback = None
        fallback_reason = None

        # let tests monkeypatch a module-level t3_dialogue with flexible signatures for speed and laziness.
        dlg_fn = globals().get("t3_dialogue")
        if callable(dlg_fn):
            try:
                # pref sig: (dialog_bundle, plan)
                res = dlg_fn(dialog_bundle, plan)
            except TypeError:
                # Fallback signature used by some tests: (ctx, state, dialog_bundle)
                res = dlg_fn(ctx, state, dialog_bundle)
            # normaliaztion: support returning just a string or (utter, metrics)
            if isinstance(res, tuple):
                utter = res[0]
                speak_metrics = res[1] if len(res) > 1 and isinstance(res[1], dict) else {}
            else:
                utter = res
                speak_metrics = {}
            backend_used = "patched"
        else:
            if backend_cfg == "llm" and adapter is not None:
                utter, speak_metrics = llm_speak(dialog_bundle, plan, adapter)
                backend_used = "llm"
            else:
                utter, speak_metrics = speak(dialog_bundle, plan)
                if backend_cfg == "llm" and adapter is None:
                    backend_fallback, fallback_reason = "rulebased", "no_adapter"
                elif backend_cfg not in ("rulebased", "llm"):
                    backend_fallback, fallback_reason = "rulebased", "invalid_backend"

        speak_ms = round((time.perf_counter() - t0) * 1000.0, 3)

        # Plan logging
        ops_counts: Dict[str, int] = {}
        for op in getattr(plan, "ops", []) or []:
            k = getattr(op, "kind", None)
            ops_counts[k] = ops_counts.get(k, 0) + 1

        policy_backend = (
            str(t3cfg.get("backend", "rulebased")) if isinstance(t3cfg, dict) else "rulebased"
        )
        append_jsonl(
            "t3_plan.jsonl",
            {
                "turn": turn_id,
                "agent": agent_id,
                "policy_backend": policy_backend,
                "backend": backend_used,
                **(
                    {"backend_fallback": backend_fallback, "fallback_reason": fallback_reason}
                    if backend_fallback
                    else {}
                ),
                "ops_counts": ops_counts,
                "requested_retrieve": bool(requested_retrieve),
                "rag_used": bool(rag_metrics.get("rag_used", False)),
                "reflection": bool(getattr(plan, "reflection", False)),
                "ms_deliberate": plan_ms,
                "ms_rag": rag_ms,
                **({"now": now} if now else {}),
            },
        )

        # Dialogue logging
        dlg_extra = {}
        if backend_used == "llm":
            adapter_name = getattr(
                adapter,
                "name",
                adapter.__class__.__name__ if hasattr(adapter, "__class__") else "Unknown",
            )
            model = str(llm_cfg.get("model", ""))
            temperature = float(llm_cfg.get("temperature", 0.2))
            dlg_extra.update(
                {
                    "backend": "llm",
                    "adapter": adapter_name,
                    "model": model,
                    "temperature": temperature,
                }
            )
        else:
            dlg_extra.update({"backend": "rulebased"})

        append_jsonl(
            "t3_dialogue.jsonl",
            {
                "turn": turn_id,
                "agent": agent_id,
                "tokens": int(speak_metrics.get("tokens", 0)),
                "truncated": bool(speak_metrics.get("truncated", False)),
                "style_prefix_used": bool(speak_metrics.get("style_prefix_used", False)),
                "snippet_count": int(speak_metrics.get("snippet_count", 0)),
                "ms": speak_ms,
                **dlg_extra,
                **({"now": now} if now else {}),
            },
        )

        # the kill switch (t4.enabled). Default True if unspecified.
        t4_cfg_full = (_get_cfg(ctx).get("t4") if isinstance(_get_cfg(ctx), dict) else {}) or {}
        t4_enabled = (
            bool(t4_cfg_full.get("enabled", True)) if isinstance(t4_cfg_full, dict) else True
        )

        # --- T4 / Apply (it honousr kill switch dw tm abt it) ---
        if t4_enabled:
            # --- T4 ---
            t0 = time.perf_counter()
            t4 = t4_filter(ctx, state, t1, t2, plan, utter)
            t4_ms = round((time.perf_counter() - t0) * 1000.0, 3)
            append_jsonl(
                "t4.jsonl",
                {
                    "turn": turn_id,
                    "agent": agent_id,
                    **t4.metrics,
                    "approved": len(getattr(t4, "approved_deltas", [])),
                    "rejected": len(getattr(t4, "rejected_ops", [])),
                    "reasons": getattr(t4, "reasons", []),
                    "ms": t4_ms,
                    **({"now": now} if now else {}),
                },
            )
            # PR70: in dry-run mode, record artifacts and return before Apply/Health
            if _dry_run:
                try:
                    setattr(ctx, "_dryrun_t4", t4)
                    setattr(ctx, "_dryrun_utter", utter if 'utter' in locals() else "")
                    # also stash a few fields used by the batch driver
                    setattr(
                        ctx,
                        "_dryrun_t1",
                        {
                            "pops": t1.metrics.get("pops"),
                            "iters": t1.metrics.get("iters"),
                            "graphs_touched": t1.metrics.get("graphs_touched"),
                        },
                    )
                    setattr(
                        ctx,
                        "_dryrun_t2",
                        {
                            "k_returned": t2.metrics.get("k_returned"),
                            "k_used": t2.metrics.get("k_used"),
                            "cache_hit": bool(cache_hit),
                        },
                    )
                except Exception:
                    pass
                return TurnResult(line=utter if 'utter' in locals() else "", events=[])
            # --- M5 boundary check after T4 ---
            if slice_ctx is not None:
                consumed = {
                    "ms": int(round((time.perf_counter() - total_t0) * 1000.0)),
                }
                reason = _should_yield(slice_ctx, consumed)
                if reason:
                    event = {
                        "turn": turn_id,
                        "slice": slice_ctx["slice_idx"],
                        "agent": agent_id,
                        "policy": (_get_cfg(ctx).get("scheduler") or {}).get(
                            "policy", "round_robin"
                        ),
                        **({"pick_reason": _get_pick_reason(ctx)} if _get_pick_reason(ctx) else {}),
                        "reason": reason,
                        "enforced": True,
                        "stage_end": "T4",
                        "quantum_ms": slice_ctx["budgets"].get("quantum_ms"),
                        "wall_ms": slice_ctx["budgets"].get("wall_ms"),
                        "budgets": {
                            k: v for k, v in slice_ctx["budgets"].items() if k != "quantum_ms"
                        },
                        "consumed": consumed,
                        "queued": [],
                        "ms": 0,
                    }
                    _write_or_capture_scheduler_event(ctx, event)
                    total_ms_now = round((time.perf_counter() - total_t0) * 1000.0, 3)
                    append_jsonl(
                        "turn.jsonl",
                        {
                            "turn": turn_id,
                            "agent": agent_id,
                            "durations_ms": {
                                "t1": t1_ms,
                                "t2": t2_ms,
                                "t4": t4_ms,
                                "apply": 0.0,
                                "total": total_ms_now,
                            },
                            "t1": {
                                "pops": t1.metrics.get("pops"),
                                "iters": t1.metrics.get("iters"),
                                "graphs_touched": t1.metrics.get("graphs_touched"),
                            },
                            "t2": {
                                "k_returned": t2.metrics.get("k_returned"),
                                "k_used": t2.metrics.get("k_used"),
                                "cache_hit": bool(cache_hit),
                            },
                            "t4": {
                                "approved": len(getattr(t4, "approved_deltas", [])),
                                "rejected": len(getattr(t4, "rejected_ops", [])),
                            },
                            "slice_idx": slice_ctx["slice_idx"],
                            "yielded": True,
                            "yield_reason": reason,
                            **({"now": now} if now else {}),
                        },
                    )
                    return TurnResult(line=utter if 'utter' in locals() else "", events=[])
            # --- GEL decay tick (optional, might learn a different mechanism; run before Apply so snapshot includes decay) ---
            graph_cfg_all2 = (
                _get_cfg(ctx).get("graph") if isinstance(_get_cfg(ctx), dict) else {}
            ) or {}
            graph_enabled2 = (
                bool(graph_cfg_all2.get("enabled", False))
                if isinstance(graph_cfg_all2, dict)
                else False
            )
            if graph_enabled2:
                t0_decay = time.perf_counter()
                try:
                    turn_idx2 = int(turn_id)
                except Exception:
                    turn_idx2 = None
                decay_metrics = gel_tick(ctx, state, decay_dt=1, turn=turn_idx2, agent=agent_id)
                decay_ms = round((time.perf_counter() - t0_decay) * 1000.0, 3)
                append_jsonl(
                    "gel.jsonl",
                    {
                        "turn": turn_id,
                        "agent": agent_id,
                        **decay_metrics,
                        "ms": decay_ms,
                        **({"now": now} if now else {}),
                    },
                )

            # --- PR24: GEL merge/split/promotion (optional; deterministic, bounded) ---
            if graph_enabled2:
                try:
                    mg_cfg = (
                        graph_cfg_all2.get("merge") if isinstance(graph_cfg_all2, dict) else {}
                    ) or {}
                    sp_cfg = (
                        graph_cfg_all2.get("split") if isinstance(graph_cfg_all2, dict) else {}
                    ) or {}
                    pr_cfg = (
                        graph_cfg_all2.get("promotion") if isinstance(graph_cfg_all2, dict) else {}
                    ) or {}

                    do_merge = bool(mg_cfg.get("enabled", False))
                    do_split = bool(sp_cfg.get("enabled", False))
                    do_promo = bool(pr_cfg.get("enabled", False))

                    if do_merge or do_split or do_promo:
                        t0_msp = time.perf_counter()
                        merge_attempts = split_attempts = 0
                        merge_applied = split_applied = promo_applied = 0

                        merges = gel_merge_candidates(ctx, state) if do_merge else []
                        merge_attempts = len(merges)
                        if do_merge:
                            cap_m = int(mg_cfg.get("cap_per_turn", 4))
                            for m in merges[:cap_m]:
                                gel_apply_merge(ctx, state, m)
                                merge_applied += 1

                        splits = gel_split_candidates(ctx, state) if do_split else []
                        split_attempts = len(splits)
                        if do_split:
                            cap_s = int(sp_cfg.get("cap_per_turn", 4))
                            for s in splits[:cap_s]:
                                gel_apply_split(ctx, state, s)
                                split_applied += 1

                        if do_promo:
                            # promotions of clusters into concept graphs derive from current merge candidates (metadata-based policies can refine later)
                            clusters = merges if do_merge else []
                            promos = gel_promote_clusters(ctx, state, clusters)
                            cap_p = int(pr_cfg.get("cap_per_turn", 2))
                            for p in promos[:cap_p]:
                                gel_apply_promotion(ctx, state, p)
                                promo_applied += 1

                        msp_ms = round((time.perf_counter() - t0_msp) * 1000.0, 3)
                        append_jsonl(
                            "gel.jsonl",
                            {
                                "turn": turn_id,
                                "agent": agent_id,
                                "merge_attempts": merge_attempts,
                                "merge_applied": merge_applied,
                                "split_attempts": split_attempts,
                                "split_applied": split_applied,
                                "promotion_applied": promo_applied,
                                "ms": msp_ms,
                                **({"now": now} if now else {}),
                            },
                        )
                except Exception:
                    # Never let optional ggelly features break the turn
                    pass
            # --- Apply & persist ---
            t0 = time.perf_counter()
            apply = apply_changes(ctx, state, t4)
            apply_ms = round((time.perf_counter() - t0) * 1000.0, 3)
            _ap = {
                "turn": turn_id,
                "agent": agent_id,
                "applied": apply.applied,
                "clamps": apply.clamps,
                "version_etag": apply.version_etag,
                "snapshot": apply.snapshot_path,
                "cache_invalidations": int(
                    (getattr(apply, "metrics", {}) or {}).get("cache_invalidations", 0)
                ),
                "ms": apply_ms,
                **({"now": now} if now else {}),
            }
            # PR76: stabilize identity fields before logging
            if "now" in _ap:
                if getattr(ctx, "now_ms", None) is not None:
                    _ap["now"] = _iso_from_ms(int(ctx.now_ms))
                else:
                    _ap.pop("now", None)
            if _os.environ.get("CI", "").lower() == "true":
                _ap["ms"] = 0.0
            append_jsonl("apply.jsonl", _ap)
            # --- M5 boundary check after Apply ---
            if slice_ctx is not None:
                consumed = {
                    "ms": int(round((time.perf_counter() - total_t0) * 1000.0)),
                }
                reason = _should_yield(slice_ctx, consumed)
                if reason:
                    event = {
                        "turn": turn_id,
                        "slice": slice_ctx["slice_idx"],
                        "agent": agent_id,
                        "policy": (_get_cfg(ctx).get("scheduler") or {}).get(
                            "policy", "round_robin"
                        ),
                        **({"pick_reason": _get_pick_reason(ctx)} if _get_pick_reason(ctx) else {}),
                        "reason": reason,
                        "enforced": True,
                        "stage_end": "Apply",
                        "quantum_ms": slice_ctx["budgets"].get("quantum_ms"),
                        "wall_ms": slice_ctx["budgets"].get("wall_ms"),
                        "budgets": {
                            k: v for k, v in slice_ctx["budgets"].items() if k != "quantum_ms"
                        },
                        "consumed": consumed,
                        "queued": [],
                        "ms": 0,
                    }
                    _write_or_capture_scheduler_event(ctx, event)
                    total_ms_now = round((time.perf_counter() - total_t0) * 1000.0, 3)
                    append_jsonl(
                        "turn.jsonl",
                        {
                            "turn": turn_id,
                            "agent": agent_id,
                            "durations_ms": {
                                "t1": t1_ms,
                                "t2": t2_ms,
                                "t4": t4_ms,
                                "apply": apply_ms,
                                "total": total_ms_now,
                            },
                            "t1": {
                                "pops": t1.metrics.get("pops"),
                                "iters": t1.metrics.get("iters"),
                                "graphs_touched": t1.metrics.get("graphs_touched"),
                            },
                            "t2": {
                                "k_returned": t2.metrics.get("k_returned"),
                                "k_used": t2.metrics.get("k_used"),
                                "cache_hit": bool(cache_hit),
                            },
                            "t4": {
                                "approved": len(getattr(t4, "approved_deltas", [])),
                                "rejected": len(getattr(t4, "rejected_ops", [])),
                            },
                            "slice_idx": slice_ctx["slice_idx"],
                            "yielded": True,
                            "yield_reason": reason,
                            **({"now": now} if now else {}),
                        },
                    )
                    return TurnResult(line=utter if 'utter' in locals() else "", events=[])
        else:
            # bypass case: give us inert placeholders; no t4/apply logs
            t4_ms = 0.0
            apply_ms = 0.0
            t4 = SimpleNamespace(
                approved_deltas=[],
                rejected_ops=[],
                reasons=["T4_DISABLED"],
                metrics={"counts": {"approved": 0}},
            )
            apply = SimpleNamespace(
                applied=0,
                clamps=0,
                version_etag=getattr(state, "version_etag", None),
                snapshot_path=None,
            )

        # --- Health summary + per-turn rollup ---
        from . import health

        health.check_and_log(ctx, state, t1, t2, t4, apply, append_jsonl)

        total_ms = round((time.perf_counter() - total_t0) * 1000.0, 3)
        append_jsonl(
            "turn.jsonl",
            {
                "turn": turn_id,
                "agent": agent_id,
                "durations_ms": {
                    "t1": t1_ms,
                    "t2": t2_ms,
                    "t4": t4_ms,
                    "apply": apply_ms,
                    "total": total_ms,
                },
                "t1": {
                    "pops": t1.metrics.get("pops"),
                    "iters": t1.metrics.get("iters"),
                    "graphs_touched": t1.metrics.get("graphs_touched"),
                },
                "t2": {
                    "k_returned": t2.metrics.get("k_returned"),
                    "k_used": t2.metrics.get("k_used"),
                    "cache_hit": bool(cache_hit),
                },
                "t4": {
                    "approved": len(getattr(t4, "approved_deltas", [])),
                    "rejected": len(getattr(t4, "rejected_ops", [])),
                },
                **(
                    {
                        "slice_idx": (slice_ctx["slice_idx"] if slice_ctx is not None else None),
                        "yielded": False,
                    }
                    if slice_ctx is not None
                    else {}
                ),
                **({"now": now} if now else {}),
            },
        )

        return TurnResult(line=utter, events=[])


def run_turn(ctx: TurnCtx, state: Dict[str, Any], input_text: str) -> TurnResult:
    """
    Thin wrapper so external callers/tests can use a stable API without
    instantiating the Orchestrator class explicitly.
    """
    return Orchestrator().run_turn(ctx, state, input_text)
