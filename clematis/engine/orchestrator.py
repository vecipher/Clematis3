from __future__ import annotations
from typing import Any, Dict
import time
from types import SimpleNamespace
from .types import TurnCtx, TurnResult
from .stages.t1 import t1_propagate
from .stages.t2 import t2_semantic
from .stages.t3 import make_plan_bundle, make_dialog_bundle, deliberate, rag_once, speak, llm_speak
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
from ..io.log import append_jsonl

from .cache import CacheManager

# --- M5: Scheduler wiring (PR26 — helpers & scaffolding; yields gated) ---
from typing import TypedDict
try:
    # Core from PR25 (pure, deterministic)
    from clematis.scheduler import next_turn as _sched_next_turn, on_yield as _sched_on_yield
except Exception:
    _sched_next_turn = None
    _sched_on_yield = None

class _SliceCtx(TypedDict):
    slice_idx: int
    started_ms: int
    budgets: Dict[str, int]
    agent_id: str

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
    b = (s.get("budgets") or {})
    out: Dict[str, int] = {}
    for k in ("t1_pops","t1_iters","t2_k","t3_ops","wall_ms"):
        v = b.get(k)
        if v is None:
            continue
        out[k] = int(v)
    out["quantum_ms"] = int(s.get("quantum_ms", 20))
    return out


def _should_yield(slice_ctx: _SliceCtx, consumed: Dict[str,int]) -> str | None:
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

        total_t0 = time.perf_counter()

        # --- M5 slice init (PR26 scaffolding) ---
        sched_enabled = _m5_enabled(ctx)
        slice_ctx: _SliceCtx | None = None
        if sched_enabled:
            budgets = _derive_budgets(ctx)
            # If PR25 core is available, record the pick (no enforcement here).
            if _sched_next_turn is not None:
                try:
                    policy = str((_get_cfg(ctx).get("scheduler") or {}).get("policy", "round_robin"))
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
        boot_loaded = state.get("_boot_loaded", False) if isinstance(state, dict) else getattr(state, "_boot_loaded", False)
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
        cache_enabled = bool(cache_cfg.get("enabled", True)) if isinstance(cache_cfg, dict) else False
        cm_existing = (state.get("_cache_mgr") if isinstance(state, dict) else getattr(state, "_cache_mgr", None))
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
        # --- M5 boundary check after T1 ---
        if slice_ctx is not None:
            consumed = {
                "ms": int(round((time.perf_counter() - total_t0) * 1000.0)),
            }
            # Map T1 metrics if present
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
                _write_or_capture_scheduler_event(ctx,event)
                total_ms_now = round((time.perf_counter() - total_t0) * 1000.0, 3)
                append_jsonl(
                    "turn.jsonl",
                    {
                        "turn": turn_id,
                        "agent": agent_id,
                        "durations_ms": {"t1": t1_ms, "t2": 0.0, "t4": 0.0, "apply": 0.0, "total": total_ms_now},
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
        cm = (state.get("_cache_mgr") if isinstance(state, dict) else getattr(state, "_cache_mgr", None))
        ver = (state.get("version_etag") if isinstance(state, dict) else getattr(state, "version_etag", None)) or "0"
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
                        "durations_ms": {"t1": t1_ms, "t2": t2_ms, "t4": 0.0, "apply": 0.0, "total": total_ms_now},
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
        graph_cfg_all = (_get_cfg(ctx).get("graph") if isinstance(_get_cfg(ctx), dict) else {}) or {}
        graph_enabled = bool(graph_cfg_all.get("enabled", False)) if isinstance(graph_cfg_all, dict) else False
        if graph_enabled:
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
                        "durations_ms": {"t1": t1_ms, "t2": t2_ms, "t4": 0.0, "apply": 0.0, "total": total_ms_now},
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
                    hits.append({
                        "id": str(r.get("id")),
                        "score": float(r.get("_score", r.get("score", 0.0)) or 0.0),
                        "owner": str(r.get("owner", "any")),
                        "quarter": str(r.get("quarter", "")),
                    })
                else:
                    rid = str(getattr(r, "id", ""))
                    if not rid:
                        continue
                    hits.append({
                        "id": rid,
                        "score": float(getattr(r, "score", 0.0) or 0.0),
                        "owner": str(getattr(r, "owner", "any")),
                        "quarter": str(getattr(r, "quarter", "")),
                    })
            return {"retrieved": hits, "metrics": getattr(t2_alt, "metrics", {})}

        requested_retrieve = any(getattr(op, "kind", None) == "RequestRetrieve" for op in getattr(plan, "ops", []) or [])
        rag_metrics = {"rag_used": False, "rag_blocked": False, "pre_s_max": 0.0, "post_s_max": 0.0, "k_retrieved": 0, "owner": None, "tier_pref": None}
        if requested_retrieve and max_rag_loops >= 1:
            t0_rag = time.perf_counter()
            plan, rag_metrics = rag_once(bundle, plan, _retrieve_fn, already_used=False)
            rag_ms = round((time.perf_counter() - t0_rag) * 1000.0, 3)
        else:
            rag_ms = 0.0

        # Dialogue synthesis (rule-based vs optional LLM backend)
        t0 = time.perf_counter()
        dialog_bundle = make_dialog_bundle(ctx, state, t1, t2, plan)

        # Backend selection
        backend_cfg = str(t3cfg.get("backend", "rulebased")) if isinstance(t3cfg, dict) else "rulebased"
        llm_cfg = t3cfg.get("llm", {}) if isinstance(t3cfg, dict) else {}
        adapter = (
            state.get("llm_adapter", None) if isinstance(state, dict) else getattr(state, "llm_adapter", None)
        ) or getattr(ctx, "llm_adapter", None)

        backend_used = "rulebased"
        backend_fallback = None
        fallback_reason = None

        # Allow tests to monkeypatch a module-level t3_dialogue with flexible signatures.
        dlg_fn = globals().get("t3_dialogue")
        if callable(dlg_fn):
            try:
                # Preferred signature: (dialog_bundle, plan)
                res = dlg_fn(dialog_bundle, plan)
            except TypeError:
                # Fallback signature used by some tests: (ctx, state, dialog_bundle)
                res = dlg_fn(ctx, state, dialog_bundle)
            # Normalize: support returning just a string or (utter, metrics)
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

        policy_backend = str(t3cfg.get("backend", "rulebased")) if isinstance(t3cfg, dict) else "rulebased"
        append_jsonl(
            "t3_plan.jsonl",
            {
                "turn": turn_id,
                "agent": agent_id,
                "policy_backend": policy_backend,
                "backend": backend_used,
                **({"backend_fallback": backend_fallback, "fallback_reason": fallback_reason} if backend_fallback else {}),
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
            adapter_name = getattr(adapter, "name", adapter.__class__.__name__ if hasattr(adapter, "__class__") else "Unknown")
            model = str(llm_cfg.get("model", ""))
            temperature = float(llm_cfg.get("temperature", 0.2))
            dlg_extra.update({"backend": "llm", "adapter": adapter_name, "model": model, "temperature": temperature})
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

        # Kill switch (t4.enabled). Default True if unspecified.
        t4_cfg_full = (_get_cfg(ctx).get("t4") if isinstance(_get_cfg(ctx), dict) else {}) or {}
        t4_enabled = bool(t4_cfg_full.get("enabled", True)) if isinstance(t4_cfg_full, dict) else True

        # --- T4 / Apply (honor kill switch) ---
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
                        "policy": (_get_cfg(ctx).get("scheduler") or {}).get("policy", "round_robin"),
                        **({"pick_reason": _get_pick_reason(ctx)} if _get_pick_reason(ctx) else {}),
                        "reason": reason,
                        "enforced": True,
                        "stage_end": "T4",
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
                            "durations_ms": {"t1": t1_ms, "t2": t2_ms, "t4": t4_ms, "apply": 0.0, "total": total_ms_now},
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
            # --- GEL decay tick (optional; run before Apply so snapshot includes decay) ---
            graph_cfg_all2 = (_get_cfg(ctx).get("graph") if isinstance(_get_cfg(ctx), dict) else {}) or {}
            graph_enabled2 = bool(graph_cfg_all2.get("enabled", False)) if isinstance(graph_cfg_all2, dict) else False
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
                    mg_cfg = (graph_cfg_all2.get("merge") if isinstance(graph_cfg_all2, dict) else {}) or {}
                    sp_cfg = (graph_cfg_all2.get("split") if isinstance(graph_cfg_all2, dict) else {}) or {}
                    pr_cfg = (graph_cfg_all2.get("promotion") if isinstance(graph_cfg_all2, dict) else {}) or {}

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
                            # Promotions derive from current merge candidates (metadata-based policies can refine later)
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
                    # Never let optional GEL features break the turn
                    pass
            # --- Apply & persist ---
            t0 = time.perf_counter()
            apply = apply_changes(ctx, state, t4)
            apply_ms = round((time.perf_counter() - t0) * 1000.0, 3)
            append_jsonl(
                "apply.jsonl",
                {
                    "turn": turn_id,
                    "agent": agent_id,
                    "applied": apply.applied,
                    "clamps": apply.clamps,
                    "version_etag": apply.version_etag,
                    "snapshot": apply.snapshot_path,
                    "cache_invalidations": int((getattr(apply, "metrics", {}) or {}).get("cache_invalidations", 0)),
                    "ms": apply_ms,
                    **({"now": now} if now else {}),
                },
            )
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
                        "policy": (_get_cfg(ctx).get("scheduler") or {}).get("policy", "round_robin"),
                        **({"pick_reason": _get_pick_reason(ctx)} if _get_pick_reason(ctx) else {}),
                        "reason": reason,
                        "enforced": True,
                        "stage_end": "Apply",
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
                            "durations_ms": {"t1": t1_ms, "t2": t2_ms, "t4": t4_ms, "apply": apply_ms, "total": total_ms_now},
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
            # Bypassed: provide inert placeholders; no t4/apply logs
            t4_ms = 0.0
            apply_ms = 0.0
            t4 = SimpleNamespace(approved_deltas=[], rejected_ops=[], reasons=["T4_DISABLED"], metrics={"counts": {"approved": 0}})
            apply = SimpleNamespace(applied=0, clamps=0, version_etag=getattr(state, "version_etag", None), snapshot_path=None)

        # --- Health summary + per-turn rollup ---
        from . import health
        health.check_and_log(ctx, state, t1, t2, t4, apply, append_jsonl)

        total_ms = round((time.perf_counter() - total_t0) * 1000.0, 3)
        append_jsonl(
            "turn.jsonl",
            {
                "turn": turn_id,
                "agent": agent_id,
                "durations_ms": {"t1": t1_ms, "t2": t2_ms, "t4": t4_ms, "apply": apply_ms, "total": total_ms},
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
                **({
                    "slice_idx": (slice_ctx["slice_idx"] if slice_ctx is not None else None),
                    "yielded": False,
                } if slice_ctx is not None else {}),
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
