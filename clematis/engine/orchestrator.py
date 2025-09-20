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
from .gel import observe_retrieval as gel_observe, tick as gel_tick
from ..io.log import append_jsonl
from .cache import CacheManager


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
