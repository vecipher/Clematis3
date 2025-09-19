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
from ..io.log import append_jsonl


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

        # --- T2 ---
        t0 = time.perf_counter()
        t2 = t2_semantic(ctx, state, input_text, t1)
        t2_ms = round((time.perf_counter() - t0) * 1000.0, 3)
        append_jsonl(
            "t2.jsonl",
            {
                "turn": turn_id,
                "agent": agent_id,
                **t2.metrics,
                "ms": t2_ms,
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
        cfg = getattr(ctx, "cfg", {}) or {}
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
        t4_cfg_full = getattr(getattr(ctx, "config", SimpleNamespace()), "t4", {}) or {}
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
            # --- Apply & persist ---
            t0 = time.perf_counter()
            apply = apply_changes(ctx, state, t4)
            apply_ms = round((time.perf_counter() - t0) * 1000.0, 3)
            append_jsonl(
                "apply.jsonl",
                {
                    "turn": turn_id,
                    "agent": agent_id,
                    "applied": int(getattr(apply, "applied", 0)),
                    "clamps": int(getattr(apply, "clamps", 0)),
                    "version_etag": getattr(apply, "version_etag", None),
                    "snapshot": getattr(apply, "snapshot_path", None),
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
                    "cache_used": t2.metrics.get("cache_used"),
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
