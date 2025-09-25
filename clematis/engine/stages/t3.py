from __future__ import annotations
from typing import Dict, Any, List, Tuple, Callable, Optional
from datetime import datetime, timezone
import os

from ..types import Plan, SpeakOp, EditGraphOp, RequestRetrieveOp
# PR8/M3-08: Optional LLM adapter types (duck-typed if unavailable)
try:  # runtime optional
    from ...adapters.llm import (
        LLMAdapter,
        LLMResult,
        FixtureLLMAdapter,
        QwenLLMAdapter,
        LLMAdapterError,
    )  # type: ignore
except Exception:  # pragma: no cover
    LLMAdapter = object  # type: ignore
    LLMResult = object   # type: ignore
    FixtureLLMAdapter = None  # type: ignore
    QwenLLMAdapter = None  # type: ignore
    class LLMAdapterError(Exception):  # type: ignore
        pass

# PR4: Pure T3 bundle assembly (no policy/RAG/dialogue yet)
# Deterministic: all lists sorted; explicit caps; empty defaults when absent.

BUNDLE_VERSION = "t3-bundle-v1"


def _iso_now(ctx) -> str:
    now = getattr(ctx, "now", None)
    if isinstance(now, str):
        return now
    if hasattr(now, "isoformat"):
        try:
            return now.astimezone(timezone.utc).isoformat()
        except Exception:
            return now.isoformat()
    return datetime.now(timezone.utc).isoformat()


def _get_cfg_caps(ctx) -> Dict[str, int]:
    cfg_t3 = getattr(ctx, "cfg", {}).t3 if hasattr(getattr(ctx, "cfg", {}), "t3") else getattr(getattr(ctx, "cfg", {}), "get", lambda *_: {})("t3", {})
    # Fall back safely if cfg is dict-like
    if isinstance(getattr(ctx, "cfg", None), dict):
        cfg_t3 = ctx.cfg.get("t3", {})
    tokens = int((cfg_t3 or {}).get("tokens", 256))
    max_ops = int((cfg_t3 or {}).get("max_ops_per_turn", 3))
    return {"tokens": tokens, "ops": max_ops}


def _get_agent_info(ctx) -> Dict[str, Any]:
    style_prefix = ""
    # Try a few conventional locations for style prefix; default empty if unknown
    try:
        style_prefix = (getattr(ctx, "agent", {}) or {}).get("style_prefix", "")  # type: ignore[attr-defined]
    except Exception:
        pass
    if not style_prefix:
        try:
            style_prefix = getattr(ctx, "style_prefix", "")
        except Exception:
            style_prefix = ""
    caps = _get_cfg_caps(ctx)
    return {
        "id": getattr(ctx, "agent_id", "agent"),
        "style_prefix": style_prefix or "",
        "caps": {"tokens": caps["tokens"], "ops": caps["ops"]},
    }


def _world_hot_labels(state) -> Tuple[List[str], int]:
    # If you maintain hot labels in state, surface them; otherwise return empty.
    labels = []
    try:
        labels = list(state.get("world_hot_labels", []))  # type: ignore[dict-item]
    except Exception:
        labels = []
    labels = sorted({str(x) for x in labels})
    return labels, len(labels)


def _extract_t1_touched_nodes(t1, cap: int = 32) -> List[Dict[str, Any]]:
    """Extract touched nodes from T1 in a permissive way.
    Expected entry shape (best‑effort): items with id/label/delta.
    Sorted by final id asc after selecting top-|delta| items.
    """
    items: List[Dict[str, Any]] = []
    # Common placements: t1.graph_deltas, t1.deltas
    candidates = getattr(t1, "graph_deltas", None) or getattr(t1, "deltas", None) or []
    if isinstance(candidates, dict):
        for nid, v in candidates.items():
            if isinstance(v, dict):
                delta = float(v.get("delta", v.get("weight", 0.0)))
            else:
                try:
                    delta = float(v)
                except Exception:
                    delta = 0.0
            items.append({"id": str(nid), "label": str(v.get("label", nid)) if isinstance(v, dict) else str(nid), "delta": float(delta)})
    elif isinstance(candidates, list):
        for e in candidates:
            if not isinstance(e, dict):
                continue
            nid = str(e.get("id", e.get("node", e.get("src", ""))))
            if not nid:
                continue
            lbl = str(e.get("label", e.get("name", nid)))
            delta = float(e.get("delta", e.get("weight", 0.0) or 0.0))
            items.append({"id": nid, "label": lbl, "delta": delta})
    # Rank by |delta| desc to pick top, then id asc for determinism in output
    items.sort(key=lambda x: (-abs(float(x.get("delta", 0.0))), str(x.get("id", ""))))
    items = items[: max(int(cap), 0)]
    items.sort(key=lambda x: str(x.get("id", "")))
    return items


def _extract_t1_metrics(t1) -> Dict[str, Any]:
    m = getattr(t1, "metrics", None)
    if isinstance(m, dict):
        # Keep only the expected keys if present, fill defaults explicitly
        return {
            "pops": int(m.get("pops", 0)),
            "iters": int(m.get("iters", 0)),
            "propagations": int(m.get("propagations", 0)),
            "radius_cap_hits": int(m.get("radius_cap_hits", 0)),
            "layer_cap_hits": int(m.get("layer_cap_hits", 0)),
            "node_budget_hits": int(m.get("node_budget_hits", 0)),
        }
    return {"pops": 0, "iters": 0, "propagations": 0, "radius_cap_hits": 0, "layer_cap_hits": 0, "node_budget_hits": 0}


def _extract_labels_from_t1(t1) -> List[str]:
    labels: List[str] = []
    # Try to collect any label-bearing fields from graph_deltas list entries
    candidates = getattr(t1, "graph_deltas", None) or getattr(t1, "deltas", None) or []
    if isinstance(candidates, list):
        for e in candidates:
            if isinstance(e, dict):
                val = e.get("label") or e.get("name")
                if val:
                    labels.append(str(val))
    labels = sorted({str(x) for x in labels})
    return labels


def _extract_t2_retrieved(t2, k_retrieval: int) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    retrieved = getattr(t2, "retrieved", [])
    for r in (retrieved or []):
        if isinstance(r, dict):
            rid = str(r.get("id"))
            score = float(r.get("_score", r.get("score", 0.0)) or 0.0)
            owner = str(r.get("owner", "any"))
            quarter = str(r.get("quarter", ""))
        else:
            rid = str(getattr(r, "id", ""))
            score = float(getattr(r, "score", 0.0) or 0.0)
            owner = str(getattr(r, "owner", "any"))
            quarter = str(getattr(r, "quarter", ""))
        if not rid:
            continue
        out.append({"id": rid, "score": score, "owner": owner, "quarter": quarter})
    # Deterministic ordering and cap
    out.sort(key=lambda e: (-float(e.get("score", 0.0)), str(e.get("id", ""))))
    return out[: max(int(k_retrieval), 0)]


def _extract_t2_metrics(t2) -> Dict[str, Any]:
    m = getattr(t2, "metrics", None)
    out: Dict[str, Any] = {}
    if isinstance(m, dict):
        # Pass through known fields; DO NOT synthesize 'tier_sequence' when absent.
        ts = m.get("tier_sequence", None)
        if isinstance(ts, list):
            out["tier_sequence"] = [str(x) for x in ts]
        sim_stats = m.get("sim_stats", None)
        if isinstance(sim_stats, dict):
            out["sim_stats"] = {
                "mean": float(sim_stats.get("mean", 0.0)),
                "max": float(sim_stats.get("max", 0.0)),
            }
        # k_returned: prefer explicit value; else derive from retrieved list length
        if "k_returned" in m:
            try:
                out["k_returned"] = int(m.get("k_returned", 0))
            except Exception:
                out["k_returned"] = 0
        else:
            try:
                out["k_returned"] = int(len(getattr(t2, "retrieved", []) or []))
            except Exception:
                out["k_returned"] = 0
        # cache_used: pass through if present; else default False
        if "cache_used" in m:
            out["cache_used"] = bool(m.get("cache_used", False))
        else:
            out["cache_used"] = False
        return out
    # Fallback: do not include 'tier_sequence' when metrics are missing
    try:
        k_ret = int(len(getattr(t2, "retrieved", []) or []))
    except Exception:
        k_ret = 0
    return {"k_returned": k_ret, "sim_stats": {"mean": 0.0, "max": 0.0}, "cache_used": False}


def _cfg_snapshot(ctx) -> Dict[str, Any]:
    cfg = getattr(ctx, "cfg", {})
    t3 = getattr(cfg, "t3", None)
    t2 = getattr(cfg, "t2", None)
    if isinstance(cfg, dict):
        t3 = cfg.get("t3", {})
        t2 = cfg.get("t2", {})
    return {
        "t3": {
            "max_rag_loops": int((t3 or {}).get("max_rag_loops", 1)),
            "tokens": int((t3 or {}).get("tokens", 256)),
            "temp": float((t3 or {}).get("temp", 0.7)),
        },
        "t2": {
            "owner_scope": str((t2 or {}).get("owner_scope", "any")),
            "k_retrieval": int((t2 or {}).get("k_retrieval", 64)),
            "sim_threshold": float((t2 or {}).get("sim_threshold", 0.3)),
        },
    }


def make_plan_bundle(ctx, state, t1, t2) -> Dict[str, Any]:
    """Assemble the compact, deterministic T3 bundle.
    Pure function: no I/O, no DB calls, no randomness.
    """
    now_iso = _iso_now(ctx)
    cfg = _cfg_snapshot(ctx)
    k_ret = cfg["t2"]["k_retrieval"]

    # M5: carry per-slice caps from orchestrator (if any)
    slice_caps = {}
    try:
        caps = getattr(ctx, "slice_budgets", None) or {}
        if isinstance(caps, dict) and caps.get("t3_ops") is not None:
            slice_caps["t3_ops"] = int(caps.get("t3_ops"))
    except Exception:
        slice_caps = {}

    # Agent & world
    agent = _get_agent_info(ctx)
    hot_labels, k_hot = _world_hot_labels(state)

    # T1
    t1_nodes = _extract_t1_touched_nodes(t1, cap=32)
    t1_metrics = _extract_t1_metrics(t1)

    # T2
    t2_hits = _extract_t2_retrieved(t2, k_retrieval=k_ret)
    t2_metrics = _extract_t2_metrics(t2)

    # Text & labels
    input_text = getattr(ctx, "input_text", "") or getattr(ctx, "text", "") or ""
    labels_from_t1 = _extract_labels_from_t1(t1)

    bundle: Dict[str, Any] = {
        "version": BUNDLE_VERSION,
        "now": now_iso,
        "agent": agent,
        "world": {"hot_labels": hot_labels, "k": k_hot},
        "t1": {"touched_nodes": t1_nodes, "metrics": t1_metrics},
        "t2": {"retrieved": t2_hits, "metrics": t2_metrics},
        "text": {"input": input_text, "labels_from_t1": labels_from_t1},
        "cfg": cfg,
        "slice_caps": slice_caps,
    }
    return bundle

# --- PR7: Dialogue bundle (deterministic, pure)
DIALOG_BUNDLE_VERSION = "t3-dialog-bundle-v1"

def make_dialog_bundle(ctx, state, t1, t2, plan=None) -> Dict[str, Any]:
    """Assemble a deterministic dialogue bundle from PR4 bundle + Plan.
    Pure: no I/O. Uses only ctx/state/t1/t2/plan and config.
    """
    base = make_plan_bundle(ctx, state, t1, t2)
    # Dialogue config snapshot
    cfg_t3 = (getattr(ctx, "cfg", {}) or {}).get("t3", {}) if isinstance(getattr(ctx, "cfg", {}), dict) else {}
    dialogue_cfg = {}
    if isinstance(cfg_t3, dict):
        dialogue_cfg = cfg_t3.get("dialogue", {}) or {}
    template = str(dialogue_cfg.get("template", "summary: {labels}. next: {intent}"))
    include_top_k = int(dialogue_cfg.get("include_top_k_snippets", 2) or 2)

    # Retrieved already sorted/capped by make_plan_bundle
    retrieved = list(base.get("t2", {}).get("retrieved", []) or [])

    return {
        "version": DIALOG_BUNDLE_VERSION,
        "now": base["now"],
        "agent": base["agent"],
        "text": base["text"],
        "retrieved": retrieved,
        "dialogue": {"template": template, "include_top_k_snippets": include_top_k},
    }

# --- PR5: Rule-based policy (no RAG) ---
# Deterministic `deliberate(bundle) -> Plan` using explicit thresholds and caps.

_DEFAULT_TAU_HIGH = 0.8
_DEFAULT_TAU_LOW = 0.4
_DEFAULT_EPS_EDIT = 0.10


def _policy_thresholds(bundle: Dict[str, Any]) -> Dict[str, float]:
    t3cfg = (bundle.get("cfg", {}).get("t3", {}) if isinstance(bundle.get("cfg", {}), dict) else {})
    pol = t3cfg.get("policy", {}) if isinstance(t3cfg, dict) else {}
    return {
        "tau_high": float(pol.get("tau_high", _DEFAULT_TAU_HIGH)),
        "tau_low": float(pol.get("tau_low", _DEFAULT_TAU_LOW)),
        "eps_edit": float(pol.get("epsilon_edit", _DEFAULT_EPS_EDIT)),
    }


def _topic_labels_from_bundle(bundle: Dict[str, Any], cap: int = 5) -> List[str]:
    labels = list(bundle.get("text", {}).get("labels_from_t1", []) or [])
    if not labels:
        # Fallback to labels from touched nodes
        labels = [str(n.get("label", n.get("id"))) for n in bundle.get("t1", {}).get("touched_nodes", [])]
    # Deterministic: dedupe + sort + cap
    labels = sorted({str(x) for x in labels})
    return labels[: max(int(cap), 0)]


def _edit_nodes_from_bundle(bundle: Dict[str, Any], eps_edit: float, cap_nodes: int) -> List[Dict[str, Any]]:
    """Select nodes with |delta| >= eps_edit; return deterministic edits.
    Returns a list of {op:"upsert_node", id: <node_id>} sorted by id asc, capped to cap_nodes.
    """
    nodes = bundle.get("t1", {}).get("touched_nodes", []) or []
    selected = []
    for n in nodes:
        try:
            if abs(float(n.get("delta", 0.0))) >= float(eps_edit):
                selected.append({"id": str(n.get("id"))})
        except Exception:
            continue
    selected.sort(key=lambda e: e["id"])  # id asc
    selected = selected[: max(int(cap_nodes), 0)]
    return [{"op": "upsert_node", "id": e["id"]} for e in selected]


def deliberate(bundle: Dict[str, Any]) -> Plan:
    """Deterministic rule-based policy.
    Inputs: PR4 bundle. Outputs: Plan with whitelisted ops, capped by config.
    No RAG here (PR6 will add it). Pure; no I/O.
    """
    cfg_t3 = bundle.get("cfg", {}).get("t3", {})
    cfg_t2 = bundle.get("cfg", {}).get("t2", {})

    base_ops = int(bundle.get("agent", {}).get("caps", {}).get("ops", 3))
    try:
        slice_cap = int(bundle.get("slice_caps", {}).get("t3_ops", base_ops))
    except Exception:
        slice_cap = base_ops
    caps_ops = min(base_ops, slice_cap)
    tokens = int(cfg_t3.get("tokens", 256))

    thresholds = _policy_thresholds(bundle)
    tau_high = thresholds["tau_high"]
    tau_low = thresholds["tau_low"]
    eps_edit = thresholds["eps_edit"]

    sim_stats = bundle.get("t2", {}).get("metrics", {}).get("sim_stats", {}) or {}
    s_max = float(sim_stats.get("max", 0.0))

    # Decide intent deterministically based on s_max
    labels = _topic_labels_from_bundle(bundle)
    if s_max >= tau_high:
        intent = "summary"
    elif s_max >= tau_low:
        intent = "assertion" if labels else "ack"
    else:
        intent = "question"

    ops: List[Any] = []
    # Always include a Speak op first (intent + labels)
    ops.append(
        SpeakOp(kind="Speak", intent=intent, topic_labels=labels, max_tokens=tokens)
    )

    # Optionally include a small EditGraph op if evidence is not too weak
    if s_max >= tau_low and len(ops) < caps_ops:
        remaining = max(caps_ops - len(ops), 0)
        edits = _edit_nodes_from_bundle(bundle, eps_edit=eps_edit, cap_nodes=remaining * 4)
        # Only emit if we have edits and stay within ops cap
        if edits:
            ops.append(EditGraphOp(kind="EditGraph", edits=edits, cap=min(len(edits), remaining * 4)))

    # Optionally include a RequestRetrieve op for low evidence (executed in PR6)
    if s_max < tau_low and len(ops) < caps_ops:
        owner = str(cfg_t2.get("owner_scope", "any"))
        k_all = int(cfg_t2.get("k_retrieval", 64))
        k = max(1, k_all // 2)
        ops.append(
            RequestRetrieveOp(
                kind="RequestRetrieve",
                query=bundle.get("text", {}).get("input", ""),
                owner=owner if owner in ("agent", "world", "any") else "any",
                k=k,
                tier_pref="cluster_semantic",
                hints={"now": bundle.get("now", ""), "sim_threshold": float(cfg_t2.get("sim_threshold", 0.3))},
            )
        )

    # Enforce ops cap deterministically (keep first N)
    if len(ops) > caps_ops:
        ops = ops[:caps_ops]

    return Plan(version="t3-plan-v1", reflection=False, ops=ops, request_retrieve=None)

# --- PR6: One-shot RAG refinement (pure) ---
# Orchestrator provides a deterministic retrieve_fn; this function remains pure.
RetrieveFn = Callable[[Dict[str, Any]], Dict[str, Any]]


def _first_request_retrieve_payload(plan: Plan) -> Optional[Dict[str, Any]]:
    for op in getattr(plan, "ops", []) or []:
        if getattr(op, "kind", None) == "RequestRetrieve":
            # Normalize to a plain dict payload
            return {
                "query": getattr(op, "query", ""),
                "owner": getattr(op, "owner", "any"),
                "k": int(getattr(op, "k", 0) or 0),
                "tier_pref": getattr(op, "tier_pref", None),
                "hints": dict(getattr(op, "hints", {}) or {}),
            }
    return None


def _normalize_rr_payload(bundle: Dict[str, Any], rr: Dict[str, Any]) -> Dict[str, Any]:
    cfg_t2 = bundle.get("cfg", {}).get("t2", {})
    hints = dict(rr.get("hints", {}) or {})
    # Ensure deterministic required hints
    hints.setdefault("now", bundle.get("now", ""))
    hints.setdefault("sim_threshold", float(cfg_t2.get("sim_threshold", 0.3)))
    owner = rr.get("owner", "any")
    owner = owner if owner in ("agent", "world", "any") else "any"
    k = int(rr.get("k", int(cfg_t2.get("k_retrieval", 1))))
    k = max(1, k)
    return {
        "query": rr.get("query", ""),
        "owner": owner,
        "k": k,
        "tier_pref": rr.get("tier_pref"),
        "hints": hints,
    }


def _normalize_retrieved_result(result: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], int, float]:
    hits: List[Dict[str, Any]] = []
    for r in (result.get("retrieved", []) or []):
        if isinstance(r, dict):
            rid = str(r.get("id"))
            score = float(r.get("_score", r.get("score", 0.0)) or 0.0)
            owner = str(r.get("owner", "any"))
            quarter = str(r.get("quarter", ""))
        else:
            rid = str(getattr(r, "id", ""))
            score = float(getattr(r, "score", 0.0) or 0.0)
            owner = str(getattr(r, "owner", "any"))
            quarter = str(getattr(r, "quarter", ""))
        if not rid:
            continue
        hits.append({"id": rid, "score": score, "owner": owner, "quarter": quarter})
    # Deterministic ordering
    hits.sort(key=lambda e: (-float(e.get("score", 0.0)), str(e.get("id", ""))))
    k_ret = len(hits)
    s_max = max([h.get("score", 0.0) for h in hits], default=0.0)
    return hits, k_ret, float(s_max)


def _refined_intent(tau_high: float, tau_low: float, labels: List[str], s_max: float) -> str:
    if s_max >= tau_high:
        return "summary"
    if s_max >= tau_low:
        return "assertion" if labels else "ack"
    return "question"


def rag_once(
    bundle: Dict[str, Any],
    plan: Plan,
    retrieve_fn: RetrieveFn,
    already_used: bool = False,
) -> Tuple[Plan, Dict[str, Any]]:
    """Single retrieval refinement. Pure given a deterministic retrieve_fn.
    - If already_used: return plan unchanged with rag_blocked.
    - If no RequestRetrieve in plan.ops: return unchanged (no-op).
    - Else: call retrieve_fn with normalized payload; refine Speak intent and possibly add one EditGraph if within cap.
    Returns: (refined_plan, rag_metrics)
    """
    # Pre metrics
    sim_stats0 = bundle.get("t2", {}).get("metrics", {}).get("sim_stats", {}) or {}
    pre_s_max = float(sim_stats0.get("max", 0.0))

    rr = _first_request_retrieve_payload(plan)
    if already_used:
        return plan, {
            "rag_used": False,
            "rag_blocked": True,
            "pre_s_max": pre_s_max,
            "post_s_max": pre_s_max,
            "k_retrieved": 0,
            "owner": rr.get("owner") if rr else None,
            "tier_pref": rr.get("tier_pref") if rr else None,
        }

    if rr is None:
        return plan, {
            "rag_used": False,
            "rag_blocked": False,
            "pre_s_max": pre_s_max,
            "post_s_max": pre_s_max,
            "k_retrieved": 0,
            "owner": None,
            "tier_pref": None,
        }

    payload = _normalize_rr_payload(bundle, rr)
    result = retrieve_fn(payload)
    hits, k_retrieved, s_max_rag = _normalize_retrieved_result(result if isinstance(result, dict) else {})

    # Compute refined evidence level
    thresholds = _policy_thresholds(bundle)
    tau_high, tau_low, eps_edit = thresholds["tau_high"], thresholds["tau_low"], thresholds["eps_edit"]
    labels = _topic_labels_from_bundle(bundle)

    post_s_max = max(pre_s_max, s_max_rag)
    new_intent = _refined_intent(tau_high, tau_low, labels, post_s_max)

    # Build new ops deterministically
    base_ops = int(bundle.get("agent", {}).get("caps", {}).get("ops", 3))
    try:
        slice_cap = int(bundle.get("slice_caps", {}).get("t3_ops", base_ops))
    except Exception:
        slice_cap = base_ops
    caps_ops = min(base_ops, slice_cap)
    tokens = int(bundle.get("cfg", {}).get("t3", {}).get("tokens", 256))

    new_ops: List[Any] = []
    has_editgraph = False
    # Replace first Speak with refined intent; keep others in order
    speak_replaced = False
    for op in plan.ops:
        k = getattr(op, "kind", None)
        if k == "Speak" and not speak_replaced:
            new_ops.append(SpeakOp(kind="Speak", intent=new_intent, topic_labels=labels, max_tokens=tokens))
            speak_replaced = True
        else:
            new_ops.append(op)
        if k == "EditGraph":
            has_editgraph = True
    # Optionally add one EditGraph if evidence now >= tau_low and none exists yet
    if (post_s_max >= tau_low) and (not has_editgraph) and (len(new_ops) < caps_ops):
        remaining = max(caps_ops - len(new_ops), 0)
        edits = _edit_nodes_from_bundle(bundle, eps_edit=eps_edit, cap_nodes=remaining * 4)
        if edits:
            new_ops.append(EditGraphOp(kind="EditGraph", edits=edits, cap=min(len(edits), remaining * 4)))

    # Enforce ops cap deterministically
    if len(new_ops) > caps_ops:
        new_ops = new_ops[:caps_ops]

    refined = Plan(version="t3-plan-v1", reflection=getattr(plan, "reflection", False), ops=new_ops, request_retrieve=getattr(plan, "request_retrieve", None))

    metrics = {
        "rag_used": True,
        "rag_blocked": False,
        "pre_s_max": pre_s_max,
        "post_s_max": post_s_max,
        "k_retrieved": int(k_retrieved),
        "owner": payload.get("owner"),
        "tier_pref": payload.get("tier_pref"),
    }
    return refined, metrics


# --- PR7: Deterministic dialogue synthesis ---

def _dedupe_sort_list(xs: List[str]) -> List[str]:
    return sorted({str(x) for x in (xs or [])})


def _format_labels(labels: List[str]) -> str:
    return ", ".join(labels)


def _top_snippet_ids(dialog_bundle: Dict[str, Any]) -> List[str]:
    n = int(dialog_bundle.get("dialogue", {}).get("include_top_k_snippets", 2) or 2)
    hits = dialog_bundle.get("retrieved", []) or []
    ids = [str(h.get("id")) for h in hits if isinstance(h, dict) and h.get("id")]
    return ids[: max(n, 0)]


def _tokenize(s: str) -> List[str]:
    # Deterministic whitespace tokenization
    return (s or "").split()


def _truncate_to_tokens(s: str, max_tokens: int) -> Tuple[str, bool, int]:
    toks = _tokenize(s)
    if max_tokens <= 0:
        return "", True, 0
    if len(toks) <= max_tokens:
        return s, False, len(toks)
    out = " ".join(toks[:max_tokens])
    return out, True, max_tokens


def _first_speak_op(plan: Plan) -> Optional[SpeakOp]:
    for op in getattr(plan, "ops", []) or []:
        if getattr(op, "kind", None) == "Speak":
            return op  # type: ignore[return-value]
    return None


def speak(dialog_bundle: Dict[str, Any], plan: Plan) -> Tuple[str, Dict[str, Any]]:
    """Produce a deterministic utterance using a simple template.
    No external I/O/LLM. Enforces token budget via deterministic truncation.
    Returns (utterance, metrics).
    """
    speak_op = _first_speak_op(plan)
    # Labels: prefer Plan's Speak labels; else fallback to bundle text labels
    plan_labels = list(getattr(speak_op, "topic_labels", []) or []) if speak_op else []
    if not plan_labels:
        plan_labels = list(dialog_bundle.get("text", {}).get("labels_from_t1", []) or [])
    labels_sorted = _dedupe_sort_list([str(x) for x in plan_labels])

    intent = getattr(speak_op, "intent", "ack") if speak_op else "ack"
    style_prefix = str(dialog_bundle.get("agent", {}).get("style_prefix", ""))
    template = str(dialog_bundle.get("dialogue", {}).get("template", "summary: {labels}. next: {intent}"))

    # Snippets: top-K ids
    snippet_ids = _top_snippet_ids(dialog_bundle)
    snippets_str = ", ".join(snippet_ids)

    # Variables for formatting
    fmt_vars = {
        "labels": _format_labels(labels_sorted),
        "intent": intent,
        "snippets": snippets_str,
        "style_prefix": style_prefix,
    }

    # Primary formatting path
    try:
        core = template.format(**fmt_vars)
    except Exception:
        # Fallback deterministic minimal rendering
        core = f"summary: {fmt_vars['labels']}. next: {fmt_vars['intent']}"

    # Ensure style prefix is included even if template lacks {style_prefix}
    if "{style_prefix}" not in template and style_prefix:
        utter = f"{style_prefix}| {core}".strip()
        style_used = True
    else:
        utter = core
        style_used = bool(style_prefix)

    # Token budget: prefer SpeakOp.max_tokens, else agent caps, else 256
    max_tokens = 256
    if speak_op and getattr(speak_op, "max_tokens", None):
        try:
            max_tokens = int(speak_op.max_tokens)
        except Exception:
            max_tokens = 256
    else:
        try:
            max_tokens = int(dialog_bundle.get("agent", {}).get("caps", {}).get("tokens", 256))
        except Exception:
            max_tokens = 256

    utter_capped, truncated, token_count = _truncate_to_tokens(utter, max_tokens)

    metrics = {
        "tokens": int(token_count),
        "truncated": bool(truncated),
        "style_prefix_used": bool(style_used and style_prefix),
        "snippet_count": int(len(snippet_ids)),
    }
    return utter_capped, metrics

# --- PR8: LLM-based dialogue (optional backend) ---

def build_llm_prompt(dialog_bundle: Dict[str, Any], plan: Plan) -> str:
    """
    Deterministic prompt assembly for LLM backends. Pure; no I/O.
    Contains the same core fields used by rule-based speak(), in a stable layout.
    """
    speak_op = _first_speak_op(plan)
    labels = list(getattr(speak_op, "topic_labels", []) or []) if speak_op else []
    if not labels:
        labels = list(dialog_bundle.get("text", {}).get("labels_from_t1", []) or [])
    labels = _dedupe_sort_list([str(x) for x in labels])

    intent = getattr(speak_op, "intent", "ack") if speak_op else "ack"
    style_prefix = str(dialog_bundle.get("agent", {}).get("style_prefix", ""))
    input_text = str(dialog_bundle.get("text", {}).get("input", ""))
    snippet_ids = _top_snippet_ids(dialog_bundle)

    # Stable, line-oriented prompt to minimize accidental non-determinism across providers.
    lines = [
        f"now: {dialog_bundle.get('now', '')}",
        f"style_prefix: {style_prefix}",
        f"intent: {intent}",
        f"labels: {', '.join(labels)}",
        f"snippets: {', '.join(snippet_ids)}",
        "instruction: Compose a concise utterance that reflects the intent and labels. Do not exceed the token budget.",
        f"input: {input_text}",
    ]
    return "\n".join(lines).strip()


def llm_speak(dialog_bundle: Dict[str, Any], plan: Plan, adapter: Any) -> Tuple[str, Dict[str, Any]]:
    """
    Produce an utterance using an injected LLM adapter. Deterministic under a deterministic adapter.
    Enforces the same token budget as rule-based speak() and prefixes style if template omits it.
    Returns (utterance, metrics) with the same metric keys as speak(), plus backend info.
    """
    # Compute labels/intent again for metrics and style use
    speak_op = _first_speak_op(plan)
    intent = getattr(speak_op, "intent", "ack") if speak_op else "ack"
    style_prefix = str(dialog_bundle.get("agent", {}).get("style_prefix", ""))
    snippet_ids = _top_snippet_ids(dialog_bundle)

    # Token budget policy: SpeakOp.max_tokens -> agent caps -> default 256
    max_tokens = 256
    if speak_op and getattr(speak_op, "max_tokens", None):
        try:
            max_tokens = int(speak_op.max_tokens)
        except Exception:
            max_tokens = 256
    else:
        try:
            max_tokens = int(dialog_bundle.get("agent", {}).get("caps", {}).get("tokens", 256))
        except Exception:
            max_tokens = 256

    # Temperature: read from adapter if present; default to 0.2
    temperature = float(getattr(adapter, "default_temperature", 0.2))

    prompt = build_llm_prompt(dialog_bundle, plan)
    # Call the adapter; it must implement .generate(prompt, max_tokens, temperature) -> LLMResult-like
    try:
        result = adapter.generate(prompt, max_tokens=max_tokens, temperature=temperature)
        # result may be a plain object; attempt attribute access first, fallback to dict
        text = getattr(result, "text", None)
        if text is None and isinstance(result, dict):
            text = result.get("text", "")
        tokens = getattr(result, "tokens", None)
        if tokens is None and isinstance(result, dict):
            tokens = int(result.get("tokens", 0))
        truncated_llm = getattr(result, "truncated", None)
        if truncated_llm is None and isinstance(result, dict):
            truncated_llm = bool(result.get("truncated", False))
    except Exception:
        # Fail closed with a deterministic minimal message
        text, tokens, truncated_llm = "[llm:error]", 0, True

    # Ensure style prefix is present even if the adapter did not apply it
    if style_prefix and not (text or "").startswith(f"{style_prefix}|"):
        text = f"{style_prefix}| {text}".strip()

    # Final budget enforcement (adapter should already respect it, but we cap deterministically)
    text_capped, truncated_final, token_count = _truncate_to_tokens(text, max_tokens)

    metrics = {
        "tokens": int(token_count),
        "truncated": bool(truncated_llm or truncated_final),
        "style_prefix_used": bool(style_prefix != ""),
        "snippet_count": int(len(snippet_ids)),
        "backend": "llm",
        "adapter": getattr(adapter, "name", adapter.__class__.__name__ if hasattr(adapter, "__class__") else "Unknown"),
    }
    return text_capped, metrics

# --- M3-08: LLM Planner (opt-in, defaults OFF) ---

def make_planner_prompt(ctx) -> str:
    """Deterministic, compact planner prompt for the LLM backend.
    Kept intentionally small so fixtures remain stable; M3-10 will add schema hardening.
    """
    import json as _json
    summary = {
        "turn": getattr(ctx, "turn_id", 0),
        "agent": getattr(ctx, "agent_id", "agent"),
    }
    return (
        "SYSTEM: Return ONLY valid JSON with keys {plan: list[str], rationale: str}. "
        "No prose. No markdown. No trailing commas.\n"
        f"STATE: {_json.dumps(summary, separators=(',',':'))}\n"
        "USER: Propose up to 4 next steps as short strings; include a brief rationale."
    )

def _get_llm_adapter_from_cfg(cfg: Dict[str, Any]):
    """Build an LLM adapter instance from cfg when backend=='llm'.
    Returns None when backend is not 'llm' or when adapters are unavailable.
    """
    try:
        t3 = cfg.get("t3", {}) if isinstance(cfg, dict) else {}
    except Exception:
        t3 = {}
    if str(t3.get("backend", "rulebased")) != "llm":
        return None
    llm = t3.get("llm", {}) if isinstance(t3, dict) else {}
    provider = str(llm.get("provider", "fixture"))
    # CI guard: enforce fixture-only provider and valid fixtures when running in CI
    try:
        ci = str(os.environ.get("CI", "")).lower() == "true"
    except Exception:
        ci = False
    if ci:
        if provider != "fixture":
            raise LLMAdapterError(f"CI requires fixture provider; got {provider}")
        fixtures = (llm.get("fixtures", {}) or {}) if isinstance(llm, dict) else {}
        enabled = fixtures.get("enabled", True)
        if not enabled:
            raise LLMAdapterError("CI requires fixtures.enabled=true")
        fx_path = fixtures.get("path") or "fixtures/llm/qwen_small.jsonl"
        if not os.path.exists(fx_path):
            raise LLMAdapterError(f"CI fixture path not found: {fx_path}")
    if provider == "fixture":
        if FixtureLLMAdapter is None:
            return None
        path = ((llm.get("fixtures", {}) or {}).get("path")
                or "fixtures/llm/qwen_small.jsonl")
        return FixtureLLMAdapter(path)
    if provider == "ollama":
        if QwenLLMAdapter is None:
            return None
        endpoint = llm.get("endpoint", "http://localhost:11434/api/generate")
        model = llm.get("model", "qwen3:4b-instruct")
        temp = float(llm.get("temp", 0.2))
        timeout_s = max(1, int(int(llm.get("timeout_ms", 10000)) / 1000))

        # Build an Ollama call_fn matching QwenLLMAdapter's expected signature
        def _ollama_call(prompt: str, *, model: str, max_tokens: int, temperature: float, timeout_s: int) -> str:
            import json as _json
            import urllib.request as _ur
            body = _json.dumps({
                "model": model,
                "prompt": prompt,
                "options": {"temperature": float(temperature), "num_predict": int(max_tokens)},
                "stream": False,
                # Hint to return strict JSON when possible
                "format": "json",
            }).encode("utf-8")
            req = _ur.Request(endpoint, data=body, headers={"Content-Type": "application/json"})
            with _ur.urlopen(req, timeout=timeout_s) as resp:
                payload = _json.loads(resp.read().decode("utf-8"))
            txt = payload.get("response")
            if not isinstance(txt, str):
                raise LLMAdapterError("Ollama returned no text response")
            return txt

        return QwenLLMAdapter(call_fn=_ollama_call, model=model, temperature=temp, timeout_s=timeout_s)
    # Unknown provider -> surface as validation error upstream; here we fail closed.
    return None

def plan_with_llm(ctx, state: Any, cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Run the opt-in LLM planner and return a minimal planner dict.
    Shape: {"plan": list[str], "rationale": str}
    On any failure, returns {"plan": [], "rationale": "fallback: invalid llm output"} and logs a reason in state.logs if present.
    Does not mutate graphs or downstream state.
    """
    import json as _json
    try:
        adapter = _get_llm_adapter_from_cfg(cfg)
    except Exception as e:
        try:
            logs = getattr(state, "logs", None)
            if isinstance(logs, list):
                prov = str((((cfg.get("t3", {}) or {}).get("llm", {}) or {}).get("provider", "unknown")))
                ci = str(os.environ.get("CI", ""))
                logs.append({"llm_error": str(e), "provider": prov, "ci": ci})
        except Exception:
            pass
        return {"plan": [], "rationale": "fallback: invalid llm output"}
    if adapter is None:
        # Not enabled or not available -> return empty plan (caller may fall back to rulebased)
        return {"plan": [], "rationale": "fallback: llm backend not active"}

    prompt = make_planner_prompt(ctx)
    try:
        llm_max = int(((cfg.get("t3", {}) or {}).get("llm", {}) or {}).get("max_tokens", 256))
        llm_temp = float(((cfg.get("t3", {}) or {}).get("llm", {}) or {}).get("temp", 0.2))
    except Exception:
        llm_max, llm_temp = 256, 0.2

    try:
        result = adapter.generate(prompt, max_tokens=llm_max, temperature=llm_temp)
        raw = getattr(result, "text", None)
        if raw is None and isinstance(result, dict):
            raw = result.get("text", "")
        obj = _json.loads(raw if isinstance(raw, str) else "")
        # Minimal shape check (strict schema arrives in M3-10)
        if not isinstance(obj, dict) or "plan" not in obj or "rationale" not in obj:
            raise ValueError("planner output missing required keys")
        if not isinstance(obj.get("plan"), list) or not isinstance(obj.get("rationale"), str):
            raise ValueError("planner output wrong types")
        return {"plan": [str(x) for x in obj["plan"]][:16], "rationale": str(obj["rationale"])[:2000]}
    except Exception as e:
        try:
            logs = getattr(state, "logs", None)
            if isinstance(logs, list):
                prov = str((((cfg.get("t3", {}) or {}).get("llm", {}) or {}).get("provider", "unknown")))
                ci = str(os.environ.get("CI", ""))
                logs.append({"llm_error": str(e), "provider": prov, "ci": ci})
        except Exception:
            pass
        return {"plan": [], "rationale": "fallback: invalid llm output"}