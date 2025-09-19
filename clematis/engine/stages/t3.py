from __future__ import annotations
from typing import Dict, Any, List, Tuple, Callable, Optional
from datetime import datetime, timezone
from ..types import Plan, SpeakOp, EditGraphOp, RequestRetrieveOp

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
    if isinstance(m, dict):
        # project down to stable fields we document
        tier_seq = list(m.get("tier_sequence", [])) if isinstance(m.get("tier_sequence", []), list) else []
        sim_stats = m.get("sim_stats", {}) if isinstance(m.get("sim_stats", {}), dict) else {}
        return {
            "tier_sequence": [str(x) for x in tier_seq],
            "k_returned": int(m.get("k_returned", len(getattr(t2, "retrieved", []) or []))),
            "sim_stats": {"mean": float(sim_stats.get("mean", 0.0)), "max": float(sim_stats.get("max", 0.0))},
            "cache_used": bool(m.get("cache_used", False)),
        }
    return {"tier_sequence": [], "k_returned": 0, "sim_stats": {"mean": 0.0, "max": 0.0}, "cache_used": False}


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
    }
    return bundle

# --- PR4 placeholder ---
# The orchestrator imports `make_dialog_bundle` as a forward stub.
# Provide a deterministic, pure placeholder that will be replaced in PR7.
DIALOG_BUNDLE_VERSION = "t3-dialog-bundle-v1"

def make_dialog_bundle(ctx, state, t1, t2, plan=None) -> Dict[str, Any]:
    """Minimal deterministic dialogue bundle placeholder.
    Pure: no I/O. Mirrors `make_plan_bundle` and projects a small subset.
    """
    base = make_plan_bundle(ctx, state, t1, t2)
    return {
        "version": DIALOG_BUNDLE_VERSION,
        "now": base["now"],
        "agent": base["agent"],
        "text": base["text"],
        # Keep retrieved list as-is (already capped/sorted by make_plan_bundle)
        "retrieved": base["t2"]["retrieved"],
        # Lightweight plan summary if provided
        "plan_summary": {
            "has_plan": bool(plan is not None),
            "ops": int(len(getattr(plan, "ops", []) or [])),
        },
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

    caps_ops = int(bundle.get("agent", {}).get("caps", {}).get("ops", 3))
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
    caps_ops = int(bundle.get("agent", {}).get("caps", {}).get("ops", 3))
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
