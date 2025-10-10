from __future__ import annotations
from typing import Any, Dict, List, Tuple
from datetime import datetime, timezone

from .types import PlanBundle

BUNDLE_VERSION = "t3-bundle-v1"


def iso_now(ctx: Any) -> str:
    now = getattr(ctx, "now", None)
    if isinstance(now, str):
        return now
    if hasattr(now, "isoformat"):
        try:
            return now.astimezone(timezone.utc).isoformat()
        except Exception:
            return now.isoformat()
    return datetime.now(timezone.utc).isoformat()


def cfg_caps(ctx: Any) -> Dict[str, int]:
    cfg_t3 = (
        getattr(ctx, "cfg", {}).t3
        if hasattr(getattr(ctx, "cfg", {}), "t3")
        else getattr(getattr(ctx, "cfg", {}), "get", lambda *_: {})("t3", {})
    )
    if isinstance(getattr(ctx, "cfg", None), dict):
        cfg_t3 = ctx.cfg.get("t3", {})
    tokens = int((cfg_t3 or {}).get("tokens", 256))
    max_ops = int((cfg_t3 or {}).get("max_ops_per_turn", 3))
    return {"tokens": tokens, "ops": max_ops}


def agent_info(ctx: Any) -> Dict[str, Any]:
    style_prefix = ""
    try:
        style_prefix = (getattr(ctx, "agent", {}) or {}).get("style_prefix", "")  # type: ignore[attr-defined]
    except Exception:
        pass
    if not style_prefix:
        try:
            style_prefix = getattr(ctx, "style_prefix", "")
        except Exception:
            style_prefix = ""
    caps = cfg_caps(ctx)
    return {
        "id": getattr(ctx, "agent_id", "agent"),
        "style_prefix": style_prefix or "",
        "caps": {"tokens": caps["tokens"], "ops": caps["ops"]},
    }


def world_hot_labels(state: Any) -> Tuple[List[str], int]:
    labels: List[str] = []
    try:
        labels = list(state.get("world_hot_labels", []))  # type: ignore[dict-item]
    except Exception:
        labels = []
    labels = sorted({str(x) for x in labels})
    return labels, len(labels)


def extract_t1_touched_nodes(t1: Any, cap: int = 32) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    candidates = getattr(t1, "graph_deltas", None) or getattr(t1, "deltas", None) or []
    if isinstance(candidates, dict):
        for nid, v in candidates.items():
            if isinstance(v, dict):
                delta = float(v.get("delta", v.get("weight", 0.0)))
                label = v.get("label", nid)
            else:
                try:
                    delta = float(v)
                except Exception:
                    delta = 0.0
                label = nid
            items.append({"id": str(nid), "label": str(label), "delta": float(delta)})
    elif isinstance(candidates, list):
        for entry in candidates:
            if not isinstance(entry, dict):
                continue
            nid = str(entry.get("id", entry.get("node", entry.get("src", ""))))
            if not nid:
                continue
            lbl = str(entry.get("label", entry.get("name", nid)))
            delta = float(entry.get("delta", entry.get("weight", 0.0) or 0.0))
            items.append({"id": nid, "label": lbl, "delta": delta})
    items.sort(key=lambda x: (-abs(float(x.get("delta", 0.0))), str(x.get("id", ""))))
    items = items[: max(int(cap), 0)]
    items.sort(key=lambda x: str(x.get("id", "")))
    return items


def extract_t1_metrics(t1: Any) -> Dict[str, Any]:
    metrics = getattr(t1, "metrics", None)
    if isinstance(metrics, dict):
        return {
            "pops": int(metrics.get("pops", 0)),
            "iters": int(metrics.get("iters", 0)),
            "propagations": int(metrics.get("propagations", 0)),
            "radius_cap_hits": int(metrics.get("radius_cap_hits", 0)),
            "layer_cap_hits": int(metrics.get("layer_cap_hits", 0)),
            "node_budget_hits": int(metrics.get("node_budget_hits", 0)),
        }
    return {
        "pops": 0,
        "iters": 0,
        "propagations": 0,
        "radius_cap_hits": 0,
        "layer_cap_hits": 0,
        "node_budget_hits": 0,
    }


def extract_labels_from_t1(t1: Any) -> List[str]:
    labels: List[str] = []
    candidates = getattr(t1, "graph_deltas", None) or getattr(t1, "deltas", None) or []
    if isinstance(candidates, list):
        for entry in candidates:
            if isinstance(entry, dict):
                val = entry.get("label") or entry.get("name")
                if val:
                    labels.append(str(val))
    return sorted({str(x) for x in labels})


def extract_t2_retrieved(t2: Any, k_retrieval: int) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    retrieved = getattr(t2, "retrieved", [])
    for record in retrieved or []:
        speaker = None
        if isinstance(record, dict):
            rid = str(record.get("id"))
            score = float(record.get("_score", record.get("score", 0.0)) or 0.0)
            owner = str(record.get("owner", "any"))
            quarter = str(record.get("quarter", ""))
            text = record.get("text")
            tags = record.get("tags")
            speaker = record.get("speaker")
        else:
            rid = str(getattr(record, "id", ""))
            score = float(getattr(record, "score", 0.0) or 0.0)
            owner = str(getattr(record, "owner", "any"))
            quarter = str(getattr(record, "quarter", ""))
            text = getattr(record, "text", None)
            tags = getattr(record, "tags", None)
            speaker = getattr(record, "speaker", None)
        if not rid:
            continue
        tags_list = list(tags) if isinstance(tags, list) else None
        if speaker is None and tags_list:
            for t in tags_list:
                if isinstance(t, str) and t.startswith("speaker:"):
                    speaker = t.split(":", 1)[1] or None
                    break
        out.append(
            {
                "id": rid,
                "score": score,
                "owner": owner,
                "quarter": quarter,
                **({"text": str(text)} if text else {}),
                **({"tags": tags_list} if tags_list else {}),
                **({"speaker": str(speaker)} if speaker else {}),
            }
        )
    out.sort(key=lambda e: (-float(e.get("score", 0.0)), str(e.get("id", ""))))
    return out[: max(int(k_retrieval), 0)]


def extract_t2_metrics(t2: Any) -> Dict[str, Any]:
    metrics_obj = getattr(t2, "metrics", None)
    if isinstance(metrics_obj, dict):
        out: Dict[str, Any] = {}
        tier_sequence = metrics_obj.get("tier_sequence")
        if isinstance(tier_sequence, list):
            out["tier_sequence"] = [str(x) for x in tier_sequence]
        sim_stats = metrics_obj.get("sim_stats")
        if isinstance(sim_stats, dict):
            out["sim_stats"] = {
                "mean": float(sim_stats.get("mean", 0.0)),
                "max": float(sim_stats.get("max", 0.0)),
            }
        if "k_returned" in metrics_obj:
            try:
                out["k_returned"] = int(metrics_obj.get("k_returned", 0))
            except Exception:
                out["k_returned"] = 0
        else:
            try:
                out["k_returned"] = int(len(getattr(t2, "retrieved", []) or []))
            except Exception:
                out["k_returned"] = 0
        if "cache_used" in metrics_obj:
            out["cache_used"] = bool(metrics_obj.get("cache_used", False))
        else:
            out["cache_used"] = False
        return out
    try:
        k_ret = int(len(getattr(t2, "retrieved", []) or []))
    except Exception:
        k_ret = 0
    return {"k_returned": k_ret, "sim_stats": {"mean": 0.0, "max": 0.0}, "cache_used": False}


def cfg_snapshot(ctx: Any) -> Dict[str, Any]:
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


def assemble_bundle(ctx: Any, state: Any, t1: Any, t2: Any) -> PlanBundle:
    now_iso = iso_now(ctx)
    cfg = cfg_snapshot(ctx)
    k_ret = cfg["t2"]["k_retrieval"]

    slice_caps: Dict[str, int] = {}
    try:
        caps = getattr(ctx, "slice_budgets", None) or {}
        if isinstance(caps, dict) and caps.get("t3_ops") is not None:
            slice_caps["t3_ops"] = int(caps.get("t3_ops"))
    except Exception:
        slice_caps = {}

    agent = agent_info(ctx)
    hot_labels, k_hot = world_hot_labels(state)

    t1_nodes = extract_t1_touched_nodes(t1, cap=32)
    t1_metrics = extract_t1_metrics(t1)

    t2_hits = extract_t2_retrieved(t2, k_retrieval=k_ret)
    t2_metrics = extract_t2_metrics(t2)

    input_text = getattr(ctx, "input_text", "") or getattr(ctx, "text", "") or ""
    labels_from_t1 = extract_labels_from_t1(t1)

    return {
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


def make_plan_bundle(ctx: Any, state: Any, t1: Any, t2: Any) -> PlanBundle:
    return assemble_bundle(ctx, state, t1, t2)


def validate_bundle(bundle: PlanBundle) -> List[str]:
    errs: List[str] = []
    for key in ("version", "now", "agent", "world", "t1", "t2", "text", "cfg"):
        if key not in bundle:
            errs.append(f"missing:{key}")
    return errs


__all__ = [
    "BUNDLE_VERSION",
    "agent_info",
    "assemble_bundle",
    "make_plan_bundle",
    "cfg_caps",
    "cfg_snapshot",
    "extract_labels_from_t1",
    "extract_t1_metrics",
    "extract_t1_touched_nodes",
    "extract_t2_metrics",
    "extract_t2_retrieved",
    "iso_now",
    "validate_bundle",
    "world_hot_labels",
]
