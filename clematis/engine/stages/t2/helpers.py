from __future__ import annotations

import datetime as dt
import hashlib
import json
from typing import Any, Dict, Iterable, List, Optional


def items_for_fusion(retrieved_list: Iterable[Any]) -> List[Dict[str, Any]]:
    retrieved = list(retrieved_list or [])
    total = len(retrieved)
    out: List[Dict[str, Any]] = []
    for idx, ref in enumerate(retrieved):
        score_surrogate = float(total - idx)
        out.append(
            {
                "id": getattr(ref, "id", None),
                "score": score_surrogate,
                "text": getattr(ref, "text", ""),
            }
        )
    return out


def gather_changed_labels(state: Dict[str, Any], t1) -> List[str]:
    store = state.get("store")
    active_graphs = state.get("active_graphs", [])
    if store is None:
        return []
    node_ids = {
        delta.get("id")
        for delta in getattr(t1, "graph_deltas", [])
        if delta.get("op") == "upsert_node"
    }
    labels: List[str] = []
    for graph_id in active_graphs:
        graph = store.get_graph(graph_id)
        for node_id in sorted(node_ids):
            node = graph.nodes.get(node_id)
            if node and node.label:
                labels.append(node.label)
    seen = set()
    out: List[str] = []
    for label in labels:
        if label not in seen:
            seen.add(label)
            out.append(label)
    return out


def build_label_map(state: Dict[str, Any]) -> Dict[str, str]:
    store = state.get("store")
    active_graphs = state.get("active_graphs", [])
    out: Dict[str, str] = {}
    if not store:
        return out
    for graph_id in active_graphs:
        graph = store.get_graph(graph_id)
        for node in graph.nodes.values():
            if node.label:
                out[node.label.lower()] = node.id
    return out


def parse_iso(ts: Optional[str]) -> dt.datetime:
    try:
        return dt.datetime.fromisoformat((ts or "").replace("Z", "+00:00")).astimezone(
            dt.timezone.utc
        )
    except Exception:
        return dt.datetime.now(dt.timezone.utc)


def owner_for_query(ctx, cfg_t2: Dict[str, Any]) -> Optional[str]:
    scope = str(cfg_t2.get("owner_scope", "any")).lower()
    if scope == "agent":
        return getattr(ctx, "agent_id", None)
    if scope == "world":
        return "world"
    return None


class EpRefShim:
    __slots__ = ("id", "text", "score")

    def __init__(self, payload: Dict[str, Any]):
        self.id = str(payload.get("id"))
        self.text = payload.get("text", "")
        score = payload.get("score", payload.get("_score", 0.0))
        try:
            self.score = float(score)
        except Exception:
            self.score = 0.0


def quality_digest(qcfg: Dict[str, Any]) -> str:
    keys = ["enabled", "lexical", "fusion", "mmr"]
    slim = {key: qcfg.get(key) for key in keys if key in qcfg}
    json_blob = json.dumps(slim, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(json_blob.encode("utf-8")).hexdigest()[:12]
