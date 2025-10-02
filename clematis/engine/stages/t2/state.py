# clematis/engine/stages/t2/state.py
from __future__ import annotations

from typing import Any, Dict, List, Optional

# Local/backends
from ....memory.index import InMemoryIndex  # type: ignore


__all__ = [
    "_init_index_from_cfg",
    "gather_changed_labels",
    "build_label_map",
]


def _init_index_from_cfg(state: dict, cfg_t2: dict):
    """
    Instantiate and cache the memory index based on t2.backend with a safe fallback.

    Returns:
        (index, backend_selected: str, fallback_reason: Optional[str])
    """
    idx = state.get("mem_index")
    if idx is not None:
        return (
            idx,
            state.get("mem_backend", str(cfg_t2.get("backend", "inmemory")).lower()),
            state.get("mem_backend_fallback_reason"),
        )

    backend = str(cfg_t2.get("backend", "inmemory")).lower()
    fallback_reason: Optional[str] = None

    if backend == "lancedb":
        try:
            # Deferred import so LanceDB remains strictly optional.
            from clematis.memory.lance_index import LanceIndex  # type: ignore

            lcfg = cfg_t2.get("lancedb", {}) or {}
            idx = LanceIndex(lcfg)
        except Exception as e:  # noqa: BLE001 — we intentionally swallow here and fall back
            idx = InMemoryIndex()
            fallback_reason = f"lancedb_unavailable: {type(e).__name__}"
            backend = "inmemory"
    elif backend == "inmemory":
        idx = InMemoryIndex()
    else:
        # Unknown backend — fall back deterministically.
        idx = InMemoryIndex()
        fallback_reason = f"unknown_backend:{backend}"
        backend = "inmemory"

    # Cache selections in state for subsequent calls.
    state["mem_index"] = idx
    state["mem_backend"] = backend
    if fallback_reason:
        state["mem_backend_fallback_reason"] = fallback_reason

    return idx, backend, fallback_reason


def gather_changed_labels(state: Dict[str, Any], t1) -> List[str]:
    """
    Walk active graphs and collect labels for nodes touched by T1 upserts.
    """
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
            if node and getattr(node, "label", None):
                labels.append(node.label)

    # stable de-dupe preserving first-seen order
    seen = set()
    out: List[str] = []
    for label in labels:
        if label not in seen:
            seen.add(label)
            out.append(label)
    return out


def build_label_map(state: Dict[str, Any]) -> Dict[str, str]:
    """
    Build a lowercase label → node_id map across active graphs.
    """
    store = state.get("store")
    active_graphs = state.get("active_graphs", [])
    out: Dict[str, str] = {}
    if not store:
        return out

    for graph_id in active_graphs:
        graph = store.get_graph(graph_id)
        # Iterate deterministically by node id
        for node_id in sorted(getattr(graph, "nodes", {}).keys()):
            node = graph.nodes.get(node_id)
            if not node:
                continue
            label = getattr(node, "label", None)
            if label:
                out[label.lower()] = node.id
    return out
