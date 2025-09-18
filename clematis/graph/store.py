from __future__ import annotations
from typing import Any, Dict, List, Literal
import hashlib
import numpy as np
from numpy.typing import NDArray
from ..engine.types import ConceptGraph, Node, Edge

class InMemoryGraphStore:
    def __init__(self) -> None:
        self._graphs: Dict[str, ConceptGraph] = {}

    def ensure(self, gid: str) -> ConceptGraph:
        if gid not in self._graphs:
            self._graphs[gid] = ConceptGraph(graph_id=gid, version_etag="v0")
        return self._graphs[gid]

    # ConceptGraphStore API
    def get_graph(self, gid: str) -> ConceptGraph:
        return self.ensure(gid)

    def upsert_nodes(self, gid: str, nodes: List[Node]) -> None:
        g = self.ensure(gid)
        for n in nodes:
            g.nodes[n.id] = n
        self._bump_etag(g)

    def upsert_edges(self, gid: str, edges: List[Edge]) -> None:
        g = self.ensure(gid)
        for e in edges:
            g.edges[e.id] = e
        self._bump_etag(g)

    def apply_deltas(self, gid: str, deltas: List[Dict[str, Any]]) -> Dict[str, Any]:
        g = self.ensure(gid)
        edits = 0
        for d in deltas:
            if d.get("op") == "upsert_edge":
                eid = d.get("id") or f"e:{d['src']}->{d['dst']}"
                g.edges[eid] = Edge(id=eid, src=d["src"], dst=d["dst"], weight=float(d["weight"]), rel=d.get("rel","associates"))
                edits += 1
            elif d.get("op") == "upsert_node":
                nid = d["id"]
                g.nodes[nid] = g.nodes.get(nid) or Node(id=nid, label=d.get("label", nid))
                edits += 1
        if edits:
            self._bump_etag(g)
        return {"edits": edits}

    def surface_view(self, gid: str, node_ids: List[str], k: int, method: Literal["PCA","TopK"]) -> Dict[str, NDArray[np.float32]]:
        # Placeholder: returns zero vectors of size k
        return {nid: np.zeros((k,), dtype=np.float32) for nid in node_ids}

    def csr(self, gid: str) -> Any:
        # Placeholder structure for adjacency
        g = self.ensure(gid)
        adj = {e.src: [] for e in g.edges.values()}
        for e in g.edges.values():
            adj.setdefault(e.src, []).append((e.dst, e))
        return adj

    def csc(self, gid: str) -> Any:
        g = self.ensure(gid)
        adj = {e.dst: [] for e in g.edges.values()}
        for e in g.edges.values():
            adj.setdefault(e.dst, []).append((e.src, e))
        return adj

    def version_etag(self, gid: str) -> str:
        g = self.ensure(gid)
        return g.version_etag

    def _bump_etag(self, g: ConceptGraph) -> None:
        h = hashlib.sha1()
        h.update(str(len(g.nodes)).encode())
        h.update(str(len(g.edges)).encode())
        g.version_etag = h.hexdigest()[:12]
