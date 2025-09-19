from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from numpy.typing import NDArray
from ..engine.types import EpisodeRef
import datetime as dt
import hashlib

class InMemoryIndex:
    def __init__(self) -> None:
        self._eps: List[Dict[str, Any]] = []

    def add(self, ep: Dict[str, Any]) -> None:
        self._eps.append(ep)

    def search_tiered(self, owner: Optional[str], q_vec: NDArray[np.float32], k: int, tier: str, hints: Dict[str, Any]) -> List[EpisodeRef]:
        # Placeholder: returns empty list
        return []
    def _filter_owner(self, eps: List[Dict[str, Any]], owner: Optional[str]) -> List[Dict[str, Any]]:
        if owner is None:
            return eps
        return [e for e in eps if e.get("owner") == owner]
    def _filter_recent(self, eps: List[Dict[str, Any]], recent_days: int, now_utc: dt.datetime) -> List[Dict[str, Any]]:
        if not recent_days or recent_days <= 0:
            return eps
        cutoff = now_utc - dt.timedelta(days=int(recent_days))
        out: List[Dict[str, Any]] = []
        for e in eps:
            ts = e.get("ts") or ""
            te = _parse_iso(ts)
            if te >= cutoff:
                out.append(e)
        return out
    def _filter_quarters(self, eps: List[Dict[str, Any]], quarters: Optional[List[str]]) -> List[Dict[str, Any]]:
        if not quarters:
            return eps
        qset = set(quarters)
        return [e for e in eps if _to_quarter(e.get("ts", "")) in qset]
    def _rank_by_cosine(self, eps: List[Dict[str, Any]], q_vec: NDArray[np.float32], k: int, sim_threshold: float) -> List[Tuple[Dict[str, Any], float]]:
        scored: List[Tuple[Dict[str, Any], float]] = []
        for e in eps:
            v = e.get("vec_full")
            if v is None:
                continue
            s = _cosine(q_vec, v)
            if s >= sim_threshold:
                scored.append((e, s))
        # Deterministic: sort by (-score, id)
        scored.sort(key=lambda t: (-t[1], str(t[0].get("id"))))
        return scored[:k]
    def search_tiered(
        self,
        owner: Optional[str],
        q_vec: NDArray[np.float32],
        k: int,
        tier: str,
        hints: Dict[str, Any],
    ) -> List[EpisodeRef]:
        """
        Deterministic, in-memory retrieval with three modes:
          - exact_semantic: owner + recent_days filter, threshold on cosine
          - cluster_semantic: route to top-M clusters by centroid sim, then rank within those clusters
          - archive: optional quarter filter (hints['archive_quarters'])
        Hints (all optional, with sensible defaults):
          - recent_days: int (default 30)
          - sim_threshold: float (default 0.0)
          - clusters_top_m: int (default 3)
          - archive_quarters: List[str], e.g., ["2024Q1","2023Q4"]
          - now: ISO8601 "now" for tests (default: current UTC)
        """
        all_eps = self._filter_owner(self._eps, owner)
        if not all_eps:
            return []
        recent_days = int(hints.get("recent_days", 30))
        sim_threshold = float(hints.get("sim_threshold", 0.0))
        clusters_top_m = int(hints.get("clusters_top_m", 3))
        archive_quarters = hints.get("archive_quarters")
        now = hints.get("now")
        now_utc = _parse_iso(now) if isinstance(now, str) else dt.datetime.now(dt.timezone.utc)
        results: List[Tuple[Dict[str, Any], float]] = []
        if tier == "exact_semantic":
            eps = self._filter_recent(all_eps, recent_days, now_utc)
            results = self._rank_by_cosine(eps, q_vec, k, sim_threshold)

        elif tier == "cluster_semantic":
            # Group by cluster id
            by_cluster: Dict[str, List[Dict[str, Any]]] = {}
            for e in all_eps:
                cid = _stable_cluster_id(e)
                by_cluster.setdefault(cid, []).append(e)
            # Compute centroids and choose top-M clusters by centroid similarity
            cluster_scores: List[Tuple[str, float]] = []
            for cid, items in by_cluster.items():
                vecs = [it.get("vec_full") for it in items if it.get("vec_full") is not None]
                if not vecs:
                    continue
                centroid = np.mean(np.stack(vecs, axis=0), axis=0).astype(np.float32)
                cs = _cosine(q_vec, centroid)
                cluster_scores.append((cid, cs))
            cluster_scores.sort(key=lambda t: (-t[1], t[0]))
            chosen = {cid for cid, _ in cluster_scores[:clusters_top_m]}

            # Rank within chosen clusters
            pool: List[Dict[str, Any]] = []
            for cid in sorted(chosen):
                pool.extend(by_cluster.get(cid, []))
            results = self._rank_by_cosine(pool, q_vec, k, sim_threshold)

        elif tier == "archive":
            eps = self._filter_quarters(all_eps, archive_quarters)
            results = self._rank_by_cosine(eps, q_vec, k, sim_threshold)

        else:
            # Unknown tier: return empty deterministic list
            results = []

        return [EpisodeRef(id=str(e["id"]), owner=str(e.get("owner", "")), score=float(s), text=str(e.get("text", ""))) for e, s in results]                  
    
def _cosine(a: NDArray[np.float32], b: NDArray[np.float32]) -> float:
    a = a.astype(np.float32, copy=False)
    b = b.astype(np.float32, copy=False)
    na = float(np.linalg.norm(a)) or 1.0
    nb = float(np.linalg.norm(b)) or 1.0
    return float(np.dot(a, b) / (na * nb))

def _parse_iso(ts: str) -> dt.datetime:
    # Accepts ISO8601; fall back to utcnow on error
    try:
        return dt.datetime.fromisoformat(ts.replace("Z", "+00:00")).astimezone(dt.timezone.utc)
    except Exception:
        return dt.datetime.now(dt.timezone.utc)
    
def _to_quarter(ts: str) -> str:
    t = _parse_iso(ts)
    q = (t.month - 1) // 3 + 1
    return f"{t.year}Q{q}"

def _stable_cluster_id(ep: Dict[str, Any]) -> str:
    # Prefer explicit cluster_id if provided
    c = (ep.get("aux") or {}).get("cluster_id")
    if c:
        return str(c)
    # Otherwise derive a stable id from episode id or text
    base = str(ep.get("id") or ep.get("text", ""))
    h = hashlib.md5(base.encode("utf-8")).hexdigest()[:8]
    return f"c:{h}"