from __future__ import annotations
from typing import Any, Dict, Iterator, List, Optional, Tuple
import numpy as np
from numpy.typing import NDArray
from ..engine.types import EpisodeRef
import datetime as dt
import hashlib


class InMemoryIndex:
    def __init__(self) -> None:
        self._eps: List[Dict[str, Any]] = []
        self._ver: int = 0

    def clear(self) -> None:
        """Remove all episodes and reset version counter."""
        self._eps.clear()
        self._ver = 0

    def add(self, ep: Dict[str, Any]) -> None:
        self._eps.append(ep)
        self._ver += 1

    def index_version(self) -> int:
        """Monotonic version that increments on each mutation (add)."""
        return self._ver

    def _filter_owner(
        self, eps: List[Dict[str, Any]], owner: Optional[str]
    ) -> List[Dict[str, Any]]:
        if owner is None:
            return eps
        return [e for e in eps if e.get("owner") == owner]

    def _filter_recent(
        self, eps: List[Dict[str, Any]], recent_days: int, now_utc: dt.datetime
    ) -> List[Dict[str, Any]]:
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

    def _filter_quarters(
        self, eps: List[Dict[str, Any]], quarters: Optional[List[str]]
    ) -> List[Dict[str, Any]]:
        if not quarters:
            return eps
        qset = set(quarters)
        return [e for e in eps if _to_quarter(e.get("ts", "")) in qset]

    def _rank_by_cosine(
        self, eps: List[Dict[str, Any]], q_vec: NDArray[np.float32], k: int, sim_threshold: float
    ) -> List[Tuple[Dict[str, Any], float]]:
        scored: List[Tuple[Dict[str, Any], float]] = []
        for e in eps:
            v = e.get("vec_full")
            if v is None:
                continue
            vec = np.asarray(v, dtype=np.float32)
            s = _cosine(q_vec, vec)
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
        return self._search_with_episodes(self._eps, owner, q_vec, k, tier, hints)

    def _search_with_episodes(
        self,
        episodes: List[Dict[str, Any]],
        owner: Optional[str],
        q_vec: NDArray[np.float32],
        k: int,
        tier: str,
        hints: Dict[str, Any],
    ) -> List[EpisodeRef]:
        all_eps = self._filter_owner(episodes, owner)
        if not all_eps:
            return []
        recent_days = int(hints.get("recent_days", 30))
        sim_threshold_hint = hints.get("sim_threshold", 0.0)
        sim_threshold = float(sim_threshold_hint if sim_threshold_hint is not None else 0.0)
        clusters_top_m = int(hints.get("clusters_top_m", 3))
        archive_quarters = hints.get("archive_quarters")
        now = hints.get("now")
        now_utc = _parse_iso(now) if isinstance(now, str) else dt.datetime.now(dt.timezone.utc)
        results: List[Tuple[Dict[str, Any], float]] = []
        if tier == "exact_semantic":
            eps = self._filter_recent(all_eps, recent_days, now_utc)
            results = self._rank_by_cosine(eps, q_vec, k, sim_threshold)

        elif tier == "cluster_semantic":
            by_cluster: Dict[str, List[Dict[str, Any]]] = {}
            for e in all_eps:
                cid = _stable_cluster_id(e)
                by_cluster.setdefault(cid, []).append(e)
            cluster_scores: List[Tuple[str, float]] = []
            for cid, items in by_cluster.items():
                vecs = [
                    np.asarray(it.get("vec_full"), dtype=np.float32)
                    for it in items
                    if it.get("vec_full") is not None
                ]
                if not vecs:
                    continue
                centroid = np.mean(np.stack(vecs, axis=0), axis=0)
                cs = _cosine(q_vec, centroid)
                cluster_scores.append((cid, cs))
            cluster_scores.sort(key=lambda t: (-t[1], t[0]))
            chosen = {cid for cid, _ in cluster_scores[:clusters_top_m]}

            pool: List[Dict[str, Any]] = []
            for cid in sorted(chosen):
                pool.extend(by_cluster.get(cid, []))
            results = self._rank_by_cosine(pool, q_vec, k, sim_threshold)

        elif tier == "archive":
            eps = self._filter_quarters(all_eps, archive_quarters)
            results = self._rank_by_cosine(eps, q_vec, k, sim_threshold)

        else:
            results = []

        out: List[EpisodeRef] = []
        for e, s in results:
            out.append(
                EpisodeRef(
                    id=str(e["id"]),
                    owner=str(e.get("owner", "")),
                    score=float(s),
                    text=str(e.get("text", "")),
                )
            )
        return out

    # ------------------------------------------------------------------
    # PR68 support: deterministic shard affordance for parallel fan-out
    # ------------------------------------------------------------------

    class _ShardView:
        """Lightweight view over a contiguous slice of episodes.

        The view references the parent index's storage without copying vectors,
        ensuring parallel reads see the same objects and identity paths remain
        unchanged when the gate is OFF.
        """

        def __init__(self, parent: "InMemoryIndex", episodes: List[Dict[str, Any]]):
            self._parent = parent
            self._episodes = episodes

        def search_tiered(
            self,
            owner: Optional[str],
            q_vec: NDArray[np.float32],
            k: int,
            tier: str,
            hints: Dict[str, Any],
        ) -> List[EpisodeRef]:
            return self._parent._search_with_episodes(self._episodes, owner, q_vec, k, tier, hints)

        def __getattr__(self, name: str) -> Any:
            # Forward other attribute access to the parent; needed for methods
            # such as index_version that parallel merge code may touch.
            return getattr(self._parent, name)

    def _iter_shards_for_t2(
        self, tier: str, suggested: int | None = None
    ) -> Iterator["InMemoryIndex | InMemoryIndex._ShardView"]:
        """Yield deterministic shard views for T2 parallel fan-out.

        When no sharding is beneficial (<=1 episode or suggested <=1), yield the
        index itself to preserve the sequential path. Otherwise, partition the
        backing list into contiguous chunks with stable ordering.
        """
        count = len(self._eps)
        if count <= 1:
            yield self
            return

        if suggested is None or suggested <= 1:
            yield self
            return

        chunks = int(suggested)
        if chunks <= 1:
            yield self
            return

        chunks = min(chunks, count)
        size = max(1, (count + chunks - 1) // chunks)

        for start in range(0, count, size):
            end = min(start + size, count)
            if end - start >= count:
                yield self
                return
            view = InMemoryIndex._ShardView(self, self._eps[start:end])
            yield view


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
