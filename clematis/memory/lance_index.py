# Copyright (c) Clematis Project
# PR3: Optional LanceDB-backed index adapter
#
# Notes
# - This module **defers** importing `lancedb`/`pyarrow` until runtime in __init__
#   so that importing the module itself never raises ImportError. The T2 factory
#   can catch ImportError from constructing LanceIndex and cleanly fall back.
# - Cosine similarity and tie-breaking are done on the Python side (numpy) to
#   preserve determinism and parity with InMemoryIndex.
# - We intentionally keep the API surface identical to InMemoryIndex:
#     * add(ep: dict) -> None
#     * search_tiered(owner, q_vec, k, tier, hints) -> List[dict-like EpisodeRef]
#     * index_version() -> int
# - The adapter stores a small `meta` table with a single counter row for a
#   monotonic version; it is incremented on each successful add().

from __future__ import annotations

import json
import logging
import math
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ----------------------------- helper utilities -----------------------------


def _parse_iso8601(ts: str) -> datetime:
    """Parse ISO8601 timestamps robustly; treat naive as UTC."""
    if ts.endswith("Z"):
        ts = ts[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(ts)
    except ValueError:
        # last resort: try without timezone
        dt = datetime.fromisoformat(ts.split("+")[0])
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _quarter_of(dt: datetime) -> str:
    q = (dt.month - 1) // 3 + 1
    return f"{dt.year}Q{q}"


def _as_float32_list(vec: Any) -> List[float]:
    arr = np.asarray(vec, dtype=np.float32)
    return arr.tolist()


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    # Safe cosine with deterministic tie behavior on zeros
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)


# ------------------------------- main adapter -------------------------------


class LanceIndex:
    """
    LanceDB-backed implementation of the InMemoryIndex interface.
    Import of lancedb/pyarrow is deferred to construction time to allow the
    caller to gracefully handle environments where Lance is unavailable.
    """

    def __init__(
        self, uri: str, table: str = "episodes", meta_table: str = "meta", create_ok: bool = True
    ):
        # Defer heavy imports here for graceful fallback
        try:
            import lancedb  # type: ignore
            import pyarrow as pa  # type: ignore
        except Exception as e:
            # Surface ImportError to caller so T2 can fallback
            raise ImportError("lancedb (and pyarrow) are required for LanceIndex") from e

        self._lancedb = __import__("lancedb")
        self._pa = __import__("pyarrow")
        self._uri = uri
        self._table_name = table
        self._meta_name = meta_table
        self._create_ok = create_ok

        # Connect DB
        self._db = self._lancedb.connect(uri)

        # Open or create meta table immediately (we need version even on empty DB)
        self._meta = self._open_or_create_meta()
        self._version_cache: Optional[int] = None

        # Episodes table is opened lazily; if it does not exist, we create it on first add().
        self._episodes = self._open_table_if_exists(self._table_name)

    # ---------------------------- table management ----------------------------

    def _open_table_if_exists(self, name: str):
        try:
            return self._db.open_table(name)
        except Exception:
            return None

    def _open_or_create_meta(self):
        pa = self._pa
        tbl = self._open_table_if_exists(self._meta_name)
        if tbl is not None:
            # Ensure a counter row exists; if table is empty, seed it
            try:
                arrow = tbl.to_arrow()
            except Exception:
                arrow = None
            if arrow is None or (hasattr(arrow, "num_rows") and arrow.num_rows == 0):
                tbl.add([{"key": "counter", "version": 0}])
            return tbl
        # Create with a single seed row
        schema = pa.schema(
            [
                pa.field("key", pa.string()),
                pa.field("version", pa.int64()),
            ]
        )
        meta = self._db.create_table(
            self._meta_name, data=[{"key": "counter", "version": 0}], schema=schema
        )
        return meta

    def _ensure_episodes_table(self, first_row: Dict[str, Any]):
        if self._episodes is not None:
            return
        # Let Lance infer schema from the first row (handles dicts and lists well)
        self._episodes = self._db.create_table(self._table_name, data=[first_row])

    # ------------------------------- public API -------------------------------

    def add(self, ep: Dict[str, Any]) -> None:
        """Upsert an episode; bump version monotonically on success.

        Expected fields (will be filled if missing):
          id:str, owner:str, text:str, tags:List[str], ts:ISO8601 str,
          vec_full: List[float32], aux: Dict[str,Any], quarter:str
        """
        # Basic normalization
        ep = dict(ep)  # copy
        if not ep.get("id"):
            ep["id"] = str(uuid.uuid4())
        if not ep.get("ts"):
            ep["ts"] = datetime.now(timezone.utc).isoformat()
        dt = _parse_iso8601(ep["ts"])  # normalized and validated
        ep.setdefault("quarter", _quarter_of(dt))
        # vector
        if "vec_full" not in ep:
            raise ValueError("Episode missing 'vec_full'")
        ep["vec_full"] = _as_float32_list(ep["vec_full"])  # JSON-serializable
        # collections
        ep.setdefault("tags", [])
        ep.setdefault("aux", {})
        if not isinstance(ep["tags"], list):
            ep["tags"] = list(ep["tags"])  # best effort
        if not isinstance(ep["aux"], dict):
            # Accept JSON string and decode; otherwise wrap in dict
            try:
                ep["aux"] = json.loads(ep["aux"])  # type: ignore[arg-type]
            except Exception:
                ep["aux"] = {"_value": ep["aux"]}

        # Ensure table exists (let Lance infer schema from this first row)
        self._ensure_episodes_table(first_row=ep)

        # Upsert behavior: delete existing id then insert
        try:
            # lancedb .delete supports a SQL-like predicate
            self._episodes.delete(f"id == '{ep['id']}'")  # type: ignore[union-attr]
        except Exception:
            # best-effort; not fatal
            pass
        # Insert
        self._episodes.add([ep])  # type: ignore[union-attr]

        # Bump version counter
        self._bump_version()

    def search_tiered(
        self,
        owner: Optional[str],
        q_vec: Iterable[float],
        k: int,
        tier: str,
        hints: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Search episodes using deterministic cosine + tie by id.

        Tier behaviors:
          - exact_semantic: optional owner filter; optional recent_days cutoff.
          - cluster_semantic: pick top-M clusters by centroid cosine, then rank within.
          - archive: optional filter by set of quarters in hints['archive_quarters'].
        """
        if self._episodes is None:
            return []

        q = np.asarray(q_vec, dtype=np.float32)
        now = hints.get("now")
        if isinstance(now, str):
            now_dt = _parse_iso8601(now)
        elif isinstance(now, datetime):
            now_dt = now.astimezone(timezone.utc)
        else:
            now_dt = datetime.now(timezone.utc)

        # Fetch full table once; for our current scale this is acceptable and keeps
        # the math identical to the in-memory implementation.
        try:
            arrow = self._episodes.to_arrow()  # type: ignore[union-attr]
            rows = arrow.to_pylist()
        except Exception:
            try:
                df = self._episodes.to_pandas()  # type: ignore[union-attr]
                rows = df.to_dict(orient="records")
            except Exception:
                # Last resort: iterate
                rows = []
                for batch in self._episodes.to_batches():  # type: ignore[union-attr]
                    rows.extend(batch.to_pylist())

        # Normalize fields and precompute
        eps: List[Dict[str, Any]] = []
        for r in rows:
            # Defensive copies
            e = dict(r)
            # aux may arrive as JSON string/bytes depending on schema inference
            aux = e.get("aux", {})
            if isinstance(aux, (bytes, bytearray)):
                try:
                    aux = json.loads(aux.decode("utf-8"))
                except Exception:
                    aux = {"_bytes": True}
            elif isinstance(aux, str):
                try:
                    aux = json.loads(aux)
                except Exception:
                    aux = {"_text": aux}
            e["aux"] = aux if isinstance(aux, dict) else {}
            # ts normalization
            try:
                e_dt = _parse_iso8601(e.get("ts"))
            except Exception:
                e_dt = now_dt
            e["_dt"] = e_dt
            # quarter fill
            e.setdefault("quarter", _quarter_of(e_dt))
            # owner
            if owner is not None and e.get("owner") != owner:
                continue
            eps.append(e)

        # Apply tier-specific prefiltering
        tier = str(tier)
        if tier == "exact_semantic":
            recent_days = hints.get("recent_days")
            if isinstance(recent_days, (int, float)) and recent_days > 0:
                cutoff = now_dt - timedelta(days=float(recent_days))
                eps = [e for e in eps if e["_dt"] >= cutoff]
        elif tier == "archive":
            qset = hints.get("archive_quarters")
            if isinstance(qset, (set, list, tuple)):
                qset = set(map(str, qset))
                eps = [e for e in eps if str(e.get("quarter")) in qset]
        elif tier == "cluster_semantic":
            pass  # handled below
        else:
            logger.warning("Unknown tier '%s'; defaulting to exact_semantic behavior", tier)

        # Vectorize and score
        def score_episode(e: Dict[str, Any]) -> float:
            v = np.asarray(e.get("vec_full", []), dtype=np.float32)
            return _cosine(q, v)

        sim_threshold = hints.get("sim_threshold")

        if tier == "cluster_semantic":
            # Group by cluster_id (from aux if present; else deterministic hash of id)
            import hashlib

            def cluster_of(e: Dict[str, Any]) -> str:
                cid = None
                aux = e.get("aux") or {}
                if isinstance(aux, dict):
                    cid = aux.get("cluster_id")
                if not cid:
                    cid = hashlib.sha1(str(e.get("id")).encode("utf-8")).hexdigest()[:8]
                return str(cid)

            clusters: Dict[str, List[Dict[str, Any]]] = {}
            for e in eps:
                clusters.setdefault(cluster_of(e), []).append(e)

            # Compute centroid sims per cluster
            centroids: List[Tuple[str, float]] = []
            for cid, items in clusters.items():
                if not items:
                    continue
                mat = np.stack(
                    [np.asarray(it.get("vec_full", []), dtype=np.float32) for it in items], axis=0
                )
                centroid = mat.mean(axis=0)
                centroids.append((cid, _cosine(q, centroid)))
            # Pick top-M clusters
            M = int(hints.get("clusters_top_m", 3))
            centroids.sort(key=lambda p: (-p[1], p[0]))
            top_cids = {cid for cid, _ in centroids[: max(M, 0)]}
            eps = [e for e in eps if cluster_of(e) in top_cids]

        # Rank by cosine then tie-break by id (asc)
        scored: List[Tuple[float, str, Dict[str, Any]]] = []
        for e in eps:
            s = score_episode(e)
            if sim_threshold is not None and s < float(sim_threshold):
                continue
            eid = str(e.get("id"))
            scored.append((s, eid, e))

        scored.sort(key=lambda t: (-t[0], t[1]))
        top = [e for (_, _, e) in scored[: max(int(k), 0)]]

        # Optionally attach the score for downstream debugging (harmless extra key)
        for s, eid, e in scored[: max(int(k), 0)]:
            e.setdefault("_score", s)

        return top

    def index_version(self) -> int:
        if self._version_cache is not None:
            return self._version_cache
        # Read from meta table
        try:
            arrow = self._meta.to_arrow()
            rows = arrow.to_pylist()
        except Exception:
            try:
                rows = self._meta.to_pandas().to_dict(orient="records")
            except Exception:
                rows = []
        ver = 0
        for r in rows:
            if r.get("key") == "counter":
                ver = int(r.get("version", 0))
                break
        self._version_cache = ver
        return ver

    # ------------------------------- meta helpers ------------------------------

    def _bump_version(self) -> None:
        # Best-effort delete+add to avoid relying on update/merge APIs
        try:
            self._meta.delete("key == 'counter'")
        except Exception:
            pass
        current = (self._version_cache or 0) + 1
        self._meta.add([{"key": "counter", "version": int(current)}])
        self._version_cache = int(current)
