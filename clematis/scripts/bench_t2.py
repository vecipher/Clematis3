#!/usr/bin/env python3
# bench_t2.py — Deterministic micro-bench for T2 retrieval (PR69)
# Read-only; no runtime semantics changes. Import-robust; does not require extras.

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    import numpy as np
except Exception as e:  # pragma: no cover
    print("ERROR: numpy is required to run bench_t2", file=sys.stderr)
    raise

# ------------------------------ utilities ----------------------------------

def _seeded_rng(seed: int):
    rs = np.random.RandomState(int(seed))
    return rs

def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0 or b.size == 0:
        return 0.0
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))

# ------------------------------ corpus -------------------------------------

@dataclass
class Episode:
    id: str
    owner: Optional[str]
    ts: str  # ISO8601 UTC
    vec_full: Sequence[float]
    aux: Dict[str, Any]


def _make_ts(base: datetime, days_offset: int) -> str:
    dt = base + timedelta(days=int(days_offset))
    return dt.astimezone(timezone.utc).isoformat()


def build_corpus(n_rows: int, dim: int, seed: int) -> List[Episode]:
    rs = _seeded_rng(seed)
    now = datetime(2025, 10, 1, tzinfo=timezone.utc)
    # Half in current quarter (Q4 2025), half in a past quarter (Q1 2025) to ensure >1 shard
    rows: List[Episode] = []
    for i in range(n_rows):
        eid = f"e{i:06d}"
        if i % 2 == 0:
            ts = _make_ts(datetime(2025, 9, 1, tzinfo=timezone.utc), int(i % 29))  # Q3/Q4 edge
        else:
            ts = _make_ts(datetime(2025, 3, 1, tzinfo=timezone.utc), int(i % 29))  # Q1
        vec = rs.normal(0.0, 1.0, size=(dim,)).astype(np.float32)
        # Light aux; leave cluster_id absent to exercise deterministic fallback
        rows.append(Episode(id=eid, owner=None, ts=ts, vec_full=vec.tolist(), aux={}))
    return rows

# ------------------------------ shard adapter ------------------------------

class _Shard:
    def __init__(self, rows: List[Episode]):
        self._rows = rows

    def search_tiered(self, owner: Optional[str], q_vec: Iterable[float], k: int,
                      tier: str, hints: Dict[str, Any]) -> List[Dict[str, Any]]:
        # Mirror Lance/InMemory search: normalize, filter, score, tie-break by id
        q = np.asarray(q_vec, dtype=np.float32)
        now = hints.get("now")
        if isinstance(now, str):
            try:
                now_dt = datetime.fromisoformat(now.replace("Z", "+00:00")).astimezone(timezone.utc)
            except Exception:
                now_dt = datetime.now(timezone.utc)
        else:
            now_dt = datetime.now(timezone.utc)

        eps: List[Dict[str, Any]] = []
        for e in self._rows:
            if owner is not None and e.owner != owner:
                continue
            eps.append({
                "id": e.id,
                "owner": e.owner,
                "ts": e.ts,
                "vec_full": e.vec_full,
                "aux": e.aux,
            })

        tier = str(tier)
        if tier == "exact_semantic":
            recent_days = hints.get("recent_days")
            if isinstance(recent_days, (int, float)) and recent_days > 0:
                cutoff = now_dt - timedelta(days=float(recent_days))
                def _dt(ts: str) -> datetime:
                    try:
                        return datetime.fromisoformat(ts.replace("Z", "+00:00")).astimezone(timezone.utc)
                    except Exception:
                        return now_dt
                eps = [e for e in eps if _dt(e["ts"]) >= cutoff]
        elif tier == "archive":
            # No-op here; corpus small
            pass
        elif tier == "cluster_semantic":
            pass

        def score(e: Dict[str, Any]) -> float:
            v = np.asarray(e.get("vec_full", []), dtype=np.float32)
            return _cosine(q, v)

        sim_threshold = hints.get("sim_threshold")
        scored: List[Tuple[float, str, Dict[str, Any]]] = []
        for e in eps:
            s = score(e)
            if sim_threshold is not None and s < float(sim_threshold):
                continue
            eid = str(e.get("id"))
            scored.append((s, eid, e))
        scored.sort(key=lambda t: (-t[0], t[1]))
        top = [e for (_, _, e) in scored[: max(int(k), 0)]]
        for s, eid, e in scored[: max(int(k), 0)]:
            e.setdefault("_score", s)
        return top


class BenchIndex:
    """Import-robust index providing `_iter_shards_for_t2` and `search_tiered`.
    Used only for the micro-bench; does not change runtime behavior.
    """
    def __init__(self, rows: List[Episode]):
        self._rows = rows

    def _iter_shards_for_t2(self, tier: str, suggested: int = 0):
        # Partition deterministically by quarter (YYYY-Qx) extracted from ts month
        def quarter_of(ts: str) -> str:
            try:
                dt = datetime.fromisoformat(ts.replace("Z", "+00:00")).astimezone(timezone.utc)
            except Exception:
                dt = datetime.now(timezone.utc)
            q = (dt.month - 1) // 3 + 1
            return f"{dt.year}-Q{q}"
        buckets: Dict[str, List[Episode]] = {}
        for e in self._rows:
            buckets.setdefault(quarter_of(e.ts), []).append(e)
        if len(buckets) <= 1:
            # Two hash buckets by id for stability
            import hashlib
            b0: List[Episode] = []
            b1: List[Episode] = []
            for e in self._rows:
                h = hashlib.sha1(e.id.encode("utf-8")).hexdigest()
                (b0 if (int(h[:8], 16) % 2 == 0) else b1).append(e)
            parts = [b0, b1]
        else:
            parts = [buckets[k] for k in sorted(buckets.keys())]
        # Map to shard objects, drop empties
        shards = [ _Shard(p) for p in parts if p ]
        return shards if len(shards) > 0 else [ _Shard(self._rows) ]

    # Fallback single-shard exact search (not used by bench when shards>1)
    def search_tiered(self, owner: Optional[str], q_vec: Iterable[float], k: int,
                      tier: str, hints: Dict[str, Any]) -> List[Dict[str, Any]]:
        return _Shard(self._rows).search_tiered(owner, q_vec, k, tier, hints)

# ------------------------------ bench core ----------------------------------

def run_bench(iters: int, workers: int, backend: str, dim: int, n_rows: int,
              parallel: bool, seed: int, k: int = 32) -> Dict[str, Any]:
    rows = build_corpus(n_rows=n_rows, dim=dim, seed=seed)

    # Index: try to import real backends; otherwise fall back to BenchIndex
    index_backend = backend.lower()
    index: Any = None
    used_fallback = False

    if index_backend == "inmemory":
        try:
            from clematis.memory.inmemory import InMemoryIndex  # type: ignore
            index = InMemoryIndex()
            if hasattr(index, "add"):
                index.add([e.__dict__ for e in rows])
            else:
                used_fallback = True
        except Exception:
            used_fallback = True
    elif index_backend == "lancedb":
        try:
            from clematis.memory.lance_index import LanceIndex  # type: ignore
            index = LanceIndex(db_path=":memory:")
            index.add([e.__dict__ for e in rows])
        except Exception:
            used_fallback = True
    else:
        used_fallback = True

    if used_fallback or index is None or not hasattr(index, "_iter_shards_for_t2"):
        index = BenchIndex(rows)

    # Build a single query vector (deterministic) and repeat iters times
    rs = _seeded_rng(seed + 1)
    q = rs.normal(0.0, 1.0, size=(dim,)).astype(np.float32)

    # Determine shards deterministically
    try:
        try:
            shards = list(index._iter_shards_for_t2("exact_semantic", suggested=workers))
        except TypeError:
            shards = list(index._iter_shards_for_t2("exact_semantic"))
    except Exception:
        shards = [index]

    shard_count = len(shards)
    effective_workers = min(max(1, int(workers)), shard_count) if parallel else 1

    hints = {
        "now": datetime(2025, 10, 1, tzinfo=timezone.utc).isoformat(),
        "sim_threshold": None,
        "clusters_top_m": 3,
        "recent_days": 365,
    }

    t0 = time.perf_counter()
    for _ in range(max(1, int(iters))):
        if parallel and shard_count > 1 and effective_workers > 1:
            # Execute per-shard searches concurrently (read-only)
            with ThreadPoolExecutor(max_workers=effective_workers) as ex:
                futs = [ex.submit(s.search_tiered, None, q, k, "exact_semantic", hints) for s in shards]
                # Deterministic collection: enumerate in submit order
                results = []
                for f in futs:
                    results.append(f.result())
            # Merge (deterministic by score desc then id asc)
            merged: List[Tuple[float, str, Dict[str, Any]]] = []
            for lst in results:
                for e in lst:
                    merged.append((float(e.get("_score", 0.0)), str(e.get("id")), e))
            merged.sort(key=lambda t: (-t[0], t[1]))
            top = [e for (_, _, e) in merged[:k]]
        else:
            # Sequential across shards, equivalent order
            merged: List[Tuple[float, str, Dict[str, Any]]] = []
            for s in shards:
                for e in s.search_tiered(None, q, k, "exact_semantic", hints):
                    merged.append((float(e.get("_score", 0.0)), str(e.get("id")), e))
            merged.sort(key=lambda t: (-t[0], t[1]))
            top = [e for (_, _, e) in merged[:k]]
    t1 = time.perf_counter()

    # Backfilled metrics (these mirror stage metrics when the gate is ON)
    t2_task_count = shard_count if (parallel and shard_count > 1) else 0
    t2_parallel_workers = effective_workers if t2_task_count > 0 else 0
    t2_partition_count = shard_count if (t2_task_count > 0 and index_backend == "lancedb") else 0

    out = {
        "queries": int(iters),
        "shards": int(shard_count),
        "workers": int(workers),
        "effective_workers": int(effective_workers),
        "backend": index_backend,
        "parallel": bool(parallel),
        "elapsed_ms": round((t1 - t0) * 1000.0, 3),
        "t2_task_count": int(t2_task_count),
        "t2_parallel_workers": int(t2_parallel_workers),
        "t2_partition_count": int(t2_partition_count),
    }
    return out

# ------------------------------ CLI ----------------------------------------

def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Deterministic micro-bench for T2 retrieval (PR69). Delegates to scripts/bench_t2.py")
    p.add_argument("--iters", type=int, default=3, help="Number of query iterations (default: 3)")
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) // 2), help="Worker cap for parallel path (default: half cores)")
    p.add_argument("--backend", choices=["inmemory", "lancedb"], default="inmemory", help="Index backend to simulate/use (default: inmemory)")
    p.add_argument("--dim", type=int, default=32, help="Embedding dimension (default: 32)")
    p.add_argument("--rows", type=int, default=128, help="Corpus size (default: 128)")
    p.add_argument("--seed", type=int, default=1337, help="Deterministic seed (default: 1337)")
    p.add_argument("--k", type=int, default=32, help="Top-K to retrieve (default: 32)")
    p.add_argument("--parallel", action="store_true", help="Enable parallel execution across shards (mirrors stage gate)")
    p.add_argument("--json", action="store_true", help="Emit JSON only")

    args = p.parse_args(argv)

    out = run_bench(
        iters=args.iters,
        workers=args.workers,
        backend=args.backend,
        dim=args.dim,
        n_rows=args.rows,
        parallel=bool(args.parallel),
        seed=args.seed,
        k=args.k,
    )

    if args.json:
        print(json.dumps(out, separators=(",", ":")))
    else:
        print(f"backend={out['backend']} parallel={out['parallel']} iters={out['queries']} shards={out['shards']} workers={out['workers']} ew={out['effective_workers']} elapsed_ms={out['elapsed_ms']}")
        print(f"t2_task_count={out['t2_task_count']} t2_parallel_workers={out['t2_parallel_workers']} t2_partition_count={out['t2_partition_count']}")

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
