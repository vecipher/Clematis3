#!/usr/bin/env python3
# bench_t2.py — Deterministic micro-bench for T2 retrieval (PR74: M9-12 Bench Kits)
# Read-only; no runtime semantics changes. Import-robust; does not require extras.
# Delegates to scripts/... (wrapper help text requirement)

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import tempfile
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
from concurrent.futures import ThreadPoolExecutor

# Hard-block accidental networking in compliant environments.
os.environ.setdefault("CLEMATIS_NETWORK_BAN", "1")

try:
    import numpy as np
except Exception as e:  # pragma: no cover
    print("ERROR: numpy is required to run bench_t2", file=sys.stderr)
    raise

# ------------------------------ utilities ----------------------------------


def _stable_json(data: Dict[str, Any], pretty: bool = False) -> str:
    if pretty:
        return json.dumps(data, sort_keys=True, indent=2, ensure_ascii=True)
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


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


def _episode_to_dict(ep: Episode) -> Dict[str, Any]:
    return {
        "id": ep.id,
        "owner": ep.owner,
        "ts": ep.ts,
        "vec_full": list(ep.vec_full),
        "aux": dict(ep.aux),
    }


def _make_ts(base: datetime, days_offset: int) -> str:
    dt = base + timedelta(days=int(days_offset))
    return dt.astimezone(timezone.utc).isoformat()


def build_corpus(n_rows: int, dim: int, seed: int) -> List[Episode]:
    rs = _seeded_rng(seed)
    # Half in recent quarter (Q3/Q4 2025), half in a past quarter (Q1 2025) to ensure >1 shard
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

    def search_tiered(
        self, owner: Optional[str], q_vec: Iterable[float], k: int, tier: str, hints: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
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
            eps.append(
                {
                    "id": e.id,
                    "owner": e.owner,
                    "ts": e.ts,
                    "vec_full": e.vec_full,
                    "aux": e.aux,
                }
            )

        tier = str(tier)
        if tier == "exact_semantic":
            recent_days = hints.get("recent_days")
            if isinstance(recent_days, (int, float)) and recent_days > 0:
                cutoff = now_dt - timedelta(days=float(recent_days))

                def _dt(ts: str) -> datetime:
                    try:
                        return datetime.fromisoformat(ts.replace("Z", "+00:00")).astimezone(
                            timezone.utc
                        )
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
        shards = [_Shard(p) for p in parts if p]
        return shards if len(shards) > 0 else [_Shard(self._rows)]

    # Fallback single-shard exact search (not used by bench when shards>1)
    def search_tiered(
        self, owner: Optional[str], q_vec: Iterable[float], k: int, tier: str, hints: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        return _Shard(self._rows).search_tiered(owner, q_vec, k, tier, hints)


# ------------------------------ bench core ----------------------------------


def run_bench(
    iters: int,
    workers: int,
    backend: str,
    dim: int,
    n_rows: int,
    parallel: bool,
    seed: int,
    k: int = 32,
    warmup: int = 1,
) -> Dict[str, Any]:
    rows = build_corpus(n_rows=n_rows, dim=dim, seed=seed)

    # Index: try to import real backends; otherwise fall back to BenchIndex
    index_backend = backend.lower()
    index: Any = None
    used_fallback = False
    lance_tmpdir: Optional[str] = None

    if index_backend == "inmemory":
        try:
            from clematis.memory.index import InMemoryIndex  # type: ignore

            idx = InMemoryIndex()
            for ep in rows:
                idx.add(_episode_to_dict(ep))
            index = idx
        except Exception:
            used_fallback = True
            index = None
    elif index_backend == "lancedb":
        try:
            from clematis.memory.lance_index import LanceIndex  # type: ignore

            lance_tmpdir = tempfile.mkdtemp(prefix="bench_t2_lance_")
            idx = LanceIndex(uri=lance_tmpdir)
            for ep in rows:
                idx.add(_episode_to_dict(ep))
            index = idx
        except Exception:
            if lance_tmpdir is not None:
                shutil.rmtree(lance_tmpdir, ignore_errors=True)
                lance_tmpdir = None
            used_fallback = True
            index = None
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
        "sim_threshold": 0.0,
        "clusters_top_m": 3,
        "recent_days": 365,
    }

    # Warmup (not timed) — prime any lazy paths deterministically
    for _ in range(max(0, int(warmup))):
        _ = shards[0].search_tiered(None, q, k, "exact_semantic", hints)

    t0 = time.perf_counter()
    total_returned = 0
    for _ in range(max(1, int(iters))):
        if parallel and shard_count > 1 and effective_workers > 1:
            # Execute per-shard searches concurrently (read-only)
            with ThreadPoolExecutor(max_workers=effective_workers) as ex:
                results = [
                    ex.submit(s.search_tiered, None, q, k, "exact_semantic", hints).result()
                    for s in shards
                ]
            # Merge (deterministic by score desc then id asc)
            merged: List[Tuple[float, str, Any]] = []
            for lst in results:
                for e in lst:
                    score = float(getattr(e, "score", 0.0))
                    eid = str(getattr(e, "id", ""))
                    merged.append((score, eid, e))
            merged.sort(key=lambda t: (-t[0], t[1]))
            top = [e for (_, _, e) in merged[:k]]
        else:
            # Sequential across shards, equivalent order
            merged: List[Tuple[float, str, Any]] = []
            for s in shards:
                for e in s.search_tiered(None, q, k, "exact_semantic", hints):
                    score = float(getattr(e, "score", 0.0))
                    eid = str(getattr(e, "id", ""))
                    merged.append((score, eid, e))
                merged.sort(key=lambda t: (-t[0], t[1]))
                top = [e for (_, _, e) in merged[:k]]
        total_returned += len(top)
    t1 = time.perf_counter()

    # Backfilled metrics (mirrors stage metrics naming used when the gate is ON)
    t2_task_count = shard_count if (parallel and shard_count > 1) else 0
    t2_parallel_workers = effective_workers if t2_task_count > 0 else 0
    t2_partition_count = shard_count if (t2_task_count > 0 and index_backend == "lancedb") else 0

    out = {
        "backend": index_backend,
        "parallel": bool(parallel),
        "queries": int(iters),
        "iters": int(iters),
        "k": int(k),
        "dim": int(dim),
        "rows": int(n_rows),
        "shards": int(shard_count),
        "workers": int(workers),
        "effective_workers": int(effective_workers),
        "elapsed_ms": round((t1 - t0) * 1000.0, 3),
        # Backfilled metrics as top-level fields for backward compatibility
        "t2_task_count": int(t2_task_count),
        "t2_parallel_workers": int(t2_parallel_workers),
        "t2_partition_count": int(t2_partition_count),
        # Bench metrics
        "metrics": {
            "total_k_returned": int(total_returned),
            "avg_returned_per_query": float(total_returned / max(1, int(iters))),
            "t2_task_count": int(t2_task_count),
            "t2_parallel_workers": int(t2_parallel_workers),
            "t2_partition_count": int(t2_partition_count),
        },
    }
    if lance_tmpdir is not None:
        shutil.rmtree(lance_tmpdir, ignore_errors=True)
    return out


# ------------------------------ CLI ----------------------------------------


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(
        prog="bench_t2",
        description="Deterministic micro-bench for T2 retrieval (PR74). Delegates to scripts/...",
    )
    p.add_argument("--iters", type=int, default=3, help="Number of query iterations (default: 3)")
    p.add_argument(
        "--workers",
        type=int,
        default=max(1, (os.cpu_count() or 4) // 2),
        help="Worker cap for parallel path (default: half cores)",
    )
    p.add_argument(
        "--backend",
        choices=["inmemory", "lancedb"],
        default="inmemory",
        help="Index backend to simulate/use (default: inmemory)",
    )
    p.add_argument("--dim", type=int, default=32, help="Embedding dimension (default: 32)")
    p.add_argument("--rows", type=int, default=256, help="Corpus size (default: 256)")
    p.add_argument("--seed", type=int, default=1337, help="Deterministic seed (default: 1337)")
    p.add_argument("--k", type=int, default=32, help="Top-K to retrieve (default: 32)")
    p.add_argument(
        "--parallel",
        action="store_true",
        help="Enable parallel execution across shards (mirrors stage gate)",
    )
    p.add_argument("--warmup", type=int, default=1, help="Warmup runs before timing (default: 1)")
    p.add_argument("--json", action="store_true", help="Emit JSON only (stable, single line)")

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
        warmup=args.warmup,
    )

    if args.json:
        print(_stable_json(out, pretty=False))
    else:
        # Two-line summary + pretty metrics for quick TTY reads
        print(
            f"backend={out['backend']} parallel={out['parallel']} "
            f"iters={out['queries']} k={out['k']} dim={out['dim']} rows={out['rows']} "
            f"shards={out['shards']} workers={out['workers']} ew={out['effective_workers']} "
            f"elapsed_ms={out['elapsed_ms']}"
        )
        print("metrics:", _stable_json(out["metrics"], pretty=True))

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
