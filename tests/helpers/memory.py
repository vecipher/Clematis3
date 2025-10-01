

from __future__ import annotations

from typing import List, Dict, Iterable, Tuple
import hashlib
import random

__all__ = [
    "build_vectors",
    "make_inmemory_rows",
    "partition_rows_round_robin",
    "shard_by_hash",
    "suggest_shards",
]


def build_vectors(*, n: int, dim: int, seed: int = 0) -> List[List[float]]:
    """Deterministic pseudo-random vectors in [-1, 1], no numpy dependency.

    This is only for tests; distribution quality is irrelevant. We avoid Python's
    hash randomization by using `random.Random(seed)` and basic arithmetic.
    """
    rng = random.Random(int(seed))
    vecs: List[List[float]] = []
    for _ in range(int(n)):
        # two 32-bit integers to create a stable float in [-1, 1]
        row = [(rng.randint(0, 2**16 - 1) / 32767.5) * 2.0 - 1.0 for _ in range(int(dim))]
        vecs.append(row)
    return vecs


def make_inmemory_rows(*, n: int = 64, dim: int = 32, seed: int = 0) -> List[Dict[str, object]]:
    """Create synthetic in-memory rows with ids and vectors for T2-oriented tests.

    Schema kept deliberately loose to avoid coupling: {"id": str, "vec": List[float]}.
    """
    vecs = build_vectors(n=n, dim=dim, seed=seed)
    rows: List[Dict[str, object]] = []
    for i, v in enumerate(vecs):
        rid = f"R{i:04d}"
        rows.append({"id": rid, "vec": v})
    return rows


def _stable_hash(s: str) -> int:
    # Use sha1 for cross-process stability (Python's built-in hash is salted).
    return int(hashlib.sha1(s.encode("utf-8")).hexdigest(), 16)


def shard_by_hash(rows: Iterable[Dict[str, object]], buckets: int) -> List[List[Dict[str, object]]]:
    """Deterministically shard rows by a stable hash of their id into `buckets`.

    Returns a list of bucket lists (possibly uneven but stable across runs).
    """
    k = max(1, int(buckets))
    out: List[List[Dict[str, object]]] = [[] for _ in range(k)]
    for r in rows:
        rid = str(r.get("id", ""))
        idx = _stable_hash(rid) % k
        out[idx].append(r)
    return out


def partition_rows_round_robin(rows: Iterable[Dict[str, object]], k: int) -> List[List[Dict[str, object]]]:
    """Deterministically split rows into `k` partitions in round-robin order."""
    kk = max(1, int(k))
    parts: List[List[Dict[str, object]]] = [[] for _ in range(kk)]
    for i, r in enumerate(rows):
        parts[i % kk].append(r)
    return parts


def suggest_shards(total_rows: int, suggested: int = 0) -> int:
    """Return a small deterministic shard count suitable for tiny tests.

    If `suggested<=1`, choose 2 when `total_rows>=2`, else 1. Cap at 4.
    """
    if suggested and suggested > 1:
        return min(int(suggested), 4)
    return 2 if int(total_rows) >= 2 else 1
