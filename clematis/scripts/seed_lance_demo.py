#!/usr/bin/env python3
"""
Seed a tiny LanceDB table for the Clematis T2 reader.

This is **local-only** helper tooling. CI never invokes this script.

Example:
    ollama pull qwen3:4b-instruct   # (unrelated; just mirrors local setup steps)
    python scripts/seed_lance_demo.py \
        --uri .lancedb \
        --table mem_index \
        --dim 64 \
        --overwrite

What it writes:
  - A LanceDB table with a few deterministic demo rows
    fields: id, owner, created_at, text, importance, owner_quarter, embedding
  - Embeddings are deterministic functions of the text (hash-seeded RNG)

Notes:
  - Keep this dataset small; it’s a smoke/demo only.
  - The main code path will *optionally* use Lance when enabled in config.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import sys
from typing import List

try:
    import numpy as np
except Exception:  # pragma: no cover
    print("[seed-lance] numpy is required: pip install numpy", file=sys.stderr)
    raise

try:
    import lancedb  # type: ignore
except Exception:  # pragma: no cover
    print("[seed-lance] lancedb is required: pip install lancedb", file=sys.stderr)
    raise


# -----------------------------
# Deterministic helpers
# -----------------------------


def _hash_seed(s: str) -> int:
    d = hashlib.sha256(s.encode("utf-8")).digest()
    return int.from_bytes(d[:8], "big", signed=False)


def text_to_vec(text: str, dim: int) -> List[float]:
    """Deterministic pseudo-random embedding for `text`.

    Uses a SHA256-derived seed for RNG; L2-normalizes the vector.
    """
    rng = np.random.default_rng(_hash_seed(text))
    v = rng.normal(size=(dim,)).astype(np.float32)
    n = float(np.linalg.norm(v))
    if n == 0:
        return [0.0] * dim
    return (v / n).tolist()


def _iso(ts: str) -> str:
    # Ensure we only write valid RFC3339-ish timestamps
    try:
        _dt.datetime.strptime(ts, "%Y-%m-%dT%H:%M:%SZ")
        return ts
    except ValueError:
        raise SystemExit(f"bad timestamp: {ts}")


def _owner_quarter(owner: str, ts: str) -> str:
    dt = _dt.datetime.strptime(ts, "%Y-%m-%dT%H:%M:%SZ")
    q = (dt.month - 1) // 3 + 1
    return f"{owner}_{dt.year}Q{q}"


# -----------------------------
# Data
# -----------------------------

_BASE_ROWS = [
    {
        "id": "ep_old",
        "text": "about apple and trees",
        "owner": "demo",
        "created_at": "2025-07-01T00:00:00Z",
    },
    {
        "id": "ep_apple",
        "text": "fresh apple pie story",
        "owner": "demo",
        "created_at": "2025-08-25T00:00:00Z",
    },
    {
        "id": "ep_banana",
        "text": "banana split tale",
        "owner": "demo",
        "created_at": "2025-08-26T00:00:00Z",
    },
    {
        "id": "ep_orange",
        "text": "orange peel facts",
        "owner": "bot",
        "created_at": "2025-08-25T12:00:00Z",
    },
    {
        "id": "ep_grape",
        "text": "grape vineyard notes",
        "owner": "demo",
        "created_at": "2025-08-24T10:00:00Z",
    },
    {
        "id": "ep_pear",
        "text": "pear tart tips",
        "owner": "bot",
        "created_at": "2025-08-23T09:30:00Z",
    },
]


def _expand_rows(n: int) -> List[dict]:
    if n <= len(_BASE_ROWS):
        return _BASE_ROWS[:n]
    out = list(_BASE_ROWS)
    fruits = [
        "kiwi",
        "mango",
        "papaya",
        "plum",
        "peach",
        "cherry",
        "apricot",
        "melon",
        "lime",
        "lemon",
        "fig",
        "date",
    ]
    i = 0
    while len(out) < n:
        f = fruits[i % len(fruits)]
        idx = len(out)
        owner = "demo" if idx % 2 == 0 else "bot"
        day = 20 - (idx % 10)  # a recent-ish spread
        ts = f"2025-08-{day:02d}T08:00:00Z"
        out.append(
            {
                "id": f"ep_{f}_{idx}",
                "text": f"note about {f}",
                "owner": owner,
                "created_at": ts,
            }
        )
        i += 1
    return out


# -----------------------------
# Main
# -----------------------------


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Seed a tiny LanceDB table for Clematis T2 reader")
    ap.add_argument("--uri", default=".lancedb", help="LanceDB URI/directory (default: .lancedb)")
    ap.add_argument("--table", default="mem_index", help="Table name (default: mem_index)")
    ap.add_argument("--dim", type=int, default=64, help="Embedding dimension (default: 64)")
    ap.add_argument("--n", type=int, default=6, help="Number of rows to write (default: 6)")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite table if it exists")
    args = ap.parse_args(argv)

    if args.dim <= 0:
        print("[seed-lance] --dim must be > 0", file=sys.stderr)
        return 2
    if args.n <= 0:
        print("[seed-lance] --n must be > 0", file=sys.stderr)
        return 2

    rows = _expand_rows(args.n)
    for r in rows:
        r["created_at"] = _iso(r["created_at"])  # validate
        r["importance"] = (hash(r["id"]) % 100) / 100.0
        r["owner_quarter"] = _owner_quarter(
            r["owner"], r["created_at"]
        )  # useful for partition-based reads
        r["embedding"] = text_to_vec(r["text"], args.dim)

    print(f"[seed-lance] connecting to: {args.uri}")
    db = lancedb.connect(args.uri)

    # Create or overwrite table deterministically
    mode = "overwrite" if args.overwrite else "create"
    try:
        tbl = db.create_table(args.table, data=rows, mode=mode)
    except Exception as e:
        # If table exists and not overwriting, open it and append missing IDs only
        if "Table already exists" in str(e) and not args.overwrite:
            print("[seed-lance] table exists; opening and upserting unique ids…", file=sys.stderr)
            tbl = db.open_table(args.table)
            # upsert by id to keep it idempotent
            # lancedb supports `tbl.add(data)`; duplicates by id may exist depending on schema
            # we’ll filter out existing ids first for a clean demo
            try:
                existing = set(x["id"] for x in tbl.to_pandas(columns=["id"]).to_dict("records"))
            except Exception:
                existing = set()
            new_rows = [r for r in rows if r["id"] not in existing]
            if new_rows:
                tbl.add(new_rows)
        else:
            raise

    total = tbl.count_rows() if hasattr(tbl, "count_rows") else None
    print(f"[seed-lance] wrote {len(rows)} rows to `{args.table}` (total={total})")

    # Print a tiny sample for sanity
    try:
        sample = tbl.search(text_to_vec("apple", args.dim)).limit(3).to_list()
        print("[seed-lance] sample nearest to 'apple':")
        for s in sample:
            print(
                "  -",
                json.dumps(
                    {k: s.get(k) for k in ("id", "owner", "created_at", "text")}, ensure_ascii=False
                ),
            )
    except Exception:
        # Older lancedb versions may not support search on lists; ignore
        pass

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
