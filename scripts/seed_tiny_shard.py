

#!/usr/bin/env python3
"""
Seed a deterministic tiny embedding store for T2 reader parity tests (Gate C).

Outputs (all JSON/JSONL, stable order):
  - vectors.jsonl: per-document records used by the embed store reader.
      {"id": "D001", "owner":"A", "quarter":"2025Q3", "vec":[...], "ts": 1, "importance": 0.11}
  - queries.jsonl: minimal query fixture (not required by tests but useful for local pokes).
      {"q":"anchor","expect_topk":["D007","D003","D002","D001","D005"]}
  - _store_meta.json: store metadata for deterministic partition discovery.
      {"partitions":["owner","quarter"], "dtype":"fp32", "precompute_norms": false}

This script is intentionally dependency-free and uses fixed vectors (no RNG)
so CI is fast and reproducible.
"""
from __future__ import annotations
import argparse
import json
import pathlib
import sys

def _write_jsonl(path: pathlib.Path, rows):
  path.write_text("\n".join(json.dumps(r, separators=(",", ":"), sort_keys=False) for r in rows))

def main() -> int:
  ap = argparse.ArgumentParser()
  ap.add_argument("--out", required=True, help="Output directory for the tiny store")
  args = ap.parse_args()

  out = pathlib.Path(args.out)
  out.mkdir(parents=True, exist_ok=True)

  # Eight tiny, hand-crafted 4D vectors: two owners (A/B), two quarters (2025Q3/Q4).
  base = [
      ("D001", "A", "2025Q3", [1.0,   0.0, 0.0, 0.0]),
      ("D002", "A", "2025Q3", [0.99,  0.01, 0.0, 0.0]),
      ("D003", "B", "2025Q3", [0.98,  0.02, 0.0, 0.0]),
      ("D004", "B", "2025Q4", [0.60,  0.80, 0.0, 0.0]),
      ("D005", "A", "2025Q4", [0.70,  0.71, 0.0, 0.0]),
      ("D006", "A", "2025Q3", [0.00,  1.00, 0.0, 0.0]),
      ("D007", "B", "2025Q3", [0.999, 0.00, 0.0, 0.0]),
      ("D008", "B", "2025Q4", [0.20,  0.98, 0.0, 0.0]),
  ]
  docs = []
  for i, (doc_id, owner, quarter, vec) in enumerate(base, start=1):
    # simple, deterministic recency/importance progression
    docs.append({
        "id": doc_id,
        "owner": owner,
        "quarter": quarter,
        "vec": vec,
        "ts": i,
        "importance": round(0.10 + 0.01 * i, 4)
    })

  _write_jsonl(out / "vectors.jsonl", docs)

  # Minimal query helper; parity tests don't consume this directly but it's handy locally.
  query = {"q": "anchor", "expect_topk": ["D007", "D003", "D002", "D001", "D005"]}
  (out / "queries.jsonl").write_text(json.dumps(query, separators=(",", ":"), sort_keys=False))

  # Store meta guides partition discovery & reader diagnostics; dtype here is advisory.
  meta = {"partitions": ["owner", "quarter"], "dtype": "fp32", "precompute_norms": False}
  (out / "_store_meta.json").write_text(json.dumps(meta, separators=(",", ":"), sort_keys=True))

  # Be quiet in CI; print the path for local convenience.
  print(str(out))
  return 0

if __name__ == "__main__":
  sys.exit(main())