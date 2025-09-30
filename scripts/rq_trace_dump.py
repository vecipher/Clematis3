#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace_dir", default="logs/quality")
    ap.add_argument("--limit", type=int, default=20)
    args = ap.parse_args()
    p = Path(args.trace_dir) / "rq_traces.jsonl"
    if not p.exists():
        print(f"No traces at {p}")
        return
    lines = p.read_text(encoding="utf-8").splitlines()[-args.limit :]
    for ln in lines:
        rec = json.loads(ln)
        print(
            f"[{rec.get('trace_schema_version')}] {rec.get('query_id')} git={rec.get('git_sha')} k={len(rec.get('items', []))}"
        )
    print(f"(showing {len(lines)} of total {sum(1 for _ in p.open(encoding='utf-8'))})")


if __name__ == "__main__":
    main()
