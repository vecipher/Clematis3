

#!/usr/bin/env python3
"""
Inspect the latest snapshot.

Usage:
  python3 scripts/inspect_snapshot.py [--dir DIR] [--format pretty|json]

Exit codes:
  0 => snapshot found and printed
  2 => no snapshot found or unreadable
"""
import argparse
import json
import os
import sys

try:
    from clematis.engine.snapshot import get_latest_snapshot_info
except Exception:
    # Fallback: add repo root to sys.path and retry
    HERE = os.path.abspath(os.path.dirname(__file__))
    ROOT = os.path.abspath(os.path.join(HERE, ".."))
    if ROOT not in sys.path:
        sys.path.insert(0, ROOT)
    from clematis.engine.snapshot import get_latest_snapshot_info  # type: ignore


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Inspect the latest snapshot")
    ap.add_argument("--dir", default="./.data/snapshots", help="Snapshot directory")
    ap.add_argument("--format", choices=["pretty", "json"], default="pretty")
    args = ap.parse_args(argv)

    info = get_latest_snapshot_info(args.dir)
    if info is None:
        print(f"No snapshot found in {args.dir}", file=sys.stderr)
        return 2

    if args.format == "json":
        print(json.dumps(info, indent=2, sort_keys=True))
        return 0

    caps = info.get("caps") or {}
    print(f"Snapshot:      {info['path']}")
    print(f"schema_version:{info['schema_version']}")
    print(f"version_etag  :{info['version_etag']}")
    print(f"nodes/edges   :{info['nodes']} / {info['edges']}")
    print(f"last_update   :{info['last_update']}")
    print(
        "caps          : "
        f"delta_l2={caps.get('delta_norm_cap_l2')} "
        f"novelty={caps.get('novelty_cap_per_node')} "
        f"churn={caps.get('churn_cap_edges')} "
        f"weights=[{caps.get('weight_min')}, {caps.get('weight_max')}]"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())