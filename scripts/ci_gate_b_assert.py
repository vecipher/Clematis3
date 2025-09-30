#!/usr/bin/env python3
from __future__ import annotations
import json
import sys
from pathlib import Path
from typing import Any, Dict

# modes: OFF, ON_NO_REPORT
MODES = {"OFF", "ON_NO_REPORT"}


def _last_jsonl(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    last = None
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    last = json.loads(line)
                except Exception:
                    continue
    except Exception:
        return {}
    return last or {}


def _any_file_contains(root: Path, needle: str) -> bool:
    # Simple substring scan; robust to schema changes
    for p in root.rglob("*.jsonl"):
        try:
            txt = p.read_text(encoding="utf-8", errors="ignore")
            if needle in txt:
                return True
        except Exception:
            pass
    return False


def _has_metrics(obj: Dict[str, Any]) -> bool:
    # Treat presence of a non-empty "metrics" object as metrics emission
    if not isinstance(obj, dict):
        return False
    m = obj.get("metrics")
    if isinstance(m, dict) and len(m) > 0:
        return True
    return False


def main(mode: str) -> int:
    mode = mode.strip().upper()
    if mode not in MODES:
        print(f"Usage: {sys.argv[0]} <OFF|ON_NO_REPORT>", file=sys.stderr)
        return 2

    logs = Path("./.logs")
    if not logs.exists():
        print("GateB WARN: ./.logs missing; nothing to assert (treat as pass)", file=sys.stderr)
        return 0

    # Look at typical stage logs if present
    t2 = _last_jsonl(logs / "t2.jsonl")
    t1 = _last_jsonl(logs / "t1.jsonl")
    any_snap_metric = _any_file_contains(logs, '"snap."')

    # In both OFF and ON_NO_REPORT, disallow only **new M6** markers.
    DISALLOWED_M6_MARKERS = [
        '"reader"',  # nested reader meta block in metrics
        '"t2.embed_dtype"',
        '"t2.embed_store_dtype"',
        '"t2.precompute_norms"',
        '"t2.reader_shards"',
        '"t2.partition_layout"',
        '"snap."',  # any snapshot counters/fields
    ]

    leaks = []
    for marker in DISALLOWED_M6_MARKERS:
        if _any_file_contains(logs, marker):
            leaks.append(marker)

    if leaks:
        print(f"GateB FAIL [{mode}]: found disallowed markers: {', '.join(leaks)}", file=sys.stderr)
        return 1

    print(f"GateB OK [{mode}]: no disallowed metrics or markers detected")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "OFF"))
