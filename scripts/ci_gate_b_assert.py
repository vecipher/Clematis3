#!/usr/bin/env python3
from __future__ import annotations
import json, sys
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
    any_snap_metric = _any_file_contains(logs, '"snap.')
    any_tier_sequence = _any_file_contains(logs, '"tier_sequence"')

    # In both OFF and ON_NO_REPORT:
    #  - There must be NO metrics objects emitted.
    #  - There must be NO snap.* metrics anywhere.
    #  - There must be NO tier_sequence (reader-only) markers.
    disallowed = []
    if _has_metrics(t2) or _has_metrics(t1):
        disallowed.append("non-empty 'metrics' object present")
    if any_snap_metric:
        disallowed.append("found 'snap.*' metrics in logs")
    if any_tier_sequence:
        disallowed.append("found 'tier_sequence' in logs")

    if disallowed:
        print(f"GateB FAIL [{mode}]: " + "; ".join(disallowed), file=sys.stderr)
        return 1

    print(f"GateB OK [{mode}]: no disallowed metrics or markers detected")
    return 0

if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "OFF"))