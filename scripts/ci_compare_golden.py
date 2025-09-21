

#!/usr/bin/env python3
"""
CI Golden Identity Guard (disabled path)

- Runs a minimal turn with scheduler.enabled=false to produce logs under ./.logs/
- Normalizes volatile fields and compares against fixtures in tests/golden/pre_m5_disabled/
- Exits non-zero on any mismatch
- Use --update to refresh fixtures from current normalized output

Note: The CI workflow currently invokes scripts/ci_compare_golden.py.
If you prefer that location, either duplicate this file there or update the workflow.
"""
from __future__ import annotations

import argparse
import difflib
import json
import os
import sys
from typing import Any, Dict, List

# Ensure repo root on sys.path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# Lazy imports after path setup
from clematis.io.config import load_config  # type: ignore
from clematis.io.paths import logs_dir      # type: ignore
from clematis.world.scenario import run_one_turn  # type: ignore

VOLATILE_KEYS = {
    "ms",
    "now",
    "timestamp",
    "ts",
    "elapsed_ms",
    "durations_ms",
    "uuid",
    "run_id",
}


def _normalize(obj: Any) -> Any:
    """Recursively remove volatile keys and sort dict keys for deterministic encoding."""
    if isinstance(obj, dict):
        # Drop None values for stability only if they are in VOLATILE_KEYS; keep semantic None elsewhere
        cleaned: Dict[str, Any] = {}
        for k in sorted(obj.keys()):
            if k in VOLATILE_KEYS:
                continue
            v = obj[k]
            cleaned[k] = _normalize(v)
        return cleaned
    if isinstance(obj, list):
        return [_normalize(x) for x in obj]
    return obj


def _normalize_jsonl(path: str) -> List[str]:
    if not os.path.exists(path):
        return []
    lines: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue
            try:
                rec = json.loads(raw)
            except Exception:
                # Skip malformed lines
                continue
            norm = _normalize(rec)
            lines.append(json.dumps(norm, sort_keys=True, separators=(",", ":")))
    return lines


def _write_lines(path: str, lines: List[str]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(line + "\n")


def _disable_scheduler(cfg: Any) -> Any:
    """Set scheduler.enabled = False on either dict-like or attr-like config objects."""
    try:
        sched = getattr(cfg, "scheduler", None)
    except Exception:
        sched = None
    if sched is None:
        try:
            # attach a dict if attribute missing
            setattr(cfg, "scheduler", {"enabled": False})
            return cfg
        except Exception:
            pass
    if isinstance(sched, dict):
        sched["enabled"] = False
    else:
        try:
            setattr(sched, "enabled", False)
        except Exception:
            # Best-effort: replace with dict
            try:
                setattr(cfg, "scheduler", {"enabled": False})
            except Exception:
                pass
    return cfg


def _run_disabled_once(cfg_path: str) -> bool:
    cfg = load_config(cfg_path)
    cfg = _disable_scheduler(cfg)

    # Clean logs dir (idempotent)
    ld = logs_dir()
    os.makedirs(ld, exist_ok=True)
    # Do not remove; CI cleans the directory. Keep it simple here.

    state: Dict[str, Any] = {}
    _ = run_one_turn("AgentA", state, "hello world", cfg)

    # When disabled, scheduler.jsonl must not exist
    sched_log = os.path.join(ld, "scheduler.jsonl")
    if os.path.exists(sched_log):
        print("ERROR: scheduler.jsonl exists while scheduler.enabled=false", file=sys.stderr)
        return False
    return True


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Compare normalized logs to pre-M5 golden (disabled path)")
    ap.add_argument("--config", default=os.path.join(REPO_ROOT, "configs", "config.yaml"))
    ap.add_argument("--golden-dir", default=os.path.join(REPO_ROOT, "tests", "golden", "pre_m5_disabled"))
    ap.add_argument("--update", action="store_true", help="Update golden fixtures from current normalized output")
    ap.add_argument("--files", nargs="*", default=None, help="Specific filenames to compare; defaults to all in golden-dir or ['turn.jsonl'] if directory missing")
    args = ap.parse_args(argv)

    if not _run_disabled_once(args.config):
        return 1

    ld = logs_dir()

    if args.files is None:
        if os.path.isdir(args.golden_dir):
            files = sorted([f for f in os.listdir(args.golden_dir) if os.path.isfile(os.path.join(args.golden_dir, f))])
            if not files:
                files = ["turn.jsonl"]
        else:
            files = ["turn.jsonl"]
    else:
        files = args.files

    diffs: List[str] = []
    for fname in files:
        actual_path = os.path.join(ld, fname)
        actual_lines = _normalize_jsonl(actual_path)

        if args.update:
            golden_path = os.path.join(args.golden_dir, fname)
            _write_lines(golden_path, actual_lines)
            continue

        golden_path = os.path.join(args.golden_dir, fname)
        if not os.path.exists(golden_path):
            print(f"Missing golden fixture: {golden_path}", file=sys.stderr)
            diffs.append(f"missing:{fname}")
            continue
        golden_lines = _normalize_jsonl(golden_path)
        if actual_lines != golden_lines:
            diff = "\n".join(
                difflib.unified_diff(
                    golden_lines,
                    actual_lines,
                    fromfile=f"golden/{fname}",
                    tofile=f"actual/{fname}",
                    lineterm="",
                )
            )
            print(f"DIFF for {fname}:\n{diff}\n", file=sys.stderr)
            diffs.append(f"diff:{fname}")

    if args.update:
        print(f"Golden updated under {args.golden_dir}")
        return 0

    if diffs:
        return 2

    print("Golden identity check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())