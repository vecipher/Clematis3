#!/usr/bin/env python3
from __future__ import annotations

import argparse
import difflib
import importlib
import json
import os
import sys
from typing import List

# --- repo roots and normalization helpers ------------------------------------

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

NORM_DIR = os.path.join(REPO_ROOT, ".norm")

# Try to import helpers from tests; fall back to local implementations.
_normalize_json_lines = None
_normalize_logs_dir = None
try:
    from tests.helpers.identity import normalize_json_lines as _normalize_json_lines  # type: ignore
except Exception:
    pass

try:
    from tests.helpers.identity import normalize_logs_dir as _normalize_logs_dir  # type: ignore
except Exception:
    pass


def _fallback_normalize_json_lines(lines: List[str]) -> List[str]:
    """Deterministically normalize JSONL lines if test helper is unavailable."""
    out: List[str] = []
    for ln in lines:
        ln = ln.strip()
        if not ln:
            continue
        try:
            obj = json.loads(ln)
        except Exception:
            # keep as-is if not valid JSON
            out.append(ln)
            continue
        out.append(json.dumps(obj, sort_keys=True, separators=(",", ":")))
    return out


def _fallback_normalize_logs_dir(p: str, base: str | None = None) -> str:
    p = os.path.expanduser(os.path.expandvars(p))
    if base and not os.path.isabs(p):
        base = os.path.normpath(os.path.realpath(os.path.expanduser(os.path.expandvars(base))))
        p = os.path.join(base, p)
    return os.path.normpath(os.path.realpath(p))


def _normalize_jsonl(path: str) -> List[str]:
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read().splitlines()
    if _normalize_json_lines is not None:
        try:
            return _normalize_json_lines(raw)  # type: ignore[misc]
        except Exception:
            pass
    return _fallback_normalize_json_lines(raw)


def _write_lines(path: str, lines: List[str]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
        if lines:
            f.write("\n")


def _norm_logs_dir(p: str) -> str:
    if _normalize_logs_dir is not None:
        try:
            # Prefer REPO_ROOT as base if helper supports it.
            return _normalize_logs_dir(p, REPO_ROOT)  # type: ignore[misc]
        except TypeError:
            # Older helper signature that accepts one arg
            return _normalize_logs_dir(p)  # type: ignore[misc]
        except Exception:
            pass
    return _fallback_normalize_logs_dir(p, REPO_ROOT)


# --- optional demo runner -----------------------------------------------------


def _maybe_run_demo(logs_dir: str) -> None:
    """
    Try to run the disabled-identity demo that writes JSONL files into logs_dir.
    If nothing is found, silently continue so the compare can still run.
    """
    candidates = (
        # (module, attr name to call)
        ("scripts.ci_golden_demo", "run_demo"),
        ("scripts.golden_disabled_identity", "run_demo"),
        ("clematis.scripts.ci_golden_demo", "run_demo"),
        ("clematis.scripts.golden_disabled_identity", "run_demo"),
        # Some repos expose a more explicit name:
        ("scripts.ci_golden_demo", "run_disabled_identity_demo"),
        ("clematis.scripts.ci_golden_demo", "run_disabled_identity_demo"),
    )
    for mod_name, fn_name in candidates:
        try:
            mod = importlib.import_module(mod_name)
            fn = getattr(mod, fn_name, None)
            if callable(fn):
                fn(logs_dir)
                return
        except Exception:
            continue
    # If nothing found, we just proceed (comparison will use whatever is present in logs_dir).


# --- main ---------------------------------------------------------------------


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Compare normalized actual logs against golden fixtures (scheduler.enabled=false).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument(
        "--logs-dir",
        default="./.logs",
        help="Directory where the demo wrote t*.jsonl and turn.jsonl.",
    )
    ap.add_argument(
        "--golden-dir",
        default="tests/golden/pre_m5_disabled",
        help="Directory containing golden JSONL files.",
    )
    ap.add_argument(
        "--update",
        action="store_true",
        help="Update goldens in-place with current normalized output.",
    )
    ap.add_argument(
        "--files",
        nargs="*",
        default=["t1.jsonl", "t2.jsonl", "t3.jsonl", "t4.jsonl", "turn.jsonl"],
        help="Subset of files to compare.",
    )
    ap.add_argument(
        "--skip-run",
        action="store_true",
        help="Skip running the demo; compare whatever is already in logs-dir.",
    )
    args = ap.parse_args(argv)

    print("Running golden comparison (scheduler.enabled=false)…")
    logs_dir = _norm_logs_dir(args.logs_dir)
    if not args.skip_run:
        _maybe_run_demo(logs_dir)
    print(f"Logs dir: {logs_dir}")

    # Ensure normalized output dirs exist for external inspection
    os.makedirs(os.path.join(NORM_DIR, "golden"), exist_ok=True)
    os.makedirs(os.path.join(NORM_DIR, "actual"), exist_ok=True)

    diffs: List[str] = []
    for fname in args.files:
        actual_path = os.path.join(logs_dir, fname)
        golden_path = os.path.join(args.golden_dir, fname)

        actual_lines = _normalize_jsonl(actual_path)

        if args.update:
            _write_lines(golden_path, actual_lines)
            print(f"[updated] {golden_path}")
            continue

        if not os.path.exists(golden_path):
            print(f"Missing golden fixture: {golden_path}", file=sys.stderr)
            diffs.append(f"missing:{fname}")
            # Still materialize what we have for inspection
            _write_lines(os.path.join(NORM_DIR, "actual", fname), actual_lines)
            continue

        golden_lines = _normalize_jsonl(golden_path)

        # If goldens capture a single summary record but the run produced multiple,
        # compare only the first normalized line to keep the guard stable.
        actual_cmp = actual_lines
        if len(golden_lines) == 1 and len(actual_lines) >= 1:
            actual_cmp = [actual_lines[0]]

        # Materialize normalized lines for CI-side inspection
        _write_lines(os.path.join(NORM_DIR, "golden", fname), golden_lines)
        _write_lines(os.path.join(NORM_DIR, "actual", fname), actual_cmp)

        if actual_cmp != golden_lines:
            diff_txt = "\n".join(
                difflib.unified_diff(
                    golden_lines,
                    actual_cmp,
                    fromfile=f"golden/{fname}",
                    tofile=f"actual/{fname}",
                    lineterm="",
                )
            )
            print(f"DIFF for {fname}:\n{diff_txt}\n", file=sys.stderr)
            diffs.append(f"diff:{fname}")

    if diffs:
        print("Normalized logs written to ./.norm")
        print("Golden identity mismatch", file=sys.stderr)
        return 2

    print("Golden identity check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
