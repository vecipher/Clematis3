"""
Golden fixtures normalizer (newline + path + JSON presentation).

Usage:
    python -m clematis.scripts.golden_normalize [PATH ...]

- By default runs in CHECK mode (prints what would change, exits 1 if drift).
- Use --write to update files in-place.
- Use --eol-only to normalize only newlines (CRLF/CR -> LF) for non‑JSONL or all files.
- Designed to be stable across OS (Windows/macOS/Linux) and Python versions.

This tool does NOT depend on test helpers to avoid import order coupling.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Mapping, MutableMapping, Any

# ---- Canonicalization knobs -------------------------------------------------

# Keys that are volatile across runs and should be dropped for golden fixtures.
DROP_KEYS: set[str] = {
    "now",
    "version_etag",
    "tier_sequence",
    "ms",
    "created_at",
}
# Prefix/suffix patterns (case-sensitive) that indicate timing fields in ms
DROP_KEY_PREFIXES: tuple[str, ...] = ("ms_",)
DROP_KEY_SUFFIXES: tuple[str, ...] = ("_ms",)

# Dict keys that likely contain filesystem paths; string values will be normalized.
PATHISH_KEYS: tuple[str, ...] = (
    "path",
    "file",
    "filepath",
    "snapshot",
    "snapshot_path",
    "artifact",
    "dir",
    "directory",
    "logs_dir",
    "log_file",
)
PATHISH_SUFFIXES: tuple[str, ...] = ("_path", "_file", "_dir", "_directory")

JSONL_EXT = ".jsonl"

SKIP_DIRS = {".venv", "__pycache__", ".git"}

# ---- Helpers ----------------------------------------------------------------

def _is_pathish_key(k: str) -> bool:
    if k in PATHISH_KEYS:
        return True
    for suf in PATHISH_SUFFIXES:
        if k.endswith(suf):
            return True
    return False

_multislash_re = re.compile(r"(?<!:)/{2,}")  # collapse multiple slashes, keep scheme like "C:/"

def normalize_path_string(s: str) -> str:
    """
    Convert Windows-style backslashes to forward slashes.
    Preserve UNC network shares (leading //).
    Collapse repeated slashes (except keep leading '//' if present).
    Do NOT alter drive letters like 'C:'.
    """
    if not s:
        return s
    # Replace backslashes with forward slashes
    s2 = s.replace("\\", "/")
    # Preserve UNC (leading //) by temporarily marking it
    is_unc = s2.startswith("//")
    # Collapse multiple slashes (not touching "http://", but these paths are filesystem)
    s2 = _multislash_re.sub("/", s2)
    if is_unc and not s2.startswith("//"):
        s2 = "/" + s2  # ensure we still have two leading slashes in total
    return s2

def _drop_volatile_keys_inplace(obj: MutableMapping[str, Any]) -> None:
    to_del: list[str] = []
    for k in list(obj.keys()):
        if k in DROP_KEYS:
            to_del.append(k)
            continue
        if any(k.startswith(pfx) for pfx in DROP_KEY_PREFIXES):
            to_del.append(k)
            continue
        if any(k.endswith(sfx) for sfx in DROP_KEY_SUFFIXES):
            to_del.append(k)
            continue
    for k in to_del:
        obj.pop(k, None)

def _transform(obj: Any) -> Any:
    """
    Recursively:
      - drop volatile keys
      - normalize path-like string values
    """
    if isinstance(obj, dict):
        _drop_volatile_keys_inplace(obj)
        for k, v in list(obj.items()):
            if isinstance(v, str) and _is_pathish_key(k):
                obj[k] = normalize_path_string(v)
            else:
                obj[k] = _transform(v)
        return obj
    if isinstance(obj, list):
        return [_transform(x) for x in obj]
    # Primitive: leave as-is
    return obj

def normalize_json_line(line: str) -> str:
    """
    Normalize a single JSON line. If not JSON, return the input stripped of CR characters.
    """
    s = line.replace("\r", "")  # CRLF/CR -> LF handled at file level, but be safe per-line too
    if not s.strip():
        return ""
    try:
        obj = json.loads(s)
    except Exception:
        return s.strip("\n")
    obj = _transform(obj)
    # Canonical minified JSON with stable key order
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"), sort_keys=True)

def lf_eol(s: str) -> str:
    """Convert CRLF/CR to LF."""
    return s.replace("\r\n", "\n").replace("\r", "\n")

def ensure_single_trailing_lf(s: str) -> str:
    s = s.rstrip("\n")
    return s + "\n"

@dataclass
class Change:
    path: Path
    before_bytes: int
    after_bytes: int

def _iter_files(paths: list[Path]) -> Iterator[Path]:
    for p in paths:
        if p.is_dir():
            for sub in p.rglob("*"):
                if sub.is_dir():
                    if sub.name in SKIP_DIRS:
                        sub.mkdir(exist_ok=True)  # no-op; just continue
                    continue
                yield sub
        elif p.is_file():
            yield p

def _normalize_file(path: Path, *, eol_only: bool) -> bytes:
    raw = path.read_bytes()
    text = lf_eol(raw.decode("utf-8", errors="replace"))
    # Only JSONL gets structural normalization unless --eol-only
    if eol_only or path.suffix.lower() != JSONL_EXT:
        return ensure_single_trailing_lf(text).encode("utf-8")

    lines_out: list[str] = []
    for line in text.splitlines():
        norm = normalize_json_line(line)
        if norm != "":
            lines_out.append(norm)
    out_text = "\n".join(lines_out)
    out_text = ensure_single_trailing_lf(out_text)
    return out_text.encode("utf-8")

def run(paths: list[str], *, write: bool, eol_only: bool, verbose: bool) -> int:
    targets = [Path(p) for p in (paths or ["tests/golden", "tests/goldens"])]
    changed: list[Change] = []
    for file in _iter_files(targets):
        if not file.exists() or file.is_dir():
            continue
        try:
            after = _normalize_file(file, eol_only=eol_only)
        except Exception as e:
            print(f"[warn] skipping {file}: {e}", file=sys.stderr)
            continue
        before = file.read_bytes()
        if before != after:
            changed.append(Change(file, len(before), len(after)))
            if verbose:
                print(f"[diff] {file}")
            if write:
                file.write_bytes(after)
    if changed and not write:
        # Print a short summary and exit non-zero to signal drift
        print(f"{len(changed)} file(s) would change. Run with --write to update.", file=sys.stderr)
        for ch in changed[:50]:
            print(f"  {ch.path}", file=sys.stderr)
        if len(changed) > 50:
            print("  ...", file=sys.stderr)
        return 1
    if verbose:
        if changed:
            print(f"Updated {len(changed)} file(s).")
        else:
            print("No changes.")
    return 0

def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="golden_normalize",
        description="Normalize golden fixtures (CRLF->LF, path separators, JSON presentation).",
    )
    p.add_argument("paths", nargs="*", help="Files or directories (default: tests/golden tests/goldens)")
    p.add_argument("--write", action="store_true", help="Write changes in-place (default: check only).")
    p.add_argument("--eol-only", action="store_true", help="Only normalize newlines (no JSON transformation).")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args(argv)

def main(argv: list[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    return run(args.paths, write=args.write, eol_only=args.eol_only, verbose=args.verbose)

if __name__ == "__main__":
    raise SystemExit(main())
