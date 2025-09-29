#!/usr/bin/env python3
"""
Deterministic size-based rotation for JSONL (or any) log files.

Usage:
  python3 scripts/rotate_logs.py --dir ./.logs --pattern "*.jsonl" --max-bytes 10000000 --backups 5

Notes:
- Rotates files whose current size is **>= max-bytes**.
- Rotation scheme (numeric suffixes):
    path.(backups-1) -> deleted (if exists)
    path.(k)         -> path.(k+1) for k = backups-2 .. 1
    path             -> path.1
- No subdirectory traversal; only files directly under --dir are considered.
- This script does not append new content; it only rotates existing files.

Exit codes:
  0 = success (possibly no files rotated)
  2 = invalid arguments / directory not found / unexpected error
"""

from __future__ import annotations

import argparse
import glob
import os
import shutil
import sys
from typing import Iterable


def _eprint(msg: str) -> None:
    print(msg, file=sys.stderr)


def rotate_one(path: str, backups: int, dry_run: bool = False) -> bool:
    """Rotate a single file in place using numeric suffixes. Returns True if rotated."""
    if backups < 1:
        return False

    # Delete the oldest backup if it exists
    oldest = f"{path}.{backups}"
    if os.path.exists(oldest):
        if dry_run:
            print(f"rm {oldest}")
        else:
            try:
                os.remove(oldest)
            except FileNotFoundError:
                pass

    # Cascade: path.(k) -> path.(k+1)
    for k in range(backups - 1, 0, -1):
        src = f"{path}.{k}"
        dst = f"{path}.{k+1}"
        if os.path.exists(src):
            if dry_run:
                print(f"mv {src} {dst}")
            else:
                try:
                    os.replace(src, dst)
                except FileNotFoundError:
                    pass

    # Finally: path -> path.1
    if os.path.exists(path):
        if dry_run:
            print(f"mv {path} {path}.1")
        else:
            try:
                os.replace(path, f"{path}.1")
            except FileNotFoundError:
                pass
        return True

    return False


def iter_targets(directory: str, pattern: str) -> Iterable[str]:
    pattern_path = os.path.join(directory, pattern)
    yield from glob.iglob(pattern_path)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Rotate log files by size (deterministic numeric suffixes)"
    )
    ap.add_argument("--dir", default="./logs", help="Directory to scan for logs (no recursion)")
    ap.add_argument("--pattern", default="*.jsonl", help="Glob pattern to match under --dir")
    ap.add_argument(
        "--max-bytes",
        type=int,
        default=10_000_000,
        help="Rotate files with size >= this many bytes",
    )
    ap.add_argument("--backups", type=int, default=5, help="How many backup generations to keep")
    ap.add_argument(
        "--dry-run", action="store_true", help="Print planned actions without changing files"
    )

    args = ap.parse_args(argv)

    directory = args.dir
    if not os.path.isdir(directory):
        _eprint(f"error: directory not found: {directory}")
        return 2

    rotated_any = False
    try:
        for path in iter_targets(directory, args.pattern):
            try:
                size = os.path.getsize(path)
            except OSError:
                continue
            if size >= args.max_bytes:
                if args.dry_run:
                    print(f"rotate {path} (size={size} >= {args.max_bytes})")
                did = rotate_one(path, backups=args.backups, dry_run=args.dry_run)
                rotated_any = rotated_any or did or args.dry_run
    except Exception as ex:
        _eprint(f"error: {ex}")
        return 2

    if not rotated_any and args.dry_run:
        print("(dry-run) nothing to rotate")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
