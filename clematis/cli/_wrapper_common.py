from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple

from ._resources import packaged_path


def _enabled(ns) -> bool:
    try:
        if getattr(ns, "debug", False):
            return True
    except Exception:
        pass
    return os.getenv("CLEMATIS_DEBUG") == "1"


def maybe_debug(
    ns, *args, resolved: Optional[str] = None, argv: Optional[List[str]] = None
) -> None:
    """
    Emit a single-line, deterministic breadcrumb to stderr (never logs).

    Back/forward compatible call forms:
      - maybe_debug(ns, resolved, argv)                      # positional (legacy)
      - maybe_debug(ns, resolved=<str>, argv=<list[str]>)    # keyword-only (current)

    Never changes stdout or exit codes.
    """
    # Accept both positional and keyword styles.
    if resolved is None and argv is None and len(args) == 2:
        resolved, argv = args
    if argv is None:
        argv = []
    if _enabled(ns):
        sys.stderr.write(f"[clematis] delegate -> {resolved} argv={list(argv)}\n")
        sys.stderr.flush()


def _has_any_flag(argv: List[str], flags: Tuple[str, ...]) -> bool:
    return any(flag in argv for flag in flags)


def inject_default_from_packaged_or_cwd(
    argv: List[str],
    flag_names: Tuple[str, ...],
    packaged_parts: Tuple[str, ...],
    cwd_rel: str,
    *,
    position: str = "prepend",
) -> List[str]:
    """
    If none of `flag_names` are present in argv, inject the first flag name
    and a value resolved as:
      1) packaged resource (if exists) at `packaged_parts`; else
      2) Path.cwd() / `cwd_rel`.

    Injection defaults to prepending the pair [flag, value] ahead of argv.

    This function NEVER mutates the input list; it returns a new list.
    """
    if _has_any_flag(argv, flag_names):
        return list(argv)

    # Prefer packaged resource; else deterministic CWD fallback.
    with packaged_path(*packaged_parts) as p:
        value_path = p if p is not None else (Path.cwd() / cwd_rel)

    pair = [flag_names[0], str(value_path)]
    if position == "append":
        return list(argv) + pair
    return pair + list(argv)
