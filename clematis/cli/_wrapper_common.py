from __future__ import annotations

import os
import sys
from dataclasses import dataclass
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


@dataclass
class WrapperArgs:
    argv: List[str]
    wants_json: bool
    wants_table: bool
    quiet: bool
    verbose: bool
    help_requested: bool


def prepare_wrapper_args(ns, *, passthrough: Tuple[str, ...] = ()) -> WrapperArgs:
    raw = list(getattr(ns, "args", []) or [])
    if raw and raw[0] == "--":
        raw = raw[1:]

    wants_json = bool(getattr(ns, "json", False))
    wants_table = bool(getattr(ns, "table", False))
    quiet = bool(getattr(ns, "quiet", False))
    verbose = bool(getattr(ns, "verbose", False))
    help_requested = False

    passthrough_flags = set(passthrough)
    filtered: List[str] = []
    i = 0
    while i < len(raw):
        tok = raw[i]
        if tok == "--":
            i += 1
            continue
        if tok in ("-h", "--help"):
            help_requested = True
            i += 1
            continue
        if tok in passthrough_flags:
            filtered.append(tok)
            if i + 1 < len(raw) and not raw[i + 1].startswith("-"):
                filtered.append(raw[i + 1])
                i += 2
            else:
                i += 1
            continue
        if tok == "--json":
            wants_json = True
            i += 1
            continue
        if tok == "--table":
            wants_table = True
            i += 1
            continue
        if tok == "--quiet":
            quiet = True
            i += 1
            continue
        if tok == "--verbose":
            verbose = True
            i += 1
            continue
        filtered.append(tok)
        i += 1

    return WrapperArgs(
        argv=filtered,
        wants_json=wants_json,
        wants_table=wants_table,
        quiet=quiet,
        verbose=verbose,
        help_requested=help_requested,
    )
