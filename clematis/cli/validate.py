#!/usr/bin/env python3
"""CLI subcommand: validate — Delegates to scripts/validate_config.py.

Behavior‑neutral wrapper so `python -m clematis validate ...` works consistently.
Parses no custom flags here; captures passthrough via REMAINDER and forwards to
the packaged shim (`clematis.scripts.validate`), which locates the real script
and adapts main(argv)/main().
"""
from __future__ import annotations

import argparse

from ._io import set_verbosity, eprint_once
from ._exit import OK, USER_ERR

from ._util import add_passthrough_subparser

_HELP = "Delegates to scripts/"
_DESC = "Delegates to scripts/validate_config.py"


def register(subparsers: argparse._SubParsersAction) -> None:
    sp = add_passthrough_subparser(
        subparsers,
        name="validate",
        help_text=_HELP,
        description=_DESC,
    )
    sp.set_defaults(command="validate", func=_run)


def _run(ns: argparse.Namespace) -> int:
    # Configure verbosity (stderr only); stdout remains reserved for command output
    set_verbosity(getattr(ns, "verbose", False), getattr(ns, "quiet", False))

    # Capture remaining args verbatim; REMAINDER swallows -h/--help, so intercept here
    rest = list(getattr(ns, "args", []) or [])
    orig_rest = list(rest)
    if any(x in ("-h", "--help") for x in rest):
        print(_HELP)
        return OK
    if rest and rest[0] == "--":
        rest = rest[1:]

    # Robust quiet: honor parsed flag or leaked token in passthrough
    quiet = bool(getattr(ns, "quiet", False) or ("--quiet" in orig_rest))

    # Hoist wrapper-only flags that users might put after `--`
    hoisted_json = False
    hoisted_table = False
    filtered = []
    for tok in rest:
        if tok == "--json":
            hoisted_json = True
            continue
        if tok == "--table":
            hoisted_table = True
            continue
        if tok == "--quiet" or tok == "--verbose":  # wrapper-only; drop if leaked past '--'
            continue
        if tok == "--":  # drop stray option-terminator if present mid-argv
            continue
        filtered.append(tok)
    rest = filtered

    # Respect output-format flags from the wrapper; ensure identity for --json
    wants_json = bool(getattr(ns, "json", False) or hoisted_json)
    wants_table = bool(getattr(ns, "table", False) or hoisted_table)

    if wants_json and wants_table:
        if not quiet:
            eprint_once("Choose exactly one of --json or --table.")
        return USER_ERR

    if wants_table:
        if not quiet:
            eprint_once("`validate` currently supports --json only.")
        return USER_ERR

    # Preserve byte-identical JSON by forwarding the flag to the underlying tool
    if wants_json and "--json" not in rest:
        rest = ["--json", *rest]

    # Delegate via packaged shim (which falls back to repo-layout if needed)
    from clematis.scripts.validate import main as _main  # type: ignore
    return _main(rest)