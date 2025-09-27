#!/usr/bin/env python3
"""CLI subcommand: validate — Delegates to scripts/validate_config.py.

Behavior‑neutral wrapper so `python -m clematis validate ...` works consistently.
Parses no custom flags here; captures passthrough via REMAINDER and forwards to
the packaged shim (`clematis.scripts.validate`), which locates the real script
and adapts main(argv)/main().
"""
from __future__ import annotations

import argparse

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
    # Capture remaining args verbatim; REMAINDER swallows -h/--help, so intercept here
    rest = list(getattr(ns, "args", []) or [])
    if any(x in ("-h", "--help") for x in rest):
        print(_HELP)
        return 0
    if rest and rest[0] == "--":
        rest = rest[1:]
    # Delegate via packaged shim (which falls back to repo-layout if needed)
    from clematis.scripts.validate import main as _main  # type: ignore
    return _main(rest)