#!/usr/bin/env python3
"""CLI subcommand: demo — Delegates to scripts/run_demo.py.

Behavior‑neutral wrapper so `python -m clematis demo ...` works consistently.
Parses no custom flags here; captures passthrough via REMAINDER and forwards to
the packaged shim (`clematis.scripts.demo`). For compatibility with the repo
layout, we set `sys.argv` and call the shim's zero‑arg `main()`.
"""
from __future__ import annotations

import argparse

from ._util import add_passthrough_subparser

_HELP = "Delegates to scripts/"
_DESC = "Delegates to scripts/run_demo.py"


def register(subparsers: argparse._SubParsersAction):
    sp = add_passthrough_subparser(
        subparsers,
        name="demo",
        help_text=_HELP,
        description=_DESC,
    )
    sp.set_defaults(command="demo", func=_run)
    return sp


def _run(ns: argparse.Namespace) -> int:
    # REMAINDER swallows -h/--help; intercept explicitly for deterministic output
    rest = list(getattr(ns, "args", []) or [])
    if any(x in ("-h", "--help") for x in rest):
        print(_HELP)
        return 0
    if rest and rest[0] == "--":
        rest = rest[1:]

    # Delegate to the packaged shim. We set sys.argv to support zero‑arg main().
    import sys as _sys
    _orig_argv = list(_sys.argv)
    try:
        _sys.argv = ["clematis.scripts.demo"] + rest
        from clematis.scripts.demo import main as _main  # type: ignore
        return _main()  # shim adapts to run_demo main()
    finally:
        _sys.argv = _orig_argv


if __name__ == "__main__":  # pragma: no cover
    # Allow direct execution for convenience
    import sys
    _dummy = type("_", (), {})()
    setattr(_dummy, "args", sys.argv[1:])
    raise SystemExit(_run(_dummy))