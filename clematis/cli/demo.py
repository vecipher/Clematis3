#!/usr/bin/env python3
"""CLI subcommand: demo — Delegates to scripts/run_demo.py.

This is the top-level CLI wrapper so `python -m clematis demo ...` works.
It forwards args to the packaged scripts shim (`clematis.scripts.demo`),
which in turn delegates to `run_demo.py`.
"""
from __future__ import annotations

import argparse
from typing import List

_HELP = "Delegates to scripts/"
_DESC = "Delegates to scripts/run_demo.py"


def register(subparsers):
    p = subparsers.add_parser(
        "demo",
        help=_HELP,
        description=_DESC,
        add_help=False,
    )
    # Capture anything after `--` without polluting top-level parsing
    p.add_argument("args", nargs=argparse.REMAINDER, help=argparse.SUPPRESS)
    p.set_defaults(command="demo", func=_run)
    return p


def _normalize_argv(passthrough: List[str] | None) -> List[str]:
    argv: List[str] = []
    if passthrough:
        argv.extend(passthrough)
    # If argparse.REMAINDER included a leading '--', drop it
    if argv and argv[0] == "--":
        argv = argv[1:]
    return argv


def _run(ns) -> int:
    argv = _normalize_argv(getattr(ns, "args", None))
    # If user asked for help, print our subcommand help phrase and exit 0
    if argv and any(x in ("-h", "--help") for x in argv):
        print(_HELP)
        return 0
    # Delegate to the packaged shim (zero-arg main). We pass argv via sys.argv.
    import sys as _sys
    _orig_argv = list(_sys.argv)
    try:
        _sys.argv = ["clematis.scripts.demo"] + argv
        from clematis.scripts.demo import main as _main  # type: ignore
        return _main()  # call zero-arg; shim adapts to run_demo main()
    finally:
        _sys.argv = _orig_argv


if __name__ == "__main__":  # pragma: no cover
    # Allow direct execution for convenience
    import sys
    _dummy = type("_", (), {})()
    setattr(_dummy, "args", sys.argv[1:])
    sys.exit(_run(_dummy))