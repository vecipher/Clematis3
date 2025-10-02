#!/usr/bin/env python3
"""CLI subcommand: demo — Delegates to scripts/run_demo.py.

Behavior‑neutral wrapper so `python -m clematis demo ...` works consistently.
Parses no custom flags here; captures passthrough via REMAINDER and forwards to
the packaged shim (`clematis.scripts.demo`). For compatibility with the repo
layout, we set `sys.argv` and call the shim's zero‑arg `main()`.
"""

from __future__ import annotations

import argparse

from ._exit import OK, USER_ERR
from ._io import eprint_once, set_verbosity
from ._util import add_passthrough_subparser
from ._wrapper_common import prepare_wrapper_args

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
    opts = prepare_wrapper_args(ns)

    if opts.help_requested:
        print(_HELP)
        return OK

    set_verbosity(opts.verbose, opts.quiet)

    if opts.wants_json and opts.wants_table:
        if not opts.quiet:
            eprint_once("Choose exactly one of --json or --table.")
        return USER_ERR
    if opts.wants_json or opts.wants_table:
        if not opts.quiet:
            eprint_once("`demo` currently does not support --json/--table.")
        return USER_ERR

    import sys as _sys

    _orig_argv = list(_sys.argv)
    try:
        _sys.argv = ["clematis.scripts.demo"] + opts.argv
        from clematis.scripts.demo import main as _main

        return _main()  # shim adapts to run_demo main()
    finally:
        _sys.argv = _orig_argv


if __name__ == "__main__":  # pragma: no cover
    # Allow direct execution for convenience
    import sys

    _dummy = type("_", (), {})()
    setattr(_dummy, "args", sys.argv[1:])
    raise SystemExit(_run(_dummy))
