#!/usr/bin/env python3
"""CLI subcommand: validate — Delegates to scripts/validate_config.py.

Behavior‑neutral wrapper so `python -m clematis validate ...` works consistently.
Parses no custom flags here; captures passthrough via REMAINDER and forwards to
the packaged shim (`clematis.scripts.validate`), which locates the real script
and adapts main(argv)/main().
"""

from __future__ import annotations

import argparse

from ._exit import OK, USER_ERR
from ._io import eprint_once, set_verbosity
from ._util import add_passthrough_subparser
from ._wrapper_common import prepare_wrapper_args

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
    opts = prepare_wrapper_args(ns)

    set_verbosity(opts.verbose, opts.quiet)

    if opts.help_requested:
        print(_HELP)
        return OK

    if opts.wants_json and opts.wants_table:
        if not opts.quiet:
            eprint_once("Choose exactly one of --json or --table.")
        return USER_ERR

    if opts.wants_table:
        if not opts.quiet:
            eprint_once("`validate` currently supports --json only.")
        return USER_ERR

    rest = opts.argv
    if opts.wants_json and "--json" not in rest:
        rest = ["--json", *rest]

    # Delegate via packaged shim (which falls back to repo-layout if needed)
    from clematis.scripts.validate import main as _main

    return _main(rest)
