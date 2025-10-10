#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from typing import List


# Fallback: slice sys.argv after the first occurrence of the subcommand.
def _slice_sys_argv_after_subcmd(subcmd: str) -> list[str]:
    """Fallback: slice raw sys.argv after the first occurrence of the subcommand.
    Works even if argparse didn't populate a remainder.
    """
    try:
        argv = sys.argv[1:]  # drop module token
        for i, tok in enumerate(argv):
            if tok == subcmd:
                return argv[i + 1 :]
        return []
    except Exception:
        return []


# Wrapper for the umbrella CLI that delegates to clematis.scripts.export_logs_for_frontend
# Contract:
#  • Accept all args after the subcommand as a raw remainder.
#  • Strip a single leading '--' sentinel (if present).
#  • Forward the remainder verbatim to scripts/export_logs_for_frontend.main(argv).
#  • Print a stderr-only breadcrumb when CLEMATIS_DEBUG=1 (no stdout changes).

_DEBUG = os.environ.get("CLEMATIS_DEBUG") == "1"


def _dbg(msg: str) -> None:
    if _DEBUG:
        sys.stderr.write(f"[clematis] delegate -> scripts.export_logs_for_frontend argv={msg!r}\n")


def _strip_single_sentinel(argv: List[str]) -> List[str]:
    if argv and argv[0] == "--":
        return argv[1:]
    return argv


def _delegate(argv: List[str]) -> int:
    _dbg(str(argv))
    try:
        from clematis.scripts.export_logs_for_frontend import main as _impl_main
    except Exception as e:
        # Keep stdout reserved for the delegated tool; only stderr here.
        sys.stderr.write(f"Module import failed: {e}\n")
        return 2
    # Ensure returned value is an int exit code.
    return int(_impl_main(argv) or 0)


def _run(ns: argparse.Namespace) -> int:
    # Prefer argparse-captured remainder, but fall back to raw sys.argv slicing
    remainder = list(getattr(ns, "remainder", []) or [])
    if not remainder:
        remainder = _slice_sys_argv_after_subcmd("export-logs")
    argv = _strip_single_sentinel(remainder)
    return _delegate(argv)


def register(subparsers: argparse._SubParsersAction) -> argparse.ArgumentParser:
    help_text = "Export logs + latest snapshot into a JSON bundle (delegates to scripts/)"
    epilog = (
        "Arguments after an optional '--' are forwarded verbatim to\n"
        "scripts/export_logs_for_frontend.py"
    )
    p = subparsers.add_parser(
        "export-logs",
        help=help_text,
        description=help_text,
        epilog=epilog,
        add_help=True,
    )
    # Capture all remaining args without this parser attempting to parse them.
    p.add_argument("remainder", nargs=argparse.REMAINDER, help=argparse.SUPPRESS)
    p.set_defaults(func=_run)
    return p
