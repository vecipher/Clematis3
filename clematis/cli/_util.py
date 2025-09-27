

from __future__ import annotations

import argparse
from typing import Any

__all__ = ["add_passthrough_subparser"]

def add_passthrough_subparser(
    subparsers: argparse._SubParsersAction, 
    name: str, 
    help_text: str, 
    description: str,
) -> argparse.ArgumentParser:
    """
    Create a subparser that *delegates to scripts/*, intercepts -h/--help, and
    captures remaining args verbatim for the script shim.

    This standardizes our CLI wrappers so help output is deterministic and the
    help-matrix test can assert the literal "Delegates to scripts/".

    Returns the created subparser.
    """
    sp = subparsers.add_parser(
        name,
        help=help_text,
        description=description,
        add_help=False,
    )
    # Provide deterministic help flag behavior while suppressing argparse's
    # auto-generated positional docs for the passthrough payload.
    sp.add_argument("-h", "--help", action="help", help="show this help message and exit")
    sp.add_argument("args", nargs=argparse.REMAINDER, help=argparse.SUPPRESS)
    return sp
