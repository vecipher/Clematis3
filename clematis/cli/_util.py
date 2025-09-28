from __future__ import annotations

import argparse

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

    Also wires common flags:
    - --json / --table (mutually exclusive): select machine-readable JSON or plain ASCII table output
    - --quiet / --verbose: control stderr verbosity; stdout remains reserved for command output
    """
    sp = subparsers.add_parser(
        name,
        help=help_text,
        description=description,
        add_help=False,
    )
    # Common output/verbosity flags shared by all wrappers
    fmt = sp.add_mutually_exclusive_group()
    fmt.add_argument("--json", action="store_true", help="JSON output (stable, machine-readable)")
    fmt.add_argument("--table", action="store_true", help="Plain table output (no color)")
    sp.add_argument("--quiet", action="store_true", help="suppress non-essential stderr")
    sp.add_argument("--verbose", action="store_true", help="increase stderr verbosity")
    # Provide deterministic help flag behavior while suppressing argparse's
    # auto-generated positional docs for the passthrough payload.
    sp.add_argument("-h", "--help", action="help", help="show this help message and exit")
    sp.add_argument("args", nargs=argparse.REMAINDER, help=argparse.SUPPRESS)
    return sp
