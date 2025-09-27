#!/usr/bin/env python3
"""clematis.cli.validate — CLI wrapper

Registers the `validate` subcommand and **Delegates to scripts/** at runtime:
- packaged: `clematis.scripts.validate` (which itself delegates to `validate_config.py`)
- repo-fallback: `scripts.validate` if needed

This keeps the top-level help consistent with the rest of the CLI and avoids
adding runtime logic here. Zero network usage.
"""
from __future__ import annotations

import argparse
from typing import List

# ---- public API expected by clematis.cli.main --------------------------------

def register(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "validate",
        help="Delegates to scripts/",
        description="Delegates to scripts/validate_config.py",
    )
    # Expose common flags so `parse_known_args` consumes them deterministically
    parser.add_argument("--json", action="store_true", help="Emit JSON report")
    parser.add_argument(
        "--check-resource",
        action="append",
        default=[],
        metavar="REL_PATH",
        help="Additional package-relative resource to verify",
    )
    parser.set_defaults(command="validate", func=_invoke)


# ---- runtime delegation -------------------------------------------------------

def _invoke(ns: argparse.Namespace, extras: List[str]) -> int:
    """Delegate to the real script entrypoint, reconstructing argv.

    We reconstruct argv from parsed flags (ns) + any passthrough `extras`
    collected by the top-level parser after a `--` separator.
    """
    argv: List[str] = []
    if getattr(ns, "json", False):
        argv.append("--json")
    for rp in getattr(ns, "check_resource", []) or []:
        argv.extend(["--check-resource", rp])
    argv.extend(extras)

    # Prefer packaged-in module
    try:
        from ..scripts import validate as _validate  # type: ignore
    except Exception:
        try:
            import scripts.validate as _validate  # type: ignore
        except Exception:
            # Final fallback: exit with a clear message consistent with scripts shim
            import sys
            sys.stderr.write(
                "clematis.cli.validate: scripts/validate not found; cannot delegate.\n"
            )
            return 2

    return _validate.main(argv)