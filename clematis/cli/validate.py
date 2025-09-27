#!/usr/bin/env python3
"""CLI subcommand: validate — Delegates to scripts/validate_config.py.

Top-level wrapper so `python -m clematis validate ...` works consistently.
Parses known flags (e.g., --json), captures passthrough via REMAINDER, and
forwards to the packaged shim (`clematis.scripts.validate`), which then
hands off to `validate_config.py`.
"""
from __future__ import annotations

import argparse
from typing import List

_HELP = "Delegates to scripts/"
_DESC = "Delegates to scripts/validate_config.py"


def register(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "validate",
        help=_HELP,
        description=_DESC,
        add_help=False,  # we intercept -h/--help via REMAINDER for deterministic output
    )
    # Known flags we want to parse deterministically here
    parser.add_argument("--json", action="store_true", help="Emit JSON report")
    parser.add_argument(
        "--check-resource",
        action="append",
        default=[],
        metavar="REL_PATH",
        help="Additional package-relative resource to verify",
    )
    # Capture any remaining args verbatim (including a bare -h/--help)
    parser.add_argument("args", nargs=argparse.REMAINDER, help=argparse.SUPPRESS)

    parser.set_defaults(command="validate", func=_run)


def _normalize_argv(ns: argparse.Namespace) -> List[str]:
    argv: List[str] = []
    if getattr(ns, "json", False):
        argv.append("--json")
    for rp in getattr(ns, "check_resource", []) or []:
        argv.extend(["--check-resource", rp])
    rest: List[str] = list(getattr(ns, "args", []) or [])
    if rest and rest[0] == "--":
        rest = rest[1:]
    argv.extend(rest)
    return argv


def _run(ns: argparse.Namespace) -> int:
    # Intercept help explicitly so tests can grep the literal
    rest = list(getattr(ns, "args", []) or [])
    if any(x in ("-h", "--help") for x in rest):
        print(_HELP)
        return 0

    argv = _normalize_argv(ns)
    # Delegate via packaged shim (which falls back to repo-layout if needed)
    from clematis.scripts.validate import main as _main  # type: ignore
    return _main(argv)


def main(argv: List[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]

    # Prefer the packaged-in module (inside the wheel)
    try:
        from . import validate_config as _vc  # type: ignore
    except Exception:
        try:
            import scripts.validate_config as _vc  # type: ignore
        except Exception:
            sys.stderr.write(
                "clematis.scripts.validate: validate_config.py not found; cannot delegate.\n"
                "Hint: install dev scripts or run `python -m clematis inspect-snapshot -- --format json`.\n"
            )
            return 2

    fn = getattr(_vc, "main", None)
    if callable(fn):
        try:
            return fn(argv)  # type: ignore[misc]
        except TypeError:
            return fn()  # type: ignore[call-arg]

    sys.stderr.write("clematis.scripts.validate: no callable main() in validate_config.py.\n")
    return 2


if __name__ == "__main__":  # pragma: no cover
    import sys
    raise SystemExit(main())