# clematis/cli/main.py
import argparse
import os
import sys
from pathlib import Path
from typing import List

from . import bench_t4, demo, inspect_snapshot, rotate_logs, seed_lance_demo, validate
from ._config import discover_config_path, maybe_log_selected


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="clematis",
        description="Clematis umbrella CLI",
        epilog="See Operator Guide: docs/operator-guide.md",
        allow_abbrev=False,
    )
    # Top-level version flag (kept for help determinism/tests)
    try:
        from clematis import __version__ as _VER  # lazy import to avoid side effects
    except Exception:
        _VER = "unknown"
    parser.add_argument(
        "--version",
        action="version",
        version=f"clematis {_VER}",
    )
    # Quiet breadcrumb only; not forwarded to subcommands.
    parser.add_argument(
        "--debug",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    # Hidden explicit config override; consumed at umbrella, not forwarded
    parser.add_argument(
        "-c",
        "--config",
        dest="config",
        help=argparse.SUPPRESS,
    )
    subparsers = parser.add_subparsers(dest="command")

    # Register wrappers (pass-through pattern)
    rotate_logs.register(subparsers)
    inspect_snapshot.register(subparsers)
    bench_t4.register(subparsers)
    seed_lance_demo.register(subparsers)
    validate.register(subparsers)
    demo.register(subparsers)

    return parser


def main(argv: List[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    parser = build_parser()

    # Prefer parsing from the first subcommand onward to tolerate unknown
    # top-level flags (which should be treated as extras for the subcommand).
    try:
        sub_actions = [a for a in parser._actions if isinstance(a, argparse._SubParsersAction)]
        choices = sub_actions[0].choices if sub_actions else {}
    except Exception:
        choices = {}

    idx = next((i for i, tok in enumerate(argv) if tok in choices), -1)
    if idx >= 0:
        # If user asked for help on the subcommand, print the subparser help
        # directly to keep usage/prog stable ("clematis <sub> ...") and match goldens.
        sub_tail = argv[idx + 1 :]
        if any(x in sub_tail for x in ("-h", "--help")):
            try:
                sub_actions = [
                    a for a in parser._actions if isinstance(a, argparse._SubParsersAction)
                ]
                sub_map = sub_actions[0].choices if sub_actions else {}
                sp = sub_map.get(argv[idx])
            except Exception:
                sp = None
            if sp is not None:
                sp.print_help(sys.stdout)
                return 0
            # Fallback: delegate to argparse rendering with a minimal argv
            parser.parse_args(argv[: idx + 1] + ["-h"])
            return 0
        # extras before subcommand, but strip umbrella-only flags (--debug, --config/-c)
        pre = []
        it = iter(argv[:idx])
        for tok in it:
            if tok == "--debug":
                continue
            if tok in ("--config", "-c"):
                # skip its argument if present
                try:
                    nxt = next(it)
                    # only skip if it isn't another option (defensive)
                    if nxt.startswith("-"):
                        pre.append(nxt)  # treat as a separate flag; user error
                except StopIteration:
                    pass
                continue
            pre.append(tok)
        # Parse only the subcommand token to bind the correct subparser/func
        ns, _ = parser.parse_known_args([argv[idx]])
        if not hasattr(ns, "func"):
            parser.print_help(sys.stderr)
            return 2
        # Propagate top-level --debug if it was in the preamble
        if "--debug" in argv[:idx]:
            setattr(ns, "debug", True)
        # Everything after the subcommand is the sub-argv for the wrapper
        sub_args = list(argv[idx + 1 :])
        if sub_args and sub_args[0] == "--":
            sub_args = sub_args[1:]
        merged = pre + sub_args
        # Discover config for umbrella only (do not inject into sub-argv)
        selected, source = discover_config_path(None, Path.cwd(), os.environ)
        if selected is not None and not getattr(ns, "config", None):
            setattr(ns, "config", str(selected))
        maybe_log_selected(selected, source, verbose=("--verbose" in merged))
        setattr(ns, "args", merged)
        return ns.func(ns)

    # Fallback: no subcommand present; behave as before.
    ns, extras = parser.parse_known_args(argv)
    if not hasattr(ns, "func"):
        parser.print_help(sys.stderr)
        return 2
    sub_args = list(getattr(ns, "args", []))
    if sub_args and sub_args[0] == "--":
        sub_args = sub_args[1:]
    merged = list(extras) + sub_args
    # Discover config for umbrella only (do not inject into sub-argv)
    selected, source = discover_config_path(None, Path.cwd(), os.environ)
    if selected is not None and not getattr(ns, "config", None):
        setattr(ns, "config", str(selected))
    maybe_log_selected(selected, source, verbose=("--verbose" in merged))
    setattr(ns, "args", merged)
    return ns.func(ns)


if __name__ == "__main__":
    raise SystemExit(main())
