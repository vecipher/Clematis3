import argparse
import importlib
import importlib.util
import inspect
import sys
from pathlib import Path

from ._exit import IO_ERR, OK, USER_ERR
from ._io import eprint_once, print_json, print_table, set_verbosity
from ._wrapper_common import (
    inject_default_from_packaged_or_cwd,
    maybe_debug,
    prepare_wrapper_args,
)

_CANDIDATES = ("clematis.scripts.rotate_logs", "scripts.rotate_logs")


def _import_script():
    last = None
    for n in _CANDIDATES:
        try:
            return importlib.import_module(n)
        except Exception as e:
            last = e
    root = Path(__file__).resolve().parents[2]
    path = root / "scripts" / "rotate_logs.py"
    if path.exists():
        spec = importlib.util.spec_from_file_location("scripts.rotate_logs", path)
        mod = importlib.util.module_from_spec(spec)
        assert spec and spec.loader
        spec.loader.exec_module(mod)
        return mod
    print(f"[clematis] rotate-logs: cannot locate {path.name}. Last error: {last}", file=sys.stderr)
    return None


def _delegate(argv):
    mod = _import_script()
    if mod is None:
        return IO_ERR
    main = getattr(mod, "main", None)
    if main is None:
        print("[clematis] rotate-logs: script has no main().", file=sys.stderr)
        return IO_ERR
    try:
        sig = inspect.signature(main)
        return main(argv) if len(sig.parameters) >= 1 else main()
    except SystemExit as e:
        return int(getattr(e, "code", 0) or 0)


def _entrypoint(ns):
    opts = prepare_wrapper_args(ns)

    if opts.help_requested:
        parser = getattr(ns, "_parser", None)
        if parser is not None:
            parser.print_help()
            return OK

    set_verbosity(opts.verbose, opts.quiet)

    argv = opts.argv
    wants_json = opts.wants_json
    wants_table = opts.wants_table
    quiet = opts.quiet

    if wants_json and wants_table:
        if not quiet:
            eprint_once("Choose exactly one of --json or --table.")
        return USER_ERR

    resolved_argv = inject_default_from_packaged_or_cwd(
        argv,
        flag_names=("--dir",),
        packaged_parts=("examples", "logs"),
        cwd_rel=".logs",
    )

    if wants_json or wants_table:

        def _flag_value(args, name):
            try:
                i = args.index(name)
                return args[i + 1] if i + 1 < len(args) else None
            except ValueError:
                return None

        summary = {
            "dry_run": ("--dry-run" in resolved_argv),
            "dir": _flag_value(resolved_argv, "--dir"),
            "pattern": _flag_value(resolved_argv, "--pattern"),
        }
        if wants_json:
            print_json(summary)
            return OK
        print_table([summary], headers=["dry_run", "dir", "pattern"])
        return OK

    maybe_debug(ns, resolved="scripts.rotate_logs", argv=resolved_argv)
    return int(_delegate(resolved_argv) or 0)


def register(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "rotate-logs",
        help="Delegates to scripts/rotate_logs.py",
        description="Delegates to scripts/rotate_logs.py",
    )
    # Common output/verbosity flags
    fmt = p.add_mutually_exclusive_group()
    fmt.add_argument("--json", action="store_true", help="JSON output (stable, machine-readable)")
    fmt.add_argument("--table", action="store_true", help="Plain table output (no color)")
    p.add_argument("--quiet", action="store_true", help="suppress non-essential stderr")
    p.add_argument("--verbose", action="store_true", help="increase stderr verbosity")
    p.add_argument("args", nargs=argparse.REMAINDER)
    p.set_defaults(func=_entrypoint, _parser=p)
