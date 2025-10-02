import argparse
import importlib
import importlib.util
import inspect
import sys
from pathlib import Path

from ._exit import IO_ERR, OK, USER_ERR
from ._io import eprint_once, set_verbosity
from ._wrapper_common import maybe_debug, prepare_wrapper_args

_CANDIDATES = ("clematis.scripts.bench_t4", "scripts.bench_t4")


def _import_script():
    last = None
    for n in _CANDIDATES:
        try:
            return importlib.import_module(n)
        except Exception as e:
            last = e
    root = Path(__file__).resolve().parents[2]
    path = root / "scripts" / "bench_t4.py"
    if path.exists():
        spec = importlib.util.spec_from_file_location("scripts.bench_t4", path)
        mod = importlib.module_from_spec(spec)
        assert spec and spec.loader
        spec.loader.exec_module(mod)
        return mod
    print(f"[clematis] bench-t4: cannot locate {path.name}. Last error: {last}", file=sys.stderr)
    return None


def _delegate(argv):
    mod = _import_script()
    if mod is None:
        return IO_ERR
    main = getattr(mod, "main", None)
    if main is None:
        print("[clematis] bench-t4: script has no main().", file=sys.stderr)
        return IO_ERR
    try:
        sig = inspect.signature(main)
        return main(argv) if len(sig.parameters) >= 1 else main()
    except SystemExit as e:
        return int(getattr(e, "code", 0) or 0)


def _entrypoint(ns: argparse.Namespace) -> int:
    opts = prepare_wrapper_args(ns)

    if opts.help_requested:
        parser = getattr(ns, "_parser", None)
        if parser is not None:
            parser.print_help()
            return OK

    set_verbosity(opts.verbose, opts.quiet)

    if opts.wants_json and opts.wants_table:
        if not opts.quiet:
            eprint_once("Choose exactly one of --json or --table.")
        return USER_ERR
    if opts.wants_json or opts.wants_table:
        if not opts.quiet:
            eprint_once("`bench-t4` currently does not support --json/--table.")
        return USER_ERR

    maybe_debug(ns, resolved="scripts.bench_t4", argv=opts.argv)
    return int(_delegate(opts.argv) or 0)


def register(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "bench-t4",
        help="Delegates to scripts/bench_t4.py",
        description="Delegates to scripts/bench_t4.py",
    )
    # Common output/verbosity flags
    fmt = p.add_mutually_exclusive_group()
    fmt.add_argument("--json", action="store_true", help="JSON output (stable, machine-readable)")
    fmt.add_argument("--table", action="store_true", help="Plain table output (no color)")
    p.add_argument("--quiet", action="store_true", help="suppress non-essential stderr")
    p.add_argument("--verbose", action="store_true", help="increase stderr verbosity")
    p.add_argument(
        "args", nargs=argparse.REMAINDER, help="Pass-through arguments for scripts/bench_t4.py."
    )
    p.set_defaults(func=_entrypoint, _parser=p)
