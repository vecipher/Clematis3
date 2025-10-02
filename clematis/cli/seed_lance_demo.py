import argparse
import importlib
import importlib.util
import inspect
import sys
from pathlib import Path

from ._exit import IO_ERR, OK, USER_ERR
from ._io import eprint_once, set_verbosity
from ._wrapper_common import maybe_debug, prepare_wrapper_args

_CANDIDATES = ("clematis.scripts.seed_lance_demo", "scripts.seed_lance_demo")


def _import_script():
    last = None
    for n in _CANDIDATES:
        try:
            return importlib.import_module(n)
        except Exception as e:
            last = e
    root = Path(__file__).resolve().parents[2]
    path = root / "scripts" / "seed_lance_demo.py"
    if path.exists():
        spec = importlib.util.spec_from_file_location("scripts.seed_lance_demo", path)
        mod = importlib.util.module_from_spec(spec)
        assert spec and spec.loader
        spec.loader.exec_module(mod)
        return mod
    print(
        f"[clematis] seed-lance-demo: cannot locate {path.name}. Last error: {last}",
        file=sys.stderr,
    )
    return None


def _delegate(argv):
    mod = _import_script()
    if mod is None:
        return IO_ERR
    main = getattr(mod, "main", None)
    if main is None:
        print("[clematis] seed-lance-demo: script has no main().", file=sys.stderr)
        return IO_ERR
    try:
        sig = inspect.signature(main)
        return main(argv) if len(sig.parameters) >= 1 else main()
    except SystemExit as e:
        return int(getattr(e, "code", 0) or 0)


def _entrypoint(ns: argparse.Namespace) -> int:
    opts = prepare_wrapper_args(ns, passthrough=("--table",))

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
            eprint_once("`seed-lance-demo` currently does not support --json/--table.")
        return USER_ERR

    maybe_debug(ns, resolved="scripts.seed_lance_demo", argv=opts.argv)
    try:
        return int(_delegate(opts.argv) or 0)
    except ValueError as e:
        msg = str(e)
        if "already exists" in msg and "Table" in msg:
            if not opts.quiet:
                eprint_once(
                    "`seed-lance-demo`: table already exists. Use --overwrite or pass --table <new_name>."
                )
            return IO_ERR
        if not opts.quiet:
            eprint_once(f"`seed-lance-demo`: {msg}")
        return IO_ERR
    except Exception as e:
        if not opts.quiet:
            eprint_once(f"`seed-lance-demo`: unexpected error: {e.__class__.__name__}")
        return IO_ERR


def register(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "seed-lance-demo",
        help="Delegates to scripts/seed_lance_demo.py",
        description="Delegates to scripts/seed_lance_demo.py",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        allow_abbrev=False,
    )
    # Common output/verbosity flags
    fmt = p.add_mutually_exclusive_group()
    fmt.add_argument("--json", action="store_true", help="JSON output (stable, machine-readable)")
    fmt.add_argument("--table", action="store_true", help="Plain table output (no color)")
    p.add_argument("--quiet", action="store_true", help="suppress non-essential stderr")
    p.add_argument("--verbose", action="store_true", help="increase stderr verbosity")
    p.add_argument(
        "args",
        nargs=argparse.REMAINDER,
        help="Pass-through arguments for scripts/seed_lance_demo.py.",
    )
    p.set_defaults(func=_entrypoint, _parser=p)
