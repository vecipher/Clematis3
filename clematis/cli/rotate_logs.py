import argparse
import importlib
import importlib.util
import inspect
import sys
from pathlib import Path

from ._exit import IO_ERR, OK, USER_ERR
from ._io import eprint_once, print_json, print_table, set_verbosity
from ._wrapper_common import inject_default_from_packaged_or_cwd, maybe_debug

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
    argv = list(ns.args or [])
    if argv and argv[0] == "--":
        argv = argv[1:]

    # Intercept help for the wrapper itself (REMAINDER would swallow it)
    if "-h" in argv or "--help" in argv:
        parser = getattr(ns, "_parser", None)
        if parser is not None:
            parser.print_help()
            return OK

    # Configure verbosity (stderr only); stdout remains reserved for command output
    set_verbosity(getattr(ns, "verbose", False), getattr(ns, "quiet", False))
    orig_argv = list(getattr(ns, "args", []) or [])
    quiet = bool(getattr(ns, "quiet", False) or ("--quiet" in orig_argv))

    # Hoist wrapper-only flags that users might put after `--`
    hoisted_json = False
    hoisted_table = False
    filtered = []
    for tok in argv:
        if tok == "--json":
            hoisted_json = True
            continue
        if tok == "--table":
            hoisted_table = True
            continue
        if tok == "--":  # drop stray option-terminator if present mid-argv
            continue
        filtered.append(tok)
    argv = filtered

    # Output format gates for PR55: rotate-logs doesn't yet implement structured output
    wants_json = bool(getattr(ns, "json", False) or hoisted_json)
    wants_table = bool(getattr(ns, "table", False) or hoisted_table)
    if wants_json and wants_table:
        if not quiet:
            eprint_once("Choose exactly one of --json or --table.")
        return USER_ERR

    # If user did not supply --dir, inject packaged examples/logs or CWD fallback.
    resolved_argv = inject_default_from_packaged_or_cwd(
        argv,
        flag_names=("--dir",),
        packaged_parts=("examples", "logs"),
        cwd_rel=".logs",
    )

    if wants_json or wants_table:
        # Build a minimal summary without invoking the delegate.
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
        else:  # wants_table
            print_table([summary], headers=["dry_run", "dir", "pattern"])
            return OK

    # No structured flags: delegate as before
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
