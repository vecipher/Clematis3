import argparse, importlib, importlib.util, inspect, sys, json, io, contextlib, os
from ._io import set_verbosity, eprint_once, print_table
from ._exit import OK, USER_ERR, IO_ERR, INTERNAL
from ._wrapper_common import maybe_debug, inject_default_from_packaged_or_cwd
from pathlib import Path

_CANDIDATES = ("clematis.scripts.inspect_snapshot", "scripts.inspect_snapshot")

def _import_script():
    last = None
    for n in _CANDIDATES:
        try:
            return importlib.import_module(n)
        except Exception as e:
            last = e
    root = Path(__file__).resolve().parents[2]
    path = root / "scripts" / "inspect_snapshot.py"
    if path.exists():
        spec = importlib.util.spec_from_file_location("scripts.inspect_snapshot", path)
        mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
        assert spec and spec.loader
        spec.loader.exec_module(mod)  # type: ignore[union-attr]
        return mod
    print(f"[clematis] inspect-snapshot: cannot locate {path.name}. Last error: {last}", file=sys.stderr)
    return None

def _delegate(argv):
    mod = _import_script()
    if mod is None:
        return IO_ERR
    main = getattr(mod, "main", None)
    if main is None:
        print("[clematis] inspect-snapshot: script has no main().", file=sys.stderr)
        return IO_ERR
    try:
        sig = inspect.signature(main)
        return main(argv) if len(sig.parameters) >= 1 else main()
    except SystemExit as e:
        return int(getattr(e, "code", 0) or 0)

def _entrypoint(ns: argparse.Namespace) -> int:
    argv = list(getattr(ns, "args", []) or [])

    if argv and argv[0] == "--":
        argv = argv[1:]

    # Intercept help for the wrapper itself (REMAINDER would swallow it)
    if "-h" in argv or "--help" in argv:
        parser = getattr(ns, "_parser", None)
        if parser is not None:
            parser.print_help()
            return OK

    # Test-only hook: allow CI/tests to force an INTERNAL error code without relying on fragile inputs.
    if os.environ.get("CLEMATIS_TEST_INTERNAL_CRASH") in ("1", "true", "TRUE", "yes", "YES"):  # pragma: no cover
        return INTERNAL

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

    # Output format handling: support --json by forwarding to underlying --format json.
    wants_json = bool(getattr(ns, "json", False) or hoisted_json)
    wants_table = bool(getattr(ns, "table", False) or hoisted_table)

    if wants_json and wants_table:
        if not quiet:
            eprint_once("Choose exactly one of --json or --table.")
        return USER_ERR

    if wants_table:
        # Ensure JSON is produced by the delegate so we can render a table
        if "--format" not in argv:
            argv = ["--format", "json", *argv]
        else:
            try:
                idx = argv.index("--format")
                fmt = argv[idx + 1] if idx + 1 < len(argv) else None
            except ValueError:
                fmt = None
            if fmt and fmt.lower() != "json":
                if not quiet:
                    eprint_once("`inspect-snapshot`: --table requires --format json if --format is supplied.")
                return USER_ERR

        # If user did not supply --dir, inject packaged examples/snapshots or CWD fallback.
        argv = inject_default_from_packaged_or_cwd(
            argv,
            flag_names=("--dir",),
            packaged_parts=("examples", "snapshots"),
            cwd_rel="snapshots",
        )
        maybe_debug(ns, resolved="scripts.inspect_snapshot", argv=argv)

        # Capture delegate JSON output and render as a simple key/value table
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            code = int(_delegate(argv) or 0)
        if code != 0:
            return code
        raw = buf.getvalue()
        try:
            payload = json.loads(raw)
        except Exception:
            if not quiet:
                eprint_once("`inspect-snapshot`: expected JSON output to render table.")
            return IO_ERR

        def _val_to_str(v):
            if isinstance(v, (dict, list)):
                return json.dumps(v, separators=(",", ":"))
            return "" if v is None else str(v)

        rows = []
        headers = []
        if isinstance(payload, dict):
            headers = ["key", "value"]
            rows = [{"key": str(k), "value": _val_to_str(v)} for k, v in payload.items()]
        elif isinstance(payload, list):
            headers = ["index", "value"]
            rows = [{"index": str(i), "value": _val_to_str(v)} for i, v in enumerate(payload)]
        else:
            headers = ["value"]
            rows = [{"value": _val_to_str(payload)}]

        print_table(rows, headers=headers)
        return OK

    if wants_json:
        # Ensure we pass --format json once
        if "--format" not in argv:
            argv = ["--format", "json", *argv]
        else:
            # If --format is provided, trust the user but fail fast if not json
            try:
                idx = argv.index("--format")
                fmt = argv[idx + 1] if idx + 1 < len(argv) else None
            except ValueError:
                fmt = None
            if fmt and fmt.lower() != "json":
                if not quiet:
                    eprint_once("`inspect-snapshot`: --json requires --format json if --format is supplied.")
                return USER_ERR

    # If user did not supply --dir, inject packaged examples/snapshots or CWD fallback.
    argv = inject_default_from_packaged_or_cwd(
        argv,
        flag_names=("--dir",),
        packaged_parts=("examples", "snapshots"),
        cwd_rel="snapshots",
    )
    maybe_debug(ns, resolved="scripts.inspect_snapshot", argv=argv)
    return int(_delegate(argv) or 0)

def register(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "inspect-snapshot",
        help="Delegates to scripts/inspect_snapshot.py",
        description="Delegates to scripts/inspect_snapshot.py",
    )
    # Common output/verbosity flags
    fmt = p.add_mutually_exclusive_group()
    fmt.add_argument("--json", action="store_true", help="JSON output (stable, machine-readable)")
    fmt.add_argument("--table", action="store_true", help="Plain table output (no color)")
    p.add_argument("--quiet", action="store_true", help="suppress non-essential stderr")
    p.add_argument("--verbose", action="store_true", help="increase stderr verbosity")
    p.add_argument("args", nargs=argparse.REMAINDER, help="Pass-through arguments for scripts/inspect_snapshot.py.")
    p.set_defaults(func=_entrypoint, _parser=p)