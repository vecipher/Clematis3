import argparse, importlib, importlib.util, inspect, sys
from pathlib import Path
from ._wrapper_common import maybe_debug

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
        mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
        assert spec and spec.loader
        spec.loader.exec_module(mod)  # type: ignore[union-attr]
        return mod
    print(f"[clematis] rotate-logs: cannot locate {path.name}. Last error: {last}", file=sys.stderr)
    return None

def _delegate(argv):
    mod = _import_script()
    if mod is None:
        return 2
    main = getattr(mod, "main", None)
    if main is None:
        print("[clematis] rotate-logs: script has no main().", file=sys.stderr)
        return 2
    try:
        sig = inspect.signature(main)
        return main(argv) if len(sig.parameters) >= 1 else main()
    except SystemExit as e:
        return int(getattr(e, "code", 0) or 0)

def _entrypoint(ns):
    argv = list(ns.args or [])
    if argv and argv[0] == "--":
        argv = argv[1:]
    # Intercept help for the wrapper itself (REM AINDER would swallow it)
    if "-h" in argv or "--help" in argv:
        parser = getattr(ns, "_parser", None)
        if parser is not None:
            parser.print_help()
            return 0
    maybe_debug(ns, resolved="scripts.rotate_logs", argv=argv)
    return int(_delegate(argv) or 0)

def register(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "rotate-logs",
        help="Delegates to scripts/rotate_logs.py",
        description="Delegates to scripts/rotate_logs.py",
    )
    p.add_argument("args", nargs=argparse.REMAINDER)
    p.set_defaults(func=_entrypoint, _parser=p)