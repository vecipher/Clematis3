import argparse, importlib, importlib.util, inspect, sys
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
        return 2
    main = getattr(mod, "main", None)
    if main is None:
        print("[clematis] inspect-snapshot: script has no main().", file=sys.stderr)
        return 2
    try:
        sig = inspect.signature(main)
        return main(argv) if len(sig.parameters) >= 1 else main()
    except SystemExit as e:
        return int(getattr(e, "code", 0) or 0)

def _entrypoint(ns: argparse.Namespace) -> int:
    argv = list(getattr(ns, "args", []) or [])
    if argv and argv[0] == "--":
        argv = argv[1:]
    return int(_delegate(argv) or 0)

def register(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "inspect-snapshot",
        help="Inspect snapshots directory and emit JSON summary (wrapper).",
        description="Delegates to scripts/inspect_snapshot.py",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        allow_abbrev=False,
    )
    p.add_argument("args", nargs=argparse.REMAINDER, help="Pass-through arguments for scripts/inspect_snapshot.py.")
    p.set_defaults(func=_entrypoint)