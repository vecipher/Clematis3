from __future__ import annotations

from typing import Optional

# Reuse the single implementation that lives in scripts/console.py
try:  # prefer packaged location
    from clematis.scripts.console import main as _console_main
except Exception:  # dev fallback when running from repo root
    try:
        from scripts.console import main as _console_main
    except Exception as e:  # helpful error if neither path is available
        raise ModuleNotFoundError(
            "Unable to import console implementation. Expected 'clematis.scripts.console' "
            "when installed, or 'scripts.console' in a source checkout. If you're in 'dist/', "
            "run from the repo root or install the package."
        ) from e


def register(subparsers) -> None:
    """
    Register 'console' as a pure pass-through subcommand that relies on the
    umbrella CLI to provide `ns.args` (see clematis/cli/main.py).
    Examples:
      python -m clematis console -- compare --a A.json --b B.json
      python -m clematis console -- step --now-ms 315532800000 --out out.json
    """
    p = subparsers.add_parser(
        "console",
        help="Deterministic local console (step/reset/status/compare)",
    )

    def _run(ns) -> int:
        # clematis/cli/main.py sets ns.args to everything after the subcommand,
        # with any leading `--` already tolerated by that layer.
        argv = list(getattr(ns, "args", []))
        if argv and argv[0] == "--":
            argv = argv[1:]
        return _console_main(argv)

    # IMPORTANT: main.py expects attribute name 'func'
    p.set_defaults(func=_run)


def main(argv: Optional[list[str]] = None) -> int:
    """
    Direct entrypoint for running the console without the umbrella CLI.
    """
    return _console_main(argv or [])
