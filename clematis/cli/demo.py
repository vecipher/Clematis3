

#!/usr/bin/env python3
"""CLI shim: **Delegates to scripts/run_demo.py**.

This wrapper exists so a future `python -m clematis demo ...` subcommand (when
wired in the CLI) can hand off to the real demo runner. We try the packaged
location `clematis.scripts.run_demo` first, then fall back to a repo-layout
`scripts.run_demo` if available. No network usage.
"""
from __future__ import annotations

import sys
from typing import List


def main(argv: List[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]

    # Prefer the packaged-in module (inside the wheel)
    try:
        from ..scripts import run_demo as _demo  # type: ignore
    except Exception:
        try:
            import scripts.run_demo as _demo  # type: ignore
        except Exception:
            sys.stderr.write(
                "clematis.scripts.demo: run_demo.py not found; cannot delegate.\n"
                "Hint: install dev scripts or use `python -m clematis bench-t4 -- --num 1 --runs 1 --json`.\n"
            )
            return 2

    return _demo.main(argv)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())