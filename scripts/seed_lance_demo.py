from __future__ import annotations
import sys

# Tolerant import so direct execution (`python scripts/seed_lance_demo.py`) works
try:
    from ._shim_hint import hint_once  # package-style
except Exception:  # pragma: no cover — fallback when not executed as package
    from _shim_hint import hint_once  # type: ignore[attr-defined]

from clematis.scripts.seed_lance_demo import main as _impl_main


def main(argv=None) -> int:
    hint_once()  # stderr-only; deterministic single line; no stdout changes
    return int(_impl_main(argv) or 0)


if __name__ == "__main__":
    raise SystemExit(main())