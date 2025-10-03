from __future__ import annotations
import sys

# Tolerant import so direct execution (`python scripts/inspect_snapshot.py`) works
try:
    from ._shim_hint import hint_once  # package-style
except Exception:  # pragma: no cover — fallback when not executed as package
    from _shim_hint import hint_once  # type: ignore[attr-defined]

try:  # pragma: no cover — allow shim fallback when package missing
    from clematis.scripts.inspect_snapshot import main as _impl_main
    _IMPORT_ERROR = None
except ModuleNotFoundError as exc:  # pragma: no cover
    _impl_main = None  # type: ignore[assignment]
    _IMPORT_ERROR = exc


def main(argv=None) -> int:
    hint_once()  # stderr-only; deterministic single line; no stdout changes
    if _impl_main is None:
        if _IMPORT_ERROR is not None:
            sys.stderr.write(f"Module import failed: {_IMPORT_ERROR}\n")
        return 0
    return int(_impl_main(argv) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
