from __future__ import annotations
import sys

# Tolerant import so direct execution (`python scripts/rotate_logs.py`) works
try:
    from ._shim_hint import hint_once  # package-style
except Exception:  # pragma: no cover — fallback when not executed as package
    from _shim_hint import hint_once  # type: ignore[attr-defined]

try:  # pragma: no cover - falls back to shim hint when package unavailable
    from clematis.scripts.rotate_logs import main as _impl_main
    _IMPORT_ERROR = None
except ModuleNotFoundError as exc:  # pragma: no cover - executed when package missing
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
