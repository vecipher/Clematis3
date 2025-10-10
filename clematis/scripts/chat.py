"""Thin adapter for the chat CLI.

This lives inside the packaged namespace so that `python -m clematis --help`
can import the chat wrapper without requiring the full repo checkout. The heavy
implementation still resides under `scripts/chat.py`; we load it lazily when
the command is actually invoked so the module import remains fast and keeps
packaged builds lightweight.
"""

from __future__ import annotations

from typing import Optional, Sequence


def _resolve_impl():
    try:
        from scripts.chat import main as _impl  # type: ignore[import]
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Chat CLI implementation not bundled; run from a source checkout where "
            "`scripts/chat.py` is available."
        ) from exc
    return _impl


def main(argv: Optional[Sequence[str]] = None) -> int:
    impl = _resolve_impl()
    return impl(list(argv) if argv is not None else None)


__all__ = ["main"]
