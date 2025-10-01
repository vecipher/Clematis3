

"""PR70: Deterministic log capture for agent-level parallel driver.

This module provides a tiny buffered logger (LogMux) and helpers to allow
stages to emit their usual JSONL lines while the driver *buffers* them per
turn, then flushes in a deterministic order during the commit phase.

Default behavior (no mux set) writes through to the real append_jsonl.
"""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar, Token
from typing import Dict, List, Optional, Tuple

# Underlying writer
from clematis.io.log import append_jsonl as _append_jsonl  # type: ignore


class LogMux:
    """Buffered writer used by the agent-level parallel driver.

    Usage:
        mux = LogMux()
        token = set_mux(mux)
        ... # run turn compute; stages call write_or_buffer(...)
        pairs = mux.dump()
        flush(pairs)
        reset_mux(token)
    """

    def __init__(self) -> None:
        self._buf: List[Tuple[str, Dict]] = []

    def write(self, stream: str, obj: Dict) -> None:
        self._buf.append((str(stream), obj))

    def dump(self) -> List[Tuple[str, Dict]]:
        return list(self._buf)

    def clear(self) -> None:
        self._buf.clear()


# Context variable holding the active mux (if any) for the current task/thread
LOG_MUX: ContextVar[Optional[LogMux]] = ContextVar("LOG_MUX", default=None)


def set_mux(mux: Optional[LogMux]) -> Token:
    """Activate a mux for the current context and return a reset token."""
    return LOG_MUX.set(mux)


def reset_mux(token: Token) -> None:
    """Reset the mux context back to a previous state using the token."""
    LOG_MUX.reset(token)


def write_or_buffer(stream: str, obj: Dict) -> None:
    """Write to active mux if present; else pass through to real writer.

    This is the preferred call-point from code that wants to be capture-aware
    without importing the concrete log writer.
    """
    try:
        mux = LOG_MUX.get()
    except Exception:
        mux = None
    if mux is not None:
        try:
            mux.write(stream, obj)
            return
        except Exception:
            # Fall back to real writer if buffering fails for any reason
            pass
    _append_jsonl(stream, obj)


def flush(pairs: List[Tuple[str, Dict]]) -> None:
    """Flush a list of (stream, obj) pairs via the real writer in order."""
    for stream, obj in pairs:
        _append_jsonl(stream, obj)


@contextmanager
def use_mux(mux: Optional[LogMux]):
    """Context manager to set/reset a mux for the current context."""
    token = set_mux(mux)
    try:
        yield mux
    finally:
        reset_mux(token)
