"""Logging helpers for the orchestrator pipeline."""

from __future__ import annotations

from typing import Tuple

from ..util.logmux import LogMux, set_mux, reset_mux
from ..util.io_logging import (
    LogStager as _LogStager,
    enable_staging as _enable_staging,
    disable_staging as _disable_staging,
    default_key_for as _default_key_for,
)

LogStager = _LogStager
enable_staging = _enable_staging
disable_staging = _disable_staging
default_key_for = _default_key_for

__all__ = (
    "LogMux",
    "LogStager",
    "enable_staging",
    "disable_staging",
    "default_key_for",
    "_begin_log_capture",
    "_end_log_capture",
)


def _begin_log_capture() -> Tuple[LogMux, object]:
    """Activate a LogMux for the current task and return (mux, token)."""
    mux = LogMux()
    token = set_mux(mux)
    return mux, token


def _end_log_capture(token: object) -> None:
    """Reset the active LogMux using the provided token."""
    try:
        reset_mux(token)
    except Exception:
        pass
