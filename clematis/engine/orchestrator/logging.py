"""Logging helpers for the orchestrator pipeline."""

from __future__ import annotations

from typing import Any, Dict, Tuple
import sys as _sys

from ..util.logmux import LogMux, set_mux, reset_mux
from ..util.io_logging import LogStager, enable_staging, disable_staging, default_key_for
from ...io.log import append_jsonl as _append_jsonl_default, _append_jsonl_unbuffered

__all__ = [
    "LogMux",
    "LogStager",
    "enable_staging",
    "disable_staging",
    "default_key_for",
    "_begin_log_capture",
    "_end_log_capture",
    "_get_logging_callable",
    "_append_unbuffered",
    "_append_jsonl",
    "append_jsonl",
]


def _get_logging_callable(name: str):
    """Return an override from the orchestrator namespace when available."""
    orch_module = _sys.modules.get("clematis.engine.orchestrator")
    if orch_module is not None:
        func = getattr(orch_module, name, None)
        if callable(func):
            return func
    return globals().get(name)


def _append_unbuffered(path: str, payload: Dict[str, Any]) -> None:
    """Write immediately to disk, respecting orchestrator monkeypatches."""
    orch_module = _sys.modules.get("clematis.engine.orchestrator")
    if orch_module is not None:
        writer = getattr(orch_module, "_append_jsonl_unbuffered", None)
        if callable(writer) and writer is not _append_unbuffered:
            writer(path, payload)
            return
    _append_jsonl_unbuffered(path, payload)


def _append_jsonl(file_path: str, payload: Dict[str, Any]) -> None:
    """Append a payload to the configured JSONL sink with override support."""
    orch_module = _sys.modules.get("clematis.engine.orchestrator")
    if orch_module is not None:
        writer = getattr(orch_module, "append_jsonl", None)
        if callable(writer) and writer is not _append_jsonl:
            writer(file_path, payload)
            return
    _append_jsonl_default(file_path, payload)


def append_jsonl(file_path: str, payload: Dict[str, Any]) -> None:
    """Public append hook; mirrors `_append_jsonl` for compatibility."""
    _append_jsonl(file_path, payload)


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
