"""Logging helpers for the orchestrator pipeline."""

from __future__ import annotations

from typing import Any, Dict, Tuple, Optional
import sys as _sys

from ..util.logmux import LogMux, set_mux, reset_mux
from ..util.io_logging import LogStager, enable_staging, disable_staging, default_key_for
from ..util import io_logging as IOL
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
    "log_t3_reflection",
    "log_t1_native_diag",
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
    """Public append hook; applies CI/identity normalization, then delegates."""
    try:
        rec = IOL.normalize_for_identity(file_path, payload)
    except Exception:
        rec = payload
    _append_jsonl(file_path, rec)


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


# Helper for t3_reflection telemetry logging
def log_t3_reflection(
    log_mux: LogMux,
    ctx,
    agent: str,
    *,
    summary_len: int,
    ops_written: int,
    embed: bool,
    backend: str,
    ms: float,
    reason: Optional[str] = None,
    extra: Optional[Dict[str, str]] = None,
) -> None:
    """
    Emit a single reflection telemetry record to the staged 't3_reflection.jsonl' stream.
    Fail-soft: any error during logging is swallowed.
    """
    payload: Dict[str, Any] = {
        "turn": int(getattr(ctx, "turn_id", 0)),
        "agent": str(agent),
        "summary_len": int(summary_len),
        "ops_written": int(ops_written),
        "embed": bool(embed),
        "backend": str(backend),
        "ms": float(ms),
        "reason": (None if reason is None else str(reason)),
    }
    if extra and extra.get("fixture_key"):
        payload["fixture_key"] = str(extra["fixture_key"])
    try:
        append_jsonl("t3_reflection.jsonl", payload)
    except Exception:
        # Never crash the turn on logging issues
        return


# Helper for native T1 diagnostics logging (non-identity stream)
def log_t1_native_diag(
    ctx,
    agent: str,
    native_t1: Dict[str, Any],
) -> None:
    """
    Emit a single diagnostics record to the staged 't1_native_diag.jsonl' stream.
    Fail-soft: any error during logging is swallowed.
    Only call this if `native_t1` is truthy (non-empty dict).
    """
    if not native_t1:
        return
    payload: Dict[str, Any] = {
        "turn": str(getattr(ctx, "turn_id", "-")),
        "agent": str(agent),
        "native_t1": dict(native_t1),
    }
    try:
        append_jsonl("t1_native_diag.jsonl", payload)
    except Exception:
        # Never crash the turn on logging issues
        return
