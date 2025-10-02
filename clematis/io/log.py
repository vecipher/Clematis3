import json
import os
from . import paths

# Logs that participate in byte-for-byte identity checks on CI
_IDENTITY_LOGS = {"t1.jsonl", "t2.jsonl", "t4.jsonl", "apply.jsonl", "turn.jsonl"}


def _append_jsonl_unbuffered(filename: str, record: dict) -> None:
    base = paths.logs_dir()
    os.makedirs(base, exist_ok=True)
    path = os.path.join(base, filename)

    # PR76: CI identity normalization at the final write sink
    try:
        name = os.path.basename(filename)
        if os.environ.get("CI", "").lower() == "true" and name in _IDENTITY_LOGS:
            rec = dict(record)
            # Drop wall-clock and zero elapsed ms
            rec.pop("now", None)
            if "ms" in rec:
                rec["ms"] = 0.0
            # Also normalize nested durations for turn.jsonl
            if name == "turn.jsonl" and isinstance(rec.get("durations_ms"), dict):
                rec["durations_ms"] = {k: 0.0 for k in rec["durations_ms"].keys()}
            record = rec
    except Exception:
        # Never block logging due to normalization
        pass

    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def append_jsonl(filename: str, record: dict, *, feature_guard: bool | None = None) -> None:
    """Append a JSON record to a log file, or buffer it if a LogMux is active.

    PR70: When the agent-level parallel driver enables a per-turn LogMux, we
    capture (stream, obj) pairs instead of writing immediately. The commit phase
    will flush them in deterministic order. When no mux is active, behavior is
    unchanged (write-through).

    PR71: When the parallel driver's staging is active, the orchestrator will bypass
    the mux at flush-time by calling _append_jsonl_unbuffered directly.
    """
    # PR76: allow callers to suppress writes when a feature is disabled
    if feature_guard is False:
        return
    # PR76: CI identity normalization — drop 'now', zero 'ms' for known logs
    try:
        name = os.path.basename(str(filename))
        if os.environ.get("CI", "").lower() == "true" and name in _IDENTITY_LOGS:
            rec = dict(record)
            # Drop wall-clock and zero elapsed ms
            rec.pop("now", None)
            if "ms" in rec:
                rec["ms"] = 0.0
            # Also normalize nested durations for turn.jsonl
            if name == "turn.jsonl" and isinstance(rec.get("durations_ms"), dict):
                rec["durations_ms"] = {k: 0.0 for k in rec["durations_ms"].keys()}
            record = rec
    except Exception:
        # Never block logging due to normalization
        pass
    # Try to detect an active LogMux without introducing a hard import dependency
    # (avoid circular import at module load; import only inside the function).
    mux = None
    try:  # best-effort, safe if module missing
        from ..engine.util.logmux import LOG_MUX  # type: ignore
        try:
            mux = LOG_MUX.get()
        except Exception:
            mux = None
    except Exception:
        mux = None

    if mux is not None:
        try:
            mux.write(str(filename), dict(record))
            return
        except Exception:
            # Fall back to write-through on any buffering error
            pass

    _append_jsonl_unbuffered(filename, record)
