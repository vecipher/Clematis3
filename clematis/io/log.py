import json
import os
from . import paths


def _append_jsonl_unbuffered(filename: str, record: dict) -> None:
    base = paths.logs_dir()
    os.makedirs(base, exist_ok=True)
    path = os.path.join(base, filename)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def append_jsonl(filename: str, record: dict) -> None:
    """Append a JSON record to a log file, or buffer it if a LogMux is active.

    PR70: When the agent-level parallel driver enables a per-turn LogMux, we
    capture (stream, obj) pairs instead of writing immediately. The commit phase
    will flush them in deterministic order. When no mux is active, behavior is
    unchanged (write-through).

    PR71: When the parallel driver's staging is active, the orchestrator will bypass
    the mux at flush-time by calling _append_jsonl_unbuffered directly.
    """
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
