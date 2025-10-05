import json
import os
from . import paths
from ..engine.util.io_logging import normalize_for_identity

# Logs that participate in byte-for-byte identity checks on CI
_IDENTITY_LOGS = {"t1.jsonl", "t2.jsonl", "t4.jsonl", "apply.jsonl", "turn.jsonl"}


def _append_jsonl_unbuffered(filename: str, record: dict) -> None:
    base = paths.logs_dir()
    os.makedirs(base, exist_ok=True)
    path = os.path.join(base, filename)

    name = os.path.basename(filename)
    record = normalize_for_identity(name, record)

    # Write deterministically across OS:
    # - Binary append avoids platform newline translation (e.g., CRLF on Windows).
    # - We always terminate records with a single LF to stabilize identity.
    line = (json.dumps(record, ensure_ascii=False) + "\n").encode("utf-8")
    with open(path, "ab") as f:
        f.write(line)


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
    name = os.path.basename(str(filename))
    record = normalize_for_identity(name, record)
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
