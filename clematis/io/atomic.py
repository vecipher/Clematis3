

from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Optional

__all__ = [
    "atomic_write_text",
    "atomic_write_bytes",
    "atomic_write_json",
    "atomic_replace",
]

# Conservative default on POSIX; harmless on Windows.
_DEFAULT_PERMS = 0o644


def _fsync_best_effort(path: Path) -> None:
    """Best-effort directory fsync for durability.

    On Windows, fsync on directories may fail—ignore in that case.
    """
    try:
        fd = os.open(str(path), os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    except Exception:
        # Directory fsync is optional; ignore platform limitations.
        pass


def atomic_replace(tmp_path: Path, final_path: Path, *, retries: int = 6, backoff_ms: int = 25) -> None:
    """Cross-platform atomic replace with a small retry window for Windows.

    Ensures parent exists, preserves retry semantics on Windows "sharing violation"
    errors, and cleans up the temporary file on failure.
    """
    final_path.parent.mkdir(parents=True, exist_ok=True)
    last_err: Optional[BaseException] = None

    for _ in range(retries):
        try:
            os.replace(str(tmp_path), str(final_path))  # atomic on POSIX and Windows
            # Best-effort durability: fsync target and parent dir
            try:
                with open(final_path, "rb", buffering=0) as f:
                    os.fsync(f.fileno())
            except Exception:
                pass
            _fsync_best_effort(final_path.parent)
            return
        except Exception as e:  # pragma: no cover - platform-specific timing
            last_err = e
            time.sleep(backoff_ms / 1000.0)

    # Replace never succeeded; attempt cleanup then re-raise.
    try:
        if tmp_path.exists():
            tmp_path.unlink()
    finally:
        if last_err:
            raise last_err


def _make_tmp(final_path: Path) -> Path:
    """Create a temporary file in the same directory as final_path.

    Using delete=False allows closing the file handle before replacement on Windows.
    """
    final_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix=final_path.name + ".",
        dir=str(final_path.parent),
        delete=False,
    ) as tf:
        return Path(tf.name)


def atomic_write_bytes(final_path: Path | str, data: bytes) -> None:
    """Atomically write bytes to *final_path*.

    Writes to a sibling temp file, fsyncs, then os.replace with retry. Attempts to
    preserve existing permissions when the target exists; otherwise applies a
    conservative default. Temp file is cleaned on failure.
    """
    final = Path(final_path)
    tmp = _make_tmp(final)
    try:
        with open(tmp, "wb", buffering=0) as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        # Try to preserve permissions from existing file
        try:
            st = final.stat()
            os.chmod(tmp, st.st_mode)
        except Exception:
            # Fall back to default perms; ignore on platforms that don't honor it
            try:
                os.chmod(tmp, _DEFAULT_PERMS)
            except Exception:
                pass
        atomic_replace(tmp, final)
    except Exception:
        try:
            if tmp.exists():
                tmp.unlink()
        finally:
            raise


def atomic_write_text(
    final_path: Path | str,
    text: str,
    *,
    encoding: str = "utf-8",
    newline: str = "\n",
) -> None:
    """Atomically write text, enforcing UTF-8 and LF by default.

    Any CRLF in *text* is normalized to LF to keep identity stable across OSes.
    """
    # Normalize CRLF to LF to enforce identity
    text = text.replace("\r\n", "\n")
    if newline != "\n":
        # Allow caller override (not recommended for identity-sensitive paths)
        text = text.replace("\n", newline)
    atomic_write_bytes(final_path, text.encode(encoding))


def atomic_write_json(
    final_path: Path | str,
    obj: Any,
    *,
    sort_keys: bool = True,
    separators: tuple[str, str] = (",", ":"),
    ensure_ascii: bool = False,
) -> None:
    """Atomically write canonical JSON.

    Uses sorted keys and compact separators by default to keep byte identity
    stable across platforms and Python versions.
    """
    payload = json.dumps(
        obj,
        sort_keys=sort_keys,
        separators=separators,
        ensure_ascii=ensure_ascii,
    )
    atomic_write_text(final_path, payload, encoding="utf-8", newline="\n")
