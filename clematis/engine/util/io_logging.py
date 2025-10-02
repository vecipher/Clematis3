

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from contextvars import ContextVar
import os

__all__ = [
    "LogKey",
    "StagedRecord",
    "LogStager",
    "enable_staging",
    "disable_staging",
    "staging_enabled",
    "default_key_for",
    "normalize_for_identity",
]

# Context flags/state (kept local to avoid global side effects when disabled)
STAGING_ENABLED: ContextVar[bool] = ContextVar("STAGING_ENABLED", default=False)
STAGING_STATE: ContextVar[Optional["LogStager"]] = ContextVar("STAGING_STATE", default=None)


# Stable within-turn stream order. Unknown streams are ordered last (ord=99).
STAGE_ORD: Dict[str, int] = {
    "t1.jsonl": 1,
    "t2.jsonl": 2,
    "t3_plan.jsonl": 3,
    "t3_dialogue.jsonl": 4,
    "t4.jsonl": 5,
    "apply.jsonl": 6,
    "health.jsonl": 7,
    "turn.jsonl": 8,
    "scheduler.jsonl": 9,
}

# Logs that participate in byte-for-byte identity checks
_IDENTITY_LOGS = {
    "t1.jsonl",
    "t2.jsonl",
    "t4.jsonl",
    "apply.jsonl",
    "turn.jsonl",
}

def normalize_for_identity(name: str, rec: Dict[str, Any]) -> Dict[str, Any]:
    """
    For CI identity checks, strip runtime noise from known identity logs:
    - zero `ms`
    - drop `now`
    No-op when CI is not set.
    """
    if os.environ.get("CI", "").lower() != "true":
        return rec
    if name not in _IDENTITY_LOGS:
        return rec
    out = dict(rec)
    if "ms" in out:
        out["ms"] = 0.0
    out.pop("now", None)
    if name == "turn.jsonl":
        durations = out.get("durations_ms")
        if isinstance(durations, dict):
            out["durations_ms"] = {k: 0.0 for k in durations.keys()}
    return out


@dataclass(frozen=True)
class LogKey:
    """Stable key for ordering staged log records.

    Attributes
    ----------
    turn_id: int
        Logical turn identifier assigned by the orchestrator for this write.
    stage_ord: int
        Stable ordinal derived from the target filename (see STAGE_ORD).
    slice_idx: int
        Position of the agent/compute slice within the current parallel batch.
    seq: int
        Monotonic sequence assigned when the record is staged (ties broken last).
    """

    turn_id: int
    stage_ord: int
    slice_idx: int
    seq: int


@dataclass
class StagedRecord:
    """A log payload staged for deterministic, ordered flushing."""

    file_path: str
    key: LogKey
    payload: Dict[str, Any]
    bytes_estimate: int


class LogStager:
    """In-memory staging buffer with deterministic ordering and a byte bound.

    Notes
    -----
    * The stager performs **no JSON serialization**; formatting is owned by the
      final writer to preserve byte-for-byte parity with the existing path.
    * Memory use is controlled by `byte_limit`. When exceeded, `stage()` raises
      `RuntimeError("LOG_STAGING_BACKPRESSURE")`. The orchestrator should drain
      and flush, then retry staging the current record exactly once.
    """

    def __init__(self, byte_limit: int = 32 * 1024 * 1024) -> None:
        self._buf: List[StagedRecord] = []
        self._seq = 0
        self._bytes = 0
        self.byte_limit = int(byte_limit)

    # ---- sequencing ----
    def next_seq(self) -> int:
        self._seq += 1
        return self._seq

    # ---- staging API ----
    def stage(self, file_path: str, key: LogKey, payload: Dict[str, Any]) -> None:
        # Rough, deterministic size estimate; avoids json.dumps cost here.
        # Use repr-like size without non-deterministic dict ordering by relying
        # on a structural bound: keys + values length (approx).
        # This is deliberately conservative; correctness does not depend on it.
        name = os.path.basename(file_path)
        payload_norm = normalize_for_identity(name, payload)
        est = sum(len(str(k)) + len(str(v)) for k, v in payload_norm.items()) + 2
        if self._bytes + est > self.byte_limit:
            raise RuntimeError("LOG_STAGING_BACKPRESSURE")
        self._buf.append(StagedRecord(file_path, key, payload_norm, est))
        self._bytes += est

    # ---- draining ----
    def drain_sorted(self) -> List[StagedRecord]:
        # Move current buffer out, then sort deterministically by composite key.
        buf = self._buf
        self._buf = []
        self._bytes = 0
        return sorted(
            buf,
            key=lambda r: (
                r.key.turn_id,
                r.key.stage_ord,
                r.key.slice_idx,
                r.key.seq,
                r.file_path,
            ),
        )


# ---- module helpers ----

def enable_staging(byte_limit: int = 32 * 1024 * 1024) -> LogStager:
    """Enable staging in the current context and return the stager instance."""
    stager = LogStager(byte_limit)
    STAGING_STATE.set(stager)
    STAGING_ENABLED.set(True)
    return stager


def disable_staging() -> None:
    """Disable staging in the current context and clear any state reference."""
    STAGING_ENABLED.set(False)
    STAGING_STATE.set(None)


def staging_enabled() -> bool:
    """Return True iff staging is enabled in this context."""
    return STAGING_ENABLED.get()


def default_key_for(*, file_path: str, turn_id: int, slice_idx: int) -> LogKey:
    """Construct a stable LogKey for a given file and turn.

    The stage ordinal is derived from the basename of `file_path`; unknown names
    receive an ordinal of 99 to sort after known streams.
    """
    stager = STAGING_STATE.get()
    if stager is None:
        raise RuntimeError("staging not enabled")
    name = os.path.basename(file_path)
    stage_ord = STAGE_ORD.get(name, 99)
    return LogKey(turn_id=turn_id, stage_ord=stage_ord, slice_idx=slice_idx, seq=stager.next_seq())
