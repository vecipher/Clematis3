

import pytest
from types import SimpleNamespace
from typing import List, Tuple, Dict, Any

# Under test
from clematis.engine.orchestrator import logging as ORCH_LOG
from clematis.engine.util import io_logging as IOL


def test_reflection_logging_timeout_reason_normalized_ci(monkeypatch):
    """
    PR86: When reflection times out, the telemetry record must include reason='reflection_timeout'
    and, under CI, the 'ms' field must be normalized to 0.0.
    """
    # Turn on CI normalization
    monkeypatch.setenv("CI", "true")

    captured: List[Tuple[str, Dict[str, Any]]] = []

    def fake_append_jsonl(stage: str, record: Dict[str, Any], mux=None) -> None:
        # Apply the same normalization the real writer uses
        norm = IOL.normalize_for_identity(stage, record)
        captured.append((stage, norm))

    # Patch the real writer used by log_t3_reflection
    monkeypatch.setattr(ORCH_LOG, "append_jsonl", fake_append_jsonl, raising=False)

    # Minimal ctx carrying a turn id
    ctx = SimpleNamespace(turn_id=3)

    # Emit a timeout-annotated telemetry record
    ORCH_LOG.log_t3_reflection(
        log_mux=None,
        ctx=ctx,
        agent="AgentTimeout",
        summary_len=0,
        ops_written=0,
        embed=False,
        backend="rulebased",
        ms=250.75,  # should be normalized to 0.0 under CI
        reason="reflection_timeout",
        extra=None,
    )

    # Assertions
    assert len(captured) == 1, "Expected a single telemetry record"
    stage, rec = captured[0]

    assert stage == "t3_reflection.jsonl"
    assert rec["turn"] == 3
    assert rec["agent"] == "AgentTimeout"
    assert rec["summary_len"] == 0
    assert rec["ops_written"] == 0
    assert rec["embed"] is False
    assert rec["backend"] == "rulebased"
    # CI normalization of time
    assert rec["ms"] == 0.0
    # Timeout reason propagated
    assert rec["reason"] == "reflection_timeout"
    # No fixture_key in rule-based path
    assert "fixture_key" not in rec
