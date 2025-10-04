import pytest
from types import SimpleNamespace
from typing import List, Tuple, Dict, Any

# Under test
from clematis.engine.orchestrator import logging as ORCH_LOG
from clematis.engine.util import io_logging as IOL


def test_reflection_logging_llm_fixture_extra_includes_fixture_key_and_normalizes_ms(monkeypatch):
    """
    PR86: When backend=='llm' and a fixture_key is present in metrics,
    the reflection telemetry log must include 'fixture_key', and in CI the 'ms' field is normalized to 0.0.

    This test bypasses the filesystem by monkeypatching the real writer used by log_t3_reflection.
    """
    # Ensure CI normalization is active
    monkeypatch.setenv("CI", "true")

    captured: List[Tuple[str, Dict[str, Any]]] = []

    def fake_append_jsonl(stage: str, record: Dict[str, Any], mux=None) -> None:
        # Apply the same identity normalization that the real writer would apply
        norm = IOL.normalize_for_identity(stage, record)
        captured.append((stage, norm))

    # Patch the real writer used by log_t3_reflection
    monkeypatch.setattr(ORCH_LOG, "append_jsonl", fake_append_jsonl, raising=False)

    # Minimal ctx carrying a turn id
    ctx = SimpleNamespace(turn_id=1)

    # First emission
    ORCH_LOG.log_t3_reflection(
        log_mux=None,
        ctx=ctx,
        agent="AgentA",
        summary_len=12,
        ops_written=1,
        embed=True,
        backend="llm",
        ms=123.456,
        reason=None,
        extra={"fixture_key": "abc123def456"},
    )

    # Second emission (identical) to verify determinism of the normalized payload
    ORCH_LOG.log_t3_reflection(
        log_mux=None,
        ctx=ctx,
        agent="AgentA",
        summary_len=12,
        ops_written=1,
        embed=True,
        backend="llm",
        ms=123.456,
        reason=None,
        extra={"fixture_key": "abc123def456"},
    )

    # Assertions
    assert len(captured) == 2, "Expected two telemetry records to be emitted"
    (stage1, rec1), (stage2, rec2) = captured

    assert stage1 == "t3_reflection.jsonl"
    assert stage2 == "t3_reflection.jsonl"

    # Core shape & types
    for rec in (rec1, rec2):
        assert rec["turn"] == 1
        assert rec["agent"] == "AgentA"
        assert rec["summary_len"] == 12
        assert rec["ops_written"] == 1
        assert rec["embed"] is True
        assert rec["backend"] == "llm"
        # CI normalization of time
        assert rec["ms"] == 0.0
        # reason should be null/None when not provided
        assert rec["reason"] is None
        # fixture_key should be present for LLM fixture path
        assert rec.get("fixture_key") == "abc123def456"

    # Determinism: normalized payloads must be byte-identical after JSON canonicalization
    s1 = IOL.stable_json_dumps(rec1)
    s2 = IOL.stable_json_dumps(rec2)
    assert s1 == s2, "Reflection telemetry payload should be stable across repeated runs"
