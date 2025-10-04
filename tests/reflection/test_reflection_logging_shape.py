import pytest
from types import SimpleNamespace
from typing import List, Tuple, Dict, Any

# Under test
from clematis.engine.orchestrator import logging as ORCH_LOG
from clematis.engine.util import io_logging as IOL


def test_reflection_logging_rulebased_shape_and_ci_normalization(monkeypatch):
    """
    PR86: Rule-based backend emits a single, schema-stable record with CI-normalized `ms`,
    and no `fixture_key` field.
    """
    # Ensure CI normalization is active
    monkeypatch.setenv("CI", "true")

    captured: List[Tuple[str, Dict[str, Any]]] = []

    def fake_append_jsonl(stage: str, record: Dict[str, Any], mux=None) -> None:
        # Apply the same normalization the real writer uses
        norm = IOL.normalize_for_identity(stage, record)
        captured.append((stage, norm))

    # Patch the real writer used by log_t3_reflection
    monkeypatch.setattr(ORCH_LOG, "append_jsonl", fake_append_jsonl, raising=False)

    # Minimal ctx carrying a turn id
    ctx = SimpleNamespace(turn_id=1)

    # Emit twice to assert deterministic payload under CI normalization
    for _ in range(2):
        ORCH_LOG.log_t3_reflection(
            log_mux=None,
            ctx=ctx,
            agent="AgentA",
            summary_len=87,
            ops_written=1,
            embed=True,
            backend="rulebased",
            ms=12.345,  # should be normalized to 0.0 under CI
            reason=None,
            extra=None,
        )

    assert len(captured) == 2
    (stage1, rec1), (stage2, rec2) = captured

    assert stage1 == "t3_reflection.jsonl"
    assert stage2 == "t3_reflection.jsonl"

    # Core shape & types
    for rec in (rec1, rec2):
        assert rec["turn"] == 1
        assert rec["agent"] == "AgentA"
        assert rec["summary_len"] == 87
        assert rec["ops_written"] == 1
        assert rec["embed"] is True
        assert rec["backend"] == "rulebased"
        # CI normalization
        assert rec["ms"] == 0.0
        # reason should be null/None when not provided
        assert rec["reason"] is None
        # No fixture_key on rule-based path
        assert "fixture_key" not in rec

    # Determinism: canonicalized JSON payloads must match
    s1 = IOL.stable_json_dumps(rec1)
    s2 = IOL.stable_json_dumps(rec2)
    assert s1 == s2


def test_reflection_logging_rulebased_no_ci_preserves_ms(monkeypatch):
    """
    PR86: When CI is not set, `ms` should not be normalized by io_logging.
    """
    # Ensure CI is disabled
    monkeypatch.delenv("CI", raising=False)

    captured: List[Tuple[str, Dict[str, Any]]] = []

    def fake_append_jsonl(stage: str, record: Dict[str, Any], mux=None) -> None:
        norm = IOL.normalize_for_identity(stage, record)
        captured.append((stage, norm))

    # Patch the real writer used by log_t3_reflection
    monkeypatch.setattr(ORCH_LOG, "append_jsonl", fake_append_jsonl, raising=False)

    ctx = SimpleNamespace(turn_id=2)
    ORCH_LOG.log_t3_reflection(
        log_mux=None,
        ctx=ctx,
        agent="AgentB",
        summary_len=5,
        ops_written=0,
        embed=False,
        backend="rulebased",
        ms=99.9,
        reason="reflection_timeout",
        extra=None,
    )

    assert len(captured) == 1
    stage, rec = captured[0]
    assert stage == "t3_reflection.jsonl"

    # ms should be preserved since CI is not set
    assert rec["ms"] == 99.9
    assert rec["reason"] == "reflection_timeout"
    # No fixture_key expected
    assert "fixture_key" not in rec
