

from __future__ import annotations

import pytest

from clematis.engine.util.io_logging import (
    LogStager,
    default_key_for,
    enable_staging,
    disable_staging,
)


def test_stage_and_sort_key():
    try:
        stager = enable_staging()
        turn_id = 5
        # Stage in intentionally jumbled order across streams and slice indices
        records = [
            ("t2.jsonl", 1, {"msg": "b"}),
            ("t1.jsonl", 0, {"msg": "a"}),
            ("scheduler.jsonl", 0, {"msg": "sched"}),
            ("apply.jsonl", 0, {"msg": "apply"}),
            ("t3_dialogue.jsonl", 0, {"msg": "dlg"}),
            ("t4.jsonl", 0, {"msg": "t4"}),
            ("turn.jsonl", 0, {"msg": "turn"}),
            ("foo.jsonl", 0, {"msg": "unknown"}),  # unknown stream -> ord 99
        ]
        for path, slice_idx, payload in records:
            key = default_key_for(file_path=path, turn_id=turn_id, slice_idx=slice_idx)
            stager.stage(path, key, payload)

        drained = stager.drain_sorted()
        # Verify deterministic ordering by the composite key
        got = [
            (r.key.turn_id, r.key.stage_ord, r.key.slice_idx, r.key.seq, r.file_path)
            for r in drained
        ]
        expect = sorted(got, key=lambda t: (t[0], t[1], t[2], t[3], t[4]))
        assert got == expect

        # The unknown stream should sort last with stage_ord=99
        unknown = [r for r in drained if r.file_path.endswith("foo.jsonl")]
        assert len(unknown) == 1
        assert unknown[0].key.stage_ord == 99
    finally:
        disable_staging()


def test_backpressure_flush_retry():
    try:
        # Force backpressure after staging one large-ish record
        stager = enable_staging(byte_limit=150)
        # ~104 bytes each under the estimation heuristic (100 chars + small overhead)
        p1 = {"a": "x" * 100}
        p2 = {"a": "y" * 100}

        k1 = default_key_for(file_path="t1.jsonl", turn_id=1, slice_idx=0)
        stager.stage("t1.jsonl", k1, p1)

        k2 = default_key_for(file_path="t2.jsonl", turn_id=1, slice_idx=0)
        with pytest.raises(RuntimeError) as exc:
            stager.stage("t2.jsonl", k2, p2)
        assert str(exc.value) == "LOG_STAGING_BACKPRESSURE"

        # Drain and verify only the first record is present
        drained1 = stager.drain_sorted()
        assert [r.file_path for r in drained1] == ["t1.jsonl"]

        # Retry staging the second record; it should now succeed
        stager.stage("t2.jsonl", k2, p2)
        drained2 = stager.drain_sorted()
        assert [r.file_path for r in drained2] == ["t2.jsonl"]
    finally:
        disable_staging()


def test_default_key_requires_enabled():
    # Ensure calling default_key_for without enabling raises
    disable_staging()
    with pytest.raises(RuntimeError):
        _ = default_key_for(file_path="t1.jsonl", turn_id=0, slice_idx=0)
