import pytest
from configs.validate import validate_config


def test_accepts_lancedb_partitions_minimal():
    cfg = {
        "perf": {"enabled": True},
        "t2": {"lancedb": {"partitions": {"by": ["owner", "quarter"], "shard_order": "lex"}}},
    }
    errs, warns = validate_config(cfg, strict=True, verbose=True)
    assert not errs


def test_warns_when_perf_off():
    cfg = {"perf": {"enabled": False}, "t2": {"lancedb": {"partitions": {"by": ["owner"]}}}}
    errs, warns = validate_config(cfg, strict=True, verbose=True)
    assert not errs
    assert any("perf.enabled=false" in w.lower() or "disabled path" in w.lower() for w in warns)


def test_rejects_bad_types():
    cfg = {"t2": {"lancedb": {"partitions": "nope"}}}
    errs, _ = validate_config(cfg, strict=True, verbose=True)
    assert errs
