

import importlib
import os
import tempfile

import pytest


def _run(query, cfg):
    try:
        t2 = importlib.import_module("clematis.engine.stages.t2")
    except Exception:
        pytest.skip("t2 module not importable")
    run_t2 = getattr(t2, "run_t2", None)
    if run_t2 is None:
        pytest.skip("run_t2 entrypoint not available")
    return run_t2(cfg, query=query)


def test_reader_mode_metric_emitted_flat():
    """With triple gate on and mode=flat, emit t2.reader_mode=flat."""
    cfg = {
        "perf": {"enabled": True, "metrics": {"report_memory": True}},
        "t2": {"reader": {"mode": "flat"}},
    }
    res = _run("apple", cfg)
    metrics = getattr(res, "metrics", {})
    assert metrics.get("t2.reader_mode") == "flat"


def test_reader_partition_fallback_when_unavailable():
    """When mode=partition but fixture is unavailable, fallback to flat and report it."""
    # Point embed_root to a non-existent directory to force unavailability
    bogus_root = os.path.join(tempfile.gettempdir(), "clematis_no_such_partition_root")
    if os.path.isdir(bogus_root):
        # Extremely unlikely, but make it non-existing deterministically
        bogus_root = bogus_root + "_X"
    cfg = {
        "perf": {"enabled": True, "metrics": {"report_memory": True}},
        "t2": {"reader": {"mode": "partition"}, "embed_root": bogus_root},
    }
    res = _run("apple", cfg)
    metrics = getattr(res, "metrics", {})
    # Should fall back to flat
    assert metrics.get("t2.reader_mode") == "flat"


def test_reader_auto_fallback_when_unavailable():
    """When mode=auto and fixture is unavailable, behave as flat and report it."""
    bogus_root = os.path.join(tempfile.gettempdir(), "clematis_no_such_partition_root_auto")
    if os.path.isdir(bogus_root):
        bogus_root = bogus_root + "_X"
    cfg = {
        "perf": {"enabled": True, "metrics": {"report_memory": True}},
        "t2": {"reader": {"mode": "auto"}, "embed_root": bogus_root},
    }
    res = _run("banana", cfg)
    metrics = getattr(res, "metrics", {})
    assert metrics.get("t2.reader_mode") == "flat"


def test_reader_mode_metric_not_emitted_without_metrics_gate():
    """If perf.metrics.report_memory=false, reader_mode metric should not be emitted."""
    cfg = {
        "perf": {"enabled": True, "metrics": {"report_memory": False}},
        "t2": {"reader": {"mode": "flat"}},
    }
    res = _run("carrot", cfg)
    metrics = getattr(res, "metrics", {})
    assert "t2.reader_mode" not in metrics