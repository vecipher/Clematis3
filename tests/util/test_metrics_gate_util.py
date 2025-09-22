import pytest
from clematis.engine.util.metrics import gate_on, emit, emit_many, maybe

def test_gate_on_variants():
    cfg1 = {"perf": {"enabled": True, "metrics": {"report_memory": True}}}
    cfg2 = type("C", (), {"perf": {"enabled": True, "metrics": {"report_memory": True}}})()
    assert gate_on(cfg1) is True
    assert gate_on(cfg2) is True
    assert gate_on({"perf": {"enabled": True, "metrics": {"report_memory": False}}}) is False
    assert gate_on({"perf": {"enabled": False}}) is False

def test_emit_and_emit_many():
    cfg_on = {"perf": {"enabled": True, "metrics": {"report_memory": True}}}
    cfg_off = {"perf": {"enabled": False}}
    evt = {}

    emit(cfg_off, evt, "a", 1)
    assert "metrics" not in evt

    emit(cfg_on, evt, "a", 1)
    assert evt["metrics"]["a"] == 1

    emit_many(cfg_on, evt, {"b": 2, "c": 3})
    assert evt["metrics"]["b"] == 2 and evt["metrics"]["c"] == 3

def test_maybe_readonly():
    evt = {"metrics": {"x": 1}}
    m = maybe(evt)
    assert m == {"x": 1}