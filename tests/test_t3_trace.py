from clematis.engine.stages.t3.trace import emit_trace


def _cfg(enabled_metrics: bool, enabled_trace: bool) -> dict:
    return {
        "perf": {"metrics": {"enabled": enabled_metrics, "trace_reason": "cfg-reason"}},
        "t3": {"trace": {"enabled": enabled_trace}},
    }


def test_emit_trace_gate_off() -> None:
    logs: list = []
    emit_trace(_cfg(False, False), "prompt", {"cfg": {}}, {"state_logs": logs})
    assert logs == []


def test_emit_trace_gate_on_and_reason_resolution() -> None:
    logs: list = []
    emit_trace(
        _cfg(True, True),
        "prompt",
        {"cfg": {}},
        {"state_logs": logs, "trace_reason": "ctx-reason"},
    )
    assert logs
    entry = logs[0]["t3_trace"]
    assert entry["reason"] == "ctx-reason"
    assert entry["prompt"] == "prompt"
    assert entry["bundle_keys"] == ["cfg"]


def test_emit_trace_falls_back_to_cfg_reason() -> None:
    logs: list = []
    emit_trace(_cfg(True, True), "prompt", {"cfg": {}}, {"state_logs": logs})
    assert logs
    entry = logs[0]["t3_trace"]
    assert entry["reason"] == "cfg-reason"
