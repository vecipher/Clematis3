from pathlib import Path
import json
from clematis.engine.stages.t2_quality_trace import emit_trace
import pytest
from configs.validate import validate_config_verbose

def run_t2_for_test(query: str, cfg: dict):
    # Minimal deterministic items for comparison
    items = [{"id": "A", "score": 0.9}, {"id": "B", "score": 0.8}]
    # Emulate the PR36 triple-gated shadow emission (no ranking mutation)
    perf = cfg.get("perf", {})
    metrics = perf.get("metrics", {})
    q = cfg.get("t2", {}).get("quality", {})
    if perf.get("enabled") and metrics.get("report_memory") and q.get("shadow") and not q.get("enabled", False):
        emit_trace(cfg, query, items, {"k": len(items), "reason": "shadow", "note": "test stub shadow"})
    return items

def _shadow_cfg():
    return {
        "perf": {"enabled": True, "metrics": {"report_memory": True}},
        "t2": {"quality": {"enabled": False, "shadow": True, "trace_dir": "logs/quality", "redact": True}},
    }

def _disabled_cfg():
    return {
        "perf": {"enabled": False, "metrics": {"report_memory": False}},
        "t2": {"quality": {"enabled": False, "shadow": False}},
    }

def test_shadow_no_rank_diff(tmp_path, monkeypatch):
    # Baseline
    cfg0 = _disabled_cfg()
    items0 = run_t2_for_test("hello world", cfg0)
    # Shadow (triple-gated)
    cfg1 = _shadow_cfg()
    cfg1["t2"]["quality"]["trace_dir"] = str(tmp_path / "logs/quality")
    items1 = run_t2_for_test("hello world", cfg1)

    assert [it["id"] for it in items1] == [it["id"] for it in items0]
    assert [it["score"] for it in items1] == [it["score"] for it in items0]

    # Trace presence
    p = Path(cfg1["t2"]["quality"]["trace_dir"]) / "rq_traces.jsonl"
    assert p.exists(), "shadow must produce rq_traces.jsonl"
    # Record is valid JSON with required headers
    ln = p.read_text(encoding="utf-8").splitlines()[-1]
    rec = json.loads(ln)
    assert rec["trace_schema_version"] >= 1
    assert "config_digest" in rec and "query_id" in rec

def test_triple_gate_required(tmp_path):
    # Any gate missing -> no trace file
    cases = [
        {"perf": {"enabled": False}, "metrics": {"report_memory": True}},
        {"perf": {"enabled": True}, "metrics": {"report_memory": False}},
        {"perf": {"enabled": True}, "metrics": {"report_memory": True}, "shadow": False},
    ]
    for c in cases:
        cfg = {
            "perf": {"enabled": c["perf"]["enabled"], "metrics": {"report_memory": c["metrics"]["report_memory"]}},
            "t2": {"quality": {"enabled": False, "shadow": c.get("shadow", False), "trace_dir": str(tmp_path / "logs")}},
        }
        _ = run_t2_for_test("q", cfg)
        p = Path(cfg["t2"]["quality"]["trace_dir"]) / "rq_traces.jsonl"
        assert not p.exists(), "trace must not be written when any gate is off"


# Validator tests
def test_quality_defaults_off_accepts():
    raw = {"t2": {"quality": {"enabled": False, "shadow": False, "trace_dir": "logs/quality", "redact": True}}}
    norm, warnings = validate_config_verbose(raw)
    q = norm["t2"]["quality"]
    assert q["enabled"] is False and q["shadow"] is False
    assert isinstance(q["trace_dir"], str) and q["trace_dir"]
    assert q["redact"] is True
    # Should not error; warnings are allowed but not required here.

def test_quality_enabled_accepted_in_pr37_with_warnings():
    raw = {"t2": {"quality": {"enabled": True}}}
    norm, warnings = validate_config_verbose(raw)
    q = norm["t2"]["quality"]
    assert q["enabled"] is True
    # Expect warnings when enabling without perf gates
    assert any(
        str(w).startswith("W[t2.quality.enabled]") or str(w).startswith("W[t2.quality.metrics]")
        for w in warnings
    ), "Expected PR37 warnings when enabling quality without perf gates"