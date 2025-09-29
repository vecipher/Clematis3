from pathlib import Path
import json
from clematis.engine.stages.t2_quality_trace import emit_trace


def _cfg(trace_dir: Path, redact: bool):
    return {
        "perf": {"enabled": True, "metrics": {"report_memory": True}},
        "t2": {
            "quality": {
                "enabled": False,
                "shadow": True,
                "trace_dir": str(trace_dir),
                "redact": redact,
            }
        },
    }


def test_emit_trace_redaction_on_off(tmp_path):
    td_red = tmp_path / "logs/quality_red"
    td_raw = tmp_path / "logs/quality_raw"
    items = [{"id": "x1", "score": 0.9, "text": "SECRET", "title": "Hello"}]
    # redacted
    emit_trace(_cfg(td_red, True), " Top Secret Query ", items, {"k": 1})
    rec_red = json.loads((td_red / "rq_traces.jsonl").read_text(encoding="utf-8").splitlines()[-1])
    assert rec_red["query"] == "[REDACTED]"
    assert rec_red["items"][0]["text"] == "[REDACTED]"
    assert rec_red["items"][0]["title"] == "[REDACTED]"
    # not redacted
    emit_trace(_cfg(td_raw, False), " Top Secret Query ", items, {"k": 1})
    rec_raw = json.loads((td_raw / "rq_traces.jsonl").read_text(encoding="utf-8").splitlines()[-1])
    assert rec_raw["query"].strip() == "Top Secret Query"
    assert rec_raw["items"][0]["text"] == "SECRET"
    assert rec_raw["items"][0]["title"] == "Hello"


def test_trace_headers_deterministic_across_runs(tmp_path):
    td1 = tmp_path / "logs/quality_a"
    td2 = tmp_path / "logs/quality_b"
    q = "hello world"
    items = [{"id": "A", "score": 1.0}]
    cfg1 = _cfg(td1, True)
    cfg2 = _cfg(td2, True)
    emit_trace(cfg1, q, items, {"k": 1})
    emit_trace(cfg2, q, items, {"k": 1})
    rec1 = json.loads((td1 / "rq_traces.jsonl").read_text(encoding="utf-8").splitlines()[-1])
    rec2 = json.loads((td2 / "rq_traces.jsonl").read_text(encoding="utf-8").splitlines()[-1])
    # deterministic headers
    assert rec1["trace_schema_version"] == rec2["trace_schema_version"]
    assert rec1["config_digest"] == rec2["config_digest"]
    assert rec1["query_id"] == rec2["query_id"]
    assert rec1["git_sha"] == rec2["git_sha"]
    assert rec1["clock"] == 0 and rec1["seed"] == 0
    assert rec2["clock"] == 0 and rec2["seed"] == 0


def test_config_digest_stable_to_key_order(tmp_path):
    tdA = tmp_path / "logs/qa"
    tdB = tmp_path / "logs/qb"
    items = [{"id": "A", "score": 0.1}]
    # Same semantic config, different insertion orders
    cfgA = {
        "perf": {"enabled": True, "metrics": {"report_memory": True}},
        "t2": {
            "quality": {"enabled": False, "shadow": True, "trace_dir": str(tdA), "redact": True}
        },
    }
    cfgB = {
        "t2": {
            "quality": {"shadow": True, "redact": True, "enabled": False, "trace_dir": str(tdB)}
        },
        "perf": {"metrics": {"report_memory": True}, "enabled": True},
    }
    emit_trace(cfgA, "q", items, {})
    emit_trace(cfgB, "q", items, {})
    recA = json.loads((tdA / "rq_traces.jsonl").read_text(encoding="utf-8").splitlines()[-1])
    recB = json.loads((tdB / "rq_traces.jsonl").read_text(encoding="utf-8").splitlines()[-1])
    assert recA["config_digest"] == recB["config_digest"]


def test_query_id_normalizes_case_and_whitespace(tmp_path):
    td1 = tmp_path / "logs/q1"
    td2 = tmp_path / "logs/q2"
    items = [{"id": "A", "score": 0.1}]
    emit_trace(_cfg(td1, True), "  Hello  World  ", items, {})
    emit_trace(_cfg(td2, True), "hello world", items, {})
    rec1 = json.loads((td1 / "rq_traces.jsonl").read_text(encoding="utf-8").splitlines()[-1])
    rec2 = json.loads((td2 / "rq_traces.jsonl").read_text(encoding="utf-8").splitlines()[-1])
    assert rec1["query_id"] == rec2["query_id"]


def test_trace_items_order_is_preserved(tmp_path):
    td = tmp_path / "logs/q_order"
    items = [{"id": "B", "score": 0.2}, {"id": "A", "score": 0.3}, {"id": "C", "score": 0.1}]
    emit_trace(_cfg(td, True), "q", items, {})
    rec = json.loads((td / "rq_traces.jsonl").read_text(encoding="utf-8").splitlines()[-1])
    assert [it["id"] for it in rec["items"]] == ["B", "A", "C"]
