

import json
from pathlib import Path

import pytest


def test_logmux_buffers_when_active():
    from clematis.engine.util.logmux import LogMux, use_mux, LOG_MUX
    from clematis.io.log import append_jsonl

    mux = LogMux()
    with use_mux(mux):
        append_jsonl("foo.jsonl", {"a": 1})
        append_jsonl("bar.jsonl", {"b": "x"})
        # While mux is active, nothing should be written; entries are buffered
        pairs = mux.dump()
        assert len(pairs) == 2
        assert pairs[0][0] == "foo.jsonl"
        assert pairs[0][1]["a"] == 1
        assert pairs[1][0] == "bar.jsonl"
        assert pairs[1][1]["b"] == "x"

    # After context exit, the active mux should be cleared
    assert LOG_MUX.get() is None


def test_logmux_write_through_when_inactive(tmp_path, monkeypatch):
    from clematis.engine.util.logmux import LOG_MUX, reset_mux
    from clematis.io.log import append_jsonl

    # Ensure no active mux
    try:
        token = LOG_MUX.set(None)
        LOG_MUX.reset(token)
    except Exception:
        pass

    # Route logs to a temporary directory
    monkeypatch.setattr(
        "clematis.io.paths.logs_dir", lambda: str(tmp_path)
    )

    append_jsonl("foo.jsonl", {"x": 1})

    p = Path(tmp_path) / "foo.jsonl"
    assert p.exists()
    data = p.read_text(encoding="utf-8").strip().splitlines()
    assert len(data) == 1
    obj = json.loads(data[0])
    assert obj["x"] == 1


def test_logmux_flush_writes_in_order(tmp_path, monkeypatch):
    from clematis.engine.util.logmux import LogMux, use_mux, flush
    from clematis.io.log import append_jsonl

    # Route logs to a temporary directory
    monkeypatch.setattr(
        "clematis.io.paths.logs_dir", lambda: str(tmp_path)
    )

    mux = LogMux()
    with use_mux(mux):
        append_jsonl("z.jsonl", {"seq": 1, "msg": "first"})
        append_jsonl("z.jsonl", {"seq": 2, "msg": "second"})

    # Now flush in the same order
    flush(mux.dump())

    p = Path(tmp_path) / "z.jsonl"
    assert p.exists()
    lines = p.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    o1 = json.loads(lines[0])
    o2 = json.loads(lines[1])
    assert (o1["seq"], o1["msg"]) == (1, "first")
    assert (o2["seq"], o2["msg"]) == (2, "second")
