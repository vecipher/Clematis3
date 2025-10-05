# tests/native/test_diag_stream_jsonl.py
from types import SimpleNamespace
import clematis.engine.orchestrator.logging as olog


class _Capture:
    def __init__(self):
        self.records = []
    def __call__(self, fname, payload):
        self.records.append((fname, payload))


def test_diag_stream_written_when_native_metrics_present(monkeypatch):
    cap = _Capture()
    # capture writes from the helper
    monkeypatch.setattr(olog, "append_jsonl", lambda fname, payload: cap(fname, payload))

    ctx = SimpleNamespace(turn_id="42")
    olog.log_t1_native_diag(ctx, agent="alice", native_t1={"used_native": 1})

    diag = [(name, p) for (name, p) in cap.records if name == "t1_native_diag.jsonl"]
    assert len(diag) == 1
    _, payload = diag[0]
    assert payload.get("turn") == "42"
    assert payload.get("agent") == "alice"
    assert payload.get("native_t1", {}).get("used_native") == 1


def test_diag_stream_absent_when_no_native_metrics(monkeypatch):
    cap = _Capture()
    monkeypatch.setattr(olog, "append_jsonl", lambda fname, payload: cap(fname, payload))

    ctx = SimpleNamespace(turn_id="7")
    # empty dict → helper should no-op
    olog.log_t1_native_diag(ctx, agent="bob", native_t1={})

    # None also should no-op
    olog.log_t1_native_diag(ctx, agent="bob", native_t1=None)

    diag = [p for (name, p) in cap.records if name == "t1_native_diag.jsonl"]
    assert len(diag) == 0
