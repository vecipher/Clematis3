


from types import SimpleNamespace
import pytest

import clematis.engine.orchestrator as orch


def test_t2_cache_uses_version_in_key(monkeypatch):
    calls = {"t2": 0}

    # Swallow logs
    monkeypatch.setattr(orch, "append_jsonl", lambda *_args, **_kw: None, raising=True)

    # Minimal T1/T3 so we can run through orchestrator
    monkeypatch.setattr(orch, "t1_propagate", lambda ctx, state, text: SimpleNamespace(metrics={"t1": True}), raising=True)

    def fake_t2(ctx, state, text, t1):
        calls["t2"] += 1
        return SimpleNamespace(metrics={"t2": True}, retrieved=[])
    monkeypatch.setattr(orch, "t2_semantic", fake_t2, raising=True)

    def fake_t3_deliberate(ctx, state, bundle):
        return SimpleNamespace(version="t3-plan-v1", ops=[], deltas=[], reflection=False)
    def fake_t3_dialogue(dialog_bundle, plan):
        return "OK"
    monkeypatch.setattr(orch, "t3_deliberate", fake_t3_deliberate, raising=False)
    monkeypatch.setattr(orch, "t3_dialogue", fake_t3_dialogue, raising=False)

    # Build ctx/state with cache enabled and T4 disabled (we don't need Apply here)
    ctx = SimpleNamespace(
        turn_id=1,
        agent_id="Cachey",
        config=SimpleNamespace(t4={
            "enabled": False,
            "cache_bust_mode": "on-apply",
            "cache": {"enabled": True, "max_entries": 16, "ttl_sec": 600},
        }),
    )
    state = {"version_etag": "1"}  # dict-style state is supported by orchestrator

    # First call: miss -> t2 called
    orch.run_turn(ctx, state, input_text="hello")
    assert calls["t2"] == 1

    # Second call with same version & same input: hit -> t2 NOT called again
    orch.run_turn(ctx, state, input_text="hello")
    assert calls["t2"] == 1  # cache hit

    # Bump version -> next call should be a MISS (key is (version, input))
    state["version_etag"] = "2"
    orch.run_turn(ctx, state, input_text="hello")
    assert calls["t2"] == 2  # re-executed due to version change


def test_t2_cache_is_query_sensitive(monkeypatch):
    calls = {"t2": 0}

    # Swallow logs
    monkeypatch.setattr(orch, "append_jsonl", lambda *_args, **_kw: None, raising=True)
    monkeypatch.setattr(orch, "t1_propagate", lambda ctx, state, text: SimpleNamespace(metrics={"t1": True}), raising=True)

    def fake_t2(ctx, state, text, t1):
        calls["t2"] += 1
        return SimpleNamespace(metrics={"t2": True}, retrieved=[])
    monkeypatch.setattr(orch, "t2_semantic", fake_t2, raising=True)

    def fake_t3_deliberate(ctx, state, bundle):
        return SimpleNamespace(version="t3-plan-v1", ops=[], deltas=[], reflection=False)
    def fake_t3_dialogue(dialog_bundle, plan):
        return "OK"
    monkeypatch.setattr(orch, "t3_deliberate", fake_t3_deliberate, raising=False)
    monkeypatch.setattr(orch, "t3_dialogue", fake_t3_dialogue, raising=False)

    ctx = SimpleNamespace(
        turn_id=1,
        agent_id="Cachey",
        config=SimpleNamespace(t4={
            "enabled": False,
            "cache": {"enabled": True, "max_entries": 16, "ttl_sec": 600},
        }),
    )
    state = {"version_etag": "7"}

    orch.run_turn(ctx, state, input_text="alpha")  # miss
    orch.run_turn(ctx, state, input_text="beta")   # miss (different query)
    assert calls["t2"] == 2

    # Repeating "alpha" now should be a hit (no new call)
    orch.run_turn(ctx, state, input_text="alpha")
    assert calls["t2"] == 2