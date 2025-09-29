# PR27 — Ready-Set hook tests
# We validate the existence, default behavior, and simple monkeypatching of
# the orchestrator-level `agent_ready(ctx, state, agent_id)` hook.

import pytest


def test_agent_ready_exists_and_defaults_true():
    orchestrator = pytest.importorskip("clematis.engine.orchestrator")
    fn = getattr(orchestrator, "agent_ready", None)
    assert callable(fn), "agent_ready hook must be defined"

    ok, reason = fn(ctx=object(), state=None, agent_id="any")
    assert ok is True, "default ready-set should return True"
    assert isinstance(reason, str) and reason == "DEFAULT_TRUE"


def test_agent_ready_is_pure_for_same_inputs():
    orchestrator = pytest.importorskip("clematis.engine.orchestrator")
    fn = getattr(orchestrator, "agent_ready")

    ctx = object()
    t1 = fn(ctx=ctx, state=None, agent_id="Ambrose")
    t2 = fn(ctx=ctx, state=None, agent_id="Ambrose")
    assert t1 == t2 == (True, "DEFAULT_TRUE"), "default hook must be deterministic/pure"


def test_agent_ready_can_be_monkeypatched_to_block_specific_agent(monkeypatch):
    """
    We don't test integration here (selection happens outside orchestrator in this codebase).
    This simply ensures the hook contract is friendly to deterministic monkeypatching.
    """
    import clematis.engine.orchestrator as orch

    def blocked(ctx, state, agent_id: str):
        if agent_id == "BlockedAgent":
            return False, "BLOCKED_BY_TEST"
        return True, "DEFAULT_TRUE"

    monkeypatch.setattr(orch, "agent_ready", blocked, raising=True)

    ok1, reason1 = orch.agent_ready(ctx=object(), state=None, agent_id="BlockedAgent")
    ok2, reason2 = orch.agent_ready(ctx=object(), state=None, agent_id="Other")
    assert ok1 is False and reason1 == "BLOCKED_BY_TEST"
    assert ok2 is True and reason2 == "DEFAULT_TRUE"
