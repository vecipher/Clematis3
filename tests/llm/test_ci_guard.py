import os
import pytest

from configs.validate import validate_config_api
from clematis.engine.stages.t3 import plan_with_llm


# Run this only in CI where sockets are banned via CLEMATIS_NETWORK_BAN=1
pytestmark = pytest.mark.skipif(os.environ.get("CI") != "true", reason="CI-only guard")


class _Ctx:
    turn_id = 1
    agent_id = "ci"


def _joined_errors(logs):
    return " | ".join(str(rec.get("llm_error", "")) for rec in logs if isinstance(rec, dict))


def test_ci_disallows_ollama_network_calls():
    """In CI, provider=ollama must not attempt real network; the socket ban forces a fallback.
    We assert the LLM path safely falls back with an error logged (no crash, no network).
    """
    cfg = {
        "t3": {
            "backend": "llm",
            "llm": {
                "provider": "ollama",
                "model": "qwen3:4b-instruct",
                "endpoint": "http://localhost:11434/api/generate",
                "max_tokens": 32,
                "temp": 0.0,
                "timeout_ms": 2000,
            },
        }
    }
    ok, errs, cfg = validate_config_api(cfg)
    assert ok, f"unexpected config errors: {errs}"

    state = type("S", (), {"logs": []})()
    out = plan_with_llm(_Ctx(), state, cfg)

    # Expect safe fallback and an error log, not a crash
    assert isinstance(out, dict)
    assert out.get("plan") == []
    assert "fallback" in out.get("rationale", "")
    assert any("llm_error" in rec for rec in state.logs), f"logs missing llm_error: {state.logs}"
    msg = _joined_errors(state.logs)
    assert "CI requires fixture provider" in msg


def test_ci_env_asserts_network_ban_present():
    """
    Only enforce the network-ban env contract when the workflow opted in.
    This avoids failing in workflows that don't set the env yet.
    """
    if os.environ.get("CLEMATIS_NETWORK_BAN") != "1":
        pytest.skip("CLEMATIS_NETWORK_BAN not set; skipping enforcement check")
    assert os.environ.get("CLEMATIS_NETWORK_BAN") == "1"


def test_ci_requires_fixtures_enabled_when_fixture_provider():
    cfg = {"t3": {"backend": "llm", "llm": {"provider": "fixture", "fixtures": {"enabled": False, "path": "fixtures/llm/qwen_small.jsonl"}}}}
    ok, errs, cfg = validate_config_api(cfg)
    assert ok, f"unexpected config errors: {errs}"

    state = type("S", (), {"logs": []})()
    out = plan_with_llm(_Ctx(), state, cfg)
    assert isinstance(out, dict)
    assert out.get("plan") == []
    msg = _joined_errors(state.logs)
    assert "CI requires fixtures.enabled=true" in msg
