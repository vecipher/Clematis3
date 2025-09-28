import os
import pytest

from configs.validate import validate_config_api
from clematis.engine.stages.t3 import plan_with_llm


pytestmark = pytest.mark.manual


class _Ctx:
    turn_id = 1
    agent_id = "demo"


def test_manual_llm_smoke_fixture_provider():
    """
    Manual smoke of the LLM planner path using the fixture provider.
    This test is intentionally skipped by default and gated by CLEMATIS_LLM_SMOKE=1.
    It does not require network and should never run in CI.
    """
    if os.environ.get("CLEMATIS_LLM_SMOKE") != "1":
        pytest.skip("manual LLM smoke disabled (set CLEMATIS_LLM_SMOKE=1 to run locally)")

    cfg = {
        "t3": {
            "backend": "llm",
            "llm": {
                "provider": "fixture",
                "fixtures": {"enabled": True, "path": "fixtures/llm/qwen_small.jsonl"},
                "max_tokens": 256,
                "temp": 0.0,
            },
        }
    }

    ok, errs, cfg = validate_config_api(cfg)
    assert ok, f"unexpected config errors: {errs}"

    state = type("S", (), {"logs": []})()
    out = plan_with_llm(_Ctx(), state, cfg)

    # Expect a parsed planner dict from the fixture mapping
    assert isinstance(out, dict)
    assert "plan" in out and isinstance(out["plan"], list)
    assert len(out["plan"]) >= 1
    assert "rationale" in out and isinstance(out["rationale"], str)
    # No LLM errors should be logged when using fixtures
    assert not state.logs, f"unexpected logs: {state.logs}"
