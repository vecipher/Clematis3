import os, pytest
from configs.validate import validate_config_api
from clematis.engine.stages.t3 import plan_with_llm

class _Ctx: 
    turn_id = 1
    agent_id = "demo"


def _cfg_with_fixture(text, tmp_path):
    """Create a one-off JSONL fixture mapping the current planner prompt hash to `text`.
    Uses tmp_path so tests don't leak files.
    """
    import json
    from clematis.adapters.llm import _prompt_hash
    from clematis.engine.stages.t3 import make_planner_prompt

    p = make_planner_prompt(_Ctx())
    h = _prompt_hash(p)

    path = tmp_path / "oneoff.jsonl"
    path.write_text(
        json.dumps({"prompt_hash": h, "completion": text}) + "\n",
        encoding="utf-8",
    )

    return {
        "t3": {
            "backend": "llm",
            "llm": {
                "provider": "fixture",
                "fixtures": {"enabled": True, "path": str(path)},
            },
        }
    }


def test_invalid_json_falls_back_and_logs(tmp_path):
    cfg = _cfg_with_fixture("not json", tmp_path)
    ok, _, cfg = validate_config_api(cfg)
    assert ok

    state = type("S", (), {"logs": []})()
    out = plan_with_llm(_Ctx(), state, cfg)

    assert out == {"plan": [], "rationale": "fallback: invalid llm output"}
    assert any(
        isinstance(x, dict)
        and "llm_validation_failed" in x
        and x.get("provider") == "fixture"
        for x in state.logs
    )


def test_valid_json_passes_through(tmp_path):
    cfg = _cfg_with_fixture('{"plan":["a"],"rationale":"ok"}', tmp_path)
    ok, _, cfg = validate_config_api(cfg)
    assert ok

    state = type("S", (), {"logs": []})()
    out = plan_with_llm(_Ctx(), state, cfg)

    assert out["plan"] == ["a"] and out["rationale"] == "ok"
    assert not state.logs


def test_fenced_json_is_rejected_and_falls_back(tmp_path):
    txt = "```json\n{\"plan\":[\"a\"],\"rationale\":\"ok\"}\n```"
    cfg = _cfg_with_fixture(txt, tmp_path)
    ok, _, cfg = validate_config_api(cfg)
    assert ok

    state = type("S", (), {"logs": []})()
    out = plan_with_llm(_Ctx(), state, cfg)
    assert out == {"plan": [], "rationale": "fallback: invalid llm output"}
    assert any("llm_validation_failed" in str(x) for x in state.logs)


def test_unknown_field_rejected_and_falls_back(tmp_path):
    cfg = _cfg_with_fixture('{"plan":["a"],"rationale":"ok","debug":true}', tmp_path)
    ok, _, cfg = validate_config_api(cfg)
    assert ok

    state = type("S", (), {"logs": []})()
    out = plan_with_llm(_Ctx(), state, cfg)
    assert out == {"plan": [], "rationale": "fallback: invalid llm output"}
    assert any("llm_validation_failed" in str(x) for x in state.logs)


def test_whitespace_only_item_rejected(tmp_path):
    cfg = _cfg_with_fixture('{"plan":["   "],"rationale":"ok"}', tmp_path)
    ok, _, cfg = validate_config_api(cfg)
    assert ok

    state = type("S", (), {"logs": []})()
    out = plan_with_llm(_Ctx(), state, cfg)
    assert out == {"plan": [], "rationale": "fallback: invalid llm output"}
