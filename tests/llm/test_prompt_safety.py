import pytest
from clematis.engine.policy.json_schemas import PLANNER_V1
from clematis.engine.policy.sanitize import parse_and_validate

_VALID = '{"plan":["a","b"],"rationale":"ok"}'

def test_accepts_minimal_valid():
    ok, obj = parse_and_validate(_VALID, PLANNER_V1)
    assert ok and obj["plan"] == ["a","b"] and obj["rationale"] == "ok"

def test_rejects_non_json_prose():
    ok, reason = parse_and_validate("Here is your plan: ['a']", PLANNER_V1)
    assert not ok and "non-JSON" in reason

def test_rejects_codefence_with_prose_outside():
    txt = "please follow\n```json\n" + _VALID + "\n```\nthanks"
    ok, reason = parse_and_validate(txt, PLANNER_V1)
    assert not ok  # prose outside fences

def test_accepts_single_fenced_json_only():
    txt = "```json\n" + _VALID + "\n```"
    ok, obj = parse_and_validate(txt, PLANNER_V1)
    assert ok and obj["plan"] == ["a","b"]

def test_rejects_unknown_fields():
    txt = '{"plan":["a"],"rationale":"ok","debug":true}'
    ok, reason = parse_and_validate(txt, PLANNER_V1)
    assert not ok and "unknown key" in reason

def test_rejects_wrong_types():
    txt = '{"plan":"not-an-array","rationale":{}}'
    ok, reason = parse_and_validate(txt, PLANNER_V1)
    assert not ok

def test_rejects_oversize_items_and_lengths():
    long_item = "x"*201
    long_rat = "y"*2001
    txt = f'{{"plan":["{long_item}"],"rationale":"ok"}}'
    ok, reason = parse_and_validate(txt, PLANNER_V1); assert not ok
    txt = f'{{"plan":["a"],"rationale":"{long_rat}"}}'
    ok, reason = parse_and_validate(txt, PLANNER_V1); assert not ok

def test_rejects_when_raw_output_too_large():
    huge = "{" + '"rationale":"' + ("z"*21000) + '","plan":[]}'
    ok, reason = parse_and_validate(huge, PLANNER_V1)
    assert not ok and "raw output too large" in reason

def test_rejects_plan_exceeding_max_items():
    plan = ["s"] * 17
    txt = '{"plan": ' + str(plan).replace("'", '"') + ', "rationale":"ok"}'
    ok, reason = parse_and_validate(txt, PLANNER_V1)
    assert not ok and "plan too long" in reason

def test_accepts_empty_plan_nonempty_rationale():
    ok, obj = parse_and_validate('{"plan":[],"rationale":"ok"}', PLANNER_V1)
    assert ok and obj["plan"] == [] and obj["rationale"] == "ok"

def test_rejects_when_fenced_json_block_too_large():
    big = "x" * 21000
    txt = f"```json\n{{\"plan\":[],\"rationale\":\"{big}\"}}\n```"
    ok, reason = parse_and_validate(txt, PLANNER_V1)
    # Depending on where the limit trips, sanitizer may report raw or inner size
    assert not ok and ("json block too large" in reason or "raw output too large" in reason)

def test_rejects_whitespace_only_plan_item():
    ok, reason = parse_and_validate('{"plan":["   "],"rationale":"ok"}', PLANNER_V1)
    assert not ok and "plan item length/content invalid" in reason

def test_rejects_unsupported_fence_language():
    txt = "```python\n{'plan': ['a'], 'rationale': 'ok'}\n```"
    ok, reason = parse_and_validate(txt, PLANNER_V1)
    assert not ok and "unsupported code fence language" in reason
