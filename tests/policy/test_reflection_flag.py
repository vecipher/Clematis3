# tests/policy/test_reflection_flag.py
import importlib
from types import SimpleNamespace as SNS

sanitize = importlib.import_module("clematis.engine.policy.sanitize")
schemas = importlib.import_module("clematis.engine.policy.json_schemas")
t3pol   = importlib.import_module("clematis.engine.stages.t3.policy")

def _build(plan_dict):
    errs = []
    sp = sanitize.sanitize_plan(plan_dict, errs)
    assert not errs, f"sanitize errors: {errs}"
    # if you have a validate step here, call it; otherwise assume sanitize precedes schema elsewhere
    p = t3pol.Plan(ops=sp.get("ops", []), version=sp.get("version", "t3-plan-v1"), **sp)
    return p

def test_reflection_defaults_false():
    p = _build({"ops":[]})
    assert p.reflection is False

def test_reflection_coercion_from_string_and_int():
    p1 = _build({"ops":[], "reflection": "true"})
    p2 = _build({"ops":[], "reflection": "0"})
    p3 = _build({"ops":[], "reflection": 1})
    assert p1.reflection is True
    assert p2.reflection is False
    assert p3.reflection is True

def test_reflection_rejects_garbage():
    errs = []
    _ = sanitize.sanitize_plan({"ops":[], "reflection": "maybe?"}, errs)
    assert any("plan.reflection" in e for e in errs)
