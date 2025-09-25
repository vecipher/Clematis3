from configs.validate import validate_config_api as validate


def test_provider_validation_rejects_unknown():
    ok, errs, _ = validate({"t3": {"backend": "llm", "llm": {"provider": "nope"}}})
    assert not ok
    joined = " | ".join(str(e) for e in errs)
    assert "t3.llm.provider" in joined or "provider" in joined

def test_backend_default_rulebased_and_llm_defaults_present():
    ok, errs, cfg = validate({})
    assert ok, f"errors: {errs}"
    assert cfg["t3"]["backend"] == "rulebased"
    # llm defaults should be materialized even when backend is rulebased
    llm = cfg["t3"]["llm"]
    assert llm["provider"] == "fixture"
    assert isinstance(llm["model"], str) and llm["model"]
    assert isinstance(llm["endpoint"], str) and llm["endpoint"]
    assert isinstance(llm["max_tokens"], int) and llm["max_tokens"] >= 1
    assert 0.0 <= float(llm["temp"]) <= 1.0
    assert isinstance(llm["timeout_ms"], int) and llm["timeout_ms"] >= 1
    assert isinstance(llm["fixtures"]["enabled"], bool)
    assert isinstance(llm["fixtures"]["path"], str) and llm["fixtures"]["path"]

def test_temp_bounds_enforced():
    ok, errs, _ = validate({"t3": {"backend":"llm", "llm": {"temp": 1.5}}})
    assert not ok
    joined = " | ".join(str(e) for e in errs)
    assert "t3.llm.temp" in joined

def test_max_tokens_must_be_positive():
    ok, errs, _ = validate({"t3": {"backend":"llm", "llm": {"max_tokens": 0}}})
    assert not ok
    joined = " | ".join(str(e) for e in errs)
    assert "t3.llm.max_tokens" in joined

def test_timeout_ms_must_be_positive():
    ok, errs, _ = validate({"t3": {"backend":"llm", "llm": {"timeout_ms": 0}}})
    assert not ok
    joined = " | ".join(str(e) for e in errs)
    assert "t3.llm.timeout_ms" in joined

def test_fixtures_path_nonempty():
    ok, errs, _ = validate({"t3": {"backend":"llm", "llm": {"fixtures": {"path": ""}}}})
    assert not ok
    joined = " | ".join(str(e) for e in errs)
    assert "t3.llm.fixtures.path" in joined

def test_unknown_llm_key_flagged():
    ok, errs, _ = validate({"t3": {"backend":"llm", "llm": {"bogus": 123}}})
    assert not ok
    joined = " | ".join(str(e) for e in errs)
    assert "t3.llm.bogus" in joined or "unknown key" in joined

def test_unknown_fixtures_key_flagged():
    ok, errs, _ = validate({"t3": {"backend":"llm", "llm": {"fixtures": {"bogus": True}}}})
    assert not ok
    joined = " | ".join(str(e) for e in errs)
    assert "t3.llm.fixtures.bogus" in joined or "unknown key" in joined
