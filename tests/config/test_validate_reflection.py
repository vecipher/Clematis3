

import copy
import pytest

import importlib
from clematis.errors import ConfigError
_val_mod = importlib.import_module("configs.validate")
for _name in ("validate", "validate_and_normalize", "validate_config", "validate_cfg"):
    if hasattr(_val_mod, _name):
        _validate = getattr(_val_mod, _name)
        break
else:  # pragma: no cover
    raise ImportError("configs.validate must expose one of: validate, validate_and_normalize, validate_config, validate_cfg")


def base_cfg():
    """
    Minimal config focused on exercising t3/reflection and scheduler.budgets.
    Keep it sparse to avoid coupling to unrelated schema parts.
    """
    return {
        "t3": {},
        "scheduler": {"budgets": {}},
    }


def test_defaults_injected_when_absent():
    cfg = base_cfg()
    out = _validate(copy.deepcopy(cfg))

    # Gate defaults
    assert out["t3"]["allow_reflection"] is False

    # Reflection defaults
    r = out["t3"]["reflection"]
    assert r["backend"] == "rulebased"
    assert r["summary_tokens"] == 128
    assert r["embed"] is True
    assert r["log"] is True
    assert r["topk_snippets"] == 3

    # Budgets defaults
    b = out["scheduler"]["budgets"]
    assert "ops_reflection" in b
    assert isinstance(b["ops_reflection"], int)
    assert b["ops_reflection"] >= 0


def test_types_enforced_and_nonneg_ints():
    cfg = base_cfg()
    cfg["t3"] = {
        "allow_reflection": True,
        "reflection": {
            "backend": "rulebased",
            "summary_tokens": 0,
            "embed": False,
            "log": True,
            "topk_snippets": 10,
        },
    }
    cfg["scheduler"]["budgets"]["ops_reflection"] = 0

    out = _validate(copy.deepcopy(cfg))

    assert out["t3"]["allow_reflection"] is True
    r = out["t3"]["reflection"]
    assert r["backend"] == "rulebased"
    assert r["summary_tokens"] == 0
    assert r["embed"] is False
    assert r["log"] is True
    assert r["topk_snippets"] == 10
    assert out["scheduler"]["budgets"]["ops_reflection"] == 0



def test_ops_reflection_negative_raises():
    cfg = base_cfg()
    cfg["scheduler"]["budgets"]["ops_reflection"] = -1
    with pytest.raises(ConfigError):
        _validate(copy.deepcopy(cfg))

@pytest.mark.parametrize("coerce_in", ["5", 3.14])
def test_ops_reflection_numericish_values_are_coerced_to_int(coerce_in):
    cfg = base_cfg()
    cfg["scheduler"]["budgets"]["ops_reflection"] = coerce_in
    out = _validate(copy.deepcopy(cfg))
    v = out["scheduler"]["budgets"]["ops_reflection"]
    assert isinstance(v, int) and v >= 0


def test_unknown_key_rejected_in_reflection():
    cfg = base_cfg()
    cfg["t3"] = {
        "allow_reflection": False,
        "reflection": {
            "backend": "rulebased",
            "summary_tokens": 128,
            "embed": True,
            "log": True,
            "topk_snippets": 3,
            "surprise": 42,  # not allowed
        },
    }
    with pytest.raises(ConfigError):
        _validate(copy.deepcopy(cfg))


def test_backend_choice_restricted():
    cfg = base_cfg()
    cfg["t3"] = {"allow_reflection": True, "reflection": {"backend": "nope"}}
    with pytest.raises(ConfigError):
        _validate(copy.deepcopy(cfg))
