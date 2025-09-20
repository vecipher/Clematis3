

# Clematis3 — M5 Scheduler Config validation tests (PR25)

import pytest

from configs.validate import (
    validate_config,
    validate_config_verbose,
)


def _expect_valid(user_cfg):
    """Validate and return the merged config; must not raise."""
    merged = validate_config(user_cfg)
    assert isinstance(merged, dict)
    return merged


def _expect_error(user_cfg, substr: str):
    """Validate and expect a ValueError mentioning substr."""
    with pytest.raises(ValueError) as ei:
        validate_config(user_cfg)
    msg = str(ei.value)
    assert substr in msg, f"Expected error message to contain '{substr}', got: {msg}"


def test_defaults_include_scheduler_block_and_values():
    merged = _expect_valid({})
    s = merged.get("scheduler")
    assert isinstance(s, dict)
    assert s.get("enabled") is False
    assert s.get("policy") in ("round_robin", "fair_queue")
    # Budgets defaults present
    b = s.get("budgets", {})
    assert isinstance(b, dict)
    for key in ("t1_pops", "t1_iters", "t2_k", "t3_ops", "wall_ms"):
        assert key in b
    # Fairness defaults present
    f = s.get("fairness", {})
    assert isinstance(f, dict)
    assert "max_consecutive_turns" in f and "aging_ms" in f


def test_policy_must_be_allowed_value():
    _expect_error({"scheduler": {"policy": "random"}}, "scheduler.policy")


def test_quantum_and_wall_bounds():
    # quantum_ms must be >= 1
    _expect_error({"scheduler": {"quantum_ms": 0}}, "scheduler.quantum_ms")
    # wall_ms must be >= quantum_ms
    _expect_error({"scheduler": {"quantum_ms": 20, "budgets": {"wall_ms": 10}}}, "scheduler.budgets.wall_ms")


@pytest.mark.parametrize("budget_key", ["t1_pops", "t1_iters", "t2_k", "t3_ops"])
def test_budgets_must_be_int_ge_0_or_none(budget_key):
    _expect_error({"scheduler": {"budgets": {budget_key: -1}}}, f"scheduler.budgets.{budget_key}")


def test_wall_ms_must_be_int_ge_1_if_present():
    _expect_error({"scheduler": {"budgets": {"wall_ms": 0}}}, "scheduler.budgets.wall_ms")


def test_fairness_bounds():
    _expect_error({"scheduler": {"fairness": {"max_consecutive_turns": 0}}}, "scheduler.fairness.max_consecutive_turns")
    _expect_error({"scheduler": {"fairness": {"aging_ms": -1}}}, "scheduler.fairness.aging_ms")


def test_valid_custom_scheduler_passes():
    cfg = {
        "scheduler": {
            "enabled": True,
            "policy": "fair_queue",
            "quantum_ms": 25,
            "budgets": {"t1_pops": None, "t1_iters": 40, "t2_k": 32, "t3_ops": 2, "wall_ms": 250},
            "fairness": {"max_consecutive_turns": 1, "aging_ms": 100},
        }
    }
    merged = _expect_valid(cfg)
    s = merged["scheduler"]
    assert s["policy"] == "fair_queue"
    assert s["budgets"]["t2_k"] == 32
    assert s["fairness"]["aging_ms"] == 100


def test_verbose_warnings_for_zero_t1_caps_and_too_small_aging():
    # both t1_iters and t1_pops set to 0 -> warning
    cfg = {
        "scheduler": {
            "quantum_ms": 20,
            "budgets": {"t1_iters": 0, "t1_pops": 0, "wall_ms": 20},
            "fairness": {"aging_ms": 5},  # < quantum_ms -> warning
        }
    }
    merged, warnings = validate_config_verbose(cfg)
    assert isinstance(warnings, list)
    joined = "\n".join(warnings)
    assert "t1_iters and t1_pops" in joined
    assert "aging_ms" in joined and "quantum_ms" in joined