

import types
import pytest


def _run_validate(raw_cfg):
    """Call validate_config_verbose and return its warnings list.
    This helper accepts multiple possible return shapes to be robust across
    minor signature differences (tuple vs direct warnings).
    """
    from configs import validate as V

    fn = getattr(V, "validate_config_verbose", None)
    assert fn is not None, "validate_config_verbose not found in configs.validate"

    try:
        res = fn(raw_cfg)
    except TypeError:
        # Some implementations may require a pre-normalized dict; pass through
        res = fn(raw_cfg, dict(raw_cfg))

    warnings = None
    if isinstance(res, tuple):
        # common patterns: (normalized, warnings) or (warnings,)
        if len(res) == 2 and isinstance(res[1], (list, tuple)):
            warnings = list(res[1])
        elif len(res) == 1 and isinstance(res[0], (list, tuple)):
            warnings = list(res[0])
    elif isinstance(res, (list, tuple)):
        warnings = list(res)

    assert warnings is not None, "validate_config_verbose did not return warnings in a recognized shape"
    return warnings


def _has_warning(warnings, needle: str) -> bool:
    return any(needle in str(w) for w in warnings)


def test_agents_flag_warns_when_perf_disabled():
    raw = {
        "perf": {
            "enabled": False,
            "parallel": {
                "enabled": False,
                "t2": False,
                "max_workers": 4,
                "agents": True,
            },
        }
    }
    warnings = _run_validate(raw)
    assert _has_warning(warnings, "perf.parallel.agents"), (
        "Expected a warning when agents=true but perf.enabled=false"
    )


def test_agents_flag_warns_when_max_workers_le_one_and_parallel_enabled():
    raw = {
        "perf": {
            "enabled": True,
            "parallel": {
                "enabled": True,
                "t2": True,
                "max_workers": 1,
                "agents": True,
            },
        }
    }
    warnings = _run_validate(raw)
    assert _has_warning(warnings, "max_workers<=1"), (
        "Expected a warning when agents=true and parallel enabled but max_workers<=1"
    )


def test_agents_flag_no_warning_when_properly_enabled():
    raw = {
        "perf": {
            "enabled": True,
            "parallel": {
                "enabled": True,
                "t2": True,
                "max_workers": 4,
                "agents": True,
            },
        }
    }
    warnings = _run_validate(raw)
    # Our PR70-specific warnings should not appear in the healthy case
    assert not _has_warning(warnings, "perf.parallel.agents"), (
        "Unexpected agents/perf.enabled warning in healthy enabled config"
    )
    assert not _has_warning(warnings, "max_workers<=1"), (
        "Unexpected max_workers<=1 warning in healthy enabled config"
    )
