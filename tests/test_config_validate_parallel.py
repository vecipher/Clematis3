from __future__ import annotations

import pytest


def _validate_api(cfg: dict):
    """
    Preferred, non-raising validation entrypoint:
    returns (ok: bool, errs: list[str], merged_cfg: dict).
    """
    import configs.validate as v  # type: ignore

    fn = getattr(v, "validate_config_api", None)
    if not callable(fn):
        pytest.skip("validate_config_api not available")

    ok, errs, merged = fn(cfg)  # type: ignore[misc]
    # Normalize errs to a list[str]
    if errs is None:
        errs_list = []
    elif isinstance(errs, (list, tuple)):
        errs_list = [str(e) for e in errs]
    else:
        errs_list = [str(errs)]
    return ok, errs_list, merged


def test_parallel_normalizes_negative_max_workers():
    ok, errors, merged = _validate_api(
        {
            "perf": {
                "enabled": False,
                "parallel": {
                    "enabled": True,
                    "max_workers": -7,  # should normalize to 0 (sequential)
                },
            }
        }
    )
    assert ok, f"unexpected errors: {errors}"
    pp = merged.get("perf", {}).get("parallel", {})
    assert pp.get("max_workers") == 0, f"expected normalization to 0, got {pp.get('max_workers')}"


def test_parallel_unknown_key_is_rejected():
    ok, errors, _ = _validate_api(
        {
            "perf": {
                "parallel": {
                    "foo": 123  # not allowed
                }
            }
        }
    )
    # We expect a clean failure with a message mentioning perf.parallel.foo and 'unknown key'
    assert not ok
    msg = "\n".join(errors)
    assert "perf.parallel.foo" in msg and "unknown key" in msg.lower()
