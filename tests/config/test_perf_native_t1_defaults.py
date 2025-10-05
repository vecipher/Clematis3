

import pytest

try:
    from configs.validate import validate_config, validate_config_verbose  # type: ignore
    HAVE_VERBOSE = True
except Exception:  # pragma: no cover
    from configs.validate import validate_config  # type: ignore
    HAVE_VERBOSE = False

# Helper to strip perf/quality/graph for identity-style comparisons
try:
    from tests.helpers.identity import _strip_perf_and_quality_and_graph as _strip
except Exception:  # pragma: no cover
    def _strip(x):
        return x  # fallback; test will still check presence of keys


def test_accepts_native_t1_block_minimal():
    cfg = {"perf": {"native": {"t1": {"enabled": False}}}}
    out = validate_config(cfg)

    assert "perf" in out
    assert "native" in out["perf"], out["perf"].keys()
    assert "t1" in out["perf"]["native"], out["perf"]["native"].keys()

    nt1 = out["perf"]["native"]["t1"]
    assert nt1.get("enabled") is False
    # backend default materializes when the block is present
    assert nt1.get("backend") == "rust"
    # strict_parity may be absent unless provided; default behavior is effectively False
    assert nt1.get("strict_parity") in (False, None)


def test_backend_normalization_valid_values():
    # Upper/lower-case normalization and accepted values
    cases = [
        ({"backend": "RUST"}, "rust"),
        ({"backend": "python"}, "python"),
    ]
    for raw, expected in cases:
        cfg = {"perf": {"native": {"t1": dict(enabled=False, **raw)}}}
        out = validate_config(cfg)
        nt1 = out["perf"]["native"]["t1"]
        assert nt1["backend"] == expected


@pytest.mark.skipif(not HAVE_VERBOSE, reason="validate_config_verbose not available")
def test_backend_invalid_reports_error_and_defaults_to_rust():
    cfg = {"perf": {"native": {"t1": {"backend": "cuda"}}}}
    try:
        res = validate_config_verbose(cfg)
    except Exception as e:  # validator may raise on error-only mode
        msg = str(e)
        assert "perf.native.t1.backend" in msg
        assert "must be one of" in msg
    else:
        # tolerant path if verbose returns (errs, out, ...)
        assert isinstance(res, tuple) and len(res) >= 2, "unexpected validate_config_verbose return"
        errs, out = res[0], res[1]
        assert any("perf.native.t1.backend" in e for e in errs)
        nt1 = out.get("perf", {}).get("native", {}).get("t1", {})
        assert nt1.get("backend") == "rust"


def test_native_block_is_inert_when_stripped():
    base = {}
    with_native = {"perf": {"native": {"t1": {"enabled": False, "backend": "rust"}}}}
    a = _strip(validate_config(base))
    b = _strip(validate_config(with_native))
    assert a == b
