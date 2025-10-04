

from configs.validate import validate_config

try:
    from tests.helpers.identity import _strip_perf_and_quality_and_graph as _strip
except Exception:  # pragma: no cover
    # Fallback: identity tests will still compare full dicts if helper is unavailable
    _strip = lambda x: x


def test_native_block_is_inert_when_stripped():
    base = {}
    with_native = {"perf": {"native": {"t1": {"enabled": False, "backend": "rust"}}}}

    a = _strip(validate_config(base))
    b = _strip(validate_config(with_native))

    assert a == b


def test_strict_parity_flag_also_inert_when_disabled():
    base = {}
    with_native = {
        "perf": {
            "native": {
                "t1": {
                    "enabled": False,
                    "backend": "rust",
                    "strict_parity": True,
                }
            }
        }
    }

    a = _strip(validate_config(base))
    b = _strip(validate_config(with_native))

    assert a == b
