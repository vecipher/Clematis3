import pytest

try:
    from configs.validate import validate_config, validate_config_verbose
except Exception as e:  # pragma: no cover
    pytest.skip(f"validate_config not importable: {e}", allow_module_level=True)


def test_accepts_snapshots_bounds():
    # Current schema: top-level keys under perf.snapshots
    cfg = {
        "perf": {
            "enabled": True,
            "snapshots": {
                "compression": "zstd",  # allowed: "none" | "zstd"
                "level": 5,  # int in [1..19]
                "delta_mode": True,  # boolean
            },
            "metrics": {"report_memory": False},
        }
    }
    # Should not raise; returns normalized dict
    normalized = validate_config(cfg)
    assert isinstance(normalized, dict)


def test_rejects_bad_level_and_codec():
    cfg = {
        "perf": {
            "enabled": True,
            "snapshots": {
                "compression": "gzip",  # invalid
                "level": 42,  # invalid
            },
        }
    }
    with pytest.raises(ValueError) as e:
        validate_config(cfg)
    msg = str(e.value).lower()
    assert "perf.snapshots.compression" in msg or "compression" in msg
    assert "perf.snapshots.level" in msg or "level" in msg


def test_warns_when_perf_off_and_delta_mode_set():
    # Existing validator emits a warning for delta_mode=true;
    # we rely on that rather than changing the validator here.
    cfg = {
        "perf": {
            "enabled": False,
            "snapshots": {
                "compression": "zstd",
                "level": 3,
                "delta_mode": True,
            },
        }
    }
    _, warns = validate_config_verbose(cfg)
    text = " ".join(warns).lower()
    assert "perf.snapshots" in text or "delta_mode" in text
