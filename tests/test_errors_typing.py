

from __future__ import annotations

import pytest

from clematis.errors import (
    ClematisError,
    ConfigError,
    SnapshotError,
    IdentityError,
    QualityShadowError,
    ReflectionError,
    CLIError,
    ParallelError,
    format_error,
)

def test_parallel_error_available():
    # Importable, subclass, and printable
    class _FakeRec:
        key = "task-1"
        exc_type = "ValueError"
        message = "boom"
    e = ParallelError([_FakeRec()])
    assert isinstance(e, ClematisError)
    assert "boom" in str(e)

def test_error_hierarchy():
    # Every typed error must be a ClematisError
    assert issubclass(ConfigError, ClematisError)
    assert issubclass(SnapshotError, ClematisError)
    assert issubclass(IdentityError, ClematisError)
    assert issubclass(QualityShadowError, ClematisError)
    assert issubclass(ReflectionError, ClematisError)
    assert issubclass(CLIError, ClematisError)
    assert issubclass(ParallelError, ClematisError)


def test_format_error_prefix_and_message():
    e = ConfigError("bad key 'foo'")
    s = format_error(e)
    assert s.startswith("ConfigError:")
    assert "bad key" in s
    # Empty message uses class name only
    e2 = SnapshotError("")
    s2 = format_error(e2)
    assert s2 == "SnapshotError"


def test_validate_config_raises_typed_error_on_wrong_version():
    from configs.validate import validate_config

    # Wrong version must raise ConfigError
    with pytest.raises(ConfigError):
        validate_config({"version": "v999", "t1": {}, "t2": {}, "t3": {}, "t4": {}})

    # Unknown top-level keys must raise ConfigError
    with pytest.raises(ConfigError):
        validate_config({"unknown": {}, "t1": {}, "t2": {}, "t3": {}, "t4": {}})


def test_snapshot_schema_validation_raises_typed_error():
    # Support either helper name depending on the repo state
    try:
        from clematis.engine.snapshot import validate_snapshot_schema as _validate, SCHEMA_VERSION
    except Exception:  # pragma: no cover - fallback name
        from clematis.engine.snapshot import validate_snapshot_header as _validate, SCHEMA_VERSION

    # Helper that tolerates different signatures (header, expected=...)
    def call_validate(header: dict):
        try:
            return _validate(header)
        except TypeError:
            # Some variants require expected=SCHEMA_VERSION
            return _validate(header, expected=SCHEMA_VERSION)

    # Missing schema_version
    with pytest.raises(SnapshotError):
        call_validate({})

    # Mismatched schema_version
    with pytest.raises(SnapshotError):
        call_validate({"schema_version": "v999"})
