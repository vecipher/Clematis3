


from __future__ import annotations

"""Typed error taxonomy (public, frozen for v3).

Only `clematis` and `clematis.errors` are public import roots. Everything else is internal.
This module exposes the operator-facing error classes and a small helper `format_error`.
"""

__all__ = [
    "ClematisError",
    "ConfigError",
    "SnapshotError",
    "IdentityError",
    "QualityShadowError",
    "ReflectionError",
    "CLIError",
    "ParallelError",
    "format_error",
]


class ClematisError(Exception):
    """Base class for all typed, operator-facing errors in Clematis."""
    pass



# Typed errors no longer inherit from ValueError; callers should catch the specific subclasses.
class ConfigError(ClematisError):
    """Configuration invalid, unknown keys, wrong version, etc."""
    pass


class SnapshotError(ClematisError):
    """Snapshot missing schema, mismatched version, corrupted header, etc."""
    pass


class IdentityError(ClematisError):
    """Identity path violations or mismatched golden artifacts."""
    pass


class QualityShadowError(ClematisError):
    """Quality-shadow tracing/adapter failures (feature off by default)."""
    pass


class ReflectionError(ClematisError):
    """Reflection-session errors (off by default for v3)."""
    pass


class CLIError(ClematisError):
    """Generic CLI failure wrapper for unexpected errors in CLI code paths."""
    pass


# Re-export ParallelError from its canonical implementation, but ensure the taxonomy
# remains importable even if that module isn't available during certain tooling runs.
try:
    # Local import to avoid import cycles in static analyzers or partial imports.
    from clematis.engine.util.parallel import ParallelError as _ParallelError  # type: ignore
except Exception:
    class _ParallelError(ClematisError):  # type: ignore[no-redef]
        """Fallback ParallelError if engine.util.parallel is unavailable at import time."""
        pass

ParallelError = _ParallelError


def format_error(e: BaseException) -> str:
    """Return a short, uniform operator-facing message like 'ConfigError: detail'."""
    name = e.__class__.__name__
    msg = str(e).strip()
    return f"{name}: {msg}" if msg else name


# Keep star-export order deterministic for tests and tooling
__all__ = sorted(__all__)
