"""Clematis v3 — public API surface.

Only `clematis` and `clematis.errors` are public. Everything else is internal.
This module also resolves `__version__` deterministically across installs.
"""
from __future__ import annotations

from typing import Any as _Any
from . import errors as errors  # re-export for star-import; noqa: F401

# Prefer stdlib importlib.metadata; fall back to importlib_metadata on older Pythons.
try:
    from importlib.metadata import version as _pkg_version, PackageNotFoundError  # type: ignore
except Exception:  # pragma: no cover
    from importlib_metadata import version as _pkg_version, PackageNotFoundError  # type: ignore


def _version_from_metadata() -> str | None:
    try:
        return _pkg_version("clematis")
    except PackageNotFoundError:
        return None
    except Exception:
        return None


def _version_from_resource() -> str | None:
    try:
        from importlib.resources import files

        p = files(__package__).joinpath("VERSION")
        return p.read_text(encoding="utf-8").strip()
    except Exception:
        return None


def _version_from_fallback_module() -> str | None:
    try:
        from ._version import __version__ as v  # type: ignore

        return v
    except Exception:
        return None


__version__ = (
    _version_from_resource()
    or _version_from_fallback_module()
    or _version_from_metadata()
    or "0+unknown"
)

# ---------------------------------------------------------------------------
# Public re-exports (frozen for v3)
# ---------------------------------------------------------------------------
# Config v1 surface – lazy-loaded via __getattr__
# Snapshot schema v1 surface – lazy-loaded via __getattr__

# Public errors submodule (users import classes from clematis.errors)

def __getattr__(name: str) -> _Any:  # PEP 562 lazy exports to avoid import-time cycles
    if name in (
        "validate_config",
        "validate_config_verbose",
        "validate_config_api",
        "CONFIG_VERSION",
    ):
        # `configs` is a top-level package, not `clematis.configs`
        from configs.validate import (
            validate_config as _validate_config,
            validate_config_verbose as _validate_config_verbose,
            validate_config_api as _validate_config_api,
            CONFIG_VERSION as _CONFIG_VERSION,
        )
        _g = globals()
        _g.update(
            {
                "validate_config": _validate_config,
                "validate_config_verbose": _validate_config_verbose,
                "validate_config_api": _validate_config_api,
                "CONFIG_VERSION": _CONFIG_VERSION,
            }
        )
        return _g[name]
    if name == "SCHEMA_VERSION":
        from .engine.snapshot import SCHEMA_VERSION as _SCHEMA_VERSION
        globals()["SCHEMA_VERSION"] = _SCHEMA_VERSION
        return _SCHEMA_VERSION
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

# Make dir(clematis) reflect our public surface

def __dir__() -> list[str]:
    return list(__all__)

# Star-export surface (deterministic ordering). Tests require __all__ to be lexicographically sorted.
__all__ = [
    "CONFIG_VERSION",
    "SCHEMA_VERSION",
    "__version__",
    "errors",
    "validate_config",
    "validate_config_api",
    "validate_config_verbose",
]
