from __future__ import annotations

"""Clematis package init.

Resolves __version__ from installed metadata so wheels/sdists/containers stay in sync.
Falls back to packaged VERSION file, then to an optional _version module, else '0+unknown'.
"""

# Prefer stdlib importlib.metadata; fall back to importlib_metadata on older Pythons.
try:
    from importlib.metadata import version as _pkg_version, PackageNotFoundError  # type: ignore
except Exception:  # pragma: no cover
    from importlib_metadata import version as _pkg_version, PackageNotFoundError  # type: ignore

__all__ = ["__version__"]

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
    _version_from_metadata()
    or _version_from_resource()
    or _version_from_fallback_module()
    or "0+unknown"
)