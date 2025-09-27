from __future__ import annotations
from contextlib import contextmanager
from importlib import resources
from importlib.resources.abc import Traversable
from pathlib import Path
from typing import Iterator, Optional


PKG = "clematis"


def _files() -> Traversable:
    """Return the package root as an importlib.resources Traversable."""
    return resources.files(PKG)


def pkg_traversable(*parts: str) -> Optional[Traversable]:
    """
    Return a Traversable handle to a packaged resource (file or directory),
    or None if it does not exist.

    Notes:
    - Traversable is not guaranteed to be a real filesystem path.
    - Use the context managers below to obtain a concrete Path when needed.
    """
    try:
        ref = _files().joinpath(*parts)
        return ref if ref.exists() else None
    except Exception:
        return None


@contextmanager
def packaged_path(*parts: str) -> Iterator[Optional[Path]]:
    """
    Yield a concrete filesystem Path to a packaged resource if it exists,
    else yield None. Works for both regular site-packages installs and
    zip-based resources by using resources.as_file(...).

    Example:
        with packaged_path("examples", "snapshots", "state.parquet") as p:
            if p is not None:
                do_something_with(p)
    """
    ref = pkg_traversable(*parts)
    if ref is None:
        yield None
        return
    try:
        with resources.as_file(ref) as path:
            yield path
    except Exception:
        yield None


@contextmanager
def example_path(name: str) -> Iterator[Optional[Path]]:
    """Context manager: yields Path to a packaged example by file/dir name."""
    with packaged_path("examples", name) as p:
        yield p


@contextmanager
def fixture_path(*parts: str) -> Iterator[Optional[Path]]:
    """Context manager: yields Path to a packaged fixture under fixtures/."""
    with packaged_path("fixtures", *parts) as p:
        yield p


# Back-compat convenience shims (non-context versions).
# These return Traversable (NOT guaranteed to be a real Path).
def pkg_path(*parts: str) -> Optional[Traversable]:
    """Deprecated: prefer pkg_traversable(...)."""
    return pkg_traversable(*parts)


def packaged_example(name: str) -> Optional[Traversable]:
    """Deprecated: prefer example_path(...) context or pkg_traversable('examples', name)."""
    return pkg_traversable("examples", name)


def packaged_fixture(name: str) -> Optional[Traversable]:
    """Deprecated: prefer fixture_path(...) context or pkg_traversable('fixtures', 'llm', name)."""
    return pkg_traversable("fixtures", "llm", name)