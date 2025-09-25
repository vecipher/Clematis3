"""Clematis package init — exposes __version__ from clematis/VERSION."""
from importlib.resources import files

__all__ = ["__version__"]

try:
    __version__ = files(__package__).joinpath("VERSION").read_text(encoding="utf-8").strip()
except Exception:
    # Conservative fallback if package data isn't available (shouldn't happen in CI).
    __version__ = "0.8.0a0"
