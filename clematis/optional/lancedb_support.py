from __future__ import annotations

import importlib.util
from types import ModuleType

from clematis.optional._require import require


def has_lancedb() -> bool:
    """Return True if the 'lancedb' module is importable."""
    return importlib.util.find_spec("lancedb") is not None


def import_lancedb() -> ModuleType:
    """Import and return the lancedb module, or raise a clear ImportError.

    Raises an ImportError instructing users to install the lancedb extra:
    `pip install 'clematis[lancedb]'`.
    """
    return require("lancedb", "lancedb")


def touch() -> None:
    """No-op that validates lancedb availability (raises if missing)."""
    import_lancedb()


__all__ = [
    "has_lancedb",
    "import_lancedb",
    "touch",
]
