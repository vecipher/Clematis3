

from __future__ import annotations

import importlib
from types import ModuleType


def require(module_name: str, extra_hint: str) -> ModuleType:
    """Import a module or raise a clear ImportError with an extras hint.

    Parameters
    ----------
    module_name: str
        The importable module name, e.g. "zstandard" or "lancedb".
    extra_hint: str
        The extras group users should install, e.g. "zstd" or "lancedb".

    Returns
    -------
    ModuleType
        The imported module object.

    Raises
    ------
    ImportError
        If the module cannot be imported, with guidance to install the
        appropriate optional extras.
    """
    try:
        return importlib.import_module(module_name)
    except Exception as e:  # noqa: BLE001
        raise ImportError(
            "Optional dependency '{mod}' is not installed. "
            "Install with: pip install 'clematis[{extra}]'".format(
                mod=module_name, extra=extra_hint
            )
        ) from e