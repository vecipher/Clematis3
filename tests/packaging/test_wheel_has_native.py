

import os
import importlib
import pytest


def _try_import_native():
    """Return the native module if present, else None.
    This test is intended to run in the wheel CI job after installing the wheel.
    In source or sdist-only environments where the extension isn't built, we skip.
    """
    try:
        return importlib.import_module("clematis.native._t1_rs")
    except Exception:
        return None


def test_wheel_has_native_and_available():
    mod = _try_import_native()
    if mod is None:
        pytest.skip(
            "native extension not present (likely source tree or sdist-only install); "
            "wheel CI will exercise this test after installing the wheel."
        )

    from clematis.native import t1

    # The wheel must ship the native extension and the wrapper should detect it.
    assert t1.available() is True, "expected native _t1_rs to be available from the installed wheel"

    # Sanity: compiled module has a real file and a compiled suffix
    assert hasattr(mod, "__file__")
    ext = os.path.splitext(mod.__file__)[1].lower()
    assert ext in (".so", ".pyd", ".dll", ".dylib"), mod.__file__

    # And the Python bridge exposes the FFI entrypoint
    assert hasattr(t1, "propagate_one_graph_rs")
