

from __future__ import annotations

import importlib.util
import pytest


def _has(mod: str) -> bool:
    return importlib.util.find_spec(mod) is not None


@pytest.mark.skipif(not _has("zstandard"), reason="zstd extra not installed")
def test_zstd_roundtrip():
    # Import from our optional shim
    from clematis.optional import zstd_support as zs  # type: ignore

    payload = b"clematis:zstd:roundtrip"
    packed = zs.compress_bytes(payload, level=5)
    assert isinstance(packed, (bytes, bytearray))
    out = zs.decompress_bytes(packed)
    assert out == payload


def test_lancedb_guard_or_ok():
    from clematis.optional import lancedb_support as ls  # type: ignore

    if not _has("lancedb"):
        with pytest.raises(ImportError) as ei:
            ls.touch()
        # Clear guidance to install extras
        assert "pip install 'clematis[lancedb]'" in str(ei.value)
    else:
        # Should be importable and touch() should not raise
        assert ls.has_lancedb() is True
        ls.touch()