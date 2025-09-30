from __future__ import annotations

import importlib.util

from clematis.optional._require import require


def has_zstd() -> bool:
    """Return True if the 'zstandard' module is importable."""
    return importlib.util.find_spec("zstandard") is not None


def _zstd():
    # Lazy import with a clear extras hint when missing
    return require("zstandard", "zstd")


def compress_bytes(data: bytes, level: int = 3) -> bytes:
    """Compress bytes with zstandard using the given compression level.

    Parameters
    ----------
    data: bytes
        Raw input payload.
    level: int
        Compression level; 1..22 typical, defaults to 3 for a reasonable trade‑off.
    """
    zstd = _zstd()
    c = zstd.ZstdCompressor(level=level)
    return c.compress(data)


def decompress_bytes(data: bytes, max_output_size: int | None = None) -> bytes:
    zstd = _zstd()
    d = zstd.ZstdDecompressor()
    if max_output_size is None:
        # zstandard API expects an int; omit the arg when unlimited
        return d.decompress(data)
    return d.decompress(data, max_output_size=max_output_size)


__all__ = [
    "has_zstd",
    "compress_bytes",
    "decompress_bytes",
]
