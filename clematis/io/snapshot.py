from __future__ import annotations

"""
Thin IO shim for PR34.

We re-export read_snapshot from the central module to avoid importing optional
dependencies at import time. This keeps local devs without `zstandard`
from seeing ImportError when simply importing clematis.engine.io.snapshot.
"""
from clematis.engine.snapshot import read_snapshot  # re-export

__all__ = ["read_snapshot"]
