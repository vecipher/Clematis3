from __future__ import annotations

import io
import os
from contextlib import redirect_stderr

from clematis.engine.snapshot import _write_lines


class DummyCtx:  # minimal stub for payload
    pass


def test_zstd_missing_prints_warning(tmp_path, monkeypatch):
    # Ensure the optional zstandard module is treated as missing
    monkeypatch.setattr("clematis.engine.snapshot._zstd", None, raising=False)

    payload_path = tmp_path / "snapshot.json"
    buf = io.StringIO()
    with redirect_stderr(buf):
        _write_lines(str(payload_path), {"schema": "snapshot:v1"}, "{}", codec="zstd", level=3)

    warning = buf.getvalue().strip()
    assert warning == "W[SNAPSHOT]: zstandard not installed; writing uncompressed"
    assert payload_path.exists()
    assert os.path.getsize(payload_path) > 0
