# tests/io/test_snapshots_env_override.py
import os, sys, json
from pathlib import Path
import pytest

def _write_one_snapshot(tmp: Path):
    # Call your public snapshot writer (whatever you use in tests elsewhere)
    from clematis.engine.snapshot import write_snapshot
    class Ctx: pass
    ctx = Ctx()
    state = {}
    # Ensure a deterministic etag, or just accept what your helper produces.
    return write_snapshot(ctx, state, version_etag="zzzz", applied=0, deltas=None)

@pytest.mark.parametrize("unicode_dir", ["テスト✓Ω", "δοκιμή", "тест"])
def test_snapshot_respects_env(tmp_path, unicode_dir, monkeypatch):
    snapdir = tmp_path / unicode_dir
    snapdir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("CLEMATIS_SNAPSHOT_DIR", str(snapdir))

    path = _write_one_snapshot(tmp_path)
    # should land inside snapdir
    assert str(path).startswith(str(snapdir))
    # readable as utf-8
    p = Path(path)
    assert p.exists()
    _ = p.read_text(encoding="utf-8")
