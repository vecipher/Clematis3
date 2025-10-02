from __future__ import annotations

from pathlib import Path

from clematis.io import paths


def test_logs_dir_prefers_single_env(monkeypatch, tmp_path):
    primary = tmp_path / "primary"
    secondary = tmp_path / "secondary"

    # When both env vars are present, CLEMATIS_LOG_DIR wins.
    monkeypatch.setenv("CLEMATIS_LOG_DIR", str(primary))
    monkeypatch.setenv("CLEMATIS_LOGS_DIR", str(secondary))

    resolved = Path(paths.logs_dir())

    assert resolved == primary.resolve()
    assert primary.exists()
    assert not secondary.exists()


def test_logs_dir_uses_legacy_env(monkeypatch, tmp_path):
    legacy = tmp_path / "legacy"

    monkeypatch.delenv("CLEMATIS_LOG_DIR", raising=False)
    monkeypatch.setenv("CLEMATIS_LOGS_DIR", str(legacy))

    resolved = Path(paths.logs_dir())

    assert resolved == legacy.resolve()
    assert legacy.exists()
