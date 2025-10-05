

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from clematis.engine.snapshot import SCHEMA_VERSION, validate_snapshot_schema


def test_validate_snapshot_schema_rejects_missing():
    with pytest.raises(Exception):
        validate_snapshot_schema({})  # no schema_version


def test_validate_snapshot_schema_rejects_wrong():
    with pytest.raises(Exception):
        validate_snapshot_schema({"schema_version": "v999"})


def test_inspect_snapshot_strict_rejects(tmp_path: Path):
    # Create a minimal snapshot JSON that lacks top-level schema_version
    snaps = tmp_path / "snaps"
    snaps.mkdir(parents=True, exist_ok=True)
    snap = snaps / "0000001.json"
    snap.write_text(json.dumps({"header": {"created_at": "2025-01-01T00:00:00Z"}}), encoding="utf-8")

    # Run inspector in strict mode (default). Expect exit code 2 and an error message.
    cmd = [sys.executable, "-m", "clematis.scripts.inspect_snapshot", "--dir", str(snaps), "--format", "json"]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    assert proc.returncode == 2, f"stdout={proc.stdout}\nstderr={proc.stderr}"
    out = (proc.stdout or "") + (proc.stderr or "")
    assert "missing 'schema_version'" in out


def test_inspect_snapshot_no_strict_warns(tmp_path: Path):
    snaps = tmp_path / "snaps"
    snaps.mkdir(parents=True, exist_ok=True)
    snap = snaps / "0000002.json"
    snap.write_text(json.dumps({"header": {"created_at": "2025-01-01T00:00:00Z"}}), encoding="utf-8")

    cmd = [sys.executable, "-m", "clematis.scripts.inspect_snapshot", "--dir", str(snaps), "--no-strict", "--format", "json"]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    # --no-strict should allow printing and return 0 with a warning
    assert proc.returncode == 0, f"stdout={proc.stdout}\nstderr={proc.stderr}"
    out = (proc.stdout or "") + (proc.stderr or "")
    assert "[warn]" in out
