import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

SCRIPT = Path("scripts/validate_config.py")


@pytest.mark.skipif(not SCRIPT.exists(), reason="validator CLI not found")
def test_cli_validator_accepts_perf_t2_cache_strict(tmp_path: Path):
    cfg = tmp_path / "good.yaml"
    cfg.write_text(
        textwrap.dedent(
            """
        perf:
          enabled: true
          metrics:
            report_memory: true
          t2:
            cache:
              max_entries: 64
              max_bytes: 4096
        # Disable legacy/stage caches to avoid strict warnings unrelated to PR32
        t2:
          cache:
            enabled: false
          quality:
            enabled: false
        t4:
          cache:
            enabled: false
            namespaces: []
        """
        ).strip()
        + "\n",
        encoding="utf-8",
    )

    res = subprocess.run(
        [sys.executable, str(SCRIPT), str(cfg), "--strict"], capture_output=True, text=True
    )
    assert res.returncode == 0, f"STDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}"


@pytest.mark.skipif(not SCRIPT.exists(), reason="validator CLI not found")
def test_cli_validator_warns_when_perf_disabled(tmp_path: Path):
    cfg = tmp_path / "warn.yaml"
    cfg.write_text(
        textwrap.dedent(
            """
        perf:
          enabled: false
          t2:
            cache:
              max_entries: 10
              max_bytes: 2048
        # Disable legacy/stage caches to keep output focused on perf warning
        t2:
          cache:
            enabled: false
          quality:
            enabled: false
        t4:
          cache:
            enabled: false
            namespaces: []
        """
        ).strip()
        + "\n",
        encoding="utf-8",
    )

    res = subprocess.run([sys.executable, str(SCRIPT), str(cfg)], capture_output=True, text=True)
    # Non-strict run should succeed, and we expect a perf.t2.cache warning in stdout
    assert res.returncode == 0
    assert "W[perf.t2.cache]" in res.stdout


@pytest.mark.skipif(not SCRIPT.exists(), reason="validator CLI not found")
def test_cli_validator_accepts_zero_caps_as_disabled(tmp_path: Path):
    cfg = tmp_path / "zero.yaml"
    cfg.write_text(
        textwrap.dedent(
            """
        perf:
          enabled: true
          t2:
            cache:
              max_entries: 0
              max_bytes: 0
        # Disable legacy/stage caches to avoid extraneous warnings
        t2:
          cache:
            enabled: false
          quality:
            enabled: false
        t4:
          cache:
            enabled: false
            namespaces: []
        """
        ).strip()
        + "\n",
        encoding="utf-8",
    )

    res = subprocess.run(
        [sys.executable, str(SCRIPT), str(cfg), "--strict"], capture_output=True, text=True
    )
    # Zero caps mean "disabled"; strict should pass and there should be no perf cache warnings
    assert res.returncode == 0, f"STDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}"
    assert "W[perf.t2.cache]" not in res.stdout
