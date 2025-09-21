import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

SCRIPT = Path("scripts/validate_config.py")

@pytest.mark.skipif(not SCRIPT.exists(), reason="validator CLI not found")
def test_cli_validator_accepts_perf_t1_caps_strict(tmp_path: Path):
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(textwrap.dedent("""
        perf:
          enabled: true
          t1:
            caps:
              frontier: 5
              visited: 3
            dedupe_window: 4
        t2:
          cache:
            enabled: false
          quality:
            enabled: false
        t4:
          cache:
            enabled: false
            namespaces: []
        """).strip() + "\n", encoding="utf-8")

    # Strict mode should succeed (no errors)
    res = subprocess.run([sys.executable, str(SCRIPT), str(cfg), "--strict"], capture_output=True, text=True)
    assert res.returncode == 0, f"STDERR:\n{res.stderr}\nSTDOUT:\n{res.stdout}"

@pytest.mark.skipif(not SCRIPT.exists(), reason="validator CLI not found")
def test_cli_validator_rejects_invalid_caps(tmp_path: Path):
    cfg = tmp_path / "bad.yaml"
    cfg.write_text(textwrap.dedent("""
        perf:
          enabled: true
          t1:
            caps:
              frontier: 0     # invalid (must be >=1)
              visited: -2     # invalid
            dedupe_window: 0  # invalid
        """).strip() + "\n", encoding="utf-8")

    res = subprocess.run([sys.executable, str(SCRIPT), str(cfg), "--strict"], capture_output=True, text=True)
    assert res.returncode != 0, "Strict validator should fail on invalid caps"
    assert "perf.t1.caps.frontier" in res.stdout or "perf.t1.dedupe_window" in res.stdout

@pytest.mark.skipif(not SCRIPT.exists(), reason="validator CLI not found")
def test_cli_validator_warns_when_perf_disabled(tmp_path: Path):
    """
    We do NOT expect strict failure here; just allow warnings in stdout/stderr.
    """
    cfg = tmp_path / "warn.yaml"
    cfg.write_text(textwrap.dedent("""
        perf:
          enabled: false
          t1:
            caps:
              frontier: 10
              visited: 5
            dedupe_window: 3
        """).strip() + "\n", encoding="utf-8")

    res = subprocess.run([sys.executable, str(SCRIPT), str(cfg)], capture_output=True, text=True)
    # Non-strict run should exit 0 even with warnings.
    assert res.returncode == 0