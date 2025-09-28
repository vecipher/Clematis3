import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

SCRIPT = Path("scripts/validate_config.py")


@pytest.mark.skipif(not SCRIPT.exists(), reason="validator CLI not found")
def test_cli_validator_accepts_t2_reader_batch_and_embed_root_strict(tmp_path: Path):
    cfg = tmp_path / "good.yaml"
    cfg.write_text(
        textwrap.dedent(
            """
      perf:
        enabled: true
      t2:
        reader_batch: 4096
        embed_root: ./data/t2
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
    # No warnings expected in strict for these fields
    assert "W[" not in res.stdout


@pytest.mark.skipif(not SCRIPT.exists(), reason="validator CLI not found")
def test_cli_validator_rejects_invalid_reader_batch_and_embed_root(tmp_path: Path):
    cfg = tmp_path / "bad.yaml"
    cfg.write_text(
        textwrap.dedent(
            """
      perf:
        enabled: true
      t2:
        reader_batch: 0        # invalid (must be >= 1)
        embed_root: ""        # invalid (must be non-empty string)
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
    assert res.returncode != 0
    # Errors are printed to STDOUT by the validator
    assert (
        "t2.reader_batch" in res.stdout or "t2.embed_root" in res.stdout
    ), f"STDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}"


@pytest.mark.skipif(not SCRIPT.exists(), reason="validator CLI not found")
def test_cli_validator_warns_partitions_with_perf_disabled(tmp_path: Path):
    cfg = tmp_path / "warn.yaml"
    cfg.write_text(
        textwrap.dedent(
            """
      perf:
        enabled: false
        t2:
          reader:
            partitions:
              enabled: true
              layout: owner_quarter
              path: ./data/t2
      # Disable unrelated caches to keep output focused
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
    assert res.returncode == 0
    assert (
        "W[perf.t2.reader]" in res.stdout
    ), f"Expected partitions+perf.disabled warning. STDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}"
