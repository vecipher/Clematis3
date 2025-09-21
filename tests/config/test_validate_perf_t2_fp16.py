

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

SCRIPT = Path("scripts/validate_config.py")


@pytest.mark.skipif(not SCRIPT.exists(), reason="validator CLI not found")
def test_cli_validator_accepts_fp16_with_precompute_norms_and_partitions(tmp_path: Path):
    cfg = tmp_path / "good.yaml"
    cfg.write_text(textwrap.dedent(
        """
        perf:
          enabled: true
          metrics:
            report_memory: true
          t2:
            embed_store_dtype: fp16
            precompute_norms: true
            reader:
              partitions:
                enabled: true
                layout: owner_quarter
                path: ./data/t2
        # Disable legacy/stage caches to keep strict quiet
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
    ).strip() + "\n", encoding="utf-8")

    res = subprocess.run([sys.executable, str(SCRIPT), str(cfg), "--strict"], capture_output=True, text=True)
    assert res.returncode == 0, f"STDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}"


@pytest.mark.skipif(not SCRIPT.exists(), reason="validator CLI not found")
def test_cli_validator_warns_fp16_without_precompute_norms(tmp_path: Path):
    cfg = tmp_path / "warn.yaml"
    cfg.write_text(textwrap.dedent(
        """
        perf:
          enabled: true
          t2:
            embed_store_dtype: fp16
            precompute_norms: false
        # Disable unrelated caches to avoid extra warnings
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
    ).strip() + "\n", encoding="utf-8")

    # Non-strict: should succeed but emit a warning recommending fp32 norms
    res = subprocess.run([sys.executable, str(SCRIPT), str(cfg)], capture_output=True, text=True)
    assert res.returncode == 0
    assert "W[perf.t2]: embed_store_dtype=fp16 without precompute_norms=true" in res.stdout


@pytest.mark.skipif(not SCRIPT.exists(), reason="validator CLI not found")
def test_cli_validator_rejects_invalid_partition_layout(tmp_path: Path):
    cfg = tmp_path / "bad_layout.yaml"
    cfg.write_text(textwrap.dedent(
        """
        perf:
          enabled: true
          t2:
            reader:
              partitions:
                enabled: true
                layout: weird_layout
        # Disable legacy/stage caches to keep strict focused
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
    ).strip() + "\n", encoding="utf-8")

    res = subprocess.run([sys.executable, str(SCRIPT), str(cfg), "--strict"], capture_output=True, text=True)
    assert res.returncode != 0
    # Error messages are printed to stdout by the validator
    assert "perf.t2.reader.partitions.layout" in res.stdout


@pytest.mark.skipif(not SCRIPT.exists(), reason="validator CLI not found")
def test_cli_validator_rejects_empty_partition_path(tmp_path: Path):
    cfg = tmp_path / "bad_path.yaml"
    cfg.write_text(textwrap.dedent(
        """
        perf:
          enabled: true
          t2:
            reader:
              partitions:
                enabled: true
                layout: owner_quarter
                path: ""
        # Disable legacy/stage caches to keep strict focused
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
    ).strip() + "\n", encoding="utf-8")

    res = subprocess.run([sys.executable, str(SCRIPT), str(cfg), "--strict"], capture_output=True, text=True)
    assert res.returncode != 0
    assert "perf.t2.reader.partitions.path" in res.stdout