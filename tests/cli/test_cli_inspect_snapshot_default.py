import os
import sys
import subprocess

PY = sys.executable


def test_inspect_snapshot_default_dir_injection():
    p = subprocess.run(
        [PY, "-m", "clematis", "inspect-snapshot", "--"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env={**os.environ, "CI": "true"},
    )
    # With a packaged demo snapshot present, the delegated script should succeed.
    assert p.returncode == 0
    assert "No snapshot found" not in p.stderr
