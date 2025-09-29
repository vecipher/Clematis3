import os, sys, subprocess
from pathlib import Path

PY = sys.executable


def _run(cmd, env):
    return subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)


def test_debug_breadcrumb_stderr_only(tmp_path: Path):
    base_cmd = [PY, "-m", "clematis", "rotate-logs", "--", "--dir", str(tmp_path), "--dry-run"]
    p1 = _run(base_cmd, env={**os.environ, "CI": "true"})
    p2 = _run(base_cmd, env={**os.environ, "CI": "true", "CLEMATIS_DEBUG": "1"})
    # Exit code and stdout must be identical; only stderr may differ.
    assert p1.returncode == p2.returncode == 0
    assert p1.stdout == p2.stdout
    assert "[clematis] delegate ->" not in p1.stderr
    assert "[clematis] delegate ->" in p2.stderr
