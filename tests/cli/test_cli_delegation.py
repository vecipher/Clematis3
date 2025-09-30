import os
import sys
import subprocess
import shlex
from pathlib import Path

PY = sys.executable


def _run(cmd, **kw):
    return subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, **kw)


def test_extras_after_sub_with_sentinel(tmp_path: Path):
    cmd = [PY, "-m", "clematis", "rotate-logs", "--", "--dir", str(tmp_path), "--dry-run"]
    p = _run(cmd, env={**os.environ, "CI": "true"})
    assert p.returncode == 0
    # no debug → no breadcrumb on stderr
    assert "[clematis] delegate ->" not in p.stderr


def test_extras_before_sub_with_sentinel(tmp_path: Path):
    cmd = [PY, "-m", "clematis", "--dir", str(tmp_path), "rotate-logs", "--", "--dry-run"]
    p = _run(cmd, env={**os.environ, "CI": "true"})
    assert p.returncode == 0
    assert "[clematis] delegate ->" not in p.stderr
