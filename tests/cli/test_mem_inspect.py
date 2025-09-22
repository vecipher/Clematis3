import json, subprocess, sys, os
from pathlib import Path
import pytest

SCRIPT = "scripts/mem_inspect.py"

@pytest.mark.skipif(not Path(SCRIPT).exists(), reason="mem_inspect.py missing")
def test_mem_inspect_runs_on_empty_dir(tmp_path):
    (tmp_path / "snapshots").mkdir()
    r = subprocess.run([sys.executable, SCRIPT, "--snapshots-dir", str(tmp_path/"snapshots")],
                       capture_output=True, text=True, check=False)
    assert r.returncode == 0, r.stderr
    data = json.loads(r.stdout)
    assert data["summary"]["count"] == 0