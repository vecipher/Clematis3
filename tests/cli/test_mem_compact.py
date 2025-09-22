import json, subprocess, sys
from pathlib import Path
import pytest

SCRIPT = "scripts/mem_compact.py"

@pytest.mark.skipif(not Path(SCRIPT).exists(), reason="mem_compact.py missing")
def test_mem_compact_dry_run_and_write(tmp_path):
    src = tmp_path / "src"; dst = tmp_path / "dst"
    src.mkdir(); dst.mkdir()
    # minimal full snapshot
    (src / "snapshot-aaaa.full.json").write_text('{"schema":1,"mode":"full","etag_to":"aaaa","codec":"none","level":0}\n{"ok":true}')
    # dry-run
    r = subprocess.run([sys.executable, SCRIPT, "--in", str(src), "--out", str(dst), "--dry-run"],
                       capture_output=True, text=True, check=False)
    assert r.returncode == 0, r.stderr
    plan = json.loads(r.stdout)
    assert "would_write" in plan and len(plan["would_write"]) == 1
    # actual write (no compression)
    dst2 = tmp_path / "dst2"
    r2 = subprocess.run([sys.executable, SCRIPT, "--in", str(src), "--out", str(dst2)],
                        capture_output=True, text=True, check=False)
    assert r2.returncode == 0, r2.stderr
    out = json.loads(r2.stdout)
    assert len(out["written"]) == 1
    assert (dst2 / "snapshot-aaaa.full.json").exists()