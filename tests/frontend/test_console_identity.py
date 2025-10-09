import json
import os
import subprocess
import sys
import pathlib

PY = sys.executable
REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]

def _run_console(*args, env=None):
    cmd = [PY, "-m", "clematis", "console", "--", *args]
    return subprocess.run(cmd, cwd=str(REPO_ROOT), capture_output=True, text=True, env=env)

def test_compare_equal(tmp_path):
    a = tmp_path/"a.json"
    b = tmp_path/"b.json"
    bundle = {"logs":{"t1":[1], "t2":[], "t4":[], "apply":[], "turn":[]}, "snapshots":[], "meta":{"x":1}}
    a.write_text(json.dumps(bundle)+"\n", encoding="utf-8")
    b.write_text(json.dumps(bundle)+"\n", encoding="utf-8")
    r = _run_console("compare", "--a", str(a), "--b", str(b))
    assert r.returncode == 0, r.stderr + r.stdout
    assert "equal" in r.stdout

def test_compare_differs(tmp_path):
    a = tmp_path/"a.json"
    b = tmp_path/"b.json"
    a.write_text(json.dumps({"logs":{"t1":[1]},"snapshots":[],"meta":{}})+"\n", "utf-8")
    b.write_text(json.dumps({"logs":{"t1":[1,2]},"snapshots":[{}],"meta":{"x":1}})+"\n", "utf-8")
    r = _run_console("compare", "--a", str(a), "--b", str(b))
    assert r.returncode == 1, r.stderr + r.stdout
    assert "counts" in r.stdout

def test_console_status_smoke():
    # Should print scheduler/budgets JSON; relies on adapter_reset picking latest snapshot.
    r = _run_console("status")
    assert r.returncode == 0, r.stderr + r.stdout
    j = json.loads(r.stdout)
    assert "scheduler" in j and "budgets" in j

def test_console_step_smoke(tmp_path):
    # Deterministic one-turn smoke; requires orchestrator to be importable
    out = tmp_path/"run.json"
    env = os.environ.copy()
    env.update({
        "TZ": "UTC",
        "PYTHONHASHSEED": "0",
        "SOURCE_DATE_EPOCH": "315532800",
        "CLEMATIS_NETWORK_BAN": "1",
    })
    r = _run_console("step", "--now-ms", "315532800000", "--out", str(out), env=env)
    assert r.returncode == 0, r.stderr + r.stdout
    assert out.exists(), "expected console to write bundle"
    j = json.loads(out.read_text("utf-8"))
    logs = j.get("logs", {})
    for k in ("t1","t2","t4","apply","turn"):
        assert k in logs, f"missing stage log: {k}"
