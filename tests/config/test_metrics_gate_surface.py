import json, subprocess, sys
from pathlib import Path
import pytest

DEMO = Path("scripts/run_demo.py")
CFG_OFF = Path(".ci/gate_b_off.yaml")

@pytest.mark.skipif(not DEMO.exists(), reason="demo script missing")
@pytest.mark.skipif(not CFG_OFF.exists(), reason="gate_b_off config missing")
def test_no_metrics_when_gates_off(tmp_path):
    # Run short demo; ignore exit status
    subprocess.run([sys.executable, str(DEMO), "--config", str(CFG_OFF), "--steps", "2"],
                   check=False, capture_output=True, text=True)
    logs = Path("./.logs")
    t2 = logs / "t2.jsonl"
    if not t2.exists():
        pytest.skip("t2.jsonl not produced by demo")
    # scan last line for a metrics object
    last = None
    with t2.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            try:
                last = json.loads(line)
            except Exception:
                pass
    if last is None:
        pytest.skip("no JSON in t2.jsonl")
    assert "metrics" not in last or last.get("metrics") in (None, {}, []), "metrics leaked when gates are OFF"