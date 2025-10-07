# tests/perf/test_perf_shadow_logs.py
from __future__ import annotations
import json, sys, subprocess, os
from pathlib import Path

PY = sys.executable

def _run_demo(out: Path, overrides: dict):
    out.mkdir(parents=True, exist_ok=True)
    cfg = out / "over.json"
    cfg.write_text(json.dumps(overrides, sort_keys=True), encoding="utf-8")
    cmd = [PY, "-m", "clematis.scripts.demo", "--out", str(out), "--config-overrides", str(cfg)]
    subprocess.run(cmd, check=True, env=os.environ.copy())

def test_shadow_logs_are_segregated_when_hybrid_on(tmp_path: Path):
    out = tmp_path / "with_hybrid"
    overrides = {"t2": {"hybrid": {"enabled": True}}}
    _run_demo(out, overrides)

    # Expect perf-suffixed or perf/ directory logs
    has_perf = any(p.name.endswith("-perf.jsonl") for p in out.glob("*.jsonl"))
    perf_dir = out / "perf"
    assert has_perf or perf_dir.exists(), "expected perf logs when hybrid is enabled"
