from __future__ import annotations
import json
import os
import subprocess
import sys
from pathlib import Path
from tests.helpers.identity import read_logs

PY = sys.executable

def _run_demo(out: Path, overrides: dict | None = None, env_extra: dict | None = None):
    out.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    if env_extra:
        env.update(env_extra)
    cfg_path = None
    if overrides:
        cfg_path = out / "overrides.json"
        cfg_path.write_text(json.dumps(overrides, sort_keys=True), encoding="utf-8")
    cmd = [PY, "-m", "clematis.scripts.demo", "--out", str(out)]
    if cfg_path:
        cmd += ["--config-overrides", str(cfg_path)]
    subprocess.run(cmd, check=True, env=env)

def _cmp_dirs(a: Path, b: Path):
    logs_a = read_logs(a)
    logs_b = read_logs(b)
    assert logs_a.keys() == logs_b.keys()
    for k in logs_a:
        assert logs_a[k] == logs_b[k], f"log {k} differs"

def test_t2_hybrid_params_do_not_matter_when_disabled(tmp_path: Path):
    # Baseline run (all defaults; flags OFF)
    a = tmp_path / "baseline"
    _run_demo(a)

    # Mutate internal knobs but keep enabled=false
    b = tmp_path / "mutated"
    overrides = {
        "t2": {
            "hybrid": {
                "enabled": False,                # still OFF
                "lambda_graph": 0.37,
                "anchor_top_m": 7,
                "walk_hops": 2,
                "max_bonus": 1.25,
                "degree_norm": "invdeg",
            }
        }
    }
    _run_demo(b, overrides=overrides)

    # No perf logs should exist when disabled
    for p in b.glob("*.jsonl"):
        assert not p.name.endswith("-perf.jsonl")
    perf_dir = b / "perf"
    assert not perf_dir.exists()

    _cmp_dirs(a, b)

def test_gel_params_do_not_matter_when_disabled(tmp_path: Path):
    a = tmp_path / "baseline"
    _run_demo(a)

    b = tmp_path / "mutated"
    overrides = {
        "gel": {
            "enabled": False,                   # still OFF
            "coactivation_threshold": 0.91,
            "decay": {"half_life_turns": 5},
        }
    }
    _run_demo(b, overrides=overrides)
    _cmp_dirs(a, b)

def test_native_t1_params_do_not_matter_when_disabled(tmp_path: Path):
    a = tmp_path / "baseline"
    _run_demo(a)

    b = tmp_path / "mutated"
    overrides = {
        "native_t1": {
            "enabled": False,                   # OFF (stub)
            "batch_size": 64,
            "vectorize": True,
        }
    }
    _run_demo(b, overrides=overrides)
    _cmp_dirs(a, b)
