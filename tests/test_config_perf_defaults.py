from __future__ import annotations

from pathlib import Path
import yaml


ROOT = Path(__file__).resolve().parents[1]
CFG_PATH = ROOT / "configs" / "config.yaml"


def _load_cfg() -> dict:
    with CFG_PATH.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def test_perf_parallel_defaults_block_present():
    cfg = _load_cfg()
    assert isinstance(cfg, dict), "config should be a mapping"
    assert "perf" in cfg and isinstance(cfg["perf"], dict), "missing 'perf' section"

    pp = cfg["perf"].get("parallel")
    assert isinstance(pp, dict), "missing 'perf.parallel' mapping"

    # Defaults OFF and sequential
    assert pp.get("enabled") is False
    assert pp.get("max_workers") == 0  # 0 (or 1) means sequential by policy
    assert pp.get("t1") is False
    assert pp.get("t2") is False
    assert pp.get("agents") is False


def test_perf_parallel_defaults_are_sequential_shape():
    cfg = _load_cfg()
    pp = cfg["perf"]["parallel"]

    # Defensive: if someone flips to 1 in future, still considered sequential.
    assert pp.get("max_workers") in (0, 1)
