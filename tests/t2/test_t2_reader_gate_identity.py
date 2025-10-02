import os
import sys
import json
import subprocess
from pathlib import Path
import pytest

try:
    # Project APIs (as used elsewhere in tests)
    from clematis.io.config import load_config
    from clematis.engine.stages.t2 import run_t2
except Exception as e:  # pragma: no cover
    pytest.skip(f"Required modules not importable: {e}", allow_module_level=True)

DISABLED_CFG = Path(".ci/disabled_path_config.yaml")
TINY_ROOT = Path("./.data/tiny")
SEED_SCRIPT = Path("scripts/seed_tiny_shard.py")


def _maybe_seed_tiny():
    """
    Seed a deterministic tiny store if the seeder exists and the tiny dir looks empty.
    Identity mode should *not* rely on the reader, but we seed to keep local runs convenient.
    """
    if TINY_ROOT.exists() and (TINY_ROOT / "vectors.jsonl").exists():
        return
    if SEED_SCRIPT.exists():
        python = sys.executable or "python3"
        subprocess.check_call([python, str(SEED_SCRIPT), "--out", str(TINY_ROOT)])
    else:
        # Minimal placeholder to satisfy any path checks; reader should not engage anyway.
        TINY_ROOT.mkdir(parents=True, exist_ok=True)
        (TINY_ROOT / "_store_meta.json").write_text("{}")


def _get(obj, *path, default=None):
    """
    Safe getter across dicts/objects for nested fields.
    """
    cur = obj
    for key in path:
        if cur is None:
            return default
        if isinstance(cur, dict):
            cur = cur.get(key, default)
        else:
            cur = getattr(cur, key, default)
    return cur


@pytest.mark.skipif(
    not DISABLED_CFG.exists(), reason="Disabled-path config missing (.ci/disabled_path_config.yaml)"
)
def test_reader_not_used_when_perf_disabled():
    """
    Gate identity: with perf.enabled=false, the T2 reader must not engage and
    must not emit reader-specific metrics. Tier sequence must not include 'embed_store'.
    """
    _maybe_seed_tiny()
    cfg = load_config(str(DISABLED_CFG))

    # Assert perf is disabled in this config
    perf_enabled = _get(cfg, "perf", "enabled", default=False)
    assert perf_enabled is False, "Test requires perf.enabled=false in disabled-path config"

    # Run T2 against the tiny corpus; reader MUST NOT engage.
    res = run_t2(cfg, corpus_dir=str(TINY_ROOT), query="anchor")

    # Extract metrics robustly
    metrics = res.get("metrics", {}) if isinstance(res, dict) else _get(res, "metrics", default={})

    # 1) Reader tier must not appear
    tier_sequence = metrics.get("tier_sequence", [])
    assert "embed_store" not in tier_sequence, f"Reader tier engaged unexpectedly: {tier_sequence}"

    # 2) No reader-specific metrics blob should be present
    assert "reader" not in metrics, (
        f"Reader metrics leaked in disabled mode: {list(metrics.keys())}"
    )
