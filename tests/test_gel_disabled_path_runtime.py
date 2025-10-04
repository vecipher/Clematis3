import os
import importlib
from pathlib import Path

import pytest

from tests.helpers.identity import (
    collect_snapshots_from_apply,
    hash_snapshots,
    normalize_log_bytes_for_identity,
)


def _maybe_get_smoke():
    """
    Only accept the canonical engine helper. Skip if missing on this branch.
    """
    try:
        core = importlib.import_module("clematis.engine.orchestrator.core")
    except Exception:
        return None
    fn = getattr(core, "run_smoke_turn", None)
    return fn if callable(fn) else None


def _read_file_map(log_dir: Path):
    """Return {relative_path: bytes} for all files under log_dir (sorted traversal)."""
    out = {}
    for p in sorted(log_dir.rglob("*")):
        if p.is_file():
            out[p.relative_to(log_dir).as_posix()] = p.read_bytes()
    return out


@pytest.mark.identity
def test_gel_disabled_path_runtime_logs_and_snapshots(tmp_path, monkeypatch):
    """
    With graph.enabled=false, a tiny smoke run should:
      • NOT produce gel.jsonl
      • Produce byte-identical logs to a baseline config w/o 'graph'
      • Produce identical snapshot sets + identical hashes (or no snapshots in both)
    """
    run_smoke = _maybe_get_smoke()
    if run_smoke is None:
        pytest.skip("run_smoke_turn not available on this branch")

    # Deterministic env
    monkeypatch.setenv("CI", "true")
    monkeypatch.setenv("CLEMATIS_NETWORK_BAN", "1")

    base_logs = tmp_path / "base"
    off_logs = tmp_path / "off"
    base_logs.mkdir(parents=True, exist_ok=True)
    off_logs.mkdir(parents=True, exist_ok=True)

    base_cfg = {"t2": {"k_retrieval": 8}}
    graph_off_cfg = {
        "t2": {"k_retrieval": 8},
        "graph": {
            "enabled": False,
            "coactivation_threshold": 0.2,
            "observe_top_k": 4,
            "update": {"mode": "additive", "alpha": 0.02, "clamp_min": -1.0, "clamp_max": 1.0},
            "decay": {"half_life_turns": 200, "floor": 0.0},
            # ops intentionally toggled ON but must remain inert while enabled=false
            "merge": {"enabled": True},
            "split": {"enabled": True},
            "promotion": {"enabled": True},
        },
    }

    monkeypatch.setenv("CLEMATIS_LOG_DIR", str(base_logs))
    run_smoke(cfg=base_cfg, log_dir=str(base_logs))

    monkeypatch.setenv("CLEMATIS_LOG_DIR", str(off_logs))
    run_smoke(cfg=graph_off_cfg, log_dir=str(off_logs))

    # 1) No gel logs
    assert not (base_logs / "gel.jsonl").exists()
    assert not (off_logs / "gel.jsonl").exists()

    # 2) Logs identical
    base_map = _read_file_map(base_logs)
    off_map = _read_file_map(off_logs)
    assert base_map.keys() == off_map.keys(), "Log file sets differ"
    base_norm = normalize_log_bytes_for_identity(base_map)
    off_norm = normalize_log_bytes_for_identity(off_map)
    for rel in base_norm.keys():
        assert base_norm[rel] == off_norm[rel], f"Log '{rel}' differs"

    # 3) Snapshots parity
    base_snaps = collect_snapshots_from_apply(base_logs)
    off_snaps = collect_snapshots_from_apply(off_logs)
    assert set(base_snaps) == set(off_snaps), "Snapshot file sets differ"
    if base_snaps:  # both non-empty sets by previous assertion
        assert hash_snapshots(base_snaps) == hash_snapshots(off_snaps), "Snapshot hashes differ"
