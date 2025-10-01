from __future__ import annotations

from pathlib import Path

import os
from clematis.world.scenario import run_one_turn
from clematis.engine.types import Config
from clematis.io.log import _append_jsonl_unbuffered
from clematis.engine.util.io_logging import enable_staging, disable_staging, default_key_for, STAGE_ORD


def test_golden_path(tmp_path, monkeypatch):
    # Ensure logs go under repo .logs (relative to package path)
    cfg = Config()
    state = {}
    line = run_one_turn("AgentA", state, "hello world", cfg)
    assert line, "Dialogue line should be non-empty"

    # Check logs exist
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    logs_dir = os.path.join(repo_root, ".logs")
    names = [
        "t1.jsonl",
        "t2.jsonl",
        "t3_plan.jsonl",
        "t3_dialogue.jsonl",
        "t4.jsonl",
        "apply.jsonl",
        "health.jsonl",
    ]
    for n in names:
        p = os.path.join(logs_dir, n)
        assert os.path.exists(p), f"Expected log file missing: {p}"
        with open(p, "r", encoding="utf-8") as f:
            line = f.readline().strip()
            assert line, f"{n} should contain at least one record"

def _write_sequential(monkeypatch, logs_dir: Path, records: list[tuple[str, int, int, dict]]):
    """Baseline: write logs in canonical order without staging (sequential run)."""
    monkeypatch.setattr("clematis.io.paths.logs_dir", lambda: str(logs_dir))
    # Canonical order: (turn_id, stage_ord, slice_idx, file_path)
    for filename, turn_id, slice_idx, payload in sorted(
        records, key=lambda r: (r[1], STAGE_ORD.get(r[0], 99), r[2], r[0])
    ):
        _append_jsonl_unbuffered(filename, payload)

essential_streams = ("turn.jsonl", "scheduler.jsonl")

def _write_parallel_staged(monkeypatch, logs_dir: Path, records: list[tuple[str, int, int, dict]]):
    """Parallel path: stage in reverse order, then flush deterministically via stager."""
    monkeypatch.setattr("clematis.io.paths.logs_dir", lambda: str(logs_dir))
    stager = enable_staging()
    try:
        for filename, turn_id, slice_idx, payload in reversed(records):
            key = default_key_for(file_path=filename, turn_id=turn_id, slice_idx=slice_idx)
            stager.stage(filename, key, payload)
        for rec in stager.drain_sorted():
            _append_jsonl_unbuffered(rec.file_path, rec.payload)
    finally:
        disable_staging()

def test_turn_and_scheduler_stable_vs_sequential(monkeypatch, tmp_path):
    # Same logical events; two agents (slice 0/1) for a single turn.
    records = [
        ("turn.jsonl", 42, 0, {"turn": 42, "agent": "A", "msg": "turn-A"}),
        ("scheduler.jsonl", 42, 0, {"turn": 42, "agent": "A", "msg": "sched-A"}),
        ("turn.jsonl", 42, 1, {"turn": 42, "agent": "B", "msg": "turn-B"}),
        ("scheduler.jsonl", 42, 1, {"turn": 42, "agent": "B", "msg": "sched-B"}),
    ]

    seq_dir = tmp_path / "seq"
    par_dir = tmp_path / "par"
    seq_dir.mkdir()
    par_dir.mkdir()

    _write_sequential(monkeypatch, seq_dir, records)
    _write_parallel_staged(monkeypatch, par_dir, records)

    for name in essential_streams:
        seq_p = seq_dir / name
        par_p = par_dir / name
        assert seq_p.exists() and par_p.exists()
        seq_lines = seq_p.read_text(encoding="utf-8").splitlines()
        par_lines = par_p.read_text(encoding="utf-8").splitlines()
        # Exact byte-for-byte equality required
        assert seq_lines == par_lines
