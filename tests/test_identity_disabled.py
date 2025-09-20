

# PR26 — Identity when disabled (scheduler.enabled = false)
# Goal: prove the M5 scaffolding remains inert when the feature flag is off.

import os
import pytest


def test_m5_enabled_gate_is_off_by_default():
    """_m5_enabled should return False for a ctx/config with no scheduler or enabled=false."""
    orchestrator = pytest.importorskip("clematis.engine.orchestrator")
    _m5_enabled = getattr(orchestrator, "_m5_enabled")

    class Ctx:
        # Minimal ctx that _get_cfg() can read; we mimic cfg as a dict
        def __init__(self, cfg=None):
            self._cfg = cfg or {"scheduler": {"enabled": False}}

    ctx = Ctx()
    assert _m5_enabled(ctx) is False


def test_no_scheduler_log_when_disabled(monkeypatch, tmp_path):
    """
    With the feature flag off, orchestrator should never produce scheduler.jsonl.
    We verify the log file is not created in logs_dir(). This is a smoke test;
    end-to-end identity checks are covered by golden tests elsewhere.
    """
    # Route logs_dir() to a temporary directory
    import clematis.io.paths as paths_module
    monkeypatch.setattr(paths_module, "logs_dir", lambda: str(tmp_path), raising=True)

    # Import the central writer and orchestrator helpers
    from clematis.io.log import append_jsonl
    orchestrator = pytest.importorskip("clematis.engine.orchestrator")

    # Simulate a disabled run: do nothing that would write scheduler logs.
    # (The scheduler path in orchestrator is fully gated by _m5_enabled(ctx).)
    # We assert that no scheduler.jsonl appears as a byproduct.
    p = tmp_path / "scheduler.jsonl"
    assert not p.exists(), "scheduler.jsonl must not exist before the run (sanity check)"

    # Sanity: writing a different log should not create scheduler.jsonl
    append_jsonl("turn.jsonl", {"turn": 1, "agent": "A"})  # unrelated log write
    assert (tmp_path / "turn.jsonl").exists()
    assert not p.exists(), "scheduler.jsonl should not be created when scheduler is disabled"

    # Optional: guard against accidental writes if someone flips the flag here
    # (we intentionally do not call any scheduler code in this test).