from __future__ import annotations

from types import SimpleNamespace as SNS

import pytest

from clematis.engine.orchestrator import _run_agents_parallel_batch
from clematis.io.log import append_jsonl

from tests.helpers.identity import (
    route_logs,
    read_lines,
    collect_snapshots_from_apply,
    hash_snapshots,
)
from tests.helpers.configs import make_cfg_par
from tests.helpers.world import make_state_disjoint, fixed_ctx


# ---------------------------------------------------------------------------
# Local fixtures: stable run_turn/apply_changes for deterministic artifacts
# ---------------------------------------------------------------------------


@pytest.fixture
def patched_run_turn(monkeypatch, tmp_path):
    from clematis.engine.orchestrator import Orchestrator

    def _stub(self, ctx, state, text):  # noqa: ANN001 - test stub signature
        agent = getattr(ctx, "agent_id", "?")
        turn = int(getattr(ctx, "turn_id", 0))
        # Emit stable entries for identity checks
        append_jsonl("t1.jsonl", {"agent": agent, "turn": turn, "msg": "t1"})
        append_jsonl("t2.jsonl", {"agent": agent, "turn": turn, "msg": "t2"})
        append_jsonl("t4.jsonl", {"agent": agent, "turn": turn, "msg": "t4"})
        if getattr(ctx, "_dry_run_until_t4", False):
            ctx._dryrun_t4 = SNS(approved_deltas=[{"op": "noop", "id": f"{agent}-1"}])
            ctx._dryrun_utter = f"ident:{agent}:{text}"
            ctx._dryrun_t1 = {"graphs_touched": [f"G-{agent}"]}
            ctx._dryrun_t2 = {"k_returned": 0, "k_used": 0}
            return SNS(line=ctx._dryrun_utter, events=[])
        return SNS(line=f"ident:{agent}:{text}", events=[])

    monkeypatch.setattr(Orchestrator, "run_turn", _stub)

    # Route logs to a temporary directory by default; each test may override
    logs_dir = tmp_path / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    route_logs(monkeypatch, logs_dir)
    return logs_dir


@pytest.fixture
def patched_apply_changes(monkeypatch):
    from clematis.engine import orchestrator as orch

    def _apply(ctx, state, t4_like):  # noqa: ANN001 - test stub
        return SNS(
            applied=len(getattr(t4_like, "approved_deltas", []) or []),
            clamps=0,
            version_etag="vtest",
            snapshot_path="snap://test",
            metrics={"cache_invalidations": 0},
        )

    monkeypatch.setattr(orch, "apply_changes", _apply)
    return True


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def _compare_artifacts(seq_dir, par_dir):
    for name in ("t1.jsonl", "t2.jsonl", "t4.jsonl", "apply.jsonl", "turn.jsonl", "scheduler.jsonl"):
        seq_lines = read_lines(seq_dir, name)
        par_lines = read_lines(par_dir, name)
        assert seq_lines == par_lines, f"Mismatch in {name}"
    seq_snaps = collect_snapshots_from_apply(seq_dir)
    par_snaps = collect_snapshots_from_apply(par_dir)
    assert len(seq_snaps) == len(par_snaps)
    assert hash_snapshots(seq_snaps) == hash_snapshots(par_snaps)


def test_deterministic_contention_toggle_identity(patched_run_turn, patched_apply_changes, monkeypatch, tmp_path):
    # Disjoint world (A,B)
    state1 = make_state_disjoint(2)
    state2 = make_state_disjoint(2)
    tasks = [("A", "hi"), ("B", "yo")]

    cfg = make_cfg_par(2)

    # Baseline parallel run (no patch)
    base_dir = tmp_path / "base"
    base_dir.mkdir(parents=True, exist_ok=True)
    route_logs(monkeypatch, base_dir)
    ctx_base = fixed_ctx(turn_id=201, cfg=cfg)
    _ = _run_agents_parallel_batch(ctx_base, state1, tasks)

    # Monkeypatch the parallel helper to deterministically flip task order inside the pool
    try:
        import clematis.engine.util.parallel as par
        orig = par.run_parallel

        def wrapped(jobs, *args, **kwargs):  # noqa: ANN001 - signature passthrough
            try:
                jobs2 = list(reversed(list(jobs)))
            except Exception:
                jobs2 = jobs
            return orig(jobs2, *args, **kwargs)

        monkeypatch.setattr(par, "run_parallel", wrapped)
    except Exception:  # pragma: no cover - environment without helper
        pytest.skip("parallel helper not available to patch")

    # Patched parallel run
    alt_dir = tmp_path / "alt"
    alt_dir.mkdir(parents=True, exist_ok=True)
    route_logs(monkeypatch, alt_dir)
    ctx_alt = fixed_ctx(turn_id=201, cfg=cfg)
    _ = _run_agents_parallel_batch(ctx_alt, state2, tasks)

    _compare_artifacts(base_dir, alt_dir)


def test_stager_backpressure_does_not_change_artifacts(patched_run_turn, patched_apply_changes, monkeypatch, tmp_path):
    # Same tasks/world; run twice: default staging vs tiny byte_limit staging.
    state1 = make_state_disjoint(2)
    state2 = make_state_disjoint(2)
    tasks = [("A", "hi"), ("B", "yo")]

    cfg = make_cfg_par(2)

    # Baseline parallel run with default staging limit
    base_dir = tmp_path / "base_bp"
    base_dir.mkdir(parents=True, exist_ok=True)
    route_logs(monkeypatch, base_dir)
    ctx_base = fixed_ctx(turn_id=202, cfg=cfg)
    _ = _run_agents_parallel_batch(ctx_base, state1, tasks)

    # Patch orchestrator's enable_staging to force a tiny byte_limit
    import clematis.engine.orchestrator as orch
    import clematis.engine.util.io_logging as io_log

    orig_enable = orch.enable_staging

    def tiny_enable():
        # 150 bytes should be small enough to trigger at least one drain
        return io_log.enable_staging(byte_limit=150)

    monkeypatch.setattr(orch, "enable_staging", tiny_enable)

    # Run with forced back-pressure
    alt_dir = tmp_path / "alt_bp"
    alt_dir.mkdir(parents=True, exist_ok=True)
    route_logs(monkeypatch, alt_dir)
    ctx_alt = fixed_ctx(turn_id=202, cfg=cfg)
    _ = _run_agents_parallel_batch(ctx_alt, state2, tasks)

    _compare_artifacts(base_dir, alt_dir)
