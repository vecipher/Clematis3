from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace as SNS

import pytest

from clematis.engine.orchestrator import _run_agents_parallel_batch

from tests.helpers.identity import (
    route_logs,
    read_lines,
    collect_snapshots_from_apply,
    hash_snapshots,
)
from tests.helpers.configs import make_cfg_seq, make_cfg_par
from tests.helpers.world import make_state_disjoint, make_tasks, fixed_ctx


# ---------------------------------------------------------------------------
# Patches: keep runtime light and outputs deterministic for identity testing
# ---------------------------------------------------------------------------


@pytest.fixture
def patched_run_turn(monkeypatch, tmp_path):
    """Patch Orchestrator.run_turn to emit stable logs for both seq & par paths.

    We intentionally make the t1/t4 entries identical across dry-run and
    sequential modes so that the artifacts match byte-for-byte between paths.
    """
    from clematis.engine.orchestrator import Orchestrator
    from clematis.io.log import append_jsonl

    def _stub(self, ctx, state, text):  # noqa: ANN001 - test stub signature
        agent = getattr(ctx, "agent_id", "?")
        turn = int(getattr(ctx, "turn_id", 0))
        # Emit identical content regardless of dry-run to ensure identity
        append_jsonl("t1.jsonl", {"agent": agent, "turn": turn, "msg": "t1"})
        append_jsonl("t2.jsonl", {"agent": agent, "turn": turn, "msg": "t2"})
        append_jsonl("t4.jsonl", {"agent": agent, "turn": turn, "msg": "t4"})
        if getattr(ctx, "_dry_run_until_t4", False):
            # Provide dry-run artifacts the parallel driver expects
            ctx._dryrun_t4 = SNS(approved_deltas=[{"op": "noop", "id": f"{agent}-1"}])
            ctx._dryrun_utter = f"ident:{agent}:{text}"
            ctx._dryrun_t1 = {"graphs_touched": [f"G-{agent}"]}
            ctx._dryrun_t2 = {"k_returned": 0, "k_used": 0}
            return SNS(line=ctx._dryrun_utter, events=[])
        else:
            # Mirror the commit-phase apply record that the parallel path writes
            append_jsonl(
                "apply.jsonl",
                {
                    "turn": turn,
                    "agent": agent,
                    "applied": 1,
                    "clamps": 0,
                    "version_etag": "vtest",
                    "snapshot": "snap://test",
                    "cache_invalidations": 0,
                    "ms": 0.0,
                },
            )
            return SNS(line=f"ident:{agent}:{text}", events=[])

    monkeypatch.setattr(Orchestrator, "run_turn", _stub)

    # Route logs to a temporary directory for this test module
    logs_dir = tmp_path / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    route_logs(monkeypatch, logs_dir)
    return logs_dir


@pytest.fixture
def patched_apply_changes(monkeypatch):
    """Patch apply_changes to avoid touching real state and return a stable shape."""
    from clematis.engine import orchestrator as orch

    def _apply(ctx, state, t4_like):  # noqa: ANN001 - test stub
        return SNS(
            applied=len(getattr(t4_like, "approved_deltas", []) or []),
            clamps=0,
            version_etag="vtest",
            snapshot_path="snap://test",  # not a real file; hashing yields empty
            metrics={"cache_invalidations": 0},
        )

    monkeypatch.setattr(orch, "apply_changes", _apply)
    return True


# ---------------------------------------------------------------------------
# Identity: sequential vs parallel produce byte-identical artifacts
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_seq_vs_par_identity_basic(patched_run_turn, patched_apply_changes, seed, tmp_path):
    # Build a tiny disjoint world with two agents (A,B)
    state_seq = make_state_disjoint(2)
    state_par = make_state_disjoint(2)
    tasks = make_tasks(2)

    # --- Sequential run (disabled path) ---
    seq_dir = tmp_path / f"seq-{seed}"
    seq_dir.mkdir(parents=True, exist_ok=True)
    from tests.helpers.identity import route_logs as _route

    _route(pytest.MonkeyPatch(), seq_dir)  # patch in a fresh monkeypatch context
    ctx_seq = fixed_ctx(turn_id=100 + seed, cfg=make_cfg_seq())
    _ = _run_agents_parallel_batch(ctx_seq, state_seq, tasks)

    # --- Parallel run (enabled path) ---
    par_dir = tmp_path / f"par-{seed}"
    par_dir.mkdir(parents=True, exist_ok=True)
    _route(pytest.MonkeyPatch(), par_dir)
    ctx_par = fixed_ctx(turn_id=100 + seed, cfg=make_cfg_par(2))
    _ = _run_agents_parallel_batch(ctx_par, state_par, tasks)

    # Compare key artifacts (raw bytes)
    for name in (
        "t1.jsonl",
        "t2.jsonl",
        "t4.jsonl",
        "apply.jsonl",
        "turn.jsonl",
        "scheduler.jsonl",
    ):
        seq_lines = read_lines(seq_dir, name)
        par_lines = read_lines(par_dir, name)
        assert seq_lines == par_lines, f"Mismatch in {name} for seed={seed}"

    # Compare snapshot lists (hashes)
    seq_snaps = collect_snapshots_from_apply(seq_dir)
    par_snaps = collect_snapshots_from_apply(par_dir)
    assert len(seq_snaps) == len(par_snaps)
    assert hash_snapshots(seq_snaps) == hash_snapshots(par_snaps)
