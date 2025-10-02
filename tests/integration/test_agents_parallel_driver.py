import json
from types import SimpleNamespace as SNS
from pathlib import Path

import pytest


@pytest.fixture
def patched_run_turn(monkeypatch, tmp_path):
    """Patch Orchestrator.run_turn to a lightweight stub that honors PR70 dry-run.
    Also route logs to a temporary directory for assertion.
    """
    from clematis.engine.orchestrator import Orchestrator

    # Route logs to tmp
    monkeypatch.setattr("clematis.io.paths.logs_dir", lambda: str(tmp_path))

    def _stub(self, ctx, state, text):  # noqa: ANN001 - test stub
        from clematis.io.log import append_jsonl

        # ensure the stub writes some logs to be captured
        if getattr(ctx, "_dry_run_until_t4", False):
            append_jsonl(
                "t1.jsonl",
                {
                    "agent": getattr(ctx, "agent_id", "?"),
                    "turn": getattr(ctx, "turn_id", 0),
                    "msg": "t1",
                },
            )
            # stash artifacts expected by the batch driver
            ctx._dryrun_t4 = SNS(
                approved_deltas=[{"op": "add", "id": f"{getattr(ctx, 'agent_id', '')}-1"}]
            )
            ctx._dryrun_utter = f"dry:{getattr(ctx, 'agent_id', '')}:{text}"
            ctx._dryrun_t1 = {"graphs_touched": [f"G-{getattr(ctx, 'agent_id', '')}"]}
            ctx._dryrun_t2 = {"k_returned": 0, "k_used": 0}
            append_jsonl(
                "t4.jsonl",
                {
                    "agent": getattr(ctx, "agent_id", "?"),
                    "turn": getattr(ctx, "turn_id", 0),
                    "msg": "t4",
                },
            )
            return SNS(line=ctx._dryrun_utter, events=[])
        else:
            append_jsonl(
                "t1.jsonl",
                {
                    "agent": getattr(ctx, "agent_id", "?"),
                    "turn": getattr(ctx, "turn_id", 0),
                    "msg": "t1_seq",
                },
            )
            return SNS(line=f"seq:{getattr(ctx, 'agent_id', '')}:{text}", events=[])

    monkeypatch.setattr(Orchestrator, "run_turn", _stub)
    return tmp_path


@pytest.fixture
def patched_apply_changes(monkeypatch):
    """Patch apply_changes to avoid touching real state; return a tiny object with fields used in commit."""
    from clematis.engine import orchestrator as orch

    def _apply(ctx, state, t4_like):  # noqa: ANN001 - test stub
        # Simulate applying deltas and return minimal shape used by the driver
        return SNS(
            applied=len(getattr(t4_like, "approved_deltas", []) or []),
            clamps=0,
            version_etag="vtest",
            snapshot_path="snap://test",
            metrics={"cache_invalidations": 0},
        )

    monkeypatch.setattr(orch, "apply_changes", _apply)
    return True


def _ctx(enabled: bool, agents: bool, max_workers: int, turn_id: int = 100):
    # Minimal TurnCtx stub accepted by orchestrator helpers
    cfg = {
        "perf": {
            "enabled": True,  # global perf.enabled
            "parallel": {
                "enabled": bool(enabled),
                "t2": True,
                "agents": bool(agents),
                "max_workers": int(max_workers),
            },
        }
    }
    return SNS(cfg=cfg, turn_id=turn_id)


def _state_disjoint():
    # State shape recognized by _resolve_graphs_for_agent: graphs_by_agent mapping
    return {"graphs_by_agent": {"A": ["G1"], "B": ["G2"]}}


def _state_overlapping():
    return {"graphs_by_agent": {"A": ["G1"], "B": ["G1"]}}


def test_off_path_sequential_fallback_returns_all(patched_run_turn):
    from clematis.engine.orchestrator import _run_agents_parallel_batch

    ctx = _ctx(enabled=False, agents=False, max_workers=1)
    state = _state_disjoint()
    tasks = [("A", "hi"), ("B", "yo")]

    results = _run_agents_parallel_batch(ctx, state, tasks)

    assert [r.line for r in results] == ["seq:A:hi", "seq:B:yo"]


def test_on_path_independent_batch_flushes_logs_and_applies(
    patched_run_turn, patched_apply_changes
):
    from clematis.engine.orchestrator import _run_agents_parallel_batch

    ctx = _ctx(enabled=True, agents=True, max_workers=2, turn_id=123)
    state = _state_disjoint()
    tasks = [("A", "hi"), ("B", "yo")]

    results = _run_agents_parallel_batch(ctx, state, tasks)

    # Compute dialogue must come from dry-run artifacts for A then B (deterministic)
    assert [r.line for r in results] == ["dry:A:hi", "dry:B:yo"]

    # Logs must be flushed to the tmp logs_dir in the same order
    logs_dir = patched_run_turn
    t1 = Path(logs_dir) / "t1.jsonl"
    t4 = Path(logs_dir) / "t4.jsonl"
    apply = Path(logs_dir) / "apply.jsonl"
    turn_id = 123  # from ctx above

    t1_lines = [json.loads(x) for x in t1.read_text(encoding="utf-8").strip().splitlines()]
    # Under PR71 staging, ordering is deterministic: A then B for the same turn
    assert [entry["agent"] for entry in t1_lines[:2]] == ["A", "B"]
    assert all(entry["turn"] == turn_id for entry in t1_lines[:2])

    t4_lines = [json.loads(x) for x in t4.read_text(encoding="utf-8").strip().splitlines()]
    assert [entry["agent"] for entry in t4_lines[:2]] == ["A", "B"]
    assert all(entry["turn"] == turn_id for entry in t4_lines[:2])

    apply_lines = [json.loads(x) for x in apply.read_text(encoding="utf-8").strip().splitlines()]
    assert len(apply_lines) == 2
    assert [entry["agent"] for entry in apply_lines] == ["A", "B"]
    assert all(entry["turn"] == turn_id for entry in apply_lines)


def test_on_path_overlapping_graphs_selects_subset(patched_run_turn):
    from clematis.engine.orchestrator import _run_agents_parallel_batch

    ctx = _ctx(enabled=True, agents=True, max_workers=2, turn_id=999)
    state = _state_overlapping()
    tasks = [("A", "hi"), ("B", "yo")]

    results = _run_agents_parallel_batch(ctx, state, tasks)

    # Only one agent should be picked for the batch due to overlapping graphs
    assert len(results) == 1
    assert results[0].line.startswith("dry:")
