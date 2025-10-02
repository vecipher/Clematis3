from __future__ import annotations

from types import SimpleNamespace as SNS

from clematis.engine import orchestrator as orch
from tests.helpers.configs import make_cfg_par
from tests.helpers.world import make_state_disjoint


def test_parallel_batch_drains_and_retries_on_backpressure(monkeypatch):
    writes: list[tuple[str, dict]] = []

    def record_write(path: str, payload: dict) -> None:
        writes.append((path, dict(payload)))

    monkeypatch.setattr(orch, "_append_jsonl_unbuffered", record_write)

    import clematis.engine.util.io_logging as io_log

    monkeypatch.setattr(orch, "enable_staging", lambda: io_log.enable_staging(byte_limit=150))
    monkeypatch.setattr(orch, "_make_readonly_snapshot", lambda state: state)

    def fake_run_turn_compute(ctx, base, agent_id, text):  # noqa: ANN001 - orchestration stub
        large1 = {"msg": "x" * 100}
        large2 = {"msg": "y" * 100}
        return {
            "turn_id": 99,
            "slice_idx": 0,
            "agent_id": agent_id,
            "logs": [("t1.jsonl", large1), ("t2.jsonl", large2)],
            "deltas": [],
            "dialogue": "ok",
            "graphs_touched": set(),
            "graph_versions": {},
        }

    monkeypatch.setattr(orch, "_run_turn_compute", fake_run_turn_compute)

    def fake_apply(ctx, state, t4_like):  # noqa: ANN001 - orchestration stub
        return SNS(
            applied=0,
            clamps=0,
            version_etag="etag",
            snapshot_path="snap",
            metrics={"cache_invalidations": 0},
        )

    monkeypatch.setattr(orch, "apply_changes", fake_apply)

    ctx = SNS(cfg=make_cfg_par(2), turn_id=1)
    state = make_state_disjoint(1)
    results = orch._run_agents_parallel_batch(ctx, state, [("A", "hi")])

    assert len(results) == 1
    # First write comes from the drain triggered by backpressure; subsequent writes
    # originate from the final flush, preserving deterministic ordering.
    assert [name for name, _ in writes][:3] == ["t1.jsonl", "t2.jsonl", "apply.jsonl"]
