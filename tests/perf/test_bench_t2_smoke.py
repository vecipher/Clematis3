

import json

import pytest


def test_run_bench_inmemory_sequential():
    from clematis.scripts import bench_t2

    out = bench_t2.run_bench(
        iters=1,
        workers=2,
        backend="inmemory",
        dim=16,
        n_rows=32,
        parallel=False,
        seed=123,
        k=16,
    )

    # Basic shape
    assert isinstance(out, dict)
    for key in (
        "queries",
        "shards",
        "workers",
        "effective_workers",
        "backend",
        "parallel",
        "elapsed_ms",
        "t2_task_count",
        "t2_parallel_workers",
        "t2_partition_count",
    ):
        assert key in out

    # In sequential mode, no parallel counters
    assert out["backend"] == "inmemory"
    assert out["parallel"] is False
    assert out["effective_workers"] == 1
    assert out["t2_task_count"] == 0
    assert out["t2_parallel_workers"] == 0


def test_run_bench_inmemory_parallel_backfills_metrics():
    from clematis.scripts import bench_t2

    out = bench_t2.run_bench(
        iters=1,
        workers=4,
        backend="inmemory",
        dim=16,
        n_rows=64,
        parallel=True,
        seed=321,
        k=16,
    )

    assert out["backend"] == "inmemory"
    assert out["parallel"] is True
    # Should see more than one shard with the synthetic corpus
    assert out["shards"] >= 2
    # Backfilled metrics should be positive in parallel mode
    assert out["t2_task_count"] >= 1
    assert out["t2_parallel_workers"] >= 1
    # Partition count is Lance-only, stays 0 here
    assert out["t2_partition_count"] == 0


def test_cli_json_smoke(capsys):
    from clematis.scripts import bench_t2

    rc = bench_t2.main([
        "--iters", "1",
        "--workers", "2",
        "--backend", "inmemory",
        "--rows", "24",
        "--dim", "8",
        "--k", "8",
        "--parallel",
        "--json",
    ])
    assert rc == 0
    captured = capsys.readouterr().out.strip()
    data = json.loads(captured)
    assert data["backend"] == "inmemory"
    assert data["parallel"] is True
    assert "elapsed_ms" in data
