#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import time
from types import SimpleNamespace
from typing import Any, Dict

from clematis.engine.types import Config
from clematis.engine.stages import t1 as t1mod
from clematis.graph.store import InMemoryGraphStore, Node, Edge


def _build_store(n_graphs: int) -> InMemoryGraphStore:
    store = InMemoryGraphStore()
    for i in range(n_graphs):
        gid = f"G{i:03d}"
        store.ensure(gid)
        store.upsert_nodes(
            gid,
            [
                Node(id=f"{gid}:seed", label="seed"),
                Node(id=f"{gid}:n1", label="n1"),
            ],
        )
        store.upsert_edges(
            gid,
            [
                Edge(
                    id=f"{gid}:e1",
                    src=f"{gid}:seed",
                    dst=f"{gid}:n1",
                    weight=1.0,
                    rel="supports",
                )
            ],
        )
    return store


def _stable_json(data: Dict[str, Any], pretty: bool = False) -> str:
    if pretty:
        return json.dumps(data, sort_keys=True, indent=2, ensure_ascii=True)
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _merge_metrics(acc: Dict[str, Any], m: Dict[str, Any]) -> None:
    # Sum numeric fields; keep max for "max_delta"; ignore unknown shapes.
    for k, v in m.items():
        if k == "max_delta":
            try:
                acc[k] = max(float(acc.get(k, 0.0)), float(v))
            except Exception:
                continue
        else:
            try:
                if isinstance(v, (int, float)):
                    acc[k] = (acc.get(k, 0) or 0) + v
            except Exception:
                continue


def main() -> None:
    ap = argparse.ArgumentParser(
        prog="bench_t1",
        description="Delegates to scripts/... Microbench for T1 propagation (deterministic).",
    )
    ap.add_argument("--graphs", type=int, default=24, help="Number of tiny graphs (default: 24)")
    ap.add_argument("--iters", type=int, default=5, help="Repetitions (default: 5)")
    ap.add_argument(
        "--workers", type=int, default=4, help="Worker hint when --parallel (default: 4)"
    )
    ap.add_argument(
        "--parallel", action="store_true", help="Enable perf.parallel for T1 (advisory)"
    )
    ap.add_argument(
        "--warmup", type=int, default=1, help="Warmup iterations before timing (default: 1)"
    )
    ap.add_argument("--json", action="store_true", help="Emit a single stable JSON line")
    args = ap.parse_args()

    # Ban accidental networking in compliant environments.
    os.environ.setdefault("CLEMATIS_NETWORK_BAN", "1")

    # Build deterministic tiny store
    store = _build_store(args.graphs)
    state: Dict[str, Any] = {
        "store": store,
        "active_graphs": [f"G{i:03d}" for i in range(args.graphs)],
    }

    task_count = len(state["active_graphs"])
    effective_workers = min(args.workers, max(1, task_count)) if args.parallel else 1

    # Build minimal config; force perf metrics gate ON for richer outputs
    cfg = Config()
    if not hasattr(cfg, "perf") or cfg.perf is None:
        setattr(cfg, "perf", {})
    elif not isinstance(cfg.perf, dict):
        try:
            cfg.perf = dict(cfg.perf)  # type: ignore[arg-type]
        except Exception:
            cfg.perf = {}
    cfg.perf["enabled"] = True
    cfg.perf.setdefault("metrics", {})
    cfg.perf["metrics"].setdefault("report_memory", True)
    if args.parallel:
        cfg.perf.setdefault("parallel", {})
        cfg.perf["parallel"].update({"enabled": True, "t1": True, "max_workers": args.workers})

    # Minimal ctx: only fields T1 actually touches
    ctx = SimpleNamespace(cfg=cfg, append_jsonl=lambda rec: None)

    # Warmup (not timed) — prime caches deterministically
    for _ in range(max(0, args.warmup)):
        _ = t1mod.t1_propagate(ctx, state, text="seed")

    # Timed section
    t0 = time.perf_counter()
    last_res = None
    metrics_acc: Dict[str, Any] = {}
    for _ in range(args.iters):
        last_res = t1mod.t1_propagate(ctx, state, text="seed")
        m = getattr(last_res, "metrics", {}) if last_res is not None else {}
        if isinstance(m, dict):
            _merge_metrics(metrics_acc, m)
    dt_ms = (time.perf_counter() - t0) * 1000.0

    # Backfill fields (only when metrics gate is ON)
    gate_on = bool(cfg.perf.get("enabled", False)) and bool(
        cfg.perf.get("metrics", {}).get("report_memory", False)
    )
    if gate_on:
        metrics_acc.setdefault("task_count", task_count)
        metrics_acc.setdefault("parallel_workers", effective_workers)

    out: Dict[str, Any] = {
        "graphs": args.graphs,
        "iters": args.iters,
        "workers": args.workers if args.parallel else 1,
        "effective_workers": effective_workers,
        "parallel": bool(args.parallel),
        "elapsed_ms": round(dt_ms, 3),
        "metrics": metrics_acc,
        # Convenience mirrors (kept for quick TTY reads)
        "parallel_workers": metrics_acc.get("parallel_workers"),
        "task_count": metrics_acc.get("task_count"),
    }

    if args.json:
        print(_stable_json(out, pretty=False))
    else:
        extras = []
        if out.get("task_count") is not None:
            extras.append(f"tasks={out['task_count']}")
        if out.get("parallel_workers") is not None:
            extras.append(f"pworkers={out['parallel_workers']}")
        extra_str = (" " + " ".join(extras)) if extras else ""
        print(
            f"T1 bench — graphs={out['graphs']} iters={out['iters']} "
            f"parallel={out['parallel']} workers={out['workers']} "
            f"(eff={out['effective_workers']}){extra_str}"
        )
        print(f"elapsed_ms={out['elapsed_ms']}")
        if out["metrics"]:
            print("metrics:", _stable_json(out["metrics"], pretty=True))


if __name__ == "__main__":
    main()
