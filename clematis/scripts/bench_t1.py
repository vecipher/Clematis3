#!/usr/bin/env python3
from __future__ import annotations
import argparse
import json
import time
from types import SimpleNamespace
from clematis.engine.types import Config
from clematis.engine.stages import t1 as t1mod
from clematis.graph.store import InMemoryGraphStore, Node, Edge

def _build_store(n_graphs: int) -> InMemoryGraphStore:
    store = InMemoryGraphStore()
    for i in range(n_graphs):
        gid = f"G{i:03d}"
        store.ensure(gid)
        store.upsert_nodes(gid, [Node(id=f"{gid}:seed", label="seed"), Node(id=f"{gid}:n1", label="n1")])
        store.upsert_edges(gid, [Edge(id=f"{gid}:e1", src=f"{gid}:seed", dst=f"{gid}:n1", weight=1.0, rel="supports")])
    return store

def main():
    ap = argparse.ArgumentParser(description="T1 microbench (deterministic)")
    ap.add_argument("--graphs", type=int, default=24)
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--parallel", action="store_true", help="enable perf.parallel for T1")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    cfg = Config()
    # Ensure `perf` exists and is a dict
    if not hasattr(cfg, "perf") or cfg.perf is None:
        setattr(cfg, "perf", {})
    elif not isinstance(cfg.perf, dict):
        try:
            cfg.perf = dict(cfg.perf)  # type: ignore[arg-type]
        except Exception:
            cfg.perf = {}
    # Enable metrics gate for bench output
    cfg.perf["enabled"] = True  # force-enable perf to ensure metrics_gate_on is true
    cfg.perf.setdefault("metrics", {})
    cfg.perf["metrics"].setdefault("report_memory", True)
    if args.parallel:
        cfg.perf.setdefault("parallel", {})
        cfg.perf["parallel"].update({"enabled": True, "t1": True, "max_workers": args.workers})

    store = _build_store(args.graphs)
    state = {"store": store, "active_graphs": [f"G{i:03d}" for i in range(args.graphs)]}

    task_count = len(state["active_graphs"])
    effective_workers = min(args.workers, max(1, task_count)) if args.parallel else 1

    ctx = SimpleNamespace(cfg=cfg, append_jsonl=lambda rec: None)
    t0 = time.perf_counter()
    res = None
    for _ in range(args.iters):
        res = t1mod.t1_propagate(ctx, state, text="seed")
    dt_ms = (time.perf_counter() - t0) * 1000.0

    m = getattr(res, "metrics", {}) if res is not None else {}
    # Backfill PR67 fields for display if the metrics gate is ON
    gate_on = bool(cfg.perf.get("enabled", False)) and bool(cfg.perf.get("metrics", {}).get("report_memory", False))
    if gate_on:
        if "task_count" not in m:
            m["task_count"] = task_count
        if "parallel_workers" not in m:
            m["parallel_workers"] = effective_workers

    out = {
        "graphs": args.graphs,
        "iters": args.iters,
        "workers": args.workers if args.parallel else 1,
        "effective_workers": effective_workers,
        "parallel": bool(args.parallel),
        "elapsed_ms": round(dt_ms, 3),
        "metrics": m,
        "parallel_workers": m.get("parallel_workers"),
        "task_count": m.get("task_count"),
    }
    if args.json:
        print(json.dumps(out, ensure_ascii=False))
    else:
        extras = []
        if out.get("task_count") is not None:
            extras.append(f"tasks={out['task_count']}")
        if out.get("parallel_workers") is not None:
            extras.append(f"pworkers={out['parallel_workers']}")
        extra_str = (" " + " ".join(extras)) if extras else ""
        print(f"T1 bench — graphs={out['graphs']} iters={out['iters']} parallel={out['parallel']} workers={out['workers']} (eff={out['effective_workers']}){extra_str}")
        print(f"elapsed_ms={out['elapsed_ms']}")
        if out["metrics"]:
            print("metrics:", json.dumps(out["metrics"], ensure_ascii=False))

if __name__ == "__main__":
    main()
