#!/usr/bin/env python3
"""
Microbench for T4 meta-filter.

Runs synthetic proposed deltas through `t4_filter` multiple times and reports timing stats.
No engine state is mutated; this is a read-only, opt-in diagnostic.

Usage examples:
  python3 scripts/bench_t4.py
  python3 scripts/bench_t4.py --num 20000 --runs 5 --seed 1337
  python3 scripts/bench_t4.py --num 8000 --runs 7 --json

Exit codes:
  0 = success
  2 = import/config error
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import statistics as stats
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

# --- import t4 with a forgiving path setup ---
try:
    from clematis.engine.stages.t4 import t4_filter as _t4_filter  # type: ignore
except Exception:
    HERE = os.path.abspath(os.path.dirname(__file__))
    ROOT = os.path.abspath(os.path.join(HERE, ".."))
    if ROOT not in sys.path:
        sys.path.insert(0, ROOT)
    from clematis.engine.stages.t4 import t4_filter as _t4_filter  # type: ignore


# --- tiny helpers to mirror the shapes T4 expects ---
class _Ctx:
    def __init__(self, cfg: Dict[str, Any]):
        self.config = cfg
        self.cfg = cfg


class _State:
    pass


class _Op:
    __slots__ = ("target_kind", "target_id", "attr", "delta", "kind", "op_idx", "idx")

    def __init__(
        self,
        target_id: str,
        delta: float,
        kind: str = "EditGraph",
        target_kind: str = "Node",
        attr: str = "weight",
        op_idx: Optional[int] = None,
        idx: Optional[int] = None,
    ):
        self.target_kind = target_kind
        self.target_id = target_id
        self.attr = attr
        self.delta = float(delta)
        self.kind = kind
        self.op_idx = op_idx
        self.idx = idx


def _mk_cfg(l2: float, novelty: float, churn: int) -> Dict[str, Any]:
    return {
        "t4": {
            "enabled": True,
            "delta_norm_cap_l2": float(l2),
            "novelty_cap_per_node": float(novelty),
            "churn_cap_edges": int(churn),
            "cooldowns": {"EditGraph": 2, "CreateGraph": 10},
            "weight_min": -1.0,
            "weight_max": 1.0,
            "cache": {
                "enabled": True,
                "namespaces": ["t2:semantic"],
                "max_entries": 512,
                "ttl_sec": 600,
            },
        }
    }


def _gen_plan(N: int, seed: int) -> Dict[str, Any]:
    rng = random.Random(seed)
    ids = [f"n{i:04d}" for i in range(256)]  # collisions encourage novelty cap engagement
    ops: List[_Op] = []
    for i in range(N):
        tgt = rng.choice(ids)
        amt = rng.uniform(-2.0, 2.0)  # some values exceed caps deliberately
        ops.append(_Op(target_id=tgt, delta=amt, op_idx=i, idx=i))
    return {"proposed_deltas": ops, "ops": ops, "deltas": ops}


def _percentile(xs: List[float], p: float) -> float:
    if not xs:
        return float("nan")
    xs = sorted(xs)
    k = (len(xs) - 1) * (p / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return xs[int(k)]
    d0 = xs[f] * (c - k)
    d1 = xs[c] * (k - f)
    return d0 + d1


def _agg_reasons(res) -> Dict[str, int]:
    reasons = getattr(res, "reasons", None)
    if reasons is None:
        metrics = getattr(res, "metrics", {}) or {}
        reasons = metrics.get("reasons", [])
    out: Dict[str, int] = {}
    for r in reasons or []:
        key = str(r)
        out[key] = out.get(key, 0) + 1
    return out


def _approved_len(res) -> int:
    approved = getattr(res, "approved_deltas", None)
    if approved is None:
        approved = getattr(res, "approved", [])
    return len(approved or [])


@dataclass
class RunStat:
    ms: float
    approved: int
    reasons: Dict[str, int]


def run_bench(
    num: int, runs: int, seed: int, l2: float, novelty: float, churn: int
) -> Dict[str, Any]:
    ctx = _Ctx(_mk_cfg(l2=l2, novelty=novelty, churn=churn))
    state = _State()

    times_ms: List[float] = []
    approved_counts: List[int] = []
    reason_agg: Dict[str, int] = {}

    for i in range(runs):
        plan = _gen_plan(num, seed + i)  # vary seed per run, but deterministically
        t0 = time.perf_counter()
        res = _t4_filter(ctx, state, {}, {}, plan, {})
        dt = (time.perf_counter() - t0) * 1000.0
        times_ms.append(dt)
        n_ok = _approved_len(res)
        approved_counts.append(n_ok)
        for k, v in _agg_reasons(res).items():
            reason_agg[k] = reason_agg.get(k, 0) + v

    summary = {
        "num": num,
        "runs": runs,
        "seed": seed,
        "l2": l2,
        "novelty": novelty,
        "churn": churn,
        "median_ms": stats.median(times_ms) if times_ms else float("nan"),
        "p95_ms": _percentile(times_ms, 95.0),
        "min_ms": min(times_ms) if times_ms else float("nan"),
        "max_ms": max(times_ms) if times_ms else float("nan"),
        "throughput_ops_per_s": (num / (stats.median(times_ms) / 1000.0))
        if times_ms and stats.median(times_ms) > 0
        else float("nan"),
        "approved_median": stats.median(approved_counts) if approved_counts else 0,
        "approved_min": min(approved_counts) if approved_counts else 0,
        "approved_max": max(approved_counts) if approved_counts else 0,
        "reasons_total": reason_agg,
        "timings_ms": times_ms,
    }
    return summary


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Microbench T4 meta-filter")
    ap.add_argument("--num", type=int, default=10000, help="Number of proposed deltas per run")
    ap.add_argument("--runs", type=int, default=5, help="How many runs to average")
    ap.add_argument("--seed", type=int, default=1337, help="Base RNG seed")
    ap.add_argument("--l2", type=float, default=1.5, help="delta_norm_cap_l2")
    ap.add_argument("--novelty", type=float, default=0.3, help="novelty_cap_per_node")
    ap.add_argument("--churn", type=int, default=64, help="churn_cap_edges")
    ap.add_argument("--json", action="store_true", help="Emit JSON instead of pretty text")

    args = ap.parse_args(argv)

    try:
        summary = run_bench(
            num=args.num,
            runs=args.runs,
            seed=args.seed,
            l2=args.l2,
            novelty=args.novelty,
            churn=args.churn,
        )
    except Exception as ex:
        print(f"error: {ex}", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0

    # Pretty
    print(
        f"N={summary['num']} runs={summary['runs']} seed={summary['seed']}  l2={summary['l2']} novelty={summary['novelty']} churn={summary['churn']}"
    )
    print(
        f"median={summary['median_ms']:.1f}ms  p95={summary['p95_ms']:.1f}ms  min={summary['min_ms']:.1f}ms  max={summary['max_ms']:.1f}ms  "
        f"thr≈{summary['throughput_ops_per_s']:.1f} ops/s"
    )
    print(
        f"approved median/min/max = {int(summary['approved_median'])}/{int(summary['approved_min'])}/{int(summary['approved_max'])}"
    )
    if summary["reasons_total"]:
        # compact reasons print sorted by frequency
        items = sorted(summary["reasons_total"].items(), key=lambda kv: (-kv[1], kv[0]))
        joined = ", ".join([f"{k}:{v}" for k, v in items])
        print(f"reasons total: {joined}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
