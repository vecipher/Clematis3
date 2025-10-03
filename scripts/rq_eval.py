"""
PR41 — Retrieval Quality (RQ) evaluation harness

Deterministic, offline evaluator for two configurations (A/B) over a fixed set of
queries and qrels. Computes recall@k, MRR@k, and nDCG@k, and emits stable JSON
(and optional CSV). Optional trace emission respects existing triple-gate rules.

This script intentionally avoids any network I/O and relies on your existing
pipeline entrypoint (`run_t2`). It does not mutate configs; any trace emission is
controlled by the provided configs and an optional context hint.
"""

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import dataclasses
import hashlib
import io
import json
import math
import os
import sys

if __package__ is None:  # Allow direct execution without installing package
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

# ------------------------------- Utilities --------------------------------


def _file_sha256(path: Optional[str]) -> str:
    if not path:
        return "absent"
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return "unreadable"


def _load_jsonl(path: str) -> List[Mapping[str, object]]:
    rows: List[Mapping[str, object]] = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            s = line.strip()
            if not s:
                continue
            try:
                rows.append(json.loads(s))
            except json.JSONDecodeError as e:
                raise SystemExit(f"Invalid JSON on line {i} of {path}: {e}")
    return rows


def _load_tsv(path: str, expect_cols: int) -> List[List[str]]:
    out: List[List[str]] = []
    with open(path, "r", encoding="utf-8") as f:
        for i, raw in enumerate(f, 1):
            s = raw.rstrip("\n\r")
            if not s:
                continue
            cols = s.split("\t")
            if len(cols) != expect_cols:
                raise SystemExit(
                    f"Expected {expect_cols} tab-separated columns on line {i} of {path}, got {len(cols)}"
                )
            out.append(cols)
    return out


# ------------------------------ Metrics -----------------------------------


def recall_at_k(truth: Sequence[str], ranked: Sequence[str], k: int) -> float:
    if not truth:
        return 0.0
    k = max(0, int(k))
    top = set(ranked[:k])
    rel = set(truth)
    return float(len(top & rel)) / float(len(rel) or 1)


def mrr_at_k(truth: Sequence[str], ranked: Sequence[str], k: int) -> float:
    k = max(0, int(k))
    rel = set(truth)
    for idx, did in enumerate(ranked[:k], 1):
        if did in rel:
            return 1.0 / float(idx)
    return 0.0


def _dcg_at_k(gains: Mapping[str, float], ranked: Sequence[str], k: int) -> float:
    s = 0.0
    for i, did in enumerate(ranked[:k], 1):
        g = float(gains.get(did, 0.0))
        if g <= 0.0:
            continue
        s += (2.0**g - 1.0) / math.log2(i + 1.0)
    return s


def ndcg_at_k(gains: Mapping[str, float], ranked: Sequence[str], k: int) -> float:
    k = max(0, int(k))
    dcg = _dcg_at_k(gains, ranked, k)
    # Ideal DCG: sort by gain desc, tie-break lex(id) for determinism
    ideal_ids = sorted((did for did, g in gains.items() if g > 0.0), key=lambda d: (-gains[d], d))[
        :k
    ]
    idcg = _dcg_at_k(gains, ideal_ids, k)
    return 0.0 if idcg <= 0.0 else dcg / idcg


# ------------------------------ Runner ------------------------------------


def _import_run_t2():
    try:
        from clematis.engine.stages.t2 import run_t2  # type: ignore

        return run_t2
    except Exception as e:
        raise SystemExit(
            "Could not import clematis.engine.stages.t2.run_t2 — ensure the package is installed and importable.\n"
            f"Import error: {e}"
        )


def _load_yaml(path: str) -> Mapping[str, object]:
    try:
        import yaml  # type: ignore

        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception as e:
        raise SystemExit(f"Failed to load YAML config '{path}': {e}")


def _extract_ranked_ids(result_obj, k: int) -> List[str]:
    """Best-effort extraction of ranked doc IDs from a T2 result.

    We try several common shapes to keep this harness portable:
      1) result.top_ids or result.ids (list[str])
      2) result.items: list[dict] with 'id' (optional 'score')
      3) result.ranking: list[tuple[id, score]]
    The list is truncated to k while preserving order.
    """
    # 1) Simple attributes
    for attr in ("top_ids", "ids"):
        ids = getattr(result_obj, attr, None)
        if isinstance(ids, list) and ids and isinstance(ids[0], str):
            return [str(x) for x in ids[:k]]

    # 2) items as dicts
    items = getattr(result_obj, "items", None)
    if isinstance(items, list) and items and isinstance(items[0], dict):
        return [str(d.get("id")) for d in items[:k] if d.get("id") is not None]

    # 3) ranking as tuples
    ranking = getattr(result_obj, "ranking", None)
    if isinstance(ranking, list) and ranking and isinstance(ranking[0], (list, tuple)):
        return [str(r[0]) for r in ranking[:k]]

    # Fallback: look for 'results' or 'candidates'
    for attr in ("results", "candidates"):
        xs = getattr(result_obj, attr, None)
        if isinstance(xs, list):
            out: List[str] = []
            for x in xs:
                if isinstance(x, dict) and "id" in x:
                    out.append(str(x["id"]))
                elif isinstance(x, (list, tuple)) and x:
                    out.append(str(x[0]))
                if len(out) >= k:
                    break
            if out:
                return out

    # Last resort: nothing
    return []


@dataclasses.dataclass(frozen=True)
class SystemEval:
    name: str
    config_path: str
    config: Mapping[str, object]
    config_digest: str
    macro: Mapping[str, float]
    per_query: List[Mapping[str, object]]


def _run_system(
    name: str,
    cfg_path: str,
    queries: List[Tuple[str, str]],
    qrels: Mapping[str, Mapping[str, float]],
    k: int,
    emit_traces: bool,
    trace_dir: Optional[str],
) -> SystemEval:
    run_t2 = _import_run_t2()
    cfg = _load_yaml(cfg_path)

    # Context hint for trace reason (runtime may ignore if not supported)
    ctx = {"trace_reason": "eval"} if emit_traces else {}

    per_rows: List[Mapping[str, object]] = []
    recalls: List[float] = []
    mrrs: List[float] = []
    ndcgs: List[float] = []

    for qid, qtext in queries:
        res = run_t2(cfg, query=qtext, ctx=ctx)  # type: ignore[arg-type]
        ranked_ids = _extract_ranked_ids(res, k=k)

        gains = qrels.get(qid, {})
        truth = [did for did, rel in gains.items() if float(rel) > 0.0]

        r = recall_at_k(truth, ranked_ids, k)
        m = mrr_at_k(truth, ranked_ids, k)
        n = ndcg_at_k(gains, ranked_ids, k)

        recalls.append(r)
        mrrs.append(m)
        ndcgs.append(n)

        per_rows.append(
            {
                "qid": qid,
                "hits": [did for did in ranked_ids if did in gains],
                "recall": r,
                "mrr": m,
                "ndcg": n,
            }
        )

    macro = {
        "recall": float(sum(recalls) / (len(recalls) or 1)),
        "mrr": float(sum(mrrs) / (len(mrrs) or 1)),
        "ndcg": float(sum(ndcgs) / (len(ndcgs) or 1)),
    }

    # Behavioral config digest (best-effort): hash of JSON dump with sorted keys
    try:
        cfg_digest = hashlib.sha256(
            json.dumps(cfg, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
    except Exception:
        cfg_digest = "unknown"

    return SystemEval(
        name=name,
        config_path=os.path.abspath(cfg_path),
        config=cfg,
        config_digest=cfg_digest,
        macro=macro,
        per_query=per_rows,
    )


# ------------------------------ CLI ---------------------------------------


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Deterministic A/B retrieval quality evaluator (PR41)")
    p.add_argument(
        "--corpus", required=False, help="Path to corpus JSONL (id,text). Used for digests only."
    )
    p.add_argument("--queries", required=True, help="Path to queries TSV: qid\tquery_text")
    p.add_argument("--truth", required=True, help="Path to qrels TSV: qid\tdoc_id\trel")
    p.add_argument("--configA", required=True, help="Path to baseline config YAML")
    p.add_argument("--configB", required=True, help="Path to candidate config YAML")
    p.add_argument("--k", type=int, default=10, help="Top-k cutoff (default 10)")
    p.add_argument("--out", required=True, help="Path to write JSON results")
    p.add_argument("--csv", required=False, help="Optional path to write CSV results")
    p.add_argument(
        "--emit-traces",
        action="store_true",
        help="Request trace emission (triple gate must be satisfied by configs)",
    )
    p.add_argument(
        "--trace-dir",
        required=False,
        help="Trace directory hint when --emit-traces (not enforced here)",
    )

    args = p.parse_args(argv)

    # Load inputs
    if args.corpus:
        try:
            _ = _load_jsonl(args.corpus)
        except SystemExit:
            # corpus content isn't strictly required for evaluation; keep digest but continue
            pass

    queries_rows = _load_tsv(args.queries, expect_cols=2)
    truth_rows = _load_tsv(args.truth, expect_cols=3)

    queries: List[Tuple[str, str]] = [(qid, q) for qid, q in queries_rows]

    qrels: Dict[str, Dict[str, float]] = {}
    for qid, did, rel in truth_rows:
        try:
            rv = float(rel)
        except Exception:
            rv = 0.0
        qrels.setdefault(qid, {})[did] = rv

    # Evaluate both systems
    sysA = _run_system(
        "A",
        args.configA,
        queries,
        qrels,
        k=args.k,
        emit_traces=args.emit_traces,
        trace_dir=args.trace_dir,
    )
    sysB = _run_system(
        "B",
        args.configB,
        queries,
        qrels,
        k=args.k,
        emit_traces=args.emit_traces,
        trace_dir=args.trace_dir,
    )

    # Macro deltas (B - A)
    delta_macro = {
        "recall": float(sysB.macro["recall"] - sysA.macro["recall"]),
        "mrr": float(sysB.macro["mrr"] - sysA.macro["mrr"]),
        "ndcg": float(sysB.macro["ndcg"] - sysA.macro["ndcg"]),
    }

    out_obj = {
        "schema_version": 1,
        "k": int(args.k),
        "corpus_digest": _file_sha256(args.corpus),
        "queries_digest": _file_sha256(args.queries),
        "qrels_digest": _file_sha256(args.truth),
        "systems": {
            "A": {
                "config_path": sysA.config_path,
                "config_digest": sysA.config_digest,
                "metrics": {"macro": sysA.macro},
            },
            "B": {
                "config_path": sysB.config_path,
                "config_digest": sysB.config_digest,
                "metrics": {"macro": sysB.macro},
            },
        },
        "delta": {"macro": delta_macro},
        "per_query": [
            {
                "qid": qid,
                "A": next((r for r in sysA.per_query if r["qid"] == qid), {}),
                "B": next((r for r in sysB.per_query if r["qid"] == qid), {}),
                "delta": {
                    "recall": float(
                        next((r for r in sysB.per_query if r["qid"] == qid), {}).get("recall", 0.0)
                        - next((r for r in sysA.per_query if r["qid"] == qid), {}).get(
                            "recall", 0.0
                        )
                    ),
                    "mrr": float(
                        next((r for r in sysB.per_query if r["qid"] == qid), {}).get("mrr", 0.0)
                        - next((r for r in sysA.per_query if r["qid"] == qid), {}).get("mrr", 0.0)
                    ),
                    "ndcg": float(
                        next((r for r in sysB.per_query if r["qid"] == qid), {}).get("ndcg", 0.0)
                        - next((r for r in sysA.per_query if r["qid"] == qid), {}).get("ndcg", 0.0)
                    ),
                },
            }
            for qid, _ in queries
        ],
    }

    # Write JSON (stable separators and sorted keys for deterministic diffs)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out_obj, f, ensure_ascii=False, sort_keys=True, indent=2)

    # Optional CSV (per-query rows for both systems)
    if args.csv:
        os.makedirs(os.path.dirname(os.path.abspath(args.csv)) or ".", exist_ok=True)
        with open(args.csv, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["qid", "system", "recall", "mrr", "ndcg", "hits"])
            for qid, _ in queries:
                arow = next((r for r in sysA.per_query if r["qid"] == qid), None)
                brow = next((r for r in sysB.per_query if r["qid"] == qid), None)
                if arow:
                    w.writerow(
                        [
                            qid,
                            "A",
                            f"{arow['recall']:.6f}",
                            f"{arow['mrr']:.6f}",
                            f"{arow['ndcg']:.6f}",
                            ",".join(arow.get("hits", [])),
                        ]
                    )
                if brow:
                    w.writerow(
                        [
                            qid,
                            "B",
                            f"{brow['recall']:.6f}",
                            f"{brow['mrr']:.6f}",
                            f"{brow['ndcg']:.6f}",
                            ",".join(brow.get("hits", [])),
                        ]
                    )

    print(f"Wrote JSON to {args.out}" + (f" and CSV to {args.csv}" if args.csv else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
