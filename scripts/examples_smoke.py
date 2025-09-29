#!/usr/bin/env python3
from __future__ import annotations

"""
PR42 — Examples smoke checker

Runs a tiny, deterministic smoke over the shipped example configs to catch drift:
  • Validates each YAML with validate_config_verbose
  • Runs a couple of queries through the pipeline (run_t2)
  • Checks determinism across repeated runs
  • When the triple gate is ON, expects the reader mode metric (t2.reader_mode)
  • Supports --examples-glob and --fail-fast

Exit non‑zero if any example fails. Keep output terse and actionable.
"""

import argparse
import glob
import json
import os
import sys

# Ensure repo root is on sys.path when invoked as a file (not -m)
try:
    import pathlib as _pathlib  # local import to avoid polluting namespace

    _REPO_ROOT = _pathlib.Path(__file__).resolve().parents[1]
    _p = str(_REPO_ROOT)
    if _p not in sys.path:
        sys.path.insert(0, _p)
except Exception:
    # Best-effort; fall back to user-provided PYTHONPATH or editable install
    pass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

# ------------------------------ imports ------------------------------------


def _import_or_die():
    try:
        from configs.validate import validate_config_verbose  # type: ignore
    except Exception as e:
        print(
            f"FATAL: could not import configs.validate.validate_config_verbose: {e}",
            file=sys.stderr,
        )
        sys.exit(2)
    try:
        from clematis.engine.stages.t2 import run_t2  # type: ignore
    except Exception as e:
        print(f"FATAL: could not import clematis.engine.stages.t2.run_t2: {e}", file=sys.stderr)
        sys.exit(2)
    return validate_config_verbose, run_t2


def _load_yaml(path: str) -> Mapping[str, Any]:
    try:
        import yaml  # type: ignore
    except Exception as e:
        print(f"FATAL: PyYAML not available: {e}", file=sys.stderr)
        sys.exit(2)
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


# ------------------------------ helpers ------------------------------------


def _gate_on(cfg: Mapping[str, Any]) -> bool:
    perf = cfg.get("perf") or {}
    t2 = cfg.get("t2") or {}
    q = t2.get("quality") or {}
    return bool(
        (perf.get("enabled") is True or str(perf.get("enabled")).lower() == "true")
        and (
            (perf.get("metrics") or {}).get("report_memory") is True
            or str((perf.get("metrics") or {}).get("report_memory")).lower() == "true"
        )
        and (
            q.get("shadow") is True
            or q.get("enabled") is True
            or str(q.get("shadow")).lower() == "true"
            or str(q.get("enabled")).lower() == "true"
        )
    )


def _extract_ranked_ids(result_obj: Any, k: int = 10) -> List[str]:
    # 1) Explicit attrs
    for attr in ("top_ids", "ids"):
        ids = getattr(result_obj, attr, None)
        if isinstance(ids, list) and (not ids or isinstance(ids[0], str)):
            return [str(x) for x in ids[:k]]
    # 2) items as dicts
    items = getattr(result_obj, "items", None)
    if isinstance(items, list) and items and isinstance(items[0], dict):
        out = [str(d.get("id")) for d in items[:k] if d.get("id") is not None]
        if out:
            return out
    # 3) ranking as tuples
    ranking = getattr(result_obj, "ranking", None)
    if isinstance(ranking, list) and ranking and isinstance(ranking[0], (list, tuple)):
        return [str(r[0]) for r in ranking[:k]]
    # 4) results/candidates fallback
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
    return []


def _metrics_dict(result_obj: Any) -> Dict[str, Any]:
    m = getattr(result_obj, "metrics", None)
    return dict(m) if isinstance(m, dict) else {}


# ------------------------------ runner -------------------------------------


def _run_example(
    path: str,
    validate_config_verbose,
    run_t2,
    queries: Sequence[str],
    repeat: int,
    check_traces: bool,
) -> Tuple[bool, str]:
    cfg = _load_yaml(path)
    try:
        normalized, warnings = validate_config_verbose(cfg)
    except ValueError as e:
        return False, f"validation errors: {str(e)}"
    cfg = normalized

    if warnings:
        print(f"W[validate] {os.path.basename(path)}: {len(warnings)} warning(s)")

    gate_on = _gate_on(cfg)
    # Disable caches to avoid cross-config pollution in smoke runs
    try:
        # Legacy t2.cache (if present)
        t2_node = cfg.setdefault("t2", {})
        t2_node.setdefault("cache", {})["enabled"] = False
        # perf.t2 byte-LRU (if present)
        perf_node = cfg.setdefault("perf", {})
        perf_t2 = perf_node.setdefault("t2", {})
        perf_t2_cache = perf_t2.setdefault("cache", {})
        perf_t2_cache["max_entries"] = 0
        perf_t2_cache["max_bytes"] = 0
    except Exception:
        # Best-effort only; if cfg isn’t a dict-like, proceed
        pass
    ctx = {}
    if gate_on:
        ctx = {"trace_reason": "examples_smoke"}

    # choose a couple of short queries
    qs = list(queries)[:2]

    # deterministic runs
    ref_ids: Optional[List[str]] = None
    ref_metrics: Optional[Dict[str, Any]] = None

    for r in range(max(1, int(repeat))):
        for q in qs:
            res = run_t2(cfg, query=q, ctx=ctx)  # type: ignore[arg-type]
            ids = _extract_ranked_ids(res, k=10)
            mets = _metrics_dict(res)
            if r == 0 and q == qs[0]:
                ref_ids, ref_metrics = ids, mets
            else:
                if ids != ref_ids:
                    return False, f"non-deterministic ids for query='{q}' (run {r+1} vs 1)"
                # metrics dict can include counters; compare keys presence for stability
                if set(mets.keys()) != set(ref_metrics.keys()):
                    return False, f"non-deterministic metric keys for query='{q}'"

    # Gate expectations
    if gate_on:
        # Reader parity metric should be present (PR40)
        if not ref_metrics or "t2.reader_mode" not in ref_metrics:
            return False, "gate on but 't2.reader_mode' metric missing"
        # If MMR enabled in config, expect selection metric when it runs
        qcfg = (cfg.get("t2") or {}).get("quality") or {}
        mmr_cfg = qcfg.get("mmr") or {}
        if (str(qcfg.get("enabled")).lower() == "true") and (
            str(mmr_cfg.get("enabled")).lower() == "true"
        ):
            if not any(k.startswith("t2q.mmr") for k in ref_metrics.keys()):
                return False, "mmr enabled but 't2q.mmr.*' metrics not found under gate"

    # Optional: trace sanity (best-effort)
    if gate_on and check_traces:
        perf = cfg.get("perf") or {}
        metrics = perf.get("metrics") or {}
        tdir = (
            metrics.get("trace_dir")
            or ((cfg.get("t2") or {}).get("quality") or {}).get("trace_dir")
            or "logs/quality"
        )
        tried = [os.path.join(str(tdir), "rq_traces.jsonl")]
        ok = False
        for candidate in tried:
            if os.path.isfile(candidate):
                try:
                    with open(candidate, "r", encoding="utf-8") as f:
                        line = f.readline()
                        if line.strip():
                            json.loads(line)
                            ok = True
                            break
                except Exception:
                    pass
        if not ok:
            return (
                False,
                f"gate on but could not verify traces in {tdir} (set --no-check-traces to ignore)",
            )

    return True, "ok"


def main(argv: Optional[Sequence[str]] = None) -> int:
    validate_config_verbose, run_t2 = _import_or_die()

    p = argparse.ArgumentParser(description="Smoke-check example YAMLs for M7")
    p.add_argument(
        "--examples", nargs="*", help="Specific example YAMLs to run (glob patterns supported)"
    )
    p.add_argument(
        "--examples-glob",
        nargs="*",
        help="Additional glob patterns for examples (e.g., 'examples/quality/*.yaml')",
    )
    p.add_argument(
        "--all", action="store_true", help="Run the default set of examples/quality/*.yaml"
    )
    p.add_argument(
        "--repeat",
        type=int,
        default=2,
        help="Repeat runs per example to check determinism (default 2)",
    )
    p.add_argument(
        "--no-check-traces",
        action="store_true",
        help="Do not verify presence of rq_traces.jsonl when gate is on",
    )
    p.add_argument(
        "--fail-fast", action="store_true", help="Exit immediately on first example failure"
    )

    args = p.parse_args(argv)

    default_globs = [
        "examples/quality/shadow.yaml",
        "examples/quality/lexical_fusion.yaml",
        "examples/quality/mmr.yaml",
        "examples/quality/normalizer_aliases.yaml",
        "examples/quality/reader_parity.yaml",
    ]

    targets: List[str] = []
    if args.examples:
        for pat in args.examples:
            targets.extend(glob.glob(pat))
    if args.examples_glob:
        for pat in args.examples_glob:
            targets.extend(glob.glob(pat))
    if args.all or not targets:
        for pat in default_globs:
            targets.extend(glob.glob(pat))

    # De-dup while preserving order
    seen = set()
    examples = []
    for pth in targets:
        if pth not in seen and pth.lower().endswith((".yaml", ".yml")) and os.path.isfile(pth):
            examples.append(pth)
            seen.add(pth)

    if not examples:
        print("No example YAMLs found.")
        return 1

    queries = [
        "apple tart",
        "banana bread",
        "zebra",
        "llm tutorial",
        "cuda install",
    ]

    failures: List[Tuple[str, str]] = []
    for path in examples:
        ok, msg = _run_example(
            path,
            validate_config_verbose,
            run_t2,
            queries,
            repeat=args.repeat,
            check_traces=(not args.no_check_traces),
        )
        status = "PASS" if ok else "FAIL"
        print(f"[{status}] {path}: {msg}")
        if not ok:
            failures.append((path, msg))
            if args.fail_fast:
                print("\nSummary: fail-fast enabled; stopping on first failure.")
                return 1

    if failures:
        print("\nSummary: some examples failed:")
        for pth, why in failures:
            print(f"  - {pth}: {why}")
        return 1

    print(f"\nSummary: {len(examples)} example(s) passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
