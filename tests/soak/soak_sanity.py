#!/usr/bin/env python3
import argparse, json, re
from pathlib import Path

def _iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8", newline="") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line: continue
            yield i, json.loads(line)

def _stable_keyset(file_path: Path) -> bool:
    base_keys = None
    for _, obj in _iter_jsonl(file_path):
        ks = tuple(sorted(obj.keys()))
        if base_keys is None:
            base_keys = ks
        elif ks != base_keys:
            return False
    return True

def _count_exceptions(log_dir: Path) -> int:
    n = 0
    # Prefer a health log if present; else scan all jsonl for "error" markers
    health = list(log_dir.glob("**/health.jsonl"))
    targets = health if health else list(log_dir.glob("**/*.jsonl"))
    pat = re.compile(r"(EXCEPTION|ERROR|Traceback)", re.IGNORECASE)
    for p in targets:
        for _, obj in _iter_jsonl(p):
            text = json.dumps(obj, ensure_ascii=False)
            if pat.search(text):
                n += 1
    return n

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log-dir", type=Path, required=True)
    ap.add_argument("--snap-dir", type=Path, required=True)
    ap.add_argument("--metrics", type=Path, required=True)
    ap.add_argument("--rss-threshold-mb", type=float, required=True)
    ap.add_argument("--rss-spike-mb", type=float, required=True)
    ap.add_argument("--schema-out", type=Path, required=True)
    args = ap.parse_args()

    metrics = json.loads(args.metrics.read_text())
    rss_peak = float(metrics.get("rss_peak_mb", 0.0))
    checkpoints = metrics.get("checkpoints", [])
    assert rss_peak <= args.rss_threshold_mb, f"RSS peak {rss_peak} MiB exceeds {args.rss_threshold_mb} MiB"

    # Per-interval spike guard
    last = None
    for cp in checkpoints:
        cur = float(cp["rss_mb"])
        if last is not None:
            delta = cur - last
            assert delta <= args.rss_spike_mb, f"RSS spike {delta:.1f} MiB exceeds {args.rss_spike_mb} MiB"
        last = cur

    # Log JSONL parse + stable keysets
    schema = {}
    for p in sorted(args.log_dir.rglob("*.jsonl")):
        # parse
        kset = None
        for _, obj in _iter_jsonl(p):
            ks = tuple(sorted(obj.keys()))
            if kset is None:
                kset = ks
            elif ks != kset:
                raise AssertionError(f"Keyset drift within {p}: {ks} vs {kset}")
        if kset:
            schema[str(p.relative_to(args.log_dir))] = list(kset)

    # Exceptions == 0
    exc = _count_exceptions(args.log_dir)
    assert exc == 0, f"Found {exc} error/exception markers in logs"

    # Snapshots directory exists (we don't enforce format here)
    assert args.snap_dir.exists(), "Snapshot dir missing"
    args.schema_out.parent.mkdir(parents=True, exist_ok=True)
    args.schema_out.write_text(json.dumps(schema, indent=2))

if __name__ == "__main__":
    main()
