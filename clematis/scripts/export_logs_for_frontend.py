#!/usr/bin/env python3
"""
Export canonical logs + latest snapshot into a single deterministic JSON bundle.

Usage:
  python -m clematis export-logs -- --out run_bundle.json [--logs-dir DIR] [--snapshots-dir DIR]
                                [--include-perf] [--strict] [--pretty] [--no-sort-keys]
                                [--max-stage-entries N]

Exit codes:
  0 => success
  2 => user/config/snapshot error (typed message printed)
"""

from __future__ import annotations
import argparse, json, os, sys, hashlib, glob

# --- tolerant import of engine utilities (works under package or repo) ---
try:
    from clematis.engine.snapshot import (
        get_latest_snapshot_info,
        SCHEMA_VERSION as SNAP_SCHEMA_VERSION,
        validate_snapshot_schema,
    )
    from clematis.io.paths import logs_dir as _default_logs_dir
except Exception:
    HERE = os.path.abspath(os.path.dirname(__file__))
    ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
    if ROOT not in sys.path:
        sys.path.insert(0, ROOT)
    from clematis.engine.snapshot import (  # type: ignore
        get_latest_snapshot_info,
        SCHEMA_VERSION as SNAP_SCHEMA_VERSION,
        validate_snapshot_schema,
    )
    from clematis.io.paths import logs_dir as _default_logs_dir  # type: ignore


STAGE_FILES = ("t1.jsonl", "t2.jsonl", "t4.jsonl", "apply.jsonl", "turn.jsonl")


def _read_jsonl(path: str, max_entries: int | None = None):
    out = []
    try:
        with open(path, "r", encoding="utf-8") as fh:
            for i, line in enumerate(fh):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    # Keep line as string to avoid silent loss; deterministic.
                    obj = {"_raw": line}
                out.append(obj)
                if max_entries is not None and len(out) >= max_entries:
                    break
    except FileNotFoundError:
        return []
    return out


def _sha256_bytes(b: bytes) -> str:
    h = hashlib.sha256()
    h.update(b)
    return h.hexdigest()


def _snapshot_payload_from_info(info: dict, strict: bool) -> tuple[dict, list[str], int]:
    """
    Returns: (snapshot_payload, warnings, exit_code_if_error_else_0)
    """
    warnings: list[str] = []
    if info is None:
        return {}, ["No snapshot found."], 2

    # Load snapshot JSON to enrich counts (mirrors scripts/inspect_snapshot.py approach)
    payload = {}
    try:
        with open(info["path"], "r", encoding="utf-8") as fh:
            payload = json.load(fh)
    except Exception as e:
        msg = f"SnapshotError: cannot read snapshot: {e}"
        if strict:
            return {}, [msg], 2
        warnings.append(msg)
        payload = {}

    # Ensure schema_version present/validated
    try:
        validate_snapshot_schema(payload, expected=SNAP_SCHEMA_VERSION)
    except Exception as e:
        msg = f"SnapshotError: {e}"
        if strict:
            return {}, [msg], 2
        warnings.append(msg)

    # Derive compact summary fields similar to inspect-snapshot
    snap = {
        "schema_version": payload.get("schema_version"),
        "version_etag": info.get("version_etag"),
        "nodes": info.get("nodes"),
        "edges": info.get("edges"),
        "gel_nodes": info.get("gel_nodes"),
        "gel_edges": info.get("gel_edges"),
        "last_update": info.get("last_update"),
        "graph_schema_version": info.get("graph_schema_version"),
        "caps": info.get("caps") or {},
        # Keep original file path for operator reference (local only)
        "path": info.get("path"),
    }
    # Drop None values to keep bundle stable and compact
    snap = {k: v for k, v in snap.items() if v is not None}
    return snap, warnings, 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Export logs + snapshot to a single JSON bundle")
    ap.add_argument("--logs-dir", default=_default_logs_dir(), help="Directory with canonical logs")
    ap.add_argument("--snapshots-dir", default="./.data/snapshots", help="Snapshot directory")
    ap.add_argument("--out", required=True, help="Output bundle path (JSON)")
    ap.add_argument("--include-perf", action="store_true", help="Include logs/perf/*-perf.jsonl files")
    ap.add_argument("--strict", action="store_true", help="Fail (exit 2) on missing/invalid snapshot schema")
    ap.add_argument("--pretty", action="store_true", help="Pretty-print JSON (indent=2)")
    ap.add_argument("--no-sort-keys", action="store_true", help="Disable JSON key sorting (default: sort for determinism)")
    ap.add_argument("--max-stage-entries", type=int, default=None, help="Cap entries per stage log (head)")
    args = ap.parse_args(argv)

    # Normalize dirs
    logs_dir = os.path.abspath(args.logs_dir)
    snaps_dir = os.path.abspath(args.snapshots_dir)
    out_path = os.path.abspath(args.out)

    # Collect stage logs deterministically
    logs: dict[str, list] = {}
    for fname in STAGE_FILES:
        path = os.path.join(logs_dir, fname)
        logs[fname.split(".")[0]] = _read_jsonl(path, args.max_stage_entries)

    # Perf logs are optional and hidden by default
    perf: dict[str, list] | None = None
    if args.include_perf:
        perf = {}
        perf_dir = os.path.join(logs_dir, "perf")
        for p in sorted(glob.glob(os.path.join(perf_dir, "*-perf.jsonl"))):
            key = os.path.basename(p)
            perf[key] = _read_jsonl(p, args.max_stage_entries)

    # Snapshot: find latest, build compact payload
    info = get_latest_snapshot_info(snaps_dir)
    snap_payload, warns, snap_rc = _snapshot_payload_from_info(info, args.strict)
    if snap_rc != 0:
        # typed, single-line operator-facing message
        print("SnapshotError: no valid snapshot found", file=sys.stdout)
        return 2
    for w in warns:
        # warnings are printed to stderr but do not change exit code unless --strict
        print(f"[warn] {w}", file=sys.stderr, flush=True)

    bundle = {
        "meta": {
            "tool": "clematis-export-logs",
            "schema": "v1",
            "stages": list(STAGE_FILES),
            "logs_dir": logs_dir,
            "snapshots_dir": snaps_dir,
        },
        "snapshot": snap_payload,
        "logs": logs,
    }
    if perf is not None:
        bundle["perf"] = perf

    # Deterministic JSON: normalize CRLF->LF on write; sorted keys by default
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    sort_keys = (not args.no_sort_keys)
    if args.pretty:
        text = json.dumps(bundle, indent=2, sort_keys=sort_keys, ensure_ascii=False)
    else:
        text = json.dumps(bundle, separators=(",", ":"), sort_keys=sort_keys, ensure_ascii=False)
    text = text.replace("\r\n", "\n")
    with open(out_path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)

    digest = _sha256_bytes(text.encode("utf-8"))
    print(f"Exported bundle: {out_path} (sha256={digest})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
