#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, os, sys, re
from pathlib import Path


def _stat(p: Path) -> int:
    try:
        return p.stat().st_size
    except Exception:
        return 0


def _read_first_json_line(p: Path):
    # Works for .json and .json.zst if the header is on first line uncompressed
    try:
        if p.suffix == ".zst":
            import zstandard as zstd  # optional

            data = zstd.ZstdDecompressor().decompress(p.read_bytes()).decode("utf-8", "ignore")
            line = data.splitlines()[0] if data else "{}"
        else:
            line = p.open("r", encoding="utf-8", errors="ignore").readline()
        return json.loads(line or "{}")
    except Exception:
        return {}


def _kind_name(name: str):
    return "delta" if ".delta." in name else ("full" if ".full." in name else "unknown")


def run(root: str, snapshots_dir: str, fmt: str, verbose: bool) -> int:
    sdir = Path(snapshots_dir)
    files = []
    if sdir.is_dir():
        for p in sorted(sdir.glob("snapshot-*.*.json")) + sorted(
            sdir.glob("snapshot-*.*.json.zst")
        ):
            hdr = _read_first_json_line(p)
            files.append(
                {
                    "name": p.name,
                    "kind": _kind_name(p.name),
                    "compressed": p.suffix == ".zst",
                    "level": int(hdr.get("level", 0)) if isinstance(hdr, dict) else 0,
                    "size_bytes": _stat(p),
                    "etag_to": hdr.get("etag_to"),
                    "delta_of": hdr.get("delta_of"),
                    "codec": hdr.get("codec", "none"),
                }
            )
    summary = {
        "count": len(files),
        "total_bytes": sum(f["size_bytes"] for f in files),
        "compressed": sum(1 for f in files if f["compressed"]),
        "delta": sum(1 for f in files if f["kind"] == "delta"),
    }
    out = {
        "root": str(Path(root).resolve()),
        "snapshots_dir": str(sdir.resolve()),
        "files": files,
        "summary": summary,
    }
    if fmt == "json":
        print(json.dumps(out, sort_keys=True))
    else:
        print(f"{'NAME':60}  KIND   CMP  LVL  BYTES       ETAG_TO      DELTA_OF")
        for f in files:
            print(
                f"{f['name'][:60]:60}  {f['kind']:<5}  {int(f['compressed'])}    {f['level']:>2}  {f['size_bytes']:>10}  {str(f['etag_to'])[:8]:>8}  {str(f.get('delta_of'))[:8]:>8}"
            )
        print(
            f"\ncount={summary['count']} total={summary['total_bytes']} compressed={summary['compressed']} delta={summary['delta']}"
        )
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Inspect snapshot files (PR35)")
    ap.add_argument("--root", default="./.data")
    ap.add_argument("--snapshots-dir", default="./.data/snapshots")
    ap.add_argument("--format", choices=["json", "table"], default="json")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()
    sys.exit(run(args.root, args.snapshots_dir, args.format, args.verbose))
