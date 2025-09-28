#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, os, sys, shutil
from pathlib import Path


def _ensure_empty_dir(p: Path):
    if p.exists():
        if any(p.iterdir()):
            raise RuntimeError(f"--out must be empty: {p}")
    else:
        p.mkdir(parents=True, exist_ok=True)


def _read_header_body(p: Path):
    data = p.read_text(encoding="utf-8", errors="ignore")
    lines = data.splitlines()
    if len(lines) >= 2:
        try:
            return json.loads(lines[0]), "\n".join(lines[1:]) + (
                "\n" if data.endswith("\n") else ""
            )
        except Exception:
            pass
    return None, data


def _write_header_body(p: Path, header: dict | None, body: str, compression: str, level: int):
    if compression == "zstd":
        import zstandard as zstd  # optional; will raise if missing

        content = (json.dumps(header) + "\n" + body) if header is not None else body
        cctx = zstd.ZstdCompressor(level=max(1, min(19, int(level or 3))))
        p.write_bytes(cctx.compress(content.encode("utf-8")))
    else:
        p.write_text(
            (json.dumps(header) + "\n" + body) if header is not None else body, encoding="utf-8"
        )


def run(
    indir: str,
    outdir: str,
    dtype: str | None,
    compression: str,
    level: int,
    delta: bool,
    dry_run: bool,
) -> int:
    src = Path(indir)
    dst = Path(outdir)
    if not src.is_dir():
        print(f"input dir not found: {src}", file=sys.stderr)
        return 2
    _ensure_empty_dir(dst)
    # Only operate on full snapshots; safe and deterministic
    inputs = sorted(
        list(src.glob("snapshot-*.full.json")) + list(src.glob("snapshot-*.full.json.zst"))
    )
    plan = []
    for ip in inputs:
        hdr, body = _read_header_body(ip)
        hdr = hdr or {}
        # apply metadata tweaks (dtype only touches header if present)
        if dtype:
            hdr["embed_store_dtype"] = dtype  # benign; readers ignore if unknown
        # normalize codec/level
        codec = "zstd" if compression == "zstd" else "none"
        lvl = max(1, min(19, int(level))) if compression == "zstd" else 0
        hdr["codec"] = codec
        hdr["level"] = lvl
        # destination name
        stem = ip.name.replace(".json.zst", ".json").replace(".json", "")
        opath = dst / f"{stem}.json{'.zst' if compression=='zstd' else ''}"
        plan.append((ip, opath, hdr, body))
    if dry_run:
        out = [{"src": str(i), "dst": str(o)} for (i, o, _, _) in plan]
        print(json.dumps({"would_write": out}, sort_keys=True))
        return 0
    for ip, op, hdr, body in plan:
        _write_header_body(op, hdr, body, compression, level)
    print(json.dumps({"written": [str(o) for (_, o, _, _) in plan]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Offline snapshot compaction (PR35)")
    ap.add_argument(
        "--in", dest="indir", required=True, help="input snapshots dir (reads full snapshots)"
    )
    ap.add_argument("--out", dest="outdir", required=True, help="output dir (must be empty or new)")
    ap.add_argument(
        "--dtype", choices=["fp32", "fp16"], help="rewrite header dtype (metadata only)"
    )
    ap.add_argument("--compression", choices=["none", "zstd"], default="none")
    ap.add_argument("--level", type=int, default=3)
    ap.add_argument("--delta", action="store_true", help="(placeholder) do not emit deltas in PR35")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    sys.exit(
        run(
            args.indir,
            args.outdir,
            args.dtype,
            args.compression,
            args.level,
            args.delta,
            args.dry_run,
        )
    )
