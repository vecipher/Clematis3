#!/usr/bin/env python3
import os
import sys
import pathlib
import hashlib
import shutil

ROOT = pathlib.Path(__file__).resolve().parents[1]
FRONTEND = ROOT / "frontend"
SRC = FRONTEND / "src"
TSDIST = FRONTEND / "tsdist" / "assets"
DIST = FRONTEND / "dist"
ASSETS_OUT = DIST / "assets"

def sha256_file(p: pathlib.Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        while True:
            b = f.read(1 << 20)
            if not b: break
            h.update(b)
    return h.hexdigest()

def write_lf(src: pathlib.Path, dst: pathlib.Path) -> None:
    data = src.read_bytes().replace(b"\r\n", b"\n")
    dst.parent.mkdir(parents=True, exist_ok=True)
    with dst.open("wb") as f: f.write(data)

def copy_tree_lf(src_dir: pathlib.Path, dst_dir: pathlib.Path, exts=(".html",".js",".css",".json",".svg",".txt",".md")) -> None:
    files = [p for p in src_dir.rglob("*") if p.is_file() and p.suffix in exts]
    files.sort(key=lambda p: str(p.relative_to(src_dir)).replace(os.sep, "/"))
    for p in files:
        rel = p.relative_to(src_dir)
        write_lf(p, dst_dir / rel)

def main() -> int:
    if not FRONTEND.exists():
        print("frontend/ missing", file=sys.stderr)
        return 2

    # Choose assets source: tsdist (preferred) or src fallback (for JS-only dev)
    assets_src = TSDIST if TSDIST.exists() else (SRC if (SRC / "app.js").exists() else None)
    if assets_src is None:
        print("No frontend assets found. Build TS (npm ci && npm run build) or keep JS sources in frontend/src/.", file=sys.stderr)
        return 3

    # Clean dist
    if DIST.exists(): shutil.rmtree(DIST)
    DIST.mkdir(parents=True, exist_ok=True)

    # Copy index.html (LF)
    index_html = FRONTEND / "index.html"
    if not index_html.exists():
        print("frontend/index.html missing", file=sys.stderr); return 4
    write_lf(index_html, DIST / "index.html")

    # Copy assets
    copy_tree_lf(assets_src, ASSETS_OUT)

    # Optional: copy styles & vendor if they live outside assets
    for extra in ("styles", "vendor"):
        src_dir = FRONTEND / extra
        if src_dir.exists():
            copy_tree_lf(src_dir, DIST / extra)

    # Print a stable summary
    files = sorted([p for p in DIST.rglob("*") if p.is_file()], key=lambda p: str(p.relative_to(DIST)).replace(os.sep, "/"))
    for p in files:
        rel = p.relative_to(DIST)
        print(f"{sha256_file(p)}  {rel}")

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
