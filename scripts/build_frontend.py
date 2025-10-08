#!/usr/bin/env python3
import os
import shutil

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FRONT = os.path.join(ROOT, "frontend")
SRC = os.path.join(FRONT, "src")
STY = os.path.join(FRONT, "styles")
VEN = os.path.join(FRONT, "vendor")
DIST = os.path.join(FRONT, "dist")
ASSETS = os.path.join(DIST, "assets")

def ensure_dir(p): os.makedirs(p, exist_ok=True)

def copy_text_lf(src, dst):
    with open(src, "rb") as f:
        data = f.read()
    # Normalize CRLF -> LF for determinism across OS
    data = data.replace(b"\r\n", b"\n")
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    with open(dst, "wb") as w:
        w.write(data)

def copy_tree_text_lf(src_dir, dst_dir):
    if not os.path.isdir(src_dir):
        return
    for root, _, files in os.walk(src_dir):
        rel = os.path.relpath(root, src_dir)
        for fn in files:
            s = os.path.join(root, fn)
            d = os.path.join(dst_dir, rel, fn) if rel != "." else os.path.join(dst_dir, fn)
            copy_text_lf(s, d)

def main():
    if os.path.exists(DIST):
        shutil.rmtree(DIST)
    ensure_dir(ASSETS)
    # index + styles + vendor (text files)
    copy_text_lf(os.path.join(FRONT, "index.html"), os.path.join(DIST, "index.html"))
    copy_tree_text_lf(STY, os.path.join(DIST, "styles"))
    copy_tree_text_lf(VEN, os.path.join(DIST, "vendor"))
    # JS assets (source -> assets/)
    # If later we compile TS, replace this with compiled JS copy.
    src_main_js = os.path.join(SRC, "main.js")
    if os.path.exists(src_main_js):
        copy_text_lf(src_main_js, os.path.join(ASSETS, "main.js"))
    # Return a stable “OK” line so Make logs are deterministic.
    print("build_frontend: dist ready")

if __name__ == "__main__":
    main()
