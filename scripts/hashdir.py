#!/usr/bin/env python3
import hashlib
import os
import sys

def norm_path(p): return p.replace("\\", "/")

def file_iter(root):
    for base, _, files in os.walk(root):
        for f in files:
            yield os.path.join(base, f)

def main():
    if len(sys.argv) != 2:
        print("usage: hashdir.py <dir>", file=sys.stderr)
        sys.exit(2)
    root = os.path.abspath(sys.argv[1])
    if not os.path.isdir(root):
        print("not a directory:", root, file=sys.stderr)
        sys.exit(2)
    h = hashlib.sha256()
    files = sorted(file_iter(root), key=lambda p: norm_path(os.path.relpath(p, root)))
    for path in files:
        rel = norm_path(os.path.relpath(path, root)).encode("utf-8")
        with open(path, "rb") as f:
            data = f.read().replace(b"\r\n", b"\n")  # normalize newlines for cross-OS stability
        h.update(b"PATH\0");
        h.update(rel);
        h.update(b"\0DATA\0");
        h.update(data);
        h.update(b"\0")
    print(h.hexdigest())

if __name__ == "__main__":
    main()
