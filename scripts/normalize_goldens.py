# scripts/normalize_goldens.py
from __future__ import annotations
import argparse, json, os, sys
from pathlib import Path
from typing import Any, Iterable

ROOTS = ["tests/golden", "tests/goldens", "tests/cli/goldens"]

# Heuristics mirrored from tests/helpers/identity.py
_PATH_KEY_NAMES = {
    "path","file","filepath","snapshot","snapshot_path","artifact",
    "dir","directory","logs_dir","log_file"
}
_PATH_KEY_SUFFIXES = ("_path","_file","_dir","_directory")

def _is_path_key(key: str) -> bool:
    lk = key.lower()
    if lk in _PATH_KEY_NAMES: return True
    return any(lk.endswith(s) for s in _PATH_KEY_SUFFIXES)

def _norm_path_string(s: str) -> str:
    if not s: return s
    unc = s.startswith("\\\\") or s.startswith("//")
    s2 = s.replace("\\", "/")
    if unc:
        s2 = "//" + s2.lstrip("/")
    while "//" in s2.lstrip("/"):
        head = "//" if unc else "/"
        s2 = head + s2.lstrip("/").replace("//", "/")
    return s2

def _norm_obj_paths(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: (_norm_path_string(v) if isinstance(v, str) and _is_path_key(k)
                    else _norm_obj_paths(v))
                for k, v in obj.items()}
    if isinstance(obj, list):
        return [_norm_obj_paths(x) for x in obj]
    return obj

def _iter_files() -> Iterable[Path]:
    for root in ROOTS:
        p = Path(root)
        if not p.exists():
            continue
        for f in p.rglob("*"):
            if f.is_file():
                yield f

def _is_probably_text(path: Path) -> bool:
    # Treat jsonl, json, md, txt, yaml, yml as text; fall back to small files
    return path.suffix.lower() in {".jsonl",".json",".md",".txt",".yaml",".yml",".rst",".man",".help",".out"}

def normalize_file(path: Path) -> bytes | None:
    orig = path.read_bytes()
    try:
        text = orig.decode("utf-8")
    except UnicodeDecodeError:
        return None  # skip binary
    # Normalize newlines to LF
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    changed = False
    if path.suffix.lower() == ".jsonl":
        out_lines = []
        for line in text.splitlines():
            s = line
            try:
                obj = json.loads(line)
                obj2 = _norm_obj_paths(obj)
                if obj2 is not obj:
                    s2 = json.dumps(obj2, ensure_ascii=False, separators=(",", ":"))
                    if s2 != s:
                        s = s2
                        changed = True
            except Exception:
                pass
            out_lines.append(s)
        new = ("\n".join(out_lines) + ("\n" if text.endswith("\n") else "")).encode("utf-8")
        if new != orig:
            return new
        return None
    else:
        # Non-JSONL: only line-endings
        new = text.encode("utf-8")
        if new != orig:
            return new
        return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true", help="Apply changes to files")
    ap.add_argument("--backup", action="store_true", help="Write .bak once per changed file (with --write)")
    args = ap.parse_args()

    changed = []
    for f in _iter_files():
        if not _is_probably_text(f):
            continue
        new = normalize_file(f)
        if new is None:
            continue
        if args.write:
            if args.backup and not (f.with_suffix(f.suffix + ".bak")).exists():
                f.with_suffix(f.suffix + ".bak").write_bytes(f.read_bytes())
            f.write_bytes(new)
        changed.append(str(f))

    if changed and not args.write:
        print("Files would change (run with --write):", *changed, sep="\n")
        sys.exit(2)
    if changed:
        print("Normalized files:", *changed, sep="\n")
    else:
        print("No changes.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
