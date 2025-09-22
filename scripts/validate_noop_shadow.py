#!/usr/bin/env python3
import sys, os
from pathlib import Path

IGNORE = {"rq_traces.jsonl"}

def list_files(root: Path):
    out = []
    for p in root.rglob("*"):
        if p.is_file():
            rel = p.relative_to(root).as_posix()
            if rel.split("/")[-1] in IGNORE:
                continue
            out.append(rel)
    return sorted(out)

def main():
    # Expect two directories set by workflow steps
    base = Path(os.environ.get("BASE_ARTIFACTS", "artifacts/base"))
    shadow = Path(os.environ.get("SHADOW_ARTIFACTS", "artifacts/shadow"))
    base_files = list_files(base)
    shadow_files = list_files(shadow)
    if base_files != shadow_files:
        print("Artifact set differs under shadow; Gate D failed.")
        diffs = set(base_files).symmetric_difference(set(shadow_files))
        for d in sorted(diffs):
            print("DIFF:", d)
        sys.exit(1)
    print("Gate D OK: shadow is a no-op (except traces).")
    sys.exit(0)

if __name__ == "__main__":
    main()