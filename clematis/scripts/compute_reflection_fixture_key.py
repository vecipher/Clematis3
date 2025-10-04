"""Utility script to compute the deterministic fixture key used by the LLM reflection backend."""

import hashlib
import json
import sys


def main() -> int:
    data = sys.stdin.read()
    if not data:
        print("{}")
        return 0
    obj = json.loads(data)
    canonical = json.dumps(obj, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:12]
    print(digest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
