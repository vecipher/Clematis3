

"""
tests/helpers/identity.py
-------------------------
Canonical normalizer for identity (disabled-path) log comparisons.

Usage in tests/workflows:
  from tests.helpers.identity import normalize_line, normalize_lines, normalize_record

Policy (conservative, explicit):
  • Drop keys that are inherently volatile in identity mode:
      - "now", "version_etag"
      - any key starting with "ms_"  (prefix timing counters)
      - any key ending with "_ms" or "_bytes" (timings/sizes)
      - exact keys: "rss_bytes"
  • Keep ordering deterministic by serializing with sorted keys.
  • Do NOT reorder lists; we only strip volatile fields.

Rationale:
  Disabled-path identity should be byte-for-byte stable after normalization.
  We deliberately keep the drop set small to avoid masking regressions.
"""
from __future__ import annotations

import json
from typing import Any, Dict, Iterable, List, Union

# ---- Drop policy ---------------------------------------------------------

DROP_EXACT_KEYS = {
    "now",
    "version_etag",
    "rss_bytes",
}

DROP_SUFFIXES = (
    "_ms",
    "_bytes",
)

DROP_PREFIXES = (
    "ms_",
)


# ---- Core normalization ---------------------------------------------------

def _should_drop(key: str) -> bool:
    if key in DROP_EXACT_KEYS:
        return True
    for suf in DROP_SUFFIXES:
        if key.endswith(suf):
            return True
    for pre in DROP_PREFIXES:
        if key.startswith(pre):
            return True
    return False


def normalize_record(obj: Any) -> Any:
    """
    Recursively drop volatile keys from a decoded JSON object (dict/list/primitive).
    Returns a new object; input is not mutated.
    """
    if isinstance(obj, dict):
        out: Dict[str, Any] = {}
        for k in sorted(obj.keys()):  # deterministic key order
            if _should_drop(k):
                continue
            out[k] = normalize_record(obj[k])
        return out
    if isinstance(obj, list):
        return [normalize_record(v) for v in obj]
    # primitives pass through unchanged
    return obj


def normalize_line(line: str) -> str:
    """
    If line is JSON, normalize it and return a compact JSON string with sorted keys.
    Otherwise, return the stripped line.
    """
    s = line.strip()
    if not s:
        return s
    try:
        obj = json.loads(s)
    except Exception:
        return s
    norm = normalize_record(obj)
    # compact separators, sorted keys for determinism
    return json.dumps(norm, sort_keys=True, separators=(",", ":"))


def normalize_lines(lines: Iterable[str]) -> List[str]:
    """Normalize an iterable of lines; returns a new list of normalized lines."""
    return [normalize_line(ln) for ln in lines]


# ---- Optional helpers for test convenience -------------------------------

def load_and_normalize(path: str) -> List[str]:
    """Read a file and return normalized lines."""
    with open(path, "r", encoding="utf-8") as f:
        return normalize_lines(f.readlines())


if __name__ == "__main__":
    # Simple CLI: read stdin and print normalized lines
    import sys
    data = sys.stdin.read().splitlines()
    for out in normalize_lines(data):
        print(out)