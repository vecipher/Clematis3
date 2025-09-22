# tests/helpers/identity.py
from __future__ import annotations
import json
from typing import Any

# Drop keys that vary across runs or are not part of disabled-path identity
_DROP_KEYS = {
    "now",
    "version_etag",
    "tier_sequence",      # added to stabilize disabled-path identity
}
_DROP_SUFFIXES = ("_ms", "ms")  # e.g., duration_ms, total_ms, etc.

def _normalize(obj: Any) -> Any:
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if k in _DROP_KEYS:
                continue
            if any(k.endswith(suf) for suf in _DROP_SUFFIXES):
                continue
            out[k] = _normalize(v)
        return out
    if isinstance(obj, list):
        return [_normalize(x) for x in obj]
    return obj

def normalize_json_line(line: str) -> str:
    try:
        obj = json.loads(line)
    except Exception:
        return line.strip()
    return json.dumps(_normalize(obj), sort_keys=True, separators=(",", ":"))

def normalize_json_lines(lines: list[str]) -> list[str]:
    return [normalize_json_line(ln) for ln in lines if ln.strip()]