

from __future__ import annotations
"""
Deterministic delta codec for snapshot JSON blobs.

The delta format is a path-based patch over nested dicts:
  {
    "_adds": { "a.b": <val>, ... },
    "_mods": { "a.c": <val>, ... },
    "_dels": ["a.d", ...]
  }

- Only dictionaries are traversed; lists/atoms are treated as atomic values.
- Paths use dot-notation with keys in their natural string form.
- Ordering is deterministic (sorted keys at each step).
- compute_delta(None, X) yields adds for all keys in X; apply_delta handles None/{} bases.

Round-trip guarantee (for JSON-serializable dicts):
    apply_delta(base, compute_delta(base, curr)) == curr
"""
from typing import Any, Dict, Tuple, Iterable
import copy

__all__ = ["compute_delta", "apply_delta"]


def _is_mapping(x: Any) -> bool:
    return isinstance(x, dict)


def _walk_diff(base: Dict[str, Any] | None,
               curr: Dict[str, Any] | None,
               prefix: Tuple[str, ...] = ()) -> Tuple[Dict[str, Any], Dict[str, Any], Iterable[str]]:
    """Return (adds, mods, dels) for the subtree at prefix."""
    b = base or {}
    c = curr or {}
    adds: Dict[str, Any] = {}
    mods: Dict[str, Any] = {}
    dels: list[str] = []

    b_keys = set(b.keys()) if _is_mapping(b) else set()
    c_keys = set(c.keys()) if _is_mapping(c) else set()

    # Deletions: in base but not in current
    for k in sorted(b_keys - c_keys):
        dels.append(".".join(prefix + (k,)))

    # Additions: in current but not in base
    for k in sorted(c_keys - b_keys):
        adds[".".join(prefix + (k,))] = c[k]

    # Mods / nested walk: intersecting keys
    for k in sorted(b_keys & c_keys):
        bv, cv = b[k], c[k]
        if _is_mapping(bv) and _is_mapping(cv):
            a2, m2, d2 = _walk_diff(bv, cv, prefix + (k,))
            adds.update(a2)
            mods.update(m2)
            dels.extend(d2)
        else:
            if bv != cv:
                mods[".".join(prefix + (k,))] = cv

    return adds, mods, dels


def compute_delta(base: Dict[str, Any] | None, curr: Dict[str, Any] | None) -> Dict[str, Any]:
    """Compute a deterministic delta turning base -> curr.

    Only dicts are traversed structurally. Lists/atoms are compared by value.
    """
    a, m, d = _walk_diff(base or {}, curr or {}, ())
    # Sort deletions for stability
    d_sorted = sorted(d)
    return {"_adds": a, "_mods": m, "_dels": d_sorted}


def _set_path(d: Dict[str, Any], path: str, value: Any) -> None:
    keys = path.split(".") if path else []
    cur = d
    for k in keys[:-1]:
        nxt = cur.get(k)
        if not _is_mapping(nxt):
            nxt = {}
            cur[k] = nxt
        cur = nxt
    if keys:
        cur[keys[-1]] = value


def _del_path(d: Dict[str, Any], path: str) -> None:
    keys = path.split(".") if path else []
    if not keys:
        return
    cur = d
    for k in keys[:-1]:
        if not _is_mapping(cur):
            return
        cur = cur.get(k, {})
    if _is_mapping(cur):
        cur.pop(keys[-1], None)


def apply_delta(base: Dict[str, Any] | None, delta: Dict[str, Any] | None) -> Dict[str, Any]:
    """Apply a delta produced by compute_delta to base and return the reconstructed dict."""
    out: Dict[str, Any] = copy.deepcopy(base or {})
    if not delta:
        return out

    adds = delta.get("_adds", {}) or {}
    mods = delta.get("_mods", {}) or {}
    dels = delta.get("_dels", []) or []

    # Apply additions and modifications in deterministic key order
    for p in sorted(adds.keys()):
        _set_path(out, p, adds[p])
    for p in sorted(mods.keys()):
        _set_path(out, p, mods[p])
    # Deletions last, deterministic order
    for p in sorted(dels):
        _del_path(out, p)

    return out


if __name__ == "__main__":  # simple smoke
    # A tiny self-test to aid local debugging
    base = {"a": 1, "b": {"c": 2, "d": 3}}
    curr = {"a": 1, "b": {"c": 20, "e": 5}}
    delta = compute_delta(base, curr)
    rebuilt = apply_delta(base, delta)
    assert rebuilt == curr, (delta, rebuilt, curr)
    print("ok")