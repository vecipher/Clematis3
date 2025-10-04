from __future__ import annotations
import json
import copy
from typing import Any, Iterable
import os

from pathlib import Path
import hashlib
import shutil

# Drop keys that vary across runs or are not part of disabled-path identity
_DROP_KEYS = {
    "now",
    "version_etag",
    "tier_sequence",  # added to stabilize disabled-path identity
    "ms",             # plain 'ms' timing field
}
# Suffixes/prefixes for volatile timing keys
_DROP_SUFFIXES = ("_ms",)       # e.g., duration_ms, total_ms
_DROP_PREFIXES = ("ms_",)       # e.g., ms_deliberate, ms_rag


def _normalize(obj: Any) -> Any:
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if k in _DROP_KEYS:
                continue
            if any(k.startswith(pre) for pre in _DROP_PREFIXES):
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


def normalize_logs_dir(p: str, base: str | None = None) -> str:
    """
    Normalize a logs directory path deterministically for comparisons:
    - Expand ~ and environment vars
    - If `base` is provided and `p` is relative, join against it
    - Resolve symlinks via realpath
    - Collapse redundant separators / trailing slashes via normpath
    """
    # Normalize base first if present
    if base:
        base = os.path.normpath(os.path.realpath(os.path.expanduser(os.path.expandvars(base))))
    # Expand and normalize p
    p = os.path.expanduser(os.path.expandvars(p))
    if base and not os.path.isabs(p):
        p = os.path.join(base, p)
    return os.path.normpath(os.path.realpath(p))


def normalize_log_bytes_for_identity(file_map: dict[str, bytes]) -> dict[str, bytes]:
    """
    Normalize a map of log files ({relative_path: bytes}) for disabled-path identity comparisons.
    Rules:
      - For any *.jsonl file, decode UTF-8 lines and apply `normalize_json_line` to each non-empty line,
        which drops keys in _DROP_KEYS and any keys ending with suffixes in _DROP_SUFFIXES or starting with prefixes in _DROP_PREFIXES.
      - Preserve the presence/absence of a trailing newline to avoid accidental diffs.
      - Non-JSONL files are returned unchanged.
    Returns a new dict; input is not mutated.
    """
    out: dict[str, bytes] = {}
    for rel, blob in file_map.items():
        if rel.endswith(".jsonl"):
            try:
                text = blob.decode("utf-8")
            except Exception:
                # If decoding fails, pass through unchanged
                out[rel] = blob
                continue
            lines = text.splitlines()
            norm_lines = normalize_json_lines(lines)
            trailing = "\n" if text.endswith("\n") else ""
            out[rel] = ("\n".join(norm_lines) + trailing).encode("utf-8")
        else:
            out[rel] = blob
    return out


# ---------------------------------------------------------------------------
# PR72 helpers: routing, IO, hashing, and snapshot collection
# ---------------------------------------------------------------------------

__all__ = [
    "normalize_json_line",
    "normalize_json_lines",
    "normalize_logs_dir",
    "normalize_log_bytes_for_identity",
    "route_logs",
    "read_lines",
    "hash_file",
    "collect_snapshots_from_apply",
    "hash_snapshots",
    "purge_dir",
    "_strip_perf_and_quality",
    "_strip_perf_and_quality_and_graph",
]

# ---------------------------------------------------------------------------
# PR72 helpers: routing, IO, hashing, and snapshot collection
# ---------------------------------------------------------------------------

def _strip_perf_and_quality(cfg: dict) -> dict:
    """
    Return a shallowly cleaned copy of cfg with:
      - top-level 'perf' subtree removed (disabled-path knobs)
      - 't2.quality' subtree removed if present
    Deterministic and side-effect free (input not mutated).
    """
    x = copy.deepcopy(cfg)
    # Drop perf.*
    x.pop("perf", None)
    # Drop t2.quality.*
    t2 = x.get("t2")
    if isinstance(t2, dict):
        t2.pop("quality", None)
    return x

def _strip_perf_and_quality_and_graph(cfg: dict) -> dict:
    """
    Same as _strip_perf_and_quality but also removes the top-level 'graph' subtree.
    Used for disabled-path identity tests where 'graph.enabled=false' must be inert.
    """
    x = _strip_perf_and_quality(cfg)
    x.pop("graph", None)
    return x


def route_logs(monkeypatch, path: str | Path) -> str:
    """Monkeypatch the logs_dir() to a deterministic location and return it."""
    p = str(path)
    monkeypatch.setattr("clematis.io.paths.logs_dir", lambda: p)
    return p


def read_lines(dir_path: str | Path, name: str) -> list[str]:
    """Read a JSONL file as raw lines (byte order preserved). Missing file → []."""
    fp = Path(dir_path) / name
    if not fp.exists():
        return []
    # Preserve exact bytes→unicode mapping using utf-8 without extra stripping
    return fp.read_text(encoding="utf-8").splitlines()


def hash_file(path: str | Path, *, chunk: int = 1 << 20) -> str:
    """SHA256 of a file's bytes; returns hex digest. Missing file → empty string."""
    p = Path(path)
    if not p.exists():
        return ""
    h = hashlib.sha256()
    with p.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def collect_snapshots_from_apply(logs_dir: str | Path) -> list[Path]:
    """Return snapshot file paths in the order they appear in apply.jsonl (if any)."""
    fp = Path(logs_dir) / "apply.jsonl"
    if not fp.exists():
        return []
    out: list[Path] = []
    for ln in fp.read_text(encoding="utf-8").splitlines():
        if not ln.strip():
            continue
        try:
            obj = json.loads(ln)
        except Exception:
            continue
        snap = obj.get("snapshot")
        if isinstance(snap, str):
            out.append(Path(snap))
    return out


def hash_snapshots(paths: Iterable[Path]) -> list[str]:
    """SHA256 digest list for existing snapshot files, in given order."""
    digs: list[str] = []
    for p in paths:
        digs.append(hash_file(p))
    return digs


def purge_dir(path: str | Path) -> None:
    """Remove a directory tree if it exists, then recreate it (empty)."""
    p = Path(path)
    if p.exists():
        shutil.rmtree(p)
    p.mkdir(parents=True, exist_ok=True)
