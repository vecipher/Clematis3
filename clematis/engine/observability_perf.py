

from __future__ import annotations

"""
Perf/quality observability helpers.

All functions here write *non-canonical* diagnostics intended for experiments,
perf tracing, and quality analyses. Outputs are routed under a dedicated
"perf" subdirectory so they are excluded from identity comparisons.

Canonical logs (e.g., t1.jsonl, t2.jsonl, t4.jsonl, health.jsonl) must NOT be
written or modified by code in this module.
"""

import json
from pathlib import Path
from typing import Iterable, Mapping, Any

__all__ = [
    "PERF_DIRNAME",
    "perf_dir",
    "write_perf_jsonl",
    "write_perf_jsonl_many",
]

# Subdirectory name used to segregate non-canonical diagnostics.
PERF_DIRNAME = "perf"


def perf_dir(root: Path) -> Path:
    """
    Return the directory path under which perf/diagnostic JSONL files should be written.
    Callers should pass the *logs* directory (or equivalent run output dir) as `root`.
    """
    return root / PERF_DIRNAME


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def write_perf_jsonl(out_root: Path, name: str, record: Mapping[str, Any]) -> None:
    """
    Append a single JSON object as a line to logs/perf/{name}.jsonl.

    Parameters
    ----------
    out_root : Path
        The base logs/output directory for the current run.
    name : str
        Logical file stem (e.g., "t2_hybrid", "gel", "native_t1").
    record : Mapping[str, Any]
        JSON-serializable mapping; keys will be written with sort_keys=True.

    Notes
    -----
    - Uses '\n' as newline to keep cross-OS consistency.
    - Identity tests ignore this directory; use for diagnostics only.
    """
    path = perf_dir(out_root) / f"{name}.jsonl"
    _ensure_parent(path)
    line = json.dumps(record, sort_keys=True, ensure_ascii=False)
    # Append (non-atomic) is acceptable for diagnostics; canonical logs use their own discipline.
    with path.open("a", encoding="utf-8", newline="\n") as f:
        f.write(line + "\n")


def write_perf_jsonl_many(out_root: Path, name: str, records: Iterable[Mapping[str, Any]]) -> None:
    """
    Append multiple JSON objects as lines to logs/perf/{name}.jsonl.

    This is a convenience wrapper; see `write_perf_jsonl` for semantics.
    """
    path = perf_dir(out_root) / f"{name}.jsonl"
    _ensure_parent(path)
    with path.open("a", encoding="utf-8", newline="\n") as f:
        for rec in records:
            f.write(json.dumps(rec, sort_keys=True, ensure_ascii=False) + "\n")
