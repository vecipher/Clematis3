# stages/t2_quality_trace.py
from __future__ import annotations
from pathlib import Path
import json
import os
import hashlib
from typing import Any, Dict, List, Optional
import unicodedata
import logging

_logger = logging.getLogger(__name__)
__TRACE_WRITE_ERROR_WARNED = False


def _norm_query(q: str) -> str:
    # Unicode normalize, trim, collapse internal whitespace, lowercase
    q = unicodedata.normalize("NFKC", q)
    q = " ".join(q.split())  # splits on any whitespace and rejoins with single space
    return q.lower()


def _query_id(q: str) -> str:
    return _sha1(_norm_query(q))


TRACE_SCHEMA_VERSION = 1


def _stable_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def _sha256(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def _config_digest(cfg: Dict[str, Any]) -> str:
    """
    Digest only the knobs that change behavior/semantics for quality/traces.
    Excludes path-like and purely cosmetic fields (e.g., trace_dir).
    """
    perf = cfg.get("perf", {}) or {}
    q = cfg.get("t2", {}).get("quality", {}) or {}
    sub = {
        "perf": {
            "enabled": bool(perf.get("enabled", False)),
            "metrics": {
                "report_memory": bool((perf.get("metrics") or {}).get("report_memory", False))
            },
        },
        "t2": {
            "quality": {
                "enabled": bool(q.get("enabled", False)),
                "shadow": bool(q.get("shadow", False)),
                # redact does affect the *shape* of traces, so include it
                "redact": bool(q.get("redact", True)),
                # NOTE: trace_dir intentionally excluded (non-semantic, env-specific)
            }
        },
    }
    return _sha256(_stable_json(sub))


def _git_sha() -> str:
    return os.environ.get("CLEMATIS_GIT_SHA", "unknown")


def _redact_items(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    # Minimal, conservative redaction for shadow mode.
    redacted = []
    for it in items:
        r = dict(it)
        # Common fields to scrub if present
        for k in ("text", "snippet", "title", "content"):
            if k in r:
                r[k] = "[REDACTED]"
        redacted.append(r)
    return redacted


def _derive_trace_dir(cfg: Dict[str, Any]) -> Path:
    """
    Prefer perf.metrics.trace_dir if set, otherwise t2.quality.trace_dir,
    otherwise default to logs/quality.
    """
    try:
        perf = cfg.get("perf") or {}
        metrics = perf.get("metrics") or {}
        td = metrics.get("trace_dir")
        if isinstance(td, str) and td.strip():
            return Path(td)
    except Exception:
        pass
    try:
        qcfg = (cfg.get("t2") or {}).get("quality", {}) or {}
        td2 = qcfg.get("trace_dir")
        if isinstance(td2, str) and td2.strip():
            return Path(td2)
    except Exception:
        pass
    return Path("logs/quality")


def emit_trace(
    cfg: Dict[str, Any], query: str, items: List[Dict[str, Any]], meta: Dict[str, Any]
) -> None:
    """
    Write a single JSONL record to trace_dir/rq_traces.jsonl.
    Must be called only when triple-gate is satisfied.
    Never raises; on failure, logs once and returns.
    """
    qcfg = (cfg.get("t2") or {}).get("quality", {}) or {}
    redact = bool(qcfg.get("redact", True))
    record = {
        "trace_schema_version": TRACE_SCHEMA_VERSION,
        "git_sha": _git_sha(),
        "config_digest": _config_digest(cfg),
        "clock": 0,
        "seed": 0,
        "query": "[REDACTED]" if redact else query,
        "query_id": _query_id(query),
        "meta": meta or {},
        "items": _redact_items(items) if redact else items,
    }
    try:
        trace_dir = _derive_trace_dir(cfg)
        trace_dir.mkdir(parents=True, exist_ok=True)
        with (trace_dir / "rq_traces.jsonl").open("a", encoding="utf-8", newline="\n") as f:
            f.write(_stable_json(record) + "\n")
    except Exception:
        global __TRACE_WRITE_ERROR_WARNED
        if not __TRACE_WRITE_ERROR_WARNED:
            try:
                _logger.exception("emit_trace: failed to write rq_traces.jsonl (one-time warning)")
            finally:
                __TRACE_WRITE_ERROR_WARNED = True
        # swallow to keep pipeline non-fatal
        return


# Back-compat/alias for callers expecting emit_quality_trace(...)
def emit_quality_trace(
    cfg: Dict[str, Any],
    query: str,
    items: List[Dict[str, Any]],
    meta: Optional[Dict[str, Any]] = None,
) -> None:
    emit_trace(cfg, query, items, meta or {})
