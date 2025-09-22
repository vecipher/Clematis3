# clematis/engine/util/metrics.py
from __future__ import annotations
from typing import Any, Mapping, MutableMapping

def gate_on(cfg: Any) -> bool:
    """
    True only when perf.enabled && perf.metrics.report_memory.
    Accepts either a plain dict or an object with .perf.
    """
    perf = {}
    if isinstance(cfg, dict):
        perf = cfg.get("perf") or {}
    else:
        perf = getattr(cfg, "perf", {}) or {}
    if not bool(perf.get("enabled", False)):
        return False
    m = perf.get("metrics") or {}
    return bool(m.get("report_memory", False))

def emit(cfg: Any, evt: MutableMapping[str, Any], key: str, value: Any) -> MutableMapping[str, Any]:
    """Emit a single metric under evt['metrics'][key] if the gate is ON. No-op otherwise."""
    if not gate_on(cfg):
        return evt
    m = evt.setdefault("metrics", {})
    if isinstance(m, dict):
        m[key] = value
    return evt

def emit_many(cfg: Any, evt: MutableMapping[str, Any], items: Mapping[str, Any] | None) -> MutableMapping[str, Any]:
    """Emit a dict of metrics if the gate is ON. No-op when items is falsy or the gate is OFF."""
    if not gate_on(cfg) or not items:
        return evt
    m = evt.setdefault("metrics", {})
    if isinstance(m, dict):
        m.update(items)
    return evt

def maybe(evt: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return evt['metrics'] if present, else {} (read-only convenience, no side effects)."""
    m = evt.get("metrics") if isinstance(evt, dict) else None
    return m if isinstance(m, dict) else {}