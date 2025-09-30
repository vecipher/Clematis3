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


# Back-compat alias for planned naming in PR67
def metrics_gate_on(cfg: Any) -> bool:
    """Alias of gate_on(cfg). True only when perf.enabled && perf.metrics.report_memory."""
    return gate_on(cfg)


def emit(cfg: Any, evt: MutableMapping[str, Any], key: str, value: Any) -> MutableMapping[str, Any]:
    """Emit a single metric under evt['metrics'][key] if the gate is ON. No-op otherwise."""
    if not gate_on(cfg):
        return evt
    m = evt.setdefault("metrics", {})
    if isinstance(m, dict):
        m[key] = value
    return evt


def emit_many(
    cfg: Any, evt: MutableMapping[str, Any], items: Mapping[str, Any] | None
) -> MutableMapping[str, Any]:
    """Emit a dict of metrics if the gate is ON. No-op when items is falsy or the gate is OFF."""
    if not gate_on(cfg) or not items:
        return evt
    m = evt.setdefault("metrics", {})
    if isinstance(m, dict):
        m.update(items)
    return evt


# Update a metrics dict in-place with items if the metrics gate is ON.
def update_gated(cfg: Any, dst: MutableMapping[str, Any], items: Mapping[str, Any] | None) -> MutableMapping[str, Any]:
    """
    Update a metrics dict in-place with `items` if the metrics gate is ON.
    Unlike `emit_many`, this expects `dst` to be the metrics mapping itself.
    Returns the same mapping for chaining.
    """
    if gate_on(cfg) and items:
        dst.update(items)
    return dst


# Convenience wrapper to match stage call-sites: update `dst` with `new` if gate is ON.
def metrics_update_gated(dst: MutableMapping[str, Any], new: Mapping[str, Any], cfg: Any) -> None:
    """Convenience wrapper to match stage call-sites: update `dst` with `new` if gate is ON."""
    if gate_on(cfg) and new:
        dst.update(new)


def maybe(evt: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return evt['metrics'] if present, else {} (read-only convenience, no side effects)."""
    m = evt.get("metrics") if isinstance(evt, dict) else None
    return m if isinstance(m, dict) else {}
