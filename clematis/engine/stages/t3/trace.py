from __future__ import annotations
from typing import Any, Dict


def _triple_gate(cfg_root: Dict[str, Any]) -> bool:
    try:
        perf = (cfg_root.get("perf") if isinstance(cfg_root, dict) else None) or {}
        metrics = (perf.get("metrics") or {})
        if not bool(metrics.get("enabled", False)):
            return False
        t3_cfg = (cfg_root.get("t3") or {}) if isinstance(cfg_root, dict) else {}
        trace_cfg = (t3_cfg.get("trace") or {}) if isinstance(t3_cfg, dict) else {}
        return bool(trace_cfg.get("enabled", False))
    except Exception:
        return False


def emit_trace(cfg_snapshot: Dict[str, Any], prompt: str, bundle: Dict[str, Any], meta: Dict[str, Any]) -> None:
    if not _triple_gate(cfg_snapshot):
        return
    try:
        logs = meta.get("state_logs") if isinstance(meta, dict) else None
        if isinstance(logs, list):
            reason = meta.get("trace_reason") if isinstance(meta, dict) else None
            if not reason:
                reason = (
                    cfg_snapshot.get("perf", {})
                    .get("metrics", {})
                    .get("trace_reason")
                    if isinstance(cfg_snapshot, dict)
                    else None
                )
            reason = reason or "t3-shadow"
            logs.append(
                {
                    "t3_trace": {
                        "reason": str(reason),
                        "prompt": str(prompt),
                        "bundle_keys": sorted(bundle.keys()),
                    }
                }
            )
    except Exception:
        pass
