from __future__ import annotations

from typing import Any, Dict


def cfg_get(obj: Any, path: list[Any], default: Any = None) -> Any:
    cur = obj
    for idx, key in enumerate(path):
        try:
            if isinstance(cur, dict):
                cur = cur.get(key, {} if idx < len(path) - 1 else default)
            else:
                cur = getattr(cur, key)
        except Exception:
            return default
    return cur


def ensure_dict(obj: Any) -> Dict[str, Any]:
    if isinstance(obj, dict):
        return dict(obj)
    if hasattr(obj, "__dict__"):
        try:
            return dict(obj.__dict__)
        except Exception:
            return {}
    return {}


def metrics_gate_on(cfg: Any) -> bool:
    if isinstance(cfg, dict):
        perf = cfg.get("perf") or {}
    else:
        perf = getattr(cfg, "perf", {}) or {}
    if not bool(perf.get("enabled", False)):
        return False
    metrics_cfg = perf.get("metrics") or {}
    return bool(metrics_cfg.get("report_memory", False))


def quality_cfg_snapshot(cfg_obj: Any) -> Dict[str, Any]:
    perf_enabled = bool(cfg_get(cfg_obj, ["perf", "enabled"], False))
    report_memory = bool(cfg_get(cfg_obj, ["perf", "metrics", "report_memory"], False))
    perf_trace_dir = cfg_get(cfg_obj, ["perf", "metrics", "trace_dir"], None)
    if isinstance(perf_trace_dir, str):
        perf_trace_dir = perf_trace_dir.strip() or None

    q_enabled = bool(cfg_get(cfg_obj, ["t2", "quality", "enabled"], False))
    q_shadow = bool(cfg_get(cfg_obj, ["t2", "quality", "shadow"], False))
    q_trace_dir = str(cfg_get(cfg_obj, ["t2", "quality", "trace_dir"], "logs/quality"))
    q_redact = bool(cfg_get(cfg_obj, ["t2", "quality", "redact"], True))

    metrics = {"report_memory": report_memory}
    if perf_trace_dir:
        metrics["trace_dir"] = perf_trace_dir

    return {
        "perf": {"enabled": perf_enabled, "metrics": metrics},
        "t2": {
            "quality": {
                "enabled": q_enabled,
                "shadow": q_shadow,
                "trace_dir": q_trace_dir,
                "redact": q_redact,
            }
        },
    }
