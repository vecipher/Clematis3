
from __future__ import annotations
from dataclasses import asdict
from typing import Any, Dict, Tuple
import os

try:
    import yaml
except ImportError:
    yaml = None

from ..engine.types import Config


# ---- small helpers --------------------------------------------------------

def _dict(obj: Any) -> Dict[str, Any]:
    return obj if isinstance(obj, dict) else {}

def _parse_bool_env(v: str) -> bool:
    return str(v).strip().lower() in {"1", "true", "yes", "on"}

def _apply_perf_env_overrides(cfg: Any) -> Any:
    """
    Merge CI env overrides into the loaded config (no effect if env vars absent).
    Supported:
      - PERF_T2_DTYPE=fp32|fp16              -> perf.t2.embed_store_dtype  (and cfg.t2.embed_store_dtype as convenience)
      - PERF_T2_PRECOMPUTE_NORMS=true|false  -> perf.t2.precompute_norms   (and cfg.t2.precompute_norms as convenience)
    This only touches the 'perf.t2' sub-tree and keeps behavior safe when 'perf' is absent.
    """
    try:
        dtype_env = os.getenv("PERF_T2_DTYPE")
        norms_env = os.getenv("PERF_T2_PRECOMPUTE_NORMS")
        if not dtype_env and norms_env is None:
            return cfg

        # Ensure we can hang a 'perf' tree even if Config dataclass doesn't define it
        if not hasattr(cfg, "perf") or getattr(cfg, "perf") is None:
            setattr(cfg, "perf", {})
        perf = getattr(cfg, "perf")
        if not isinstance(perf, dict):
            # be defensive; do nothing if it's an unexpected type
            return cfg
        t2 = perf.setdefault("t2", {})

        if dtype_env:
            vv = dtype_env.strip().lower()
            if vv in {"fp16", "fp32"}:
                t2["embed_store_dtype"] = vv
                # convenience mirror onto cfg.t2 if it's a dict
                if hasattr(cfg, "t2") and isinstance(cfg.t2, dict):
                    cfg.t2["embed_store_dtype"] = vv
        if norms_env is not None:
            bn = _parse_bool_env(norms_env)
            t2["precompute_norms"] = bn
            if hasattr(cfg, "t2") and isinstance(cfg.t2, dict):
                cfg.t2["precompute_norms"] = bn
    except Exception:
        # Never break loads due to env handling
        return cfg
    return cfg


# ---- loader ---------------------------------------------------------------

def load_config(path: str | None = None) -> Config:
    """
    Load YAML config if available; otherwise return defaults.
    Behavior:
      * Recognized top-level keys set directly on Config: k_surface, surface_method, t1, t2, t3, t4, budgets, flags, scheduler.
      * Unknown keys (e.g., 'perf') are attached as attributes on the Config instance if possible.
      * Safe CI env overrides for PR33.5: PERF_T2_DTYPE, PERF_T2_PRECOMPUTE_NORMS.
    """
    # If PyYAML unavailable or file missing, return defaults.
    if not path or (yaml is None):
        cfg = Config()
        return _apply_perf_env_overrides(cfg)

    try:
        with open(path, "r", encoding="utf-8") as f:
            data: Dict[str, Any] = yaml.safe_load(f) or {}
        cfg = Config()

        recognized = {"k_surface", "surface_method", "t1", "t2", "t3", "t4", "budgets", "flags", "scheduler"}
        # Set recognized sections (shallow assign to match previous behavior)
        for k in recognized:
            if k in data:
                setattr(cfg, k, data[k])

        # Attach any unknown top-level keys (e.g., 'perf') onto cfg for downstream access
        for k, v in data.items():
            if k not in recognized:
                try:
                    setattr(cfg, k, v)
                except Exception:
                    # If dataclass prohibits setting, skip silently (keeps old behavior)
                    pass

        # Apply minimal env overrides for Gate C matrix
        cfg = _apply_perf_env_overrides(cfg)
        return cfg
    except FileNotFoundError:
        cfg = Config()
        return _apply_perf_env_overrides(cfg)
