from __future__ import annotations
from dataclasses import asdict
from typing import Any, Dict, Tuple
import os
import json, hashlib

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

def _canon_perf(perf: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build a canonical perf dict with explicit defaults and stable key order.
    This removes structural/insertion-order differences between implicit and explicit OFF.
    """
    par = perf.get("parallel", {}) if isinstance(perf, dict) else {}
    met = perf.get("metrics", {}) if isinstance(perf, dict) else {}
    # Return a NEW dict with canonical insertion order and filled defaults
    return {
        "parallel": {
            "enabled": bool((par.get("enabled", False))),
            "max_workers": int((par.get("max_workers", 1))),
            "t1": bool((par.get("t1", False))),
            "t2": bool((par.get("t2", False))),
            "agents": bool((par.get("agents", False))),
        },
        "metrics": {
            "enabled": bool((met.get("enabled", False))),
            "report_memory": bool((met.get("report_memory", False))),
        },
    }

def _materialize_perf_on_cfg(cfg: Any) -> Any:
    """
    Ensure cfg.perf exists and is canonicalized to identity-off defaults when missing.
    """
    try:
        cur_perf = getattr(cfg, "perf", None)
    except Exception:
        cur_perf = None
    if not isinstance(cur_perf, dict):
        cur_perf = {}
    # Canonicalize and attach
    try:
        setattr(cfg, "perf", _canon_perf(cur_perf))
    except Exception:
        # If cfg is restrictive, try to use its __dict__ directly
        d = getattr(cfg, "__dict__", None)
        if isinstance(d, dict):
            d["perf"] = _canon_perf(cur_perf)
    return cfg


# ---- identity helpers ----------------------------------------------------

def _to_plain_dict(cfg: Any) -> Dict[str, Any]:
    """
    Best-effort conversion of a Config-like object to a plain dict that includes
    both dataclass fields and any dynamically attached attributes (e.g., 'perf').
    """
    # Prefer __dict__ because we attach unknown keys onto the instance
    d: Dict[str, Any] = {}
    try:
        if hasattr(cfg, "__dict__") and isinstance(cfg.__dict__, dict):
            d.update(cfg.__dict__)
        else:
            try:
                # fall back to dataclasses.asdict if possible
                d.update(asdict(cfg))  # type: ignore[arg-type]
            except Exception:
                if isinstance(cfg, dict):
                    d.update(cfg)
    except Exception:
        if isinstance(cfg, dict):
            d.update(cfg)
    return d


def config_identity_basis(cfg: Any) -> Dict[str, Any]:
    """
    Canonical, stable view of config for hashing/seeding and safe logging.
    - Excludes non-identity inputs (argv, raw YAML, paths, env snapshots).
    - Includes 'perf' in canonicalized form so implicit OFF == explicit OFF.
    """
    d = _to_plain_dict(cfg)
    # Drop non-identity fields if present
    for k in ("raw_yaml", "argv", "source_path", "env"):
        d.pop(k, None)
    # Ensure 'perf' is present in a canonical shape
    perf = d.get("perf", {})
    d["perf"] = _canon_perf(perf if isinstance(perf, dict) else {})
    return d


def config_identity_sha(cfg: Any) -> str:
    """
    Stable SHA-256 of the identity basis (keys sorted, compact separators).
    Use this for seeds and for any 'config_sha' fields in logs.
    """
    basis = config_identity_basis(cfg)
    payload = json.dumps(basis, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


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
        cfg = _apply_perf_env_overrides(cfg)
        cfg = _materialize_perf_on_cfg(cfg)
        return cfg

    try:
        with open(path, "r", encoding="utf-8") as f:
            data: Dict[str, Any] = yaml.safe_load(f) or {}
        cfg = Config()

        recognized = {
            "k_surface",
            "surface_method",
            "t1",
            "t2",
            "t3",
            "t4",
            "budgets",
            "flags",
            "scheduler",
        }
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
        cfg = _materialize_perf_on_cfg(cfg)
        return cfg
    except FileNotFoundError:
        cfg = Config()
        cfg = _apply_perf_env_overrides(cfg)
        cfg = _materialize_perf_on_cfg(cfg)
        return cfg
