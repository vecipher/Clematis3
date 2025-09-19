"""
Lightweight configuration validation and normalization for Clematis3.

Public API:
    validate_config(cfg: dict) -> dict

- Raises ValueError with clear messages (field paths + constraints) on invalid input.
- Returns a **new** normalized dict; the input is not mutated.
- No external dependencies.
"""
from __future__ import annotations
from typing import Any, Dict, List

__all__ = ["validate_config"]


# ------------------------------
# Utilities
# ------------------------------

def _ensure_dict(x: Any) -> Dict[str, Any]:
    if isinstance(x, dict):
        return dict(x)
    if hasattr(x, "__dict__"):
        try:
            return dict(x.__dict__)
        except Exception:
            return {}
    return {}


def _coerce_bool(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    if isinstance(v, str):
        s = v.strip().lower()
        if s in {"1", "true", "yes", "on"}:
            return True
        if s in {"0", "false", "no", "off"}:
            return False
    return False


def _coerce_int(v: Any, default: int = 0) -> int:
    try:
        return int(v)
    except Exception:
        return int(default)


def _coerce_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return float(default)


def _deep_merge(dst: Dict[str, Any], src: Dict[str, Any]) -> Dict[str, Any]:
    """Merge keys from src into dst (shallow for non-dicts, deep for dicts) without mutating inputs."""
    out = dict(dst)
    for k, v in (src or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        elif k not in out:
            out[k] = v
    return out


# ------------------------------
# Defaults
# ------------------------------

DEFAULTS: Dict[str, Any] = {
    "t1": {
        "cache": {"max_entries": 512, "ttl_s": 300},
    },
    "t2": {
        "backend": "inmemory",
        "k_retrieval": 10,
        "sim_threshold": 0.0,
        "cache": {"max_entries": 512, "ttl_s": 300},
        "ranking": {"alpha_sim": 1.0, "beta_recency": 0.0, "gamma_importance": 0.0},
    },
    "t3": {
        "max_rag_loops": 1,
        "max_ops_per_turn": 8,
        "backend": "rulebased",
    },
    "t4": {
        "enabled": True,
        # Filter caps
        "delta_norm_cap_l2": 1.5,
        "novelty_cap_per_node": 0.3,
        "churn_cap_edges": 64,
        "cooldowns": {},
        # Apply + snapshots
        "weight_min": -1.0,
        "weight_max": 1.0,
        "snapshot_every_n_turns": 1,
        "snapshot_dir": "./.data/snapshots",
        # Cache coherency
        "cache_bust_mode": "on-apply",
        "cache": {
            "enabled": True,
            "namespaces": ["t2:semantic"],
            "max_entries": 512,
            "ttl_sec": 600,
        },
    },
}


# ------------------------------
# Validation helpers
# ------------------------------

def _err(errors: List[str], path: str, msg: str) -> None:
    errors.append(f"{path} {msg}")


def _ensure_subdict(cfg: Dict[str, Any], key: str) -> Dict[str, Any]:
    v = _ensure_dict(cfg.get(key))
    if key not in cfg:
        cfg[key] = v
    return v


# ------------------------------
# Main validator
# ------------------------------

def validate_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate and normalize a configuration dictionary.

    Returns a NEW dict with defaults merged and certain fields coerced.
    Raises ValueError on invalid configurations with actionable messages.
    """
    # Normalize top-level shape and merge defaults non-destructively
    cfg_in = _ensure_dict(cfg)
    merged = _deep_merge(cfg_in, DEFAULTS)

    # Keep raw user-provided subdicts for precedence-sensitive fields
    raw_t4 = _ensure_dict(cfg_in.get("t4"))
    raw_t4_cache = _ensure_dict(raw_t4.get("cache"))

    errors: List[str] = []

    # ---- T1 ----
    t1 = _ensure_subdict(merged, "t1")
    c1 = _ensure_subdict(t1, "cache")
    # types/coercions
    c1["max_entries"] = _coerce_int(c1.get("max_entries", 512))
    c1["ttl_s"] = _coerce_int(c1.get("ttl_s", 300))
    # optional numeric knobs in some repos
    if "iter_cap" in t1:
        t1["iter_cap"] = _coerce_int(t1.get("iter_cap"))
        if t1["iter_cap"] < 0:
            _err(errors, "t1.iter_cap", "must be >= 0")
    if "queue_budget" in t1:
        t1["queue_budget"] = _coerce_int(t1.get("queue_budget"))
        if t1["queue_budget"] < 0:
            _err(errors, "t1.queue_budget", "must be >= 0")
    if "node_budget" in t1:
        t1["node_budget"] = _coerce_float(t1.get("node_budget"))
        if t1["node_budget"] <= 0:
            _err(errors, "t1.node_budget", "must be > 0")
    if c1["max_entries"] < 0:
        _err(errors, "t1.cache.max_entries", "must be >= 0")
    if c1["ttl_s"] < 0:
        _err(errors, "t1.cache.ttl_s", "must be >= 0")
    # write-back: ensure normalized cache is attached
    t1["cache"] = c1

    # ---- T2 ----
    t2 = _ensure_subdict(merged, "t2")
    backend = str(t2.get("backend", "inmemory"))
    if backend not in {"inmemory", "lancedb"}:
        _err(errors, "t2.backend", f"must be one of {{inmemory,lancedb}}, got: {backend!r}")
    t2["k_retrieval"] = _coerce_int(t2.get("k_retrieval", 10))
    if t2["k_retrieval"] < 1:
        _err(errors, "t2.k_retrieval", "must be >= 1")
    t2["sim_threshold"] = _coerce_float(t2.get("sim_threshold", 0.0))
    if not (-1.0 <= t2["sim_threshold"] <= 1.0):
        _err(errors, "t2.sim_threshold", "must be in [-1.0, 1.0]")

    c2 = _ensure_subdict(t2, "cache")
    c2["max_entries"] = _coerce_int(c2.get("max_entries", 512))
    c2["ttl_s"] = _coerce_int(c2.get("ttl_s", 300))
    if c2["max_entries"] < 0:
        _err(errors, "t2.cache.max_entries", "must be >= 0")
    if c2["ttl_s"] < 0:
        _err(errors, "t2.cache.ttl_s", "must be >= 0")

    r2 = _ensure_subdict(t2, "ranking")
    for k in ("alpha_sim", "beta_recency", "gamma_importance"):
        r2[k] = _coerce_float(r2.get(k, 0.0))
        if not (0.0 <= r2[k] <= 1.0):
            _err(errors, f"t2.ranking.{k}", "must be in [0, 1]")
    # write-back: ensure normalized subdicts are attached
    t2["cache"] = c2
    t2["ranking"] = r2

    # ---- T3 ----
    t3 = _ensure_subdict(merged, "t3")
    t3["max_rag_loops"] = _coerce_int(t3.get("max_rag_loops", 1))
    if t3["max_rag_loops"] not in {0, 1}:
        _err(errors, "t3.max_rag_loops", "must be 0 or 1 (only one-shot supported)")
    t3["max_ops_per_turn"] = _coerce_int(t3.get("max_ops_per_turn", 8))
    if not (1 <= t3["max_ops_per_turn"] <= 16):
        _err(errors, "t3.max_ops_per_turn", "must be in [1, 16]")
    t3_backend = str(t3.get("backend", "rulebased"))
    if t3_backend not in {"rulebased", "llm"}:
        _err(errors, "t3.backend", f"must be one of {{rulebased,llm}}, got: {t3_backend!r}")

    # ---- T4 ----
    t4 = _ensure_subdict(merged, "t4")
    t4["enabled"] = _coerce_bool(t4.get("enabled", True))
    t4["delta_norm_cap_l2"] = _coerce_float(t4.get("delta_norm_cap_l2", 1.5))
    if t4["delta_norm_cap_l2"] <= 0:
        _err(errors, "t4.delta_norm_cap_l2", "must be > 0")
    t4["novelty_cap_per_node"] = _coerce_float(t4.get("novelty_cap_per_node", 0.3))
    if not (0 < t4["novelty_cap_per_node"] <= 1.0):
        _err(errors, "t4.novelty_cap_per_node", "must be in (0, 1]")
    t4["churn_cap_edges"] = _coerce_int(t4.get("churn_cap_edges", 64))
    if t4["churn_cap_edges"] < 0:
        _err(errors, "t4.churn_cap_edges", "must be >= 0")

    # cooldowns map
    cooldowns = _ensure_dict(t4.get("cooldowns", {}))
    for k, v in cooldowns.items():
        if not isinstance(k, str):
            _err(errors, "t4.cooldowns", "keys must be strings (op kinds)")
        vi = _coerce_int(v)
        if vi < 0:
            _err(errors, f"t4.cooldowns[{k}]", "must be >= 0")
        cooldowns[k] = vi
    t4["cooldowns"] = cooldowns

    # apply/snapshot
    t4["weight_min"] = _coerce_float(t4.get("weight_min", -1.0))
    t4["weight_max"] = _coerce_float(t4.get("weight_max", 1.0))
    if not (t4["weight_min"] < t4["weight_max"]):
        _err(errors, "t4.weight_min/weight_max", "must satisfy weight_min < weight_max")

    t4["snapshot_every_n_turns"] = _coerce_int(t4.get("snapshot_every_n_turns", 1))
    if t4["snapshot_every_n_turns"] < 1:
        _err(errors, "t4.snapshot_every_n_turns", "must be >= 1")
    snap_dir = t4.get("snapshot_dir", "./.data/snapshots")
    if not isinstance(snap_dir, str) or not snap_dir.strip():
        _err(errors, "t4.snapshot_dir", "must be a non-empty string path")
    else:
        t4["snapshot_dir"] = snap_dir

    # cache coherency
    t4["cache_bust_mode"] = str(t4.get("cache_bust_mode", "on-apply"))
    if t4["cache_bust_mode"] not in {"none", "on-apply"}:
        _err(errors, "t4.cache_bust_mode", "must be one of {none,on-apply}")

    c4 = _ensure_subdict(t4, "cache")
    c4["enabled"] = _coerce_bool(c4.get("enabled", True))
    c4["max_entries"] = _coerce_int(c4.get("max_entries", 512))
    if c4["max_entries"] < 0:
        _err(errors, "t4.cache.max_entries", "must be >= 0")

    # Normalize TTL with correct precedence: user ttl_sec > user ttl_s > current/default
    if "ttl_sec" in raw_t4_cache:
        c4["ttl_sec"] = _coerce_int(raw_t4_cache.get("ttl_sec"), 600)
    elif "ttl_s" in raw_t4_cache:
        c4["ttl_sec"] = _coerce_int(raw_t4_cache.get("ttl_s"), 600)
    else:
        c4["ttl_sec"] = _coerce_int(c4.get("ttl_sec", 600), 600)
    if c4["ttl_sec"] < 0:
        _err(errors, "t4.cache.ttl_sec", "must be >= 0")

    # namespaces list[str]
    namespaces = c4.get("namespaces", ["t2:semantic"])
    if not isinstance(namespaces, list) or not all(isinstance(x, str) for x in namespaces):
        _err(errors, "t4.cache.namespaces", "must be a list of strings")
        # normalize to default if invalid
        c4["namespaces"] = ["t2:semantic"]
    else:
        c4["namespaces"] = namespaces
    # write-back: ensure normalized cache is attached
    t4["cache"] = c4

    # If we collected errors, raise a single ValueError with all messages (stable order)
    if errors:
        raise ValueError("\n".join(errors))

    # Return merged/normalized config
    merged["t1"] = t1
    merged["t2"] = t2
    merged["t3"] = t3
    merged["t4"] = t4
    return merged
