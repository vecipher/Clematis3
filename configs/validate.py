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

__all__ = ["validate_config", "validate_config_verbose"]


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
    # GEL defaults (contracts + PR22)
    "graph": {
        "enabled": False,
        "coactivation_threshold": 0.20,
        "observe_top_k": 64,
        "pair_cap_per_obs": 2048,
        "update": {
            "mode": "additive",          # additive | proportional
            "alpha": 0.02,
            "clamp_min": -1.0,
            "clamp_max": 1.0,
        },
        "decay": {
            "half_life_turns": 200,
            "floor": 0.0,
        },
        # tolerated by contracts (no behavior in PR22)
        "merge": {"enabled": False, "min_size": 3},
        "split": {"enabled": False},
    },
}


# ------------------------------
# Allowed keys, namespaces, and suggestion helpers (for PR17)
# ------------------------------

# Known cache namespaces for orchestrator-level cache coherence
KNOWN_CACHE_NAMESPACES = {"t2:semantic"}

# Allowed key sets per section
ALLOWED_TOP = {"t1", "t2", "t3", "t4", "graph", "k_surface", "surface_method", "budgets", "flags"}
ALLOWED_T1 = {"cache", "iter_cap", "queue_budget", "node_budget", "decay", "edge_type_mult", "radius_cap"}
ALLOWED_T2 = {"backend", "k_retrieval", "sim_threshold", "cache", "ranking",
              "tiers", "exact_recent_days", "clusters_top_m", "owner_scope",
              "residual_cap_per_turn", "lancedb", "archive"}
ALLOWED_T3 = {"max_rag_loops", "max_ops_per_turn", "backend",
              "tokens", "temp", "allow_reflection", "dialogue", "policy", "llm"}
ALLOWED_T4 = {
    "enabled", "delta_norm_cap_l2", "novelty_cap_per_node", "churn_cap_edges",
    "cooldowns", "weight_min", "weight_max", "snapshot_every_n_turns", "snapshot_dir",
    "cache_bust_mode", "cache"
}
ALLOWED_CACHE_FIELDS = {"enabled", "namespaces", "max_entries", "ttl_sec", "ttl_s"}
ALLOWED_RANKING_FIELDS = {"alpha_sim", "beta_recency", "gamma_importance"}

ALLOWED_GRAPH = {"enabled", "coactivation_threshold", "observe_top_k", "pair_cap_per_obs", "update", "decay", "merge", "split"}
ALLOWED_GRAPH_UPDATE = {"mode", "alpha", "clamp_min", "clamp_max"}
ALLOWED_GRAPH_DECAY = {"half_life_turns", "floor"}
ALLOWED_GRAPH_MERGE = {"enabled", "min_size"}
ALLOWED_GRAPH_SPLIT = {"enabled"}

def _lev(a: str, b: str) -> int:
    """Tiny Levenshtein distance (edit distance) for did-you-mean suggestions."""
    la, lb = len(a), len(b)
    dp = list(range(lb + 1))
    for i, ca in enumerate(a, 1):
        prev = dp[0]
        dp[0] = i
        for j, cb in enumerate(b, 1):
            ins = dp[j] + 1
            dele = dp[j - 1] + 1
            sub = prev + (0 if ca == cb else 1)
            prev, dp[j] = dp[j], min(ins, dele, sub)
    return dp[-1]

def _suggest_key(bad: str, allowed: set[str]) -> str | None:
    """Return closest allowed key within distance ≤2, else None."""
    best_key, best_dist = None, 99
    for k in allowed:
        d = _lev(bad, k)
        if d < best_dist:
            best_key, best_dist = k, d
    return best_key if best_dist <= 2 else None


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

    errors: List[str] = []

    # Capture raw sections for precedence/suggestions (user-provided view)
    raw_t1 = _ensure_dict(cfg_in.get("t1"))
    raw_t2 = _ensure_dict(cfg_in.get("t2"))
    raw_t3 = _ensure_dict(cfg_in.get("t3"))
    raw_t4 = _ensure_dict(cfg_in.get("t4"))
    raw_t1_cache = _ensure_dict(raw_t1.get("cache"))
    raw_t2_cache = _ensure_dict(raw_t2.get("cache"))
    raw_t2_ranking = _ensure_dict(raw_t2.get("ranking"))
    raw_t4_cache = _ensure_dict(raw_t4.get("cache"))
    raw_graph = _ensure_dict(cfg_in.get("graph"))
    raw_graph_update = _ensure_dict(raw_graph.get("update"))
    raw_graph_decay = _ensure_dict(raw_graph.get("decay"))
    raw_graph_merge = _ensure_dict(raw_graph.get("merge"))
    raw_graph_split = _ensure_dict(raw_graph.get("split"))

    # Unknown key detection (top-level and per-section), with suggestions
    for k in cfg_in.keys():
        if k not in ALLOWED_TOP:
            sug = _suggest_key(k, ALLOWED_TOP)
            if sug:
                _err(errors, k, f"unknown top-level key (did you mean '{sug}')")
            else:
                _err(errors, k, "unknown top-level key")

    for k in raw_t1.keys():
        if k not in ALLOWED_T1:
            sug = _suggest_key(k, ALLOWED_T1)
            hint = f" (did you mean '{sug}')" if sug else ""
            _err(errors, f"t1.{k}", f"unknown key{hint}")

    for k in raw_t2.keys():
        if k not in ALLOWED_T2:
            sug = _suggest_key(k, ALLOWED_T2)
            hint = f" (did you mean '{sug}')" if sug else ""
            _err(errors, f"t2.{k}", f"unknown key{hint}")

    for k in raw_t3.keys():
        if k not in ALLOWED_T3:
            sug = _suggest_key(k, ALLOWED_T3)
            hint = f" (did you mean '{sug}')" if sug else ""
            _err(errors, f"t3.{k}", f"unknown key{hint}")

    for k in raw_t4.keys():
        if k not in ALLOWED_T4:
            sug = _suggest_key(k, ALLOWED_T4)
            hint = f" (did you mean '{sug}')" if sug else ""
            _err(errors, f"t4.{k}", f"unknown key{hint}")

    for k in raw_t1_cache.keys():
        if k not in ALLOWED_CACHE_FIELDS:
            sug = _suggest_key(k, ALLOWED_CACHE_FIELDS)
            hint = f" (did you mean '{sug}')" if sug else ""
            _err(errors, f"t1.cache.{k}", f"unknown key{hint}")

    for k in raw_t2_cache.keys():
        if k not in ALLOWED_CACHE_FIELDS:
            sug = _suggest_key(k, ALLOWED_CACHE_FIELDS)
            hint = f" (did you mean '{sug}')" if sug else ""
            _err(errors, f"t2.cache.{k}", f"unknown key{hint}")

    for k in raw_t2_ranking.keys():
        if k not in ALLOWED_RANKING_FIELDS:
            sug = _suggest_key(k, ALLOWED_RANKING_FIELDS)
            hint = f" (did you mean '{sug}')" if sug else ""
            _err(errors, f"t2.ranking.{k}", f"unknown key{hint}")

    for k in raw_t4_cache.keys():
        if k not in ALLOWED_CACHE_FIELDS:
            sug = _suggest_key(k, ALLOWED_CACHE_FIELDS)
            hint = f" (did you mean '{sug}')" if sug else ""
            _err(errors, f"t4.cache.{k}", f"unknown key{hint}")

    # Graph (GEL) unknown key checks
    for k in raw_graph.keys():
        if k not in ALLOWED_GRAPH:
            sug = _suggest_key(k, ALLOWED_GRAPH)
            hint = f" (did you mean '{sug}')" if sug else ""
            _err(errors, f"graph.{k}", f"unknown key{hint}")

    for k in raw_graph_update.keys():
        if k not in ALLOWED_GRAPH_UPDATE:
            sug = _suggest_key(k, ALLOWED_GRAPH_UPDATE)
            hint = f" (did you mean '{sug}')" if sug else ""
            _err(errors, f"graph.update.{k}", f"unknown key{hint}")

    for k in raw_graph_decay.keys():
        if k not in ALLOWED_GRAPH_DECAY:
            sug = _suggest_key(k, ALLOWED_GRAPH_DECAY)
            hint = f" (did you mean '{sug}')" if sug else ""
            _err(errors, f"graph.decay.{k}", f"unknown key{hint}")

    for k in raw_graph_merge.keys():
        if k not in ALLOWED_GRAPH_MERGE:
            sug = _suggest_key(k, ALLOWED_GRAPH_MERGE)
            hint = f" (did you mean '{sug}')" if sug else ""
            _err(errors, f"graph.merge.{k}", f"unknown key{hint}")

    for k in raw_graph_split.keys():
        if k not in ALLOWED_GRAPH_SPLIT:
            sug = _suggest_key(k, ALLOWED_GRAPH_SPLIT)
            hint = f" (did you mean '{sug}')" if sug else ""
            _err(errors, f"graph.split.{k}", f"unknown key{hint}")

    merged = _deep_merge(cfg_in, DEFAULTS)

    # Keep raw user-provided subdicts for precedence-sensitive fields
    raw_t4_cache = _ensure_dict(raw_t4.get("cache"))

    # ---- T1 ----
    t1 = _ensure_subdict(merged, "t1")
    c1 = _ensure_subdict(t1, "cache")
    # types/coercions
    c1["max_entries"] = _coerce_int(c1.get("max_entries", 512))
    # TTL alias precedence: user ttl_s > user ttl_sec > current/default
    if "ttl_s" in raw_t1_cache:
        c1["ttl_s"] = _coerce_int(raw_t1_cache.get("ttl_s"), 300)
    elif "ttl_sec" in raw_t1_cache:
        c1["ttl_s"] = _coerce_int(raw_t1_cache.get("ttl_sec"), 300)
    else:
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
    # TTL alias precedence: user ttl_s > user ttl_sec > current/default
    if "ttl_s" in raw_t2_cache:
        c2["ttl_s"] = _coerce_int(raw_t2_cache.get("ttl_s"), 300)
    elif "ttl_sec" in raw_t2_cache:
        c2["ttl_s"] = _coerce_int(raw_t2_cache.get("ttl_sec"), 300)
    else:
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
    # Enforce inclusive bounds for weights
    if not (-1.0 <= t4["weight_min"] <= 1.0):
        _err(errors, "t4.weight_min", "must be in [-1.0, 1.0]")
    if not (-1.0 <= t4["weight_max"] <= 1.0):
        _err(errors, "t4.weight_max", "must be in [-1.0, 1.0]")

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
    # Validate namespaces against the known set
    for ns in c4["namespaces"]:
        if ns not in KNOWN_CACHE_NAMESPACES:
            _err(errors, f"t4.cache.namespaces[{ns}]", f"unknown namespace (allowed: {sorted(KNOWN_CACHE_NAMESPACES)})")
    # write-back: ensure normalized cache is attached
    t4["cache"] = c4


    # ---- GRAPH (GEL) ----
    g = _ensure_subdict(merged, "graph")
    g["enabled"] = _coerce_bool(g.get("enabled", False))

    # Scalars
    g["coactivation_threshold"] = _coerce_float(g.get("coactivation_threshold", 0.20))
    if not (0.0 <= g["coactivation_threshold"] <= 1.0):
        _err(errors, "graph.coactivation_threshold", "must be in [0, 1]")

    g["observe_top_k"] = _coerce_int(g.get("observe_top_k", 64))
    if g["observe_top_k"] < 1:
        _err(errors, "graph.observe_top_k", "must be >= 1")

    g["pair_cap_per_obs"] = _coerce_int(g.get("pair_cap_per_obs", 2048))
    if g["pair_cap_per_obs"] < 0:
        _err(errors, "graph.pair_cap_per_obs", "must be >= 0")

    # Update sub-block
    gu = _ensure_subdict(g, "update")
    gu_mode = str(gu.get("mode", "additive"))
    if gu_mode not in {"additive", "proportional"}:
        _err(errors, "graph.update.mode", "must be one of {additive,proportional}")
    gu["mode"] = gu_mode
    gu["alpha"] = _coerce_float(gu.get("alpha", 0.02))
    if not (gu["alpha"] > 0.0):
        _err(errors, "graph.update.alpha", "must be > 0")
    gu["clamp_min"] = _coerce_float(gu.get("clamp_min", -1.0))
    gu["clamp_max"] = _coerce_float(gu.get("clamp_max", 1.0))
    if not (gu["clamp_min"] < gu["clamp_max"]):
        _err(errors, "graph.update.clamp_min/clamp_max", "must satisfy clamp_min < clamp_max")
    g["update"] = gu

    # Decay sub-block
    gd = _ensure_subdict(g, "decay")
    gd["half_life_turns"] = _coerce_int(gd.get("half_life_turns", 200))
    if gd["half_life_turns"] < 1:
        _err(errors, "graph.decay.half_life_turns", "must be >= 1")
    gd["floor"] = _coerce_float(gd.get("floor", 0.0))
    if gd["floor"] < 0:
        _err(errors, "graph.decay.floor", "must be >= 0")
    # floor should not exceed clamp_max if both present
    try:
        if gd["floor"] > gu["clamp_max"]:
            _err(errors, "graph.decay.floor", "must be <= graph.update.clamp_max")
    except Exception:
        pass
    g["decay"] = gd

    # Merge & split blocks (tolerated for contracts; minimal checks)
    gm = _ensure_subdict(g, "merge")
    if "enabled" in gm:
        gm["enabled"] = _coerce_bool(gm.get("enabled"))
    if "min_size" in gm:
        gm["min_size"] = _coerce_int(gm.get("min_size"))
        if gm["min_size"] < 2:
            _err(errors, "graph.merge.min_size", "must be >= 2")
    g["merge"] = gm

    gs = _ensure_subdict(g, "split")
    if "enabled" in gs:
        gs["enabled"] = _coerce_bool(gs.get("enabled"))
    g["split"] = gs

    # If we collected errors, raise a single ValueError with all messages (stable order)
    if errors:
        raise ValueError("\n".join(errors))

    # Return merged/normalized config
    merged["t1"] = t1
    merged["t2"] = t2
    merged["t3"] = t3
    merged["t4"] = t4
    merged["graph"] = g
    return merged


# ------------------------------
# Verbose API: return warnings as a second value
# ------------------------------
from typing import Tuple

def validate_config_verbose(cfg: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
    """
    Validate and normalize configuration, returning (normalized_cfg, warnings).

    Warnings currently include:
    - Duplicate cache namespace overlap between t2.stage cache and t4.orchestrator cache.
    """
    normalized = validate_config(cfg)  # will raise ValueError on errors

    warnings: List[str] = []
    try:
        t2c = _ensure_dict(_ensure_dict(normalized.get("t2")).get("cache"))
        t4c = _ensure_dict(_ensure_dict(normalized.get("t4")).get("cache"))
        if _coerce_bool(t2c.get("enabled", True)) and "t2:semantic" in (t4c.get("namespaces") or []):
            warnings.append("W[t4.cache.namespaces]: duplicate-cache-namespace 't2:semantic' overlaps with stage cache (t2.cache.enabled=true); deterministic but TTLs may be confusing.")
    except Exception:
        # Be conservative: never crash on warning collection
        pass

    return normalized, warnings
