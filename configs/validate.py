"""
"Lightweight" configuration validation and normalization for Clematis3.

Public API:
    validate_config(cfg: dict) -> dict

- Raises ValueError with clear messages (field paths + constraints) on invalid input.
- Returns a **new** normalized dict; the input is not mutated.
- No external dependencies.
"""
from __future__ import annotations
from typing import Any, Dict, List

__all__ = ["validate_config", "validate_config_verbose", "validate_config_api"]


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
        "hybrid": {
            "enabled": False,
            "use_graph": True,
            "anchor_top_m": 8,
            "walk_hops": 1,                 # 1 or 2
            "edge_threshold": 0.10,         # [0,1]
            "lambda_graph": 0.25,           # [0,1]
            "damping": 0.50,                # [0,1], used when walk_hops=2
            "degree_norm": "none",          # none | invdeg
            "max_bonus": 0.50,              # >= 0
            "k_max": 128,                   # >= 1 (cap work)
        },
    },
    "t3": {
        "max_rag_loops": 1,
        "max_ops_per_turn": 8,
        "backend": "rulebased",
        "llm": {
            "provider": "fixture",
            "model": "qwen3:4b-instruct-q4_K_M",
            "endpoint": "http://localhost:11434/api/generate",
            "max_tokens": 256,
            "temp": 0.2,
            "timeout_ms": 10000,
            "fixtures": {
                "enabled": True,
                "path": "fixtures/llm/qwen_small.jsonl",
            },
        },
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
    # GEL defaults (contracts + PR22, PR24)
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
        # PR24: richer defaults for contracts
        "merge": {
            "enabled": False,
            "min_size": 3,
            "min_avg_w": 0.20,
            "max_diameter": 2,
            "cap_per_turn": 4,
        },
        "split": {
            "enabled": False,
            "weak_edge_thresh": 0.05,
            "min_component_size": 2,
            "cap_per_turn": 4,
        },
        "promotion": {
            "enabled": False,
            "label_mode": "lexmin",     # lexmin | concat_k
            "topk_label_ids": 3,
            "attach_weight": 0.5,        # [-1,1]
            "cap_per_turn": 2,
        },
    },
    "scheduler": {
        "enabled": False,
        "policy": "round_robin",      # or "fair_queue"
        "quantum_ms": 20,             # used in PR26
        "budgets": {
            "t1_pops": None,          # int or None
            "t1_iters": 50,
            "t2_k": 64,               # cap on results USED (not fetched)
            "t3_ops": 3,
            "wall_ms": 200,           # hard per-slice wall (PR26)
        },
        "fairness": {
            "max_consecutive_turns": 1,  # enforced in PR27
            "aging_ms": 200,             # bucket size for fair-queue priority
        },
    },
}


# ------------------------------
# Allowed keys, namespaces, and suggestion helpers (for PR17)
# ------------------------------

# Known cache namespaces for orchestrator-level cache coherence
KNOWN_CACHE_NAMESPACES = {"t2:semantic"}

# Allowed key sets per section
ALLOWED_TOP = {"t1", "t2", "t3", "t4", "graph", "k_surface", "surface_method", "budgets", "flags", "scheduler", "perf"}
ALLOWED_T1 = {"cache", "iter_cap", "queue_budget", "node_budget", "decay", "edge_type_mult", "radius_cap"}
ALLOWED_T2 = {"backend", "k_retrieval", "sim_threshold", "cache", "ranking", "hybrid",
              "tiers", "exact_recent_days", "clusters_top_m", "owner_scope",
              "residual_cap_per_turn", "lancedb", "archive", "quality", "embed_root", "reader_batch", "reader"}
ALLOWED_T3 = {"max_rag_loops", "max_ops_per_turn", "backend",
              "tokens", "temp", "allow_reflection", "dialogue", "policy", "llm"}
ALLOWED_T3_LLM = {"provider", "model", "endpoint", "max_tokens", "temp", "timeout_ms", "fixtures"}
ALLOWED_T3_LLM_FIXTURES = {"enabled", "path"}
ALLOWED_T4 = {
    "enabled", "delta_norm_cap_l2", "novelty_cap_per_node", "churn_cap_edges",
    "cooldowns", "weight_min", "weight_max", "snapshot_every_n_turns", "snapshot_dir",
    "cache_bust_mode", "cache"
}
ALLOWED_CACHE_FIELDS = {"enabled", "namespaces", "max_entries", "ttl_sec", "ttl_s"}
ALLOWED_RANKING_FIELDS = {"alpha_sim", "beta_recency", "gamma_importance"}

ALLOWED_T2_HYBRID = {
    "enabled", "use_graph", "anchor_top_m", "walk_hops", "edge_threshold",
    "lambda_graph", "damping", "degree_norm", "max_bonus", "k_max",
}

# PR40: allowed keys for t2.reader
ALLOWED_T2_READER = {"mode"}

ALLOWED_GRAPH = {"enabled", "coactivation_threshold", "observe_top_k", "pair_cap_per_obs", "update", "decay", "merge", "split", "promotion"}
ALLOWED_GRAPH_UPDATE = {"mode", "alpha", "clamp_min", "clamp_max"}
ALLOWED_GRAPH_DECAY = {"half_life_turns", "floor"}
ALLOWED_GRAPH_MERGE = {"enabled", "min_size", "min_avg_w", "max_diameter", "cap_per_turn"}
ALLOWED_GRAPH_SPLIT = {"enabled", "weak_edge_thresh", "min_component_size", "cap_per_turn"}
ALLOWED_GRAPH_PROMOTION = {"enabled", "label_mode", "topk_label_ids", "attach_weight", "cap_per_turn"}
_ALLOWED_SCHED_POLICIES = {"round_robin", "fair_queue"}

# PERF (M6) allowed keys
ALLOWED_PERF = {"enabled", "t1", "t2", "snapshots", "metrics"}
ALLOWED_PERF_T1 = {"queue_cap", "dedupe_window", "cache", "caps"}
ALLOWED_PERF_T1_CACHE = {"max_entries", "max_bytes"}
ALLOWED_PERF_T1_CAPS = {"frontier", "visited"}
ALLOWED_PERF_T2 = {"embed_dtype", "embed_store_dtype", "precompute_norms", "cache", "reader"}
ALLOWED_PERF_T2_CACHE = {"max_entries", "max_bytes"}
ALLOWED_PERF_T2_READER = {"partitions"}
ALLOWED_PERF_T2_READER_PARTITIONS = {"enabled", "layout", "path", "by"}
ALLOWED_PERF_SNAP = {"compression", "level", "delta_mode", "every_n_turns"}
ALLOWED_PERF_METRICS = {"report_memory"}

ALLOWED_T2_QUALITY = {"enabled", "shadow", "trace_dir", "redact", "normalizer", "aliasing", "lexical", "fusion", "mmr", "cache"}
ALLOWED_T2_QUALITY_NORMALIZER = {"enabled", "case", "unicode", "stopwords", "stemmer", "min_token_len"}
ALLOWED_T2_QUALITY_ALIASING = {"enabled", "map_path", "max_expansions_per_token"}
# Accept both canonical nested BM25 keys and flat convenience keys used in PR37 examples
ALLOWED_T2_QUALITY_LEXICAL = {"enabled", "bm25", "bm25_k1", "bm25_b", "stopwords"}
ALLOWED_T2_QUALITY_BM25 = {"k1", "b", "doclen_floor"}
# Fusion supports an explicit 'mode' (PR37: only 'score_interp' allowed)
ALLOWED_T2_QUALITY_FUSION = {"enabled", "mode", "alpha_semantic", "score_norm"}
ALLOWED_T2_QUALITY_MMR = {"enabled", "lambda", "lambda_relevance", "diversity_by_owner", "diversity_by_token", "k", "k_final"}
ALLOWED_T2_QUALITY_CACHE = {"salt"}

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

def _validate_config_normalize_impl(cfg: Dict[str, Any]) -> Dict[str, Any]:
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
    raw_t3_llm = _ensure_dict(raw_t3.get("llm"))
    raw_t3_llm_fixtures = _ensure_dict(raw_t3_llm.get("fixtures"))
    raw_t4 = _ensure_dict(cfg_in.get("t4"))
    raw_t1_cache = _ensure_dict(raw_t1.get("cache"))
    raw_t2_cache = _ensure_dict(raw_t2.get("cache"))
    raw_t2_ranking = _ensure_dict(raw_t2.get("ranking"))
    raw_t2_hybrid = _ensure_dict(raw_t2.get("hybrid"))
    # PR40: capture t2.reader
    raw_t2_reader = _ensure_dict(raw_t2.get("reader"))
    raw_t4_cache = _ensure_dict(raw_t4.get("cache"))
    raw_graph = _ensure_dict(cfg_in.get("graph"))
    raw_graph_update = _ensure_dict(raw_graph.get("update"))
    raw_graph_decay = _ensure_dict(raw_graph.get("decay"))
    raw_graph_merge = _ensure_dict(raw_graph.get("merge"))
    raw_graph_split = _ensure_dict(raw_graph.get("split"))
    raw_graph_promotion = _ensure_dict(raw_graph.get("promotion"))

    # PERF and T2.quality raw captures
    raw_perf = _ensure_dict(cfg_in.get("perf"))
    raw_perf_t1 = _ensure_dict(raw_perf.get("t1"))
    raw_perf_t1_cache = _ensure_dict(raw_perf_t1.get("cache"))
    raw_perf_t1_caps = _ensure_dict(raw_perf_t1.get("caps"))
    raw_perf_t2 = _ensure_dict(raw_perf.get("t2"))
    raw_perf_t2_cache = _ensure_dict(raw_perf_t2.get("cache"))
    raw_perf_t2_reader = _ensure_dict(raw_perf_t2.get("reader"))
    raw_perf_t2_reader_partitions = _ensure_dict(raw_perf_t2_reader.get("partitions"))
    raw_perf_snap = _ensure_dict(raw_perf.get("snapshots"))
    raw_perf_metrics = _ensure_dict(raw_perf.get("metrics"))

    raw_t2_quality = _ensure_dict(raw_t2.get("quality"))
    raw_q_norm = _ensure_dict(raw_t2_quality.get("normalizer"))
    raw_q_alias = _ensure_dict(raw_t2_quality.get("aliasing"))
    raw_q_lex = _ensure_dict(raw_t2_quality.get("lexical"))
    raw_q_bm25 = _ensure_dict(raw_q_lex.get("bm25"))
    raw_q_fusion = _ensure_dict(raw_t2_quality.get("fusion"))
    raw_q_mmr = _ensure_dict(raw_t2_quality.get("mmr"))
    raw_q_cache = _ensure_dict(raw_t2_quality.get("cache"))

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

    for k in raw_t3_llm.keys():
        if k not in ALLOWED_T3_LLM:
            sug = _suggest_key(k, ALLOWED_T3_LLM)
            hint = f" (did you mean '{sug}')" if sug else ""
            _err(errors, f"t3.llm.{k}", f"unknown key{hint}")

    for k in raw_t3_llm_fixtures.keys():
        if k not in ALLOWED_T3_LLM_FIXTURES:
            sug = _suggest_key(k, ALLOWED_T3_LLM_FIXTURES)
            hint = f" (did you mean '{sug}')" if sug else ""
            _err(errors, f"t3.llm.fixtures.{k}", f"unknown key{hint}")

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

    for k in raw_t2_hybrid.keys():
        if k not in ALLOWED_T2_HYBRID:
            sug = _suggest_key(k, ALLOWED_T2_HYBRID)
            hint = f" (did you mean '{sug}')" if sug else ""
            _err(errors, f"t2.hybrid.{k}", f"unknown key{hint}")

    # PR40: unknown key check for t2.reader
    for k in raw_t2_reader.keys():
        if k not in ALLOWED_T2_READER:
            sug = _suggest_key(k, ALLOWED_T2_READER)
            hint = f" (did you mean '{sug}')" if sug else ""
            _err(errors, f"t2.reader.{k}", f"unknown key{hint}")

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

    for k in raw_graph_promotion.keys():
        if k not in ALLOWED_GRAPH_PROMOTION:
            sug = _suggest_key(k, ALLOWED_GRAPH_PROMOTION)
            hint = f" (did you mean '{sug}')" if sug else ""
            _err(errors, f"graph.promotion.{k}", f"unknown key{hint}")

    # PERF unknown key checks
    if raw_perf:
        for k in raw_perf.keys():
            if k not in ALLOWED_PERF:
                sug = _suggest_key(k, ALLOWED_PERF)
                hint = f" (did you mean '{sug}')" if sug else ""
                _err(errors, f"perf.{k}", f"unknown key{hint}")
        for k in raw_perf_t1.keys():
            if k not in ALLOWED_PERF_T1:
                sug = _suggest_key(k, ALLOWED_PERF_T1); hint = f" (did you mean '{sug}')" if sug else ""
                _err(errors, f"perf.t1.{k}", f"unknown key{hint}")
        for k in raw_perf_t1_cache.keys():
            if k not in ALLOWED_PERF_T1_CACHE:
                sug = _suggest_key(k, ALLOWED_PERF_T1_CACHE); hint = f" (did you mean '{sug}')" if sug else ""
                _err(errors, f"perf.t1.cache.{k}", f"unknown key{hint}")
        for k in raw_perf_t1_caps.keys():
            if k not in ALLOWED_PERF_T1_CAPS:
                sug = _suggest_key(k, ALLOWED_PERF_T1_CAPS); hint = f" (did you mean '{sug}')" if sug else ""
                _err(errors, f"perf.t1.caps.{k}", f"unknown key{hint}")
        for k in raw_perf_t2.keys():
            if k not in ALLOWED_PERF_T2:
                sug = _suggest_key(k, ALLOWED_PERF_T2); hint = f" (did you mean '{sug}')" if sug else ""
                _err(errors, f"perf.t2.{k}", f"unknown key{hint}")
        for k in raw_perf_t2_cache.keys():
            if k not in ALLOWED_PERF_T2_CACHE:
                sug = _suggest_key(k, ALLOWED_PERF_T2_CACHE); hint = f" (did you mean '{sug}')" if sug else ""
                _err(errors, f"perf.t2.cache.{k}", f"unknown key{hint}")
        for k in raw_perf_t2_reader.keys():
            if k not in ALLOWED_PERF_T2_READER:
                sug = _suggest_key(k, ALLOWED_PERF_T2_READER); hint = f" (did you mean '{sug}')" if sug else ""
                _err(errors, f"perf.t2.reader.{k}", f"unknown key{hint}")
        for k in raw_perf_t2_reader_partitions.keys():
            if k not in ALLOWED_PERF_T2_READER_PARTITIONS:
                sug = _suggest_key(k, ALLOWED_PERF_T2_READER_PARTITIONS); hint = f" (did you mean '{sug}')" if sug else ""
                _err(errors, f"perf.t2.reader.partitions.{k}", f"unknown key{hint}")
        for k in raw_perf_snap.keys():
            if k not in ALLOWED_PERF_SNAP:
                sug = _suggest_key(k, ALLOWED_PERF_SNAP); hint = f" (did you mean '{sug}')" if sug else ""
                _err(errors, f"perf.snapshots.{k}", f"unknown key{hint}")
        for k in raw_perf_metrics.keys():
            if k not in ALLOWED_PERF_METRICS:
                sug = _suggest_key(k, ALLOWED_PERF_METRICS); hint = f" (did you mean '{sug}')" if sug else ""
                _err(errors, f"perf.metrics.{k}", f"unknown key{hint}")

    # T2.quality unknown key checks (RQ prep only)
    if raw_t2_quality:
        for k in raw_t2_quality.keys():
            if k not in ALLOWED_T2_QUALITY:
                sug = _suggest_key(k, ALLOWED_T2_QUALITY); hint = f" (did you mean '{sug}')" if sug else ""
                _err(errors, f"t2.quality.{k}", f"unknown key{hint}")
        for k in raw_q_norm.keys():
            if k not in ALLOWED_T2_QUALITY_NORMALIZER:
                sug = _suggest_key(k, ALLOWED_T2_QUALITY_NORMALIZER); hint = f" (did you mean '{sug}')" if sug else ""
                _err(errors, f"t2.quality.normalizer.{k}", f"unknown key{hint}")
        for k in raw_q_alias.keys():
            if k not in ALLOWED_T2_QUALITY_ALIASING:
                sug = _suggest_key(k, ALLOWED_T2_QUALITY_ALIASING); hint = f" (did you mean '{sug}')" if sug else ""
                _err(errors, f"t2.quality.aliasing.{k}", f"unknown key{hint}")
        for k in raw_q_lex.keys():
            if k not in ALLOWED_T2_QUALITY_LEXICAL:
                sug = _suggest_key(k, ALLOWED_T2_QUALITY_LEXICAL); hint = f" (did you mean '{sug}')" if sug else ""
                _err(errors, f"t2.quality.lexical.{k}", f"unknown key{hint}")
        for k in raw_q_bm25.keys():
            if k not in ALLOWED_T2_QUALITY_BM25:
                sug = _suggest_key(k, ALLOWED_T2_QUALITY_BM25); hint = f" (did you mean '{sug}')" if sug else ""
                _err(errors, f"t2.quality.lexical.bm25.{k}", f"unknown key{hint}")
        for k in raw_q_fusion.keys():
            if k not in ALLOWED_T2_QUALITY_FUSION:
                sug = _suggest_key(k, ALLOWED_T2_QUALITY_FUSION); hint = f" (did you mean '{sug}')" if sug else ""
                _err(errors, f"t2.quality.fusion.{k}", f"unknown key{hint}")
        for k in raw_q_mmr.keys():
            if k not in ALLOWED_T2_QUALITY_MMR:
                sug = _suggest_key(k, ALLOWED_T2_QUALITY_MMR); hint = f" (did you mean '{sug}')" if sug else ""
                _err(errors, f"t2.quality.mmr.{k}", f"unknown key{hint}")
        for k in raw_q_cache.keys():
            if k not in ALLOWED_T2_QUALITY_CACHE:
                sug = _suggest_key(k, ALLOWED_T2_QUALITY_CACHE); hint = f" (did you mean '{sug}')" if sug else ""
                _err(errors, f"t2.quality.cache.{k}", f"unknown key{hint}")

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
    # Optional PR33 knobs (only validate if provided)
    if "reader_batch" in t2:
        t2["reader_batch"] = _coerce_int(t2.get("reader_batch", 8192))
        if t2["reader_batch"] < 1:
            _err(errors, "t2.reader_batch", "must be >= 1")
    if "embed_root" in t2:
        er = t2.get("embed_root")
        if not isinstance(er, str) or not er.strip():
            _err(errors, "t2.embed_root", "must be a non-empty string path")
        else:
            t2["embed_root"] = er

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

    # hybrid re-ranker (PR23)
    h2 = _ensure_subdict(t2, "hybrid")
    h2["enabled"] = _coerce_bool(h2.get("enabled", False))
    h2["use_graph"] = _coerce_bool(h2.get("use_graph", True))

    h2["anchor_top_m"] = _coerce_int(h2.get("anchor_top_m", 8))
    if h2["anchor_top_m"] < 1:
        _err(errors, "t2.hybrid.anchor_top_m", "must be >= 1")

    h2["walk_hops"] = _coerce_int(h2.get("walk_hops", 1))
    if h2["walk_hops"] not in {1, 2}:
        _err(errors, "t2.hybrid.walk_hops", "must be 1 or 2")

    h2["edge_threshold"] = _coerce_float(h2.get("edge_threshold", 0.10))
    if not (0.0 <= h2["edge_threshold"] <= 1.0):
        _err(errors, "t2.hybrid.edge_threshold", "must be in [0, 1]")

    h2["lambda_graph"] = _coerce_float(h2.get("lambda_graph", 0.25))
    if not (0.0 <= h2["lambda_graph"] <= 1.0):
        _err(errors, "t2.hybrid.lambda_graph", "must be in [0, 1]")

    h2["damping"] = _coerce_float(h2.get("damping", 0.50))
    if not (0.0 <= h2["damping"] <= 1.0):
        _err(errors, "t2.hybrid.damping", "must be in [0, 1]")

    deg = str(h2.get("degree_norm", "none"))
    if deg not in {"none", "invdeg"}:
        _err(errors, "t2.hybrid.degree_norm", "must be one of {none,invdeg}")
    h2["degree_norm"] = deg

    h2["max_bonus"] = _coerce_float(h2.get("max_bonus", 0.50))
    if h2["max_bonus"] < 0.0:
        _err(errors, "t2.hybrid.max_bonus", "must be >= 0")

    h2["k_max"] = _coerce_int(h2.get("k_max", 128))
    if h2["k_max"] < 1:
        _err(errors, "t2.hybrid.k_max", "must be >= 1")

    # write back
    t2["hybrid"] = h2

    # PR40: reader mode normalization
    # reader mode (PR40): flat | partition | auto
    r2_reader = _ensure_subdict(t2, "reader")
    mode_raw = str(raw_t2_reader.get("mode", "flat")).lower() if raw_t2_reader else str(r2_reader.get("mode", "flat")).lower()
    if mode_raw not in {"flat", "partition", "auto"}:
        _err(errors, "t2.reader.mode", "must be one of {flat,partition,auto}")
        mode_raw = "flat"
    r2_reader["mode"] = mode_raw
    t2["reader"] = r2_reader

    # Optional LanceDB partitions (config surface only; no runtime changes in M6)
    raw_lancedb_val = raw_t2.get("lancedb", None)
    if raw_lancedb_val is not None and not isinstance(raw_lancedb_val, dict):
        _err(errors, "t2.lancedb", "must be an object")
    raw_lancedb = _ensure_dict(raw_lancedb_val)
    if raw_lancedb:
        ldb_out: Dict[str, Any] = {}
        raw_parts_val = raw_lancedb.get("partitions", None)
        if raw_parts_val is not None and not isinstance(raw_parts_val, dict):
            _err(errors, "t2.lancedb.partitions", "must be an object")
        raw_parts = _ensure_dict(raw_parts_val)
        if raw_parts:
            pp: Dict[str, Any] = {}
            if "by" in raw_parts:
                by = raw_parts.get("by")
                if not (isinstance(by, list) and all(isinstance(x, str) for x in by)):
                    _err(errors, "t2.lancedb.partitions.by", "must be a list of strings (e.g., ['owner','quarter'])")
                else:
                    pp["by"] = list(by)
            if "shard_order" in raw_parts:
                so = str(raw_parts.get("shard_order")).lower()
                if so not in {"lex", "score"}:
                    _err(errors, "t2.lancedb.partitions.shard_order", "must be 'lex' or 'score'")
                else:
                    pp["shard_order"] = so
            if pp:
                ldb_out["partitions"] = pp
        if ldb_out:
            t2["lancedb"] = ldb_out

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

    # t3.llm normalization (config-only in M3-07; no runtime wiring yet)
    llm = _ensure_subdict(t3, "llm")
    provider = str(llm.get("provider", "fixture"))
    if provider not in {"fixture", "ollama"}:
        _err(errors, "t3.llm.provider", "must be one of {fixture,ollama}")
    llm["provider"] = provider

    model = llm.get("model", "qwen2:4b-instruct-q4_K_M")
    if not isinstance(model, str) or not model.strip():
        _err(errors, "t3.llm.model", "must be a non-empty string")
    else:
        llm["model"] = model

    endpoint = llm.get("endpoint", "http://localhost:11434/api/generate")
    if not isinstance(endpoint, str) or not endpoint.strip():
        _err(errors, "t3.llm.endpoint", "must be a non-empty string")
    else:
        llm["endpoint"] = endpoint

    llm["max_tokens"] = _coerce_int(llm.get("max_tokens", 256))
    if llm["max_tokens"] < 1:
        _err(errors, "t3.llm.max_tokens", "must be >= 1")

    llm["temp"] = _coerce_float(llm.get("temp", 0.2))
    if not (0.0 <= llm["temp"] <= 1.0):
        _err(errors, "t3.llm.temp", "must be in [0,1]")

    llm["timeout_ms"] = _coerce_int(llm.get("timeout_ms", 10000))
    if llm["timeout_ms"] < 1:
        _err(errors, "t3.llm.timeout_ms", "must be >= 1")

    fixtures = _ensure_subdict(llm, "fixtures")
    fixtures["enabled"] = _coerce_bool(fixtures.get("enabled", True))
    fpath = fixtures.get("path", "fixtures/llm/qwen_small.jsonl")
    if not isinstance(fpath, str) or not fpath.strip():
        _err(errors, "t3.llm.fixtures.path", "must be a non-empty string")
    else:
        fixtures["path"] = fpath
    llm["fixtures"] = fixtures
    t3["llm"] = llm

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

    # Merge block (PR24)
    gm = _ensure_subdict(g, "merge")
    gm["enabled"] = _coerce_bool(gm.get("enabled", False))
    if "min_size" in gm:
        gm["min_size"] = _coerce_int(gm.get("min_size"))
        if gm["min_size"] < 2:
            _err(errors, "graph.merge.min_size", "must be >= 2")
    else:
        gm["min_size"] = 3
    if "min_avg_w" in gm:
        gm["min_avg_w"] = _coerce_float(gm.get("min_avg_w"))
        if not (0.0 <= gm["min_avg_w"] <= 1.0):
            _err(errors, "graph.merge.min_avg_w", "must be in [0, 1]")
    else:
        gm["min_avg_w"] = 0.20
    if "max_diameter" in gm:
        gm["max_diameter"] = _coerce_int(gm.get("max_diameter"))
        if gm["max_diameter"] < 1:
            _err(errors, "graph.merge.max_diameter", "must be >= 1")
    else:
        gm["max_diameter"] = 2
    if "cap_per_turn" in gm:
        gm["cap_per_turn"] = _coerce_int(gm.get("cap_per_turn"))
        if gm["cap_per_turn"] < 0:
            _err(errors, "graph.merge.cap_per_turn", "must be >= 0")
    else:
        gm["cap_per_turn"] = 4
    g["merge"] = gm

    # Split block (PR24)
    gs = _ensure_subdict(g, "split")
    gs["enabled"] = _coerce_bool(gs.get("enabled", False))
    if "weak_edge_thresh" in gs:
        gs["weak_edge_thresh"] = _coerce_float(gs.get("weak_edge_thresh"))
        if not (0.0 <= gs["weak_edge_thresh"] <= 1.0):
            _err(errors, "graph.split.weak_edge_thresh", "must be in [0, 1]")
    else:
        gs["weak_edge_thresh"] = 0.05
    if "min_component_size" in gs:
        gs["min_component_size"] = _coerce_int(gs.get("min_component_size"))
        if gs["min_component_size"] < 2:
            _err(errors, "graph.split.min_component_size", "must be >= 2")
    else:
        gs["min_component_size"] = 2
    if "cap_per_turn" in gs:
        gs["cap_per_turn"] = _coerce_int(gs.get("cap_per_turn"))
        if gs["cap_per_turn"] < 0:
            _err(errors, "graph.split.cap_per_turn", "must be >= 0")
    else:
        gs["cap_per_turn"] = 4
    # Cross-field sanity: if both thresholds present, weak_edge_thresh should not exceed merge.min_avg_w
    try:
        if gs["weak_edge_thresh"] > gm["min_avg_w"]:
            _err(errors, "graph.split.weak_edge_thresh", "should be <= graph.merge.min_avg_w for consistency")
    except Exception:
        pass
    g["split"] = gs

    # Promotion block (PR24)
    gp = _ensure_subdict(g, "promotion")
    gp["enabled"] = _coerce_bool(gp.get("enabled", False))
    mode = str(gp.get("label_mode", "lexmin"))
    if mode not in {"lexmin", "concat_k"}:
        _err(errors, "graph.promotion.label_mode", "must be one of {lexmin,concat_k}")
    gp["label_mode"] = mode
    gp["topk_label_ids"] = _coerce_int(gp.get("topk_label_ids", 3))
    if gp["topk_label_ids"] < 1:
        _err(errors, "graph.promotion.topk_label_ids", "must be >= 1")
    gp["attach_weight"] = _coerce_float(gp.get("attach_weight", 0.5))
    if not (-1.0 <= gp["attach_weight"] <= 1.0):
        _err(errors, "graph.promotion.attach_weight", "must be in [-1, 1]")
    gp["cap_per_turn"] = _coerce_int(gp.get("cap_per_turn", 2))
    if gp["cap_per_turn"] < 0:
        _err(errors, "graph.promotion.cap_per_turn", "must be >= 0")
    g["promotion"] = gp

    # ---- SCHEDULER ----
    s = _ensure_subdict(merged, "scheduler")
    # enabled
    s["enabled"] = _coerce_bool(s.get("enabled", False))
    # policy
    pol = str(s.get("policy", "round_robin"))
    if pol not in _ALLOWED_SCHED_POLICIES:
        _err(errors, "scheduler.policy", f"must be one of {_ALLOWED_SCHED_POLICIES}")
    s["policy"] = pol
    # quantum_ms
    qms = _coerce_int(s.get("quantum_ms", 20))
    if qms < 1:
        _err(errors, "scheduler.quantum_ms", "must be >= 1")
    s["quantum_ms"] = qms

    # budgets block
    sb = _ensure_subdict(s, "budgets")
    def _budget_int_or_none(name: str, min_val: int) -> None:
        v = sb.get(name, None)
        if v is None:
            return
        vi = _coerce_int(v)
        if vi < min_val:
            _err(errors, f"scheduler.budgets.{name}", f"must be >= {min_val} (or null)")
        sb[name] = vi

    _budget_int_or_none("t1_pops", 0)
    _budget_int_or_none("t1_iters", 0)
    _budget_int_or_none("t2_k", 0)
    _budget_int_or_none("t3_ops", 0)
    _budget_int_or_none("wall_ms", 1)

    wall = sb.get("wall_ms")
    if isinstance(wall, int) and wall < qms:
        _err(errors, "scheduler.budgets.wall_ms", "must be >= scheduler.quantum_ms")
    s["budgets"] = sb

    # fairness block
    sf = _ensure_subdict(s, "fairness")
    sf["max_consecutive_turns"] = _coerce_int(sf.get("max_consecutive_turns", 1))
    if sf["max_consecutive_turns"] < 1:
        _err(errors, "scheduler.fairness.max_consecutive_turns", "must be >= 1")
    sf["aging_ms"] = _coerce_int(sf.get("aging_ms", 200))
    if sf["aging_ms"] < 0:
        _err(errors, "scheduler.fairness.aging_ms", "must be >= 0")
    s["fairness"] = sf

    # ---- PERF (M6; config-only) ----
    if raw_perf:
        p: Dict[str, Any] = {}
        p["enabled"] = _coerce_bool(raw_perf.get("enabled", False))

        # perf.t1
        if raw_perf_t1:
            pt1: Dict[str, Any] = {}
            # caps sub-block (PR31)
            caps_out: Dict[str, Any] = {}
            if raw_perf_t1_caps:
                if "frontier" in raw_perf_t1_caps:
                    qc = _coerce_int(raw_perf_t1_caps.get("frontier"))
                    if qc < 1: _err(errors, "perf.t1.caps.frontier", "must be >= 1")
                    caps_out["frontier"] = qc
                if "visited" in raw_perf_t1_caps:
                    vc = _coerce_int(raw_perf_t1_caps.get("visited"))
                    if vc < 1: _err(errors, "perf.t1.caps.visited", "must be >= 1")
                    caps_out["visited"] = vc
            # legacy queue_cap → caps.frontier (if caps.frontier not provided)
            if "queue_cap" in raw_perf_t1:
                qc_legacy = _coerce_int(raw_perf_t1.get("queue_cap"))
                if qc_legacy < 1: _err(errors, "perf.t1.queue_cap", "must be >= 1")
                # keep legacy key in normalized output (for transparency)
                pt1["queue_cap"] = qc_legacy
                if "frontier" not in caps_out:
                    caps_out["frontier"] = qc_legacy
            if caps_out:
                pt1["caps"] = caps_out
            # dedupe window
            if "dedupe_window" in raw_perf_t1:
                dw = _coerce_int(raw_perf_t1.get("dedupe_window"))
                if dw < 1: _err(errors, "perf.t1.dedupe_window", "must be >= 1")
                pt1["dedupe_window"] = dw
            # optional local cache fields under perf.t1
            if raw_perf_t1_cache:
                pc: Dict[str, Any] = {}
                if "max_entries" in raw_perf_t1_cache:
                    me = _coerce_int(raw_perf_t1_cache.get("max_entries"))
                    if me < 0: _err(errors, "perf.t1.cache.max_entries", "must be >= 0")
                    pc["max_entries"] = me
                if "max_bytes" in raw_perf_t1_cache:
                    mb = _coerce_int(raw_perf_t1_cache.get("max_bytes"))
                    if mb < 0: _err(errors, "perf.t1.cache.max_bytes", "must be >= 0")
                    pc["max_bytes"] = mb
                if pc: pt1["cache"] = pc
            if pt1: p["t1"] = pt1

        # perf.t2
        if raw_perf_t2:
            pt2: Dict[str, Any] = {}
            if "embed_dtype" in raw_perf_t2:
                ed = str(raw_perf_t2.get("embed_dtype")).lower()
                if ed not in {"fp32", "fp16"}:
                    _err(errors, "perf.t2.embed_dtype", "must be one of {fp32,fp16}")
                pt2["embed_dtype"] = ed
            if "embed_store_dtype" in raw_perf_t2:
                esd = str(raw_perf_t2.get("embed_store_dtype")).lower()
                if esd not in {"fp32", "fp16"}:
                    _err(errors, "perf.t2.embed_store_dtype", "must be one of {fp32,fp16}")
                pt2["embed_store_dtype"] = esd
            if "precompute_norms" in raw_perf_t2:
                pt2["precompute_norms"] = _coerce_bool(raw_perf_t2.get("precompute_norms"))
            if raw_perf_t2_cache:
                pc2: Dict[str, Any] = {}
                if "max_entries" in raw_perf_t2_cache:
                    me2 = _coerce_int(raw_perf_t2_cache.get("max_entries"))
                    if me2 < 0: _err(errors, "perf.t2.cache.max_entries", "must be >= 0")
                    pc2["max_entries"] = me2
                if "max_bytes" in raw_perf_t2_cache:
                    mb2 = _coerce_int(raw_perf_t2_cache.get("max_bytes"))
                    if mb2 < 0: _err(errors, "perf.t2.cache.max_bytes", "must be >= 0")
                    pc2["max_bytes"] = mb2
                if pc2: pt2["cache"] = pc2
            # reader (PR33): partitions config
            raw_perf_t2_reader = _ensure_dict(raw_perf_t2.get("reader"))
            if raw_perf_t2_reader:
                rd_out: Dict[str, Any] = {}
                prt = _ensure_dict(raw_perf_t2_reader.get("partitions"))
                if prt:
                    pp_out: Dict[str, Any] = {}
                    if "enabled" in prt:
                        pp_out["enabled"] = _coerce_bool(prt.get("enabled"))
                    if "layout" in prt:
                        lay = str(prt.get("layout")).lower()
                        if lay not in {"owner_quarter", "none"}:
                            _err(errors, "perf.t2.reader.partitions.layout", "must be one of {owner_quarter,none}")
                        pp_out["layout"] = lay
                    if "path" in prt:
                        pth = prt.get("path")
                        if not isinstance(pth, str) or not pth.strip():
                            _err(errors, "perf.t2.reader.partitions.path", "must be a non-empty string")
                        pp_out["path"] = pth
                    if "by" in prt:
                        byv = prt.get("by")
                        if not isinstance(byv, (list, tuple)) or not byv or not all(isinstance(x, str) and x.strip() for x in byv):
                            _err(errors, "perf.t2.reader.partitions.by", "must be a non-empty list of strings")
                        else:
                            # Restrict to known partition fields for PR33.5
                            allowed_fields = {"owner", "quarter"}
                            for fld in byv:
                                if fld not in allowed_fields:
                                    _err(errors, f"perf.t2.reader.partitions.by[{fld}]", f"unknown partition field (allowed: {sorted(allowed_fields)})")
                            pp_out["by"] = list(byv)
                    if pp_out:
                        rd_out["partitions"] = pp_out
                if rd_out:
                    pt2["reader"] = rd_out
            if pt2: p["t2"] = pt2

        # perf.snapshots
        if raw_perf_snap:
            ps: Dict[str, Any] = {}
            if "compression" in raw_perf_snap:
                comp = str(raw_perf_snap.get("compression")).lower()
                if comp not in {"none", "zstd"}:
                    _err(errors, "perf.snapshots.compression", "must be one of {none,zstd}")
                ps["compression"] = comp
            if "level" in raw_perf_snap:
                lvl = _coerce_int(raw_perf_snap.get("level"))
                if not (1 <= lvl <= 19):
                    _err(errors, "perf.snapshots.level", "must be in [1,19]")
                ps["level"] = lvl
            if "delta_mode" in raw_perf_snap:
                ps["delta_mode"] = _coerce_bool(raw_perf_snap.get("delta_mode"))
            if "every_n_turns" in raw_perf_snap:
                ent = _coerce_int(raw_perf_snap.get("every_n_turns"))
                if ent < 1: _err(errors, "perf.snapshots.every_n_turns", "must be >= 1")
                ps["every_n_turns"] = ent
            if ps: p["snapshots"] = ps

        # perf.metrics
        if raw_perf_metrics:
            pm: Dict[str, Any] = {}
            if "report_memory" in raw_perf_metrics:
                pm["report_memory"] = _coerce_bool(raw_perf_metrics.get("report_memory"))
            if pm: p["metrics"] = pm

        merged["perf"] = p

    # ---- T2.QUALITY (M6 prep only; no runtime wiring) ----
    if raw_t2_quality:
        q: Dict[str, Any] = {}
        q["enabled"] = _coerce_bool(raw_t2_quality.get("enabled", False))

        # PR36: shadow tracing controls (no-op path)
        q["shadow"] = _coerce_bool(raw_t2_quality.get("shadow", False))

        # trace_dir: must be a non-empty string; default to logs/quality
        if "trace_dir" in raw_t2_quality:
            td = raw_t2_quality.get("trace_dir")
            if not isinstance(td, str) or not td.strip():
                _err(errors, "t2.quality.trace_dir", "must be a non-empty string path")
            else:
                q["trace_dir"] = td
        else:
            q["trace_dir"] = "logs/quality"

        # privacy: redact text fields in traces by default
        q["redact"] = _coerce_bool(raw_t2_quality.get("redact", True))
        # PR37: lexical BM25 and fusion knobs (alpha)

        # lexical: defaults and bounds
        lex = q.setdefault("lexical", {})
        lex.setdefault("bm25_k1", 1.2)
        lex.setdefault("bm25_b", 0.75)
        lex.setdefault("stopwords", "en-basic")
        if not isinstance(lex["bm25_k1"], (int, float)) or lex["bm25_k1"] < 0:
            _err(errors, "t2.quality.lexical.bm25_k1", "must be a number >= 0")
        if not isinstance(lex["bm25_b"], (int, float)) or not (0.0 <= float(lex["bm25_b"]) <= 1.0):
            _err(errors, "t2.quality.lexical.bm25_b", "must be a number in [0,1]")
        if lex["stopwords"] not in ("none", "en-basic"):
            _err(errors, "t2.quality.lexical.stopwords", 'must be one of {"none","en-basic"}')

        # fusion: mode and alpha (only score_interp supported in PR37)
        fus = q.setdefault("fusion", {})
        fus.setdefault("mode", "score_interp")
        fus.setdefault("alpha_semantic", 0.6)
        if fus["mode"] != "score_interp":
            _err(errors, "t2.quality.fusion.mode", 'only "score_interp" is supported in PR37')
        try:
            alpha = float(fus["alpha_semantic"])
        except Exception:
            _err(errors, "t2.quality.fusion.alpha_semantic", "must be a number in [0,1]")
            alpha = 0.6
        if not (0.0 <= alpha <= 1.0):
            _err(errors, "t2.quality.fusion.alpha_semantic", "must be a number in [0,1]")
        else:
            fus["alpha_semantic"] = alpha
        # PR36 guard: enabling the quality path is unsupported until PR37 lands, removed for PR37
        # if q["enabled"]:
        #     _err(errors, "t2.quality.enabled", "unsupported in PR36; set to false until PR37 (lexical+fusion) lands.")

        # normalizer (PR39): default ON when quality.enabled=true
        qn: Dict[str, Any] = {}
        # Default enablement follows quality.enabled
        q_enabled_flag = _coerce_bool(raw_t2_quality.get("enabled", False))
        if "enabled" in raw_q_norm:
            qn["enabled"] = _coerce_bool(raw_q_norm.get("enabled"))
        else:
            qn["enabled"] = True if q_enabled_flag else False
        # case (only 'lower')
        if "case" in raw_q_norm:
            cs = str(raw_q_norm.get("case")).lower()
            if cs != "lower":
                _err(errors, "t2.quality.normalizer.case", "must be 'lower'")
            qn["case"] = "lower"
        else:
            qn["case"] = "lower"
        # unicode (only 'NFKC')
        if "unicode" in raw_q_norm:
            uc = str(raw_q_norm.get("unicode")).upper()
            if uc != "NFKC":
                _err(errors, "t2.quality.normalizer.unicode", "must be 'NFKC'")
            qn["unicode"] = "NFKC"
        else:
            qn["unicode"] = "NFKC"
        # Back-compat keys (accepted but not required)
        if "stopwords" in raw_q_norm:
            sw = raw_q_norm.get("stopwords")
            if not isinstance(sw, str) or not sw:
                _err(errors, "t2.quality.normalizer.stopwords", "must be a non-empty string")
            qn["stopwords"] = sw
        if "stemmer" in raw_q_norm:
            st = str(raw_q_norm.get("stemmer"))
            if st not in {"none", "porter-lite"}:
                _err(errors, "t2.quality.normalizer.stemmer", "must be one of {none,porter-lite}")
            qn["stemmer"] = st
        if "min_token_len" in raw_q_norm:
            mtl = _coerce_int(raw_q_norm.get("min_token_len"))
            if mtl < 1:
                _err(errors, "t2.quality.normalizer.min_token_len", "must be >= 1")
            qn["min_token_len"] = mtl
        # Always attach normalized normalizer (explicit defaults are helpful for downstream)
        q["normalizer"] = qn

        # aliasing
        if raw_q_alias:
            qa: Dict[str, Any] = {}
            if "enabled" in raw_q_alias:
                qa["enabled"] = _coerce_bool(raw_q_alias.get("enabled"))
            if "map_path" in raw_q_alias:
                mp = raw_q_alias.get("map_path")
                if not isinstance(mp, str) or not mp:
                    _err(errors, "t2.quality.aliasing.map_path", "must be a non-empty string path")
                qa["map_path"] = mp
            if "max_expansions_per_token" in raw_q_alias:
                mep = _coerce_int(raw_q_alias.get("max_expansions_per_token"))
                if mep < 0: _err(errors, "t2.quality.aliasing.max_expansions_per_token", "must be >= 0")
                qa["max_expansions_per_token"] = mep
            if qa: q["aliasing"] = qa

        # lexical
        if raw_q_lex:
            ql: Dict[str, Any] = {}
            if "enabled" in raw_q_lex:
                ql["enabled"] = _coerce_bool(raw_q_lex.get("enabled"))
            if raw_q_bm25:
                qb: Dict[str, Any] = {}
                if "k1" in raw_q_bm25: qb["k1"] = _coerce_float(raw_q_bm25.get("k1"))
                if "b" in raw_q_bm25:
                    bval = _coerce_float(raw_q_bm25.get("b"))
                    qb["b"] = bval
                if "doclen_floor" in raw_q_bm25:
                    dlf = _coerce_int(raw_q_bm25.get("doclen_floor"))
                    if dlf < 0: _err(errors, "t2.quality.lexical.bm25.doclen_floor", "must be >= 0")
                    qb["doclen_floor"] = dlf
                if qb: ql["bm25"] = qb
            if ql: q["lexical"] = ql

        # fusion
        if raw_q_fusion:
            qf: Dict[str, Any] = {}
            if "enabled" in raw_q_fusion:
                qf["enabled"] = _coerce_bool(raw_q_fusion.get("enabled"))
            if "alpha_semantic" in raw_q_fusion:
                qf["alpha_semantic"] = _coerce_float(raw_q_fusion.get("alpha_semantic"))
            if "score_norm" in raw_q_fusion:
                sn = str(raw_q_fusion.get("score_norm"))
                if sn not in {"zscore", "minmax"}:
                    _err(errors, "t2.quality.fusion.score_norm", "must be one of {zscore,minmax}")
                qf["score_norm"] = sn
            if qf: q["fusion"] = qf

        # mmr (PR38): accept canonical keys (lambda, k) and legacy aliases (lambda_relevance, k_final)
        if raw_q_mmr:
            qm: Dict[str, Any] = {}
            # enabled
            if "enabled" in raw_q_mmr:
                qm["enabled"] = _coerce_bool(raw_q_mmr.get("enabled"))
            # lambda: prefer canonical 'lambda', fall back to 'lambda_relevance', default 0.5 when mmr present
            lam_present = False
            lam_val = 0.5
            if "lambda" in raw_q_mmr:
                lam_present = True
                lam_val = _coerce_float(raw_q_mmr.get("lambda"))
            elif "lambda_relevance" in raw_q_mmr:
                lam_present = True
                lam_val = _coerce_float(raw_q_mmr.get("lambda_relevance"))
            # range check if provided explicitly
            if lam_present and not (0.0 <= lam_val <= 1.0):
                _err(errors, "t2.quality.mmr.lambda", "must be in [0,1]")
            # write canonical + legacy mirror (for downstream readers/tests)
            if lam_present or "enabled" in qm:
                qm["lambda"] = lam_val
                qm["lambda_relevance"] = lam_val
            # diversity flags (accepted but not required in PR38)
            if "diversity_by_owner" in raw_q_mmr:
                qm["diversity_by_owner"] = _coerce_bool(raw_q_mmr.get("diversity_by_owner"))
            if "diversity_by_token" in raw_q_mmr:
                qm["diversity_by_token"] = _coerce_bool(raw_q_mmr.get("diversity_by_token"))
            # k: prefer canonical 'k', fall back to 'k_final'
            k_present = False
            if "k" in raw_q_mmr:
                k_present = True
                kv = _coerce_int(raw_q_mmr.get("k"))
                if kv < 1:
                    _err(errors, "t2.quality.mmr.k", "must be >= 1")
                else:
                    qm["k"] = kv
                    qm["k_final"] = kv
            elif "k_final" in raw_q_mmr:
                k_present = True
                kv = _coerce_int(raw_q_mmr.get("k_final"))
                if kv < 1:
                    _err(errors, "t2.quality.mmr.k_final", "must be >= 1")
                else:
                    qm["k_final"] = kv
                    qm["k"] = kv
            if qm:
                q["mmr"] = qm

        # cache
        if raw_q_cache:
            qc: Dict[str, Any] = {}
            if "salt" in raw_q_cache:
                qc["salt"] = str(raw_q_cache.get("salt"))
            if qc: q["cache"] = qc

        # attach normalized quality only if provided
        t2["quality"] = q

    # If we collected errors, raise a single ValueError with all messages (stable order)
    if errors:
        raise ValueError("\n".join(errors))

    # Return merged/normalized config
    merged["t1"] = t1
    merged["t2"] = t2
    merged["t3"] = t3
    merged["t4"] = t4
    merged["graph"] = g
    merged["scheduler"] = s
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
    normalized = _validate_config_normalize_impl(cfg)  # will raise ValueError on errors

    warnings: List[str] = []
    try:
        t2c = _ensure_dict(_ensure_dict(normalized.get("t2")).get("cache"))
        t4c = _ensure_dict(_ensure_dict(normalized.get("t4")).get("cache"))
        if _coerce_bool(t2c.get("enabled", True)) and "t2:semantic" in (t4c.get("namespaces") or []):
            warnings.append("W[t4.cache.namespaces]: duplicate-cache-namespace 't2:semantic' overlaps with stage cache (t2.cache.enabled=true); deterministic but TTLs may be confusing.")
    except Exception:
        # Be conservative: never crash on warning collection
        pass
    try:
        sched = _ensure_dict(normalized.get("scheduler"))
        budgets = _ensure_dict(sched.get("budgets"))
        fairness = _ensure_dict(sched.get("fairness"))

        qms = _coerce_int(sched.get("quantum_ms", 20))
        t1_iters = budgets.get("t1_iters")
        t1_pops = budgets.get("t1_pops")

        # Both zero can stall T1 when the scheduler is enabled
        if t1_iters == 0 and t1_pops == 0:
            warnings.append("W[scheduler]: both t1_iters and t1_pops are 0; T1 may not progress when scheduler is enabled.")

        aging = _coerce_int(fairness.get("aging_ms", 200))
        if aging < qms:
            warnings.append("W[scheduler]: fairness.aging_ms < quantum_ms; aging tiers may be too fine-grained to have effect.")
    except Exception:
        pass
    try:
        perf = _ensure_dict(normalized.get("perf"))
        if perf:
            pt1v = _ensure_dict(perf.get("t1"))
            capsv = _ensure_dict(pt1v.get("caps"))
            perf_on = _coerce_bool(perf.get("enabled", False))
            # Warn if caps/dedupe configured while perf is disabled
            if not perf_on:
                if (_coerce_int(pt1v.get("queue_cap", 0)) > 0 or
                    _coerce_int(pt1v.get("dedupe_window", 0)) > 0 or
                    _coerce_int(capsv.get("frontier", 0)) > 0 or
                    _coerce_int(capsv.get("visited", 0)) > 0):
                    warnings.append("W[perf.t1]: caps/dedupe configured while perf.enabled=false; features remain disabled (identity path).")
                # PR32: caches configured while perf is disabled → identity path (no effect)
                pt1c = _ensure_dict(pt1v.get("cache"))
                pt2 = _ensure_dict(perf.get("t2"))
                pt2c = _ensure_dict(pt2.get("cache"))
                if (_coerce_int(pt1c.get("max_entries", 0)) > 0 or _coerce_int(pt1c.get("max_bytes", 0)) > 0):
                    warnings.append("W[perf.t1.cache]: cache configured while perf.enabled=false; cache remains disabled (identity path).")
                if (_coerce_int(pt2c.get("max_entries", 0)) > 0 or _coerce_int(pt2c.get("max_bytes", 0)) > 0):
                    warnings.append("W[perf.t2.cache]: cache configured while perf.enabled=false; cache remains disabled (identity path).")
                # PR33: reader partitions configured while perf is disabled → identity path (no effect)
                pt2r = _ensure_dict(pt2.get("reader"))
                prt = _ensure_dict(pt2r.get("partitions"))
                if (_coerce_bool(prt.get("enabled", False))
                    or ("by" in prt) or ("layout" in prt) or ("path" in prt)):
                    warnings.append("W[perf.t2.reader]: partitions configured while perf.enabled=false; reader remains disabled (identity path).")
                # PR34: snapshots configured while perf is disabled → identity path (no effect)
                ps = _ensure_dict(perf.get("snapshots"))
                if ps:
                    comp = str(ps.get("compression", "none")).lower()
                    lvl_present = ("level" in ps)
                    delta_set = _coerce_bool(ps.get("delta_mode", False))
                    every_present = ("every_n_turns" in ps)
                    if (comp != "none") or lvl_present or delta_set or every_present:
                        warnings.append("W[perf.snapshots]: snapshots configured while perf.enabled=false; features remain disabled (identity path).")
            # Warn if both legacy and new frontier caps are present
            if "queue_cap" in pt1v and "frontier" in capsv:
                warnings.append("W[perf.t1]: both queue_cap and caps.frontier set; caps.frontier will be used by runtime.")
            pt2 = _ensure_dict(perf.get("t2"))
            esd = str(pt2.get("embed_store_dtype", "")).lower()
            pre = _coerce_bool(pt2.get("precompute_norms", False))
            if esd == "fp16" and not pre:
                warnings.append("W[perf.t2]: embed_store_dtype=fp16 without precompute_norms=true; fp32 norms recommended for score parity.")
            ps = _ensure_dict(perf.get("snapshots"))
            if _coerce_bool(ps.get("delta_mode", False)):
                warnings.append("W[perf.snapshots]: delta_mode=true may fallback to full snapshot when baseline/etag is missing.")
        # LanceDB partitions configured while perf is OFF → ignored in disabled path
        try:
            t2n = _ensure_dict(normalized.get("t2"))
            ldb = _ensure_dict(t2n.get("lancedb"))
            parts = _ensure_dict(ldb.get("partitions"))
            if parts and not perf_on:
                warnings.append("W[t2.lancedb.partitions]: configured while perf.enabled=false; ignored in disabled path.")
        except Exception:
            pass
        # PR40: t2.reader.mode warnings
        try:
            t2n = _ensure_dict(normalized.get("t2"))
            rmode = str(_ensure_dict(t2n.get("reader")).get("mode", "flat")).lower()
            perf = _ensure_dict(normalized.get("perf"))
            perf_on = _coerce_bool(perf.get("enabled", False))
            if rmode in {"partition", "auto"} and not perf_on:
                warnings.append("W[t2.reader.mode]: partition/auto configured while perf.enabled=false; reader parity path & metrics are disabled (flat behavior).")
        except Exception:
            pass
        q = _ensure_dict(_ensure_dict(normalized.get("t2")).get("quality"))
        if q:
            # PR39: normalizer & aliasing warnings
            qn = _ensure_dict(q.get("normalizer"))
            if qn:
                if _coerce_bool(qn.get("enabled", False)) and not _coerce_bool(q.get("enabled", False)):
                    warnings.append("W[t2.quality.normalizer]: normalizer.enabled=true while t2.quality.enabled=false; no effect.")
                case = str(qn.get("case", "lower"))
                if case != "lower":
                    warnings.append("W[t2.quality.normalizer.case]: only 'lower' is supported; using 'lower'.")
                uni = str(qn.get("unicode", "NFKC"))
                if uni != "NFKC":
                    warnings.append("W[t2.quality.normalizer.unicode]: only 'NFKC' is supported; using 'NFKC'.")
            qa = _ensure_dict(q.get("aliasing"))
            if qa:
                if not _coerce_bool(q.get("enabled", False)):
                    warnings.append("W[t2.quality.aliasing]: configured while t2.quality.enabled=false; no effect.")
                mp = qa.get("map_path")
                if isinstance(mp, str) and mp:
                    try:
                        import os
                        if not os.path.isfile(mp):
                            warnings.append("W[t2.quality.aliasing.map_path]: file not found or unreadable; aliasing will be a no-op.")
                    except Exception:
                        pass
            # Shadow triple-gate reminder: traces only emit when perf.enabled && perf.metrics.report_memory && shadow==true && enabled==false
            try:
                perf = _ensure_dict(normalized.get("perf"))
                perf_on = _coerce_bool(perf.get("enabled", False))
                perf_metrics = _ensure_dict(perf.get("metrics"))
                report_mem = _coerce_bool(perf_metrics.get("report_memory", False))
                if _coerce_bool(q.get("shadow", False)):
                    if not (perf_on and report_mem and not _coerce_bool(q.get("enabled", False))):
                        warnings.append("W[t2.quality.shadow]: requires perf.enabled=true AND perf.metrics.report_memory=true AND t2.quality.enabled=false; traces will not emit otherwise.")
            except Exception:
                pass
            # PR37: enabled-path warnings
            try:
                perf = _ensure_dict(normalized.get("perf"))
                perf_on = _coerce_bool(perf.get("enabled", False))
                perf_metrics = _ensure_dict(perf.get("metrics"))
                report_mem = _coerce_bool(perf_metrics.get("report_memory", False))
                if _coerce_bool(q.get("enabled", False)) and not perf_on:
                    warnings.append("W[t2.quality.enabled]: perf.enabled=false; quality fusion will not execute.")
                if _coerce_bool(q.get("enabled", False)) and perf_on and not report_mem:
                    warnings.append("W[t2.quality.metrics]: perf.metrics.report_memory=false; quality metrics/traces will not emit under Gate B.")
            except Exception:
                pass
            qf = _ensure_dict(q.get("fusion"))
            if "alpha_semantic" in qf:
                a = _coerce_float(qf.get("alpha_semantic"))
                if not (0.0 <= a <= 1.0):
                    warnings.append("W[t2.quality.fusion.alpha_semantic]: expected in [0,1].")
            qm = _ensure_subdict(q, "mmr")
            if qm:
                # enabled-without-quality: no effect
                if _coerce_bool(qm.get("enabled", False)) and not _coerce_bool(q.get("enabled", False)):
                    warnings.append("W[t2.quality.mmr]: mmr.enabled=true while t2.quality.enabled=false; MMR has no effect.")
                # lambda checks (canonical + legacy)
                lam = None
                if "lambda" in qm:
                    lam = _coerce_float(qm.get("lambda"))
                    if not (0.0 <= lam <= 1.0):
                        warnings.append("W[t2.quality.mmr.lambda]: expected in [0,1].")
                elif "lambda_relevance" in qm:
                    lam = _coerce_float(qm.get("lambda_relevance"))
                    if not (0.0 <= lam <= 1.0):
                        warnings.append("W[t2.quality.mmr.lambda_relevance]: expected in [0,1].")
                # k checks (canonical + legacy) and compare with t2.k_retrieval
                k_val = None
                if "k" in qm:
                    k_val = _coerce_int(qm.get("k"))
                    if k_val < 1:
                        warnings.append("W[t2.quality.mmr.k]: expected >= 1.")
                elif "k_final" in qm:
                    k_val = _coerce_int(qm.get("k_final"))
                    if k_val < 1:
                        warnings.append("W[t2.quality.mmr.k_final]: expected >= 1.")
                if k_val is not None:
                    try:
                        kret = _coerce_int(_ensure_dict(normalized.get("t2")).get("k_retrieval", 10))
                        if k_val > kret:
                            # Choose label based on raw cfg to preserve legacy expectation
                            raw_t2 = _ensure_dict(cfg.get("t2")) if isinstance(cfg, dict) else {}
                            raw_q = _ensure_dict(raw_t2.get("quality"))
                            raw_mmr = _ensure_dict(raw_q.get("mmr"))
                            if "k_final" in raw_mmr and "k" not in raw_mmr:
                                k_label = "k_final"
                            elif "k" in raw_mmr:
                                k_label = "k"
                            else:
                                k_label = "k"
                            warnings.append(
                                f"W[t2.quality.mmr.{k_label}]: {k_label} exceeds t2.k_retrieval; results may truncate in M7 wiring."
                            )
                    except Exception:
                        pass
                # PR38 scope: owner-diversity is ignored; token-diversity is implicit
                if _coerce_bool(qm.get("diversity_by_owner", False)):
                    warnings.append("W[t2.quality.mmr.diversity_by_owner]: ignored in PR38; token-based diversity only.")
                if "diversity_by_token" in qm:
                    warnings.append("W[t2.quality.mmr.diversity_by_token]: flag is accepted but PR38 always uses token-based diversity.")
    except Exception:
        pass
    return normalized, warnings


# Public API (unified):
# - validate_config(cfg) -> dict (normalized)  [primary path]
# - validate_config(cfg, strict=..., verbose=...) -> (errors, warnings)  [compat path for a few tests]

def validate_config_api(cfg: Dict[str, Any]):
    """Stable, test-friendly API for M3-07.

    Returns a tuple: (ok: bool, errs: list[str], cfg_or_none).
    - On success: (True, [], normalized_cfg)
    - On validation error: (False, [messages...], None)
    Does not raise; wraps the strict normalizer.
    """
    try:
        normalized = _validate_config_normalize_impl(cfg)
        return True, [], normalized
    except ValueError as e:
        msg = str(e).strip()
        errs = msg.split("\n") if msg else ["invalid configuration"]
        return False, errs, None

def validate_config(cfg: Dict[str, Any], **kwargs):
    """Validate configuration.

    Primary form: validate_config(cfg) -> dict (normalized) and raises ValueError on errors.
    Compatibility form (if any kwargs like 'strict' or 'verbose' are provided):
      returns (errors, warnings) instead of raising, matching older tests.
    """
    if kwargs:  # compat mode triggered by presence of any kwargs
        try:
            # Run normalization to surface any errors
            _ = _validate_config_normalize_impl(cfg)
            # Collect warnings using the verbose API
            _, warnings = validate_config_verbose(cfg)
            return [], warnings
        except ValueError as e:
            msg = str(e).strip()
            errs = msg.split("\n") if msg else ["invalid configuration"]
            return errs, []
    # No kwargs: return normalized dict or raise on error
    return _validate_config_normalize_impl(cfg)
    