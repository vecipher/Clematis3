from __future__ import annotations

from typing import Any, Dict, List

from .config import cfg_get


def t2_parallel_enabled(cfg_obj: Any, backend: str, index) -> bool:
    try:
        if not cfg_get(cfg_obj, ["perf", "parallel", "enabled"], False):
            return False
        if not cfg_get(cfg_obj, ["perf", "parallel", "t2"], False):
            return False
        max_workers = int(cfg_get(cfg_obj, ["perf", "parallel", "max_workers"], 1) or 1)
        if max_workers <= 1:
            return False
        backend_lower = str(backend or "inmemory").lower()
        if backend_lower not in ("inmemory", "lancedb"):
            return False
        if not hasattr(index, "_iter_shards_for_t2"):
            return False
        try:
            try:
                iterator = index._iter_shards_for_t2("exact_semantic", suggested=max_workers)  # type: ignore[attr-defined]
            except TypeError:
                iterator = index._iter_shards_for_t2("exact_semantic")  # type: ignore[attr-defined]
            shards = list(iterator)
            return len(shards) > 1
        except Exception:
            return False
    except Exception:
        return False


def collect_shard_hits(
    shard,
    tiers: List[str],
    owner_query,
    q_vec,
    k_retrieval: int,
    now_str: str | None,
    sim_threshold: float,
    clusters_top_m: int,
) -> Dict[str, List[Dict[str, Any]]]:
    out: Dict[str, List[Dict[str, Any]]] = {}
    for tier in tiers:
        hints: Dict[str, Any] = {"sim_threshold": sim_threshold}
        if tier == "exact_semantic":
            hints.update({"recent_days": None})
        elif tier == "cluster_semantic":
            hints.update({"clusters_top_m": int(clusters_top_m)})
        elif tier == "archive":
            pass
        else:
            continue
        if now_str:
            hints["now"] = now_str
        try:
            hits = shard.search_tiered(
                owner=owner_query, q_vec=q_vec, k=k_retrieval, tier=tier, hints=hints
            )
        except Exception:
            hits = []
        normalised: List[Dict[str, Any]] = []
        for hit in hits or []:
            if isinstance(hit, dict):
                data = dict(hit)
                data["id"] = str(data.get("id"))
                if "score" not in data:
                    data["score"] = float(data.get("_score", 0.0))
                normalised.append(data)
            else:
                data = {
                    "id": str(getattr(hit, "id", "")),
                    "text": getattr(hit, "text", ""),
                    "score": float(getattr(hit, "score", 0.0)),
                }
                normalised.append(data)
        out[tier] = normalised
    return out
