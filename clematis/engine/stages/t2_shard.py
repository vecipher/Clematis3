from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Tuple

__all__ = [
    "RawHit",
    "sort_key",
    "merge_hits",
    "_qscore",
    "merge_tier_hits_across_shards_dict",
]


@dataclass(frozen=True)
class RawHit:
    """Lightweight candidate used by sharded T2 helpers."""
    episode_id: str
    score: float
    payload: Dict[str, Any]


def _qscore(score: float) -> int:
    """
    Quantize a float to an integer for stable ordering across platforms.
    1e9 granularity mirrors the T2 helper; non-finite or bad values map to 0.
    """
    try:
        s = float(score)
        if s != s:  # NaN check
            return 0
        return int(round(s * 1_000_000_000))
    except Exception:
        return 0


def sort_key(hit: RawHit) -> Tuple[int, str]:
    """Deterministic sort key: primary -qscore(score), secondary episode_id lex asc."""
    return (-_qscore(hit.score), str(hit.episode_id))


def merge_hits(buckets: Iterable[List[RawHit]]) -> List[RawHit]:
    """
    Deterministically merge multiple RawHit buckets:
    - keep the best by (score desc, id asc) per episode_id
    - final list sorted by the same key
    """
    best: Dict[str, RawHit] = {}
    for bucket in buckets:
        for h in bucket:
            prev = best.get(h.episode_id)
            if prev is None or sort_key(h) < sort_key(prev):
                best[h.episode_id] = h
    return sorted(best.values(), key=sort_key)


def merge_tier_hits_across_shards_dict(
    shard_hits_by_tier: Iterable[Dict[str, List[Dict[str, Any]]]],
    tiers: List[str],
    k_retrieval: int,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """
    Dict-based merge used by T2's parallel path.

    Each element of `shard_hits_by_tier` is a mapping {tier -> [hit_dict,...]} where
    each hit_dict contains at least:
      - "id": str-like (episode id)
      - "score" or "_score": float-like
    We walk tiers in order, merging cross-shard buckets with deterministic ordering:
      sort by (-qscore(score), id_asc), de-duplicate by id, stop at k_retrieval.

    Returns (merged_hits, used_tiers).
    """
    seen: set[str] = set()
    out: List[Dict[str, Any]] = []
    used_tiers: List[str] = []

    for tier in tiers:
        # collect all hits for this tier across shards
        bucket: List[Dict[str, Any]] = []
        for d in shard_hits_by_tier:
            bucket.extend((d.get(tier) or []))

        # Append tier to sequence regardless (mirrors sequential walk)
        used_tiers.append(tier)
        if not bucket:
            continue

        # Stable sort: primary = -score, secondary = id asc (lex)
        def _key(h: Dict[str, Any]) -> Tuple[int, str]:
            s = h.get("score", h.get("_score", 0.0))
            return (-_qscore(s), str(h.get("id")))

        bucket.sort(key=_key)

        for h in bucket:
            hid = str(h.get("id"))
            if hid in seen:
                continue
            out.append(h)
            seen.add(hid)
            if len(out) >= k_retrieval:
                return out, used_tiers

    return out, used_tiers
