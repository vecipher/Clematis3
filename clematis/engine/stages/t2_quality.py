from __future__ import annotations

"""
PR37 — Lexical BM25 + Fusion (alpha)

This module adds a deterministic lexical scorer over the current candidate set and a
rank-based fusion step that combines semantic and lexical signals. It is designed to
be stable, side-effect free, and independent of corpus-global statistics.

Contract:
  - No RNG; deterministic math; ties break by lex(id).
  - Input `items` is a list of dicts with at least {"id", "score"} where "score"
    reflects the current semantic ordering (e.g., a rank surrogate).
  - Fusion is rank-based (reciprocal rank with a fixed constant) to avoid sensitivity
    to raw score scales.
  - Returns (fused_items, meta) where fused_items preserves shape and ordering fields.

Metrics surfaced via meta (picked up by t2.py under the triple gate):
  - "t2q.fusion_mode"        -> "score_interp" (emitted in t2.py)
  - "t2q.alpha_semantic"     -> float (emitted in t2.py)
  - "t2q.lex_hits"           -> int (sum of unique query-term matches across candidates)
"""

import math
import unicodedata
from typing import Any, Dict, List, Tuple, FrozenSet

from .t2_quality_mmr import MMRItem, mmr_reorder_full
from .t2_quality_norm import normalize_text as _qnorm, tokenize as _qtokenize, load_alias_map as _load_alias_map, apply_aliases as _apply_aliases

__all__ = ["fuse", "maybe_apply_mmr"]

# Small, deterministic English stopword set for PR37 (can be extended later)
_STOP_EN_BASIC = {"a", "an", "the", "of", "to", "in", "and", "or"}

# Reciprocal-rank smoothing constant (rank starts at 1)
_RRANK_C = 60

_EPS = 1e-12


# ---------- Normalization & tokenization ----------

def _norm_text(s: str) -> str:
  """Unicode-normalize (NFKC) and lowercase."""
  return unicodedata.normalize("NFKC", s or "").lower()


def _tokenize(s: str, stopset) -> List[str]:
  toks = _norm_text(s).split()
  if stopset:
    toks = [t for t in toks if t not in stopset]
  return toks


# ---------- Rank utilities ----------

def _reciprocal_rank_scores(ids_in_order: List[str], C: int = _RRANK_C) -> Dict[str, float]:
  """
  Map each id to a reciprocal-rank score using its 1-based position.
  Ties must be resolved before this via deterministic sorting.
  """
  return {doc_id: 1.0 / (rank + C) for rank, doc_id in enumerate(ids_in_order, start=1)}


# ---------- Lexical BM25 over the candidate set ----------

def _bm25_scores(query: str, items: List[Dict[str, Any]], k1: float, b: float, stopset, qcfg: Dict[str, Any]) -> Tuple[Dict[str, float], Dict[str, int], Dict[str, int]]:
  """
  Compute BM25-style lexical scores over the provided candidate set.
  Returns:
    - scores: dict id -> float score
    - df:     dict term -> document frequency within the candidate set
    - hits:   dict id -> count of UNIQUE query terms present in that item
  Uses PR39 normalizer/aliasing when quality paths are enabled.
  """
  norm_cfg = (qcfg.get("normalizer") or {})
  use_norm = bool(norm_cfg.get("enabled", True))
  alias_cfg = (qcfg.get("aliasing") or {})
  map_path = alias_cfg.get("map_path")
  amap = _load_alias_map(map_path) if map_path else {}

  def _lex_tokens(text: str) -> List[str]:
    toks = _qtokenize(text, stopset=stopset) if use_norm else _tokenize(text, stopset)
    if amap:
      toks = _apply_aliases(toks, amap)
    return toks

  q_terms = _lex_tokens(query)
  if not q_terms:
    return ({it["id"]: 0.0 for it in items}, {}, {it["id"]: 0 for it in items})

  # Build per-doc token lists and lengths
  docs: List[List[str]] = []
  doc_lens: List[int] = []
  for it in items:
    toks = _lex_tokens(str(it.get("text", "")))
    docs.append(toks)
    doc_lens.append(len(toks))

  N = max(1, len(items))
  avgdl = (sum(doc_lens) / float(N)) if N else 1.0

  # Document frequency per term (presence/absence, not counts)
  df: Dict[str, int] = {}
  for toks in docs:
    seen = set()
    for t in toks:
      if t in q_terms and t not in seen:
        df[t] = df.get(t, 0) + 1
        seen.add(t)

  # IDF with small stabilizer
  idf = {t: math.log((N - df.get(t, 0) + 0.5) / (df.get(t, 0) + 0.5) + 1e-9) for t in set(q_terms)}

  scores: Dict[str, float] = {}
  unique_hits: Dict[str, int] = {}
  for it, toks, dl in zip(items, docs, doc_lens):
    tf: Dict[str, int] = {}
    present_unique = set()
    for t in toks:
      if t in idf:
        tf[t] = tf.get(t, 0) + 1
        present_unique.add(t)

    s = 0.0
    for t in q_terms:
      f = tf.get(t, 0)
      if f == 0:
        continue
      denom = f + k1 * (1 - b + b * (dl / (avgdl or 1.0)))
      denom = denom if denom > _EPS else _EPS
      s += idf[t] * ((f * (k1 + 1.0)) / denom)

    scores[it["id"]] = float(s)
    unique_hits[it["id"]] = len(present_unique)

  return scores, df, unique_hits


# ---------- Fusion ----------

def fuse(query: str, items: List[Dict[str, Any]], *, cfg: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
  """
  Combine semantic and lexical signals using rank-based interpolation.

  Args:
    query: raw query string.
    items: list of dicts with at least {"id", "score"}; "text" is optional.
    cfg:   config tree; uses:
           t2.quality.lexical.{bm25_k1,bm25_b,stopwords}
           t2.quality.fusion.{mode,alpha_semantic}

  Returns:
    (fused_items, meta)
      fused_items: list of dicts with original keys plus "score_fused".
      meta:        dict with optional quality metrics (e.g., "t2q.lex_hits").
  """
  qcfg = (cfg.get("t2") or {}).get("quality") or {}
  if not qcfg.get("enabled", False):
    # Identity path: return as-is (plus empty meta)
    return items, {}

  # Defaults are already validated upstream, but be defensive here
  lex_cfg = qcfg.get("lexical", {}) or {}
  k1 = float(lex_cfg.get("bm25_k1", 1.2))
  b = float(lex_cfg.get("bm25_b", 0.75))
  stopwords = lex_cfg.get("stopwords", "en-basic")
  stopset = _STOP_EN_BASIC if stopwords == "en-basic" else None

  fus_cfg = qcfg.get("fusion", {}) or {}
  mode = str(fus_cfg.get("mode", "score_interp"))
  alpha = float(fus_cfg.get("alpha_semantic", 0.6))
  if mode != "score_interp":
    # Unsupported mode in PR37 -> behave like identity
    return items, {}

  # 1) Lexical scores over candidates
  lex_scores, _, unique_hits = _bm25_scores(query, items, k1=k1, b=b, stopset=stopset, qcfg=qcfg)

  # 2) Deterministic ranks for both signals (break ties lex(id))
  def _id_or(it): return it.get("id")
  sem_ids = [it["id"] for it in sorted(items, key=lambda d: (-float(d.get("score", 0.0)), str(_id_or(d))))]
  lex_ids = [it["id"] for it in sorted(items, key=lambda d: (-float(lex_scores.get(d["id"], 0.0)), str(_id_or(d))))]

  sem_rr = _reciprocal_rank_scores(sem_ids)
  lex_rr = _reciprocal_rank_scores(lex_ids)

  # 3) Interpolate ranks
  fused: List[Dict[str, Any]] = []
  for it in items:
    doc_id = it["id"]
    s_sem = sem_rr.get(doc_id, 0.0)
    s_lex = lex_rr.get(doc_id, 0.0)
    s_fused = alpha * s_sem + (1.0 - alpha) * s_lex
    fused.append({**it, "score_fused": float(s_fused)})

  # 4) Final ordering: fused desc, ties by lex(id)
  fused.sort(key=lambda d: (-float(d["score_fused"]), str(d["id"])))

  # Meta: sum of unique query-term hits across candidates (integer)
  lex_hits_total = 0
  for doc_id in (it["id"] for it in items):
    lex_hits_total += int(unique_hits.get(doc_id, 0))

  # Provide a per-id fused score map to aid enabled-path traces (optional to consume)
  score_fused_map = {d["id"]: float(d.get("score_fused", 0.0)) for d in fused}

  meta = {
    "t2q.lex_hits": int(lex_hits_total),
    "score_fused_map": score_fused_map,
  }
  return fused, meta


# ---------- MMR (PR38) glue ----------

def _tokenize_for_mmr(s: str, stopset) -> List[str]:
  """Mirror PR37 normalization for MMR token sets."""
  toks = _norm_text(s).split()
  if stopset:
    toks = [t for t in toks if t not in stopset]
  return toks

def _tokens_for_item_mmr(it: Dict[str, Any], stopset) -> FrozenSet[str]:
  """Use provided tokens if present; otherwise tokenize the item's text."""
  toks = it.get("tokens")
  if toks:
    # Normalize tokens defensively to match PR37 lower/NFKC
    normed = []
    for t in toks:
      try:
        tt = _norm_text(str(t))
        if tt and (not stopset or tt not in stopset):
          normed.append(tt)
      except Exception:
        continue
    return frozenset(normed)
  return frozenset(_tokenize_for_mmr(str(it.get("text", "")), stopset))

def maybe_apply_mmr(fused: List[Dict[str, Any]], qcfg: Dict[str, Any]) -> List[Dict[str, Any]]:
  """Apply deterministic MMR reordering over PR37 fused results if enabled.

  Expectations for `fused` items:
    - Each item contains `id` (str) and `score_fused` (float). If `score_fused`
      is missing, we fall back to `score`.
  `qcfg` is the `t2.quality` sub-config.
  """
  if not qcfg or not qcfg.get("enabled", False):
    return fused
  mmr_cfg = (qcfg.get("mmr") or {})
  if not mmr_cfg.get("enabled", False):
    return fused

  lam = float(mmr_cfg.get("lambda", 0.5))
  if lam < 0.0:
    lam = 0.0
  elif lam > 1.0:
    lam = 1.0
  k = mmr_cfg.get("k")
  if isinstance(k, int) and k < 1:
    k = None

  # Reuse PR37/PR39 lexical stopword choice and PR39 normalizer/aliasing
  lex_cfg = qcfg.get("lexical", {}) or {}
  stopwords = lex_cfg.get("stopwords", "en-basic")
  stopset = _STOP_EN_BASIC if stopwords == "en-basic" else None

  norm_cfg = (qcfg.get("normalizer") or {})
  use_norm = bool(norm_cfg.get("enabled", True))
  alias_cfg = (qcfg.get("aliasing") or {})
  map_path = alias_cfg.get("map_path")
  amap = _load_alias_map(map_path) if map_path else {}

  def _mmr_toks_for_item(c: Dict[str, Any]) -> FrozenSet[str]:
    # Prefer text; fallback to provided tokens
    text = c.get("text", "")
    if text:
      toks = _qtokenize(text, stopset=stopset) if use_norm else _tokenize(text, stopset)
    else:
      raw = c.get("tokens") or []
      toks = []
      for t in raw:
        tt = _qnorm(str(t)) if use_norm else _norm_text(str(t))
        if tt and (not stopset or tt not in stopset):
          toks.append(tt)
    if amap:
      toks = _apply_aliases(toks, amap)
    return frozenset(toks)

  items: List[MMRItem] = []
  for c in fused:
    cid = str(c.get("id"))
    rel = float(c.get("score_fused", c.get("score", 0.0)))
    toks = _mmr_toks_for_item(c)
    items.append(MMRItem(id=cid, rel=rel, toks=toks))

  order = mmr_reorder_full(items, k=k, lam=lam)
  # Reorder without mutating input list
  return [fused[i] for i in order]
