

import math
from pathlib import Path

import numpy as np
import pytest

from clematis.engine.util.embed_store import write_shard, open_reader


def _make_corpus(n: int, d: int, seed: int = 1234):
    """Deterministic corpus with values aligned to fp16 grid (parity-friendly).
    We use int steps divided by 128 so fp16 quantization is exact for this set.
    """
    rng = np.random.default_rng(seed)
    ints = rng.integers(-64, 64, size=(n, d), dtype=np.int16)
    arr = (ints.astype(np.float32) / 128.0).astype(np.float32)
    ids = [f"doc{idx:04d}" for idx in range(n)]
    return ids, arr


def _make_queries(m: int, d: int, seed: int = 5678):
    rng = np.random.default_rng(seed)
    ints = rng.integers(-64, 64, size=(m, d), dtype=np.int16)
    qs = (ints.astype(np.float32) / 128.0).astype(np.float32)
    return qs


def _topk(ids, vecs_fp32, norms_fp32, q_fp32, k: int, *, cosine=True):
    # Compute scores in fp32 deterministically
    q = q_fp32.astype(np.float32, copy=False)
    if cosine:
        qn = float(np.linalg.norm(q, ord=2))
        # Safe guard against zero norm (unlikely with our grid)
        denom = (norms_fp32 * (qn if qn != 0.0 else 1.0))
        scores = (vecs_fp32 @ q) / np.where(denom == 0.0, 1.0, denom)
    else:
        scores = vecs_fp32 @ q
    # Stable tie-break: (-score, lex(id))
    items = [(-float(scores[i]), ids[i], float(scores[i])) for i in range(len(ids))]
    items.sort(key=lambda t: (t[0], t[1]))
    top = items[:k]
    top_ids = [t[1] for t in top]
    top_scores = np.array([t[2] for t in top], dtype=np.float32)
    return top_ids, top_scores


@pytest.mark.parametrize("precompute_norms", [False, True])
def test_fp16_store_fp32_math_parity(tmp_path: Path, precompute_norms: bool):
    N, D = 200, 64
    ids, embeds_fp32 = _make_corpus(N, D, seed=20240815)
    qs = _make_queries(5, D, seed=20240816)

    # Baseline (pure fp32 in-memory)
    base_vecs = embeds_fp32.astype(np.float32, copy=False)
    base_norms = np.linalg.norm(base_vecs, ord=2, axis=1).astype(np.float32, copy=False)

    # Write fp16 shard and read via EmbedReader
    shard_dir = tmp_path / ("shard_norms" if precompute_norms else "shard_no_norms")
    write_shard(shard_dir, ids, embeds_fp32, dtype="fp16", precompute_norms=precompute_norms)
    reader = open_reader(shard_dir)

    # Slurp the entire shard via reader (fp32 views)
    r_ids, r_vecs, r_norms = reader.load_all()
    assert r_vecs.dtype == np.float32
    assert len(r_ids) == len(ids) == N
    assert r_vecs.shape == (N, D)
    assert r_norms is not None and r_norms.shape == (N,)

    # For several K, verify Top-K ids identical and score deltas small
    for k in (5, 10, 50):
        for qi, q in enumerate(qs):
            base_ids_k, base_scores_k = _topk(ids, base_vecs, base_norms, q, k)
            read_ids_k, read_scores_k = _topk(r_ids, r_vecs, r_norms, q, k)
            assert read_ids_k == base_ids_k, f"Top-{k} mismatch for query {qi}:\nbase={base_ids_k}\nread={read_ids_k}"
            # Scores should match to within 1e-6 due to fp16-aligned grid
            max_delta = float(np.max(np.abs(read_scores_k - base_scores_k))) if len(base_scores_k) else 0.0
            assert max_delta <= 1e-6, f"Score drift {max_delta} exceeds 1e-6 for Top-{k} query {qi}"


def test_fp32_store_sanity(tmp_path: Path):
    """Sanity: fp32 storage path behaves like baseline (exact)."""
    N, D = 128, 32
    ids, embeds_fp32 = _make_corpus(N, D, seed=4242)
    qs = _make_queries(3, D, seed=4343)

    base_vecs = embeds_fp32.astype(np.float32, copy=False)
    base_norms = np.linalg.norm(base_vecs, ord=2, axis=1).astype(np.float32, copy=False)

    shard_dir = tmp_path / "shard_fp32"
    write_shard(shard_dir, ids, embeds_fp32, dtype="fp32", precompute_norms=True)
    reader = open_reader(shard_dir)
    r_ids, r_vecs, r_norms = reader.load_all()

    for k in (5, 20):
        for qi, q in enumerate(qs):
            base_ids_k, base_scores_k = _topk(ids, base_vecs, base_norms, q, k)
            read_ids_k, read_scores_k = _topk(r_ids, r_vecs, r_norms, q, k)
            assert read_ids_k == base_ids_k
            max_delta = float(np.max(np.abs(read_scores_k - base_scores_k))) if len(base_scores_k) else 0.0
            assert max_delta == 0.0