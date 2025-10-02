import numpy as np
import pytest
from pathlib import Path

from clematis.engine.util.embed_store import write_shard, open_reader


def _make_corpus(n: int, d: int, seed: int = 20240901):
    """Deterministic corpus on an fp16-friendly grid (exact cast to/from fp16)."""
    rng = np.random.default_rng(seed)
    ints = rng.integers(-64, 64, size=(n, d), dtype=np.int16)
    arr = (ints.astype(np.float32) / 128.0).astype(np.float32)
    ids = [f"doc{idx:04d}" for idx in range(n)]
    return ids, arr


def _make_queries(m: int, d: int, seed: int = 20240902):
    rng = np.random.default_rng(seed)
    ints = rng.integers(-64, 64, size=(m, d), dtype=np.int16)
    qs = (ints.astype(np.float32) / 128.0).astype(np.float32)
    return qs


def _topk(ids, vecs_fp32, norms_fp32, q_fp32, k: int, *, cosine=True):
    q = q_fp32.astype(np.float32, copy=False)
    if cosine:
        qn = float(np.linalg.norm(q, ord=2))
        denom = norms_fp32 * (qn if qn != 0.0 else 1.0)
        scores = (vecs_fp32 @ q) / np.where(denom == 0.0, 1.0, denom)
    else:
        scores = vecs_fp32 @ q
    items = [(-float(scores[i]), ids[i], float(scores[i])) for i in range(len(ids))]
    items.sort(key=lambda t: (t[0], t[1]))
    top = items[:k]
    top_ids = [t[1] for t in top]
    top_scores = np.array([t[2] for t in top], dtype=np.float32)
    return top_ids, top_scores


def test_partition_owner_quarter_parity_fp16(tmp_path: Path):
    N, D = 300, 64
    ids, embeds_fp32 = _make_corpus(N, D)
    qs = _make_queries(4, D)

    # --- Unpartitioned ---
    root_unpart = tmp_path / "unpart"
    shard_unpart = root_unpart / "shard-000"
    write_shard(shard_unpart, ids, embeds_fp32, dtype="fp16", precompute_norms=True)
    r_u = open_reader(shard_unpart)
    u_ids, u_vecs, u_norms = r_u.load_all()
    assert len(u_ids) == N and u_vecs.shape == (N, D)

    # --- Partitioned owner/quarter with two shards ---
    root_part = tmp_path / "part"
    ownerA = root_part / "ownerA" / "2025Q3"
    ownerB = root_part / "ownerB" / "2025Q3"
    shard_a = ownerA / "shard-A"
    shard_b = ownerB / "shard-B"

    mid = N // 2
    write_shard(shard_a, ids[:mid], embeds_fp32[:mid], dtype="fp16", precompute_norms=True)
    write_shard(shard_b, ids[mid:], embeds_fp32[mid:], dtype="fp16", precompute_norms=True)

    r_p = open_reader(root_part, partitions={"enabled": True, "layout": "owner_quarter"})
    p_ids, p_vecs, p_norms = r_p.load_all()
    assert len(p_ids) == N and p_vecs.shape == (N, D)
    assert p_vecs.dtype == np.float32 and p_norms is not None

    # Top-K parity across layouts
    for k in (5, 10, 50):
        for qi, q in enumerate(qs):
            base_ids_k, base_scores_k = _topk(u_ids, u_vecs, u_norms, q, k)
            part_ids_k, part_scores_k = _topk(p_ids, p_vecs, p_norms, q, k)
            assert part_ids_k == base_ids_k, (
                f"Top-{k} mismatch for query {qi}:\nunpart={base_ids_k}\npart=  {part_ids_k}"
            )
            max_delta = (
                float(np.max(np.abs(part_scores_k - base_scores_k))) if len(base_scores_k) else 0.0
            )
            assert max_delta <= 1e-6, f"Score drift {max_delta} exceeds 1e-6 for Top-{k} query {qi}"


def test_partition_quarter_direct_meta(tmp_path: Path):
    """Support the case where meta.json sits directly under the QUARTER dir (no shard subdirs)."""
    N, D = 128, 32
    ids, embeds_fp32 = _make_corpus(N, D, seed=111)
    root = tmp_path / "layout_qmeta"
    quarter = root / "teamZ" / "2025Q2"
    # Write shard directly into the quarter directory
    write_shard(quarter, ids, embeds_fp32, dtype="fp16", precompute_norms=True)

    reader = open_reader(root, partitions={"enabled": True, "layout": "owner_quarter"})
    r_ids, r_vecs, r_norms = reader.load_all()
    assert len(r_ids) == N and r_vecs.shape == (N, D)
    assert r_vecs.dtype == np.float32 and r_norms is not None
