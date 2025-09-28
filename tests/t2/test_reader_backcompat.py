from pathlib import Path

import numpy as np
import pytest

from clematis.engine.util.embed_store import write_shard, open_reader


def test_fp32_reader_without_precomputed_norms_computes_on_the_fly(tmp_path: Path):
    """
    Back-compat path we guarantee: fp32 storage with no norms.bin.
    Reader should compute norms on the fly (fp32) and return fp32 vectors.
    """
    N, D = 128, 32
    rng = np.random.default_rng(20250921)
    # Use an fp16-friendly grid purely to keep arithmetic exact in assertions
    ints = rng.integers(-64, 64, size=(N, D), dtype=np.int16)
    embeds_fp32 = (ints.astype(np.float32) / 128.0).astype(np.float32)
    ids = [f"doc{idx:04d}" for idx in range(N)]

    shard_dir = tmp_path / "legacy_fp32_no_norms"
    write_shard(shard_dir, ids, embeds_fp32, dtype="fp32", precompute_norms=False)

    reader = open_reader(shard_dir)
    r_ids, r_vecs, r_norms = reader.load_all()

    assert r_vecs.dtype == np.float32
    assert r_norms is not None
    assert len(r_ids) == N and r_vecs.shape == (N, D) and r_norms.shape == (N,)

    # Norms computed by the reader should match direct fp32 norms
    expected = np.linalg.norm(embeds_fp32.astype(np.float32, copy=False), ord=2, axis=1).astype(
        np.float32, copy=False
    )
    assert np.allclose(r_norms, expected, rtol=0.0, atol=0.0)


def test_missing_meta_is_rejected(tmp_path: Path):
    """
    Current reader requires meta.json. A shard missing meta.json should raise FileNotFoundError.
    (True 'no-meta' back-compat can be added later if we decide to support it.)
    """
    shard_dir = tmp_path / "no_meta_shard"
    shard_dir.mkdir(parents=True, exist_ok=True)
    # Create minimal embeddings.bin and ids.tsv without meta.json
    N, D = 10, 8
    vecs = np.zeros((N, D), dtype=np.float32)
    vecs.tofile(shard_dir / "embeddings.bin")
    with (shard_dir / "ids.tsv").open("w", encoding="utf-8") as f:
        for i in range(N):
            f.write(f"id{i}\n")

    with pytest.raises(FileNotFoundError):
        _ = open_reader(shard_dir)
