from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest

import clematis.engine.stages.t2 as t2mod
from clematis.engine.util.embed_store import write_shard

HAVE_T2 = hasattr(t2mod, "t2_semantic")

class _Ctx(SimpleNamespace):
    pass

class _State(SimpleNamespace):
    pass

class _Index(SimpleNamespace):
    pass

class _DummyEnc:
    def __init__(self, dim: int):
        self.dim = int(dim)
    def encode(self, texts):
        # Deterministic unit-norm vector so cosine math is stable.
        v = np.ones((len(texts), self.dim), dtype=np.float32)
        v /= np.linalg.norm(v, ord=2, axis=1, keepdims=True)
        return v

def _ns(d):
    if isinstance(d, dict):
        return SimpleNamespace(**{k: _ns(v) for k, v in d.items()})
    return d

@pytest.mark.skipif(not HAVE_T2, reason="t2_semantic not available")
def test_reader_flip_matches_direct_cosine(tmp_path: Path):
    # Build a small fp16-friendly corpus
    N, D = 120, 32
    rng = np.random.default_rng(20250921)
    ints = rng.integers(-64, 64, size=(N, D), dtype=np.int16)
    embeds = (ints.astype(np.float32) / 128.0).astype(np.float32)
    ids = [f"doc{idx:04d}" for idx in range(N)]

    # Partitioned owner/quarter layout with a single shard under the quarter
    root = tmp_path / "data" / "t2"
    quarter = root / "ownerA" / "2025Q3"
    write_shard(quarter, ids, embeds, dtype="fp16", precompute_norms=True)

    # Minimal in-memory episodes to hydrate text
    eps = [{"id": _id, "text": f"text for {_id}"} for _id in ids]

    # Build config namespaces
    cfg_dict = {
        "perf": {
            "enabled": True,
            "metrics": {"report_memory": True},
            "t2": {
                "embed_store_dtype": "fp16",
                "precompute_norms": True,
                "reader": {
                    "partitions": {
                        "enabled": True,
                        "layout": "owner_quarter",
                        "path": str(root),
                    }
                },
            },
        },
        "t2": {  # stage-level optional knobs
            "embed_root": str(root),
            "reader_batch": 4096,
            "tiers": ["exact_semantic", "cluster_semantic", "archive"],  # legacy default
        },
    }
    cfg_ns = _ns(cfg_dict)
    # Ensure perf.t2.reader.partitions is dict-like (open_reader expects .get)
    cfg_ns.perf.t2.reader = {"partitions": cfg_dict["perf"]["t2"]["reader"]["partitions"]}
    # Ensure cfg.t2 is dict-like for code paths that call .get
    cfg_ns.t2 = cfg_dict["t2"]
    ctx = _Ctx(cfg=cfg_ns)
    state = {"mem_index": _Index(_eps=eps), "mem_backend": "inmemory"}

    # Provide a deterministic encoder via ctx (t2.py prefers ctx.enc if present)
    ctx.enc = _DummyEnc(D)

    # Run the reader-backed retrieval
    q_text = "any deterministic query"
    res = t2mod.t2_semantic(ctx, state, q_text, None)

    # Sanity checks
    assert isinstance(res.retrieved, list) and len(res.retrieved) > 0
    assert res.metrics.get("tier_sequence") == ["embed_store"]
    K = len(res.retrieved)

    # Compute expected Top-K directly from the shard using the same unit query vector
    q = np.ones((D,), dtype=np.float32)
    q /= float(np.linalg.norm(q, ord=2))
    norms = np.linalg.norm(embeds, ord=2, axis=1).astype(np.float32, copy=False)
    denom = np.where(norms == 0.0, 1.0, norms) * np.linalg.norm(q, ord=2)
    scores = (embeds @ q) / np.where(denom == 0.0, 1.0, denom)
    items = [(-float(scores[i]), ids[i], float(scores[i])) for i in range(N)]
    items.sort(key=lambda t: (t[0], t[1]))
    top_ids = [t[1] for t in items[:K]]
    top_scores = np.array([t[2] for t in items[:K]], dtype=np.float32)

    # Extract ids/scores from result
    out_ids = []
    out_scores = []
    for r in res.retrieved:
        rid = getattr(r, "id", None)
        if rid is None and isinstance(r, dict):
            rid = r.get("id")
        out_ids.append(str(rid))
        sc = getattr(r, "score", None)
        if sc is None and isinstance(r, dict):
            sc = r.get("score")
        out_scores.append(float(sc) if sc is not None else 0.0)
    out_scores = np.array(out_scores, dtype=np.float32)

    assert out_ids == top_ids, f"IDs mismatch:\nexpected={top_ids}\nactual=  {out_ids}"
    # Score parity within epsilon (identical math path)
    max_delta = float(np.max(np.abs(out_scores - top_scores))) if top_scores.size else 0.0
    assert max_delta <= 1e-6, f"Score drift {max_delta} exceeds 1e-6"