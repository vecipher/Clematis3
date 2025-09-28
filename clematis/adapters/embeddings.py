from __future__ import annotations
from typing import List
import numpy as np
from numpy.typing import NDArray
import hashlib


class DummyEmbeddingAdapter:
    def encode(self, texts: List[str]) -> List[NDArray[np.float32]]:
        # Deterministic dummy vectors
        return [np.random.default_rng(0).random(32, dtype=np.float32) for _ in texts]


# Deterministic, content-dependent embedding stub.
# - Stable across runs/platforms
# - Distinct per input string
# - Unit-normalized vectors (optional)
class DeterministicEmbeddingAdapter:
    """
    Deterministic, content-dependent embedding stub.
    - Stable across runs/platforms
    - Distinct per input string
    - Unit-normalized vectors (optional)
    """

    def __init__(self, dim: int = 32, normalize: bool = True) -> None:
        self.dim = int(dim)
        self.normalize = bool(normalize)

    def encode(self, texts: List[str]) -> List[NDArray[np.float32]]:
        vecs: List[NDArray[np.float32]] = []
        for t in texts:
            # Use SHAKE-256 to generate dim*4 bytes deterministically
            raw = hashlib.shake_256(t.encode("utf-8")).digest(self.dim * 4)
            arr = np.frombuffer(raw, dtype=np.uint32).astype(np.float32)
            # Map to [-1, 1]
            arr = (arr / np.float32(0xFFFFFFFF)) * 2.0 - 1.0
            if self.normalize:
                n = float(np.linalg.norm(arr)) or 1.0
                arr = (arr / n).astype(np.float32)
            vecs.append(arr.astype(np.float32))
        return vecs


# Alias for clarity with planned BGE usage in T2
BGEAdapter = DeterministicEmbeddingAdapter
