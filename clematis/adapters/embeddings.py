from __future__ import annotations
from typing import List
import numpy as np
from numpy.typing import NDArray

class DummyEmbeddingAdapter:
    def encode(self, texts: List[str]) -> List[NDArray[np.float32]]:
        # Deterministic dummy vectors
        return [np.random.default_rng(0).random(32, dtype=np.float32) for _ in texts]
