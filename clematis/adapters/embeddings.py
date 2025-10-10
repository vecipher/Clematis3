from __future__ import annotations
from typing import List, Optional
import numpy as np
from numpy.typing import NDArray
import hashlib
import os
import logging

logger = logging.getLogger(__name__)


class _DevDummyEmbeddingAdapter:
    """Legacy adapter kept for documentation examples (not exported).

    Prefer ``DeterministicEmbeddingAdapter`` / ``BGEAdapter`` in code so embeddings
    remain content-dependent and stable across runs.
    """

    def encode(self, texts: List[str]) -> List[NDArray[np.float32]]:
        rng = np.random.default_rng(0)
        return [rng.random(32, dtype=np.float32) for _ in texts]


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


# Alias for clarity with planned BGE usage in T2 (deterministic fallback)
class BGEAdapter:
    """
    Wrapper that attempts to load the real BGE v1.5 encoder when available.

    By default we keep the deterministic adapter for reproducibility. Set the
    environment variable ``CLEMATIS_USE_REAL_BGE=1`` (or pass ``use_real=True``)
    to enable the SentenceTransformer-backed encoder.
    """

    _ENV_FLAG = "CLEMATIS_USE_REAL_BGE"

    def __init__(
        self,
        dim: int = 32,
        normalize: bool = True,
        *,
        use_real: Optional[bool] = None,
        model_name: str = "BAAI/bge-base-en-v1.5",
        device: Optional[str] = None,
    ) -> None:
        flag = use_real
        if flag is None:
            flag = os.getenv(self._ENV_FLAG, "").strip().lower() in {"1", "true", "yes", "on"}

        self.normalize = bool(normalize)
        self._stub = DeterministicEmbeddingAdapter(dim=dim, normalize=normalize)
        self._model = None
        self._use_real = bool(flag)
        self.dim = self._stub.dim

        if self._use_real:
            try:
                from sentence_transformers import SentenceTransformer  # type: ignore

                logger.info("Loading real BGE encoder '%s' (device=%s)", model_name, device or "auto")
                self._model = SentenceTransformer(model_name, device=device)
                try:
                    self.dim = int(self._model.get_sentence_embedding_dimension())  # type: ignore[attr-defined]
                except Exception:
                    # Fallback: infer dim from a dummy encode
                    sample = self._model.encode(["probe"], convert_to_numpy=True)
                    self.dim = int(np.asarray(sample[0]).shape[-1])
            except Exception as exc:
                logger.warning(
                    "Falling back to deterministic BGE adapter (failed to load '%s': %s)",
                    model_name,
                    exc,
                )
                self._model = None
                self._use_real = False

    def encode(self, texts: List[str]) -> List[NDArray[np.float32]]:
        if self._model is not None:
            vectors = self._model.encode(
                texts,
                convert_to_numpy=True,
                normalize_embeddings=self.normalize,
            )
            return [np.asarray(vec, dtype=np.float32) for vec in vectors]
        return self._stub.encode(texts)


__all__ = ["DeterministicEmbeddingAdapter", "BGEAdapter"]
