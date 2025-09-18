from __future__ import annotations
from typing import Any, Dict, List, Optional
import numpy as np
from numpy.typing import NDArray
from ..engine.types import EpisodeRef

class InMemoryIndex:
    def __init__(self) -> None:
        self._eps: List[Dict[str, Any]] = []

    def add(self, ep: Dict[str, Any]) -> None:
        self._eps.append(ep)

    def search_tiered(self, owner: Optional[str], q_vec: NDArray[np.float32], k: int, tier: str, hints: Dict[str, Any]) -> List[EpisodeRef]:
        # Placeholder: returns empty list
        return []
