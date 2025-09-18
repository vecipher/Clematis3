from dataclasses import dataclass, field
from typing import Any, Dict, List
from numpy.typing import NDArray
import numpy as np

@dataclass
class MemoryEpisode:
    id: str
    owner: str
    text: str
    tags: List[str]
    ts: str
    vec_full: NDArray[np.float32] | None = None
    aux: Dict[str, Any] = field(default_factory=dict)
