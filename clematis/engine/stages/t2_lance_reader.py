

from __future__ import annotations

"""
PR40 — Partition-aware reader shim (optional; flat remains default)

This module provides a *deterministic* partitioned reader wrapper that can be
used to iterate an existing flat reader in partition order without changing
content or scores. If no partition metadata is available, it degrades to the
original flat iteration (parity preserved).

Design goals
  • No hard dependency on external storage libs. If a Lance/Arrow dataset or
    on-disk fixture is required by your environment, gate availability via
    simple file/dir checks and fall back to the flat reader.
  • Determinism: no RNG/wall‑clock. Partition keys are ordered lexicographically;
    items keep their original within‑partition order.
  • Safety: never raise during capability checks; callers decide to fall back.

Intended usage (from t2.py):
  spec = LancePartitionSpec(root=FIXTURE_ROOT, by=("owner", "quarter"))
  preader = PartitionedReader(spec, flat_iter=open_reader_iter, id_to_meta=meta_map)
  if preader.available():
      for ids, embeds, norms in preader.iter_blocks(batch=1024):
          ...  # identical content to flat reader, grouped by partition
  else:
      ...  # use flat reader directly

Notes
  • `open_reader_iter` is any zero-arg callable returning an iterator of
    (ids: List[str], embeds: np.ndarray, norms: np.ndarray) blocks.
  • `id_to_meta` is an optional mapping from id -> dict of metadata fields used
    to compute partition keys. If omitted, the wrapper yields the flat order.
"""

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, Iterator, List, Mapping, MutableMapping, Optional, Sequence, Tuple
import os
import numpy as np

__all__ = [
    "LancePartitionSpec",
    "PartitionedReader",
]


@dataclass(frozen=True)
class LancePartitionSpec:
    """Describes a partitioned dataset.

    Attributes
    ----------
    root : str
        Dataset root or fixture directory. Availability checks are kept light
        (exists and is a directory).
    by : Tuple[str, ...]
        Ordered tuple of metadata fields that define the partition key.
    """

    root: str
    by: Tuple[str, ...] = ()


class PartitionedReader:
    """Deterministic partition wrapper over an existing flat reader.

    Parameters
    ----------
    spec : LancePartitionSpec
        Partition descriptor (root path and fields to partition by).
    flat_iter : Callable[[], Iterable[Tuple[List[str], np.ndarray, np.ndarray]]]
        Zero-arg callable that returns an *iterator* over flat reader blocks.
    id_to_meta : Optional[Mapping[str, Mapping[str, object]]]
        Optional mapping from id -> metadata dict (used to compute partition
        keys). If absent or incomplete, the wrapper yields the flat order.
    """

    def __init__(
        self,
        spec: LancePartitionSpec,
        flat_iter: Optional[Callable[[], Iterable[Tuple[List[str], np.ndarray, np.ndarray]]]] = None,
        id_to_meta: Optional[Mapping[str, Mapping[str, object]]] = None,
    ) -> None:
        self._spec = spec
        self._flat_iter = flat_iter
        self._meta = id_to_meta or {}

    # ------------------------------ Capability ------------------------------
    def available(self) -> bool:
        """Return True if the partitioned path looks usable.

        This is intentionally conservative: if `root` is a directory we deem
        the fixture "available" and let the caller attempt to use it. If not,
        we report False so the caller can fall back to the flat reader.
        """
        try:
            root = self._spec.root
            return bool(root) and os.path.isdir(root)
        except Exception:
            return False

    # ----------------------------- Iteration API ----------------------------
    def iter_blocks(self, batch: int) -> Iterator[Tuple[List[str], np.ndarray, np.ndarray]]:
        """Yield blocks grouped by partition, preserving content and order.

        If no `flat_iter` is provided or no partition metadata is available,
        this yields the original flat blocks unchanged. When metadata is
        provided, it *re-groups* each flat block into per-partition sub-blocks
        and yields them with partition keys in lexicographic order.
        """
        if self._flat_iter is None:
            # Nothing to iterate; yield nothing deterministically.
            return
            yield  # pragma: no cover (generator form, never executed)

        # Build a stable ordering of partition keys up-front as we see them.
        # We avoid randomization and preserve within-partition arrival order.
        for ids, embeds, norms in self._flat_iter():
            if not self._spec.by or not self._meta:
                # No partitioning information — passthrough for parity.
                yield ids, embeds, norms
                continue

            # Partition the current block deterministically.
            buckets: Dict[Tuple[object, ...], List[int]] = {}
            for idx, id_ in enumerate(ids):
                meta = self._meta.get(id_, {})
                key = tuple(meta.get(field, "") for field in self._spec.by)
                buckets.setdefault(key, []).append(idx)

            # Emit per-partition slices in lexicographic key order.
            for key in sorted(buckets.keys(), key=lambda t: tuple(str(x) for x in t)):
                sel = buckets[key]
                # Keep original order within the partition
                sub_ids = [ids[i] for i in sel]
                sub_emb = embeds[sel]
                sub_nrm = norms[sel]
                yield sub_ids, sub_emb, sub_nrm