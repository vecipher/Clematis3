from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "write_shard",
    "open_reader",
    "EmbedReader",
]

SCHEMA_VERSION = "t2:1"


# ------------------------------- Writer ------------------------------------
def write_shard(
    shard_dir: str | Path,
    ids: Sequence[str],
    embeds: np.ndarray,
    *,
    dtype: str = "fp32",
    precompute_norms: bool = False,
) -> None:
    """
    Write a single embedding shard to disk.

    Layout inside shard_dir:
      - meta.json         (small JSON with dtype/shape/flags)
      - embeddings.bin    (raw contiguous float16/float32 array of shape [N, D])
      - ids.tsv           (one id per line)
      - norms.bin         (optional float32 [N] if precompute_norms=True)
    """
    shard = Path(shard_dir)
    shard.mkdir(parents=True, exist_ok=True)

    if not isinstance(embeds, np.ndarray):
        raise TypeError("embeds must be a numpy.ndarray")
    if embeds.ndim != 2:
        raise ValueError("embeds must have shape (N, D)")
    n, d = int(embeds.shape[0]), int(embeds.shape[1])
    if len(ids) != n:
        raise ValueError(f"ids length ({len(ids)}) must match embeds.shape[0] ({n})")

    # Normalize dtype
    dtype = str(dtype or "fp32").lower()
    if dtype not in {"fp16", "fp32"}:
        raise ValueError("dtype must be 'fp16' or 'fp32'")
    out_dtype = np.float16 if dtype == "fp16" else np.float32

    # Write ids.tsv deterministically
    ids_path = shard / "ids.tsv"
    with ids_path.open("w", encoding="utf-8", newline="\n") as f:
        for _id in ids:
            f.write(f"{_id}\n")

    # Write embeddings.bin (row-major)
    emb_path = shard / "embeddings.bin"
    np.ascontiguousarray(embeds, dtype=out_dtype).tofile(emb_path)

    # Optional norms.bin (always float32 for numeric parity)
    norms_flag = bool(precompute_norms)
    if norms_flag:
        norms = np.linalg.norm(np.asanyarray(embeds, dtype=np.float32), ord=2, axis=1)
        norms = norms.astype(np.float32, copy=False)
        norms_path = shard / "norms.bin"
        norms.tofile(norms_path)

    # meta.json
    meta = {
        "schema": SCHEMA_VERSION,
        "embed_dtype": dtype,
        "norms": norms_flag,
        "dim": d,
        "count": n,
    }
    (shard / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )


# ------------------------------- Reader ------------------------------------
@dataclass
class _ShardInfo:
    path: Path
    dtype: str
    dim: int
    count: int
    has_norms: bool


class EmbedReader:
    """
    Partition-friendly, deterministic reader for T2 embedding shards.

    Usage:
      reader = open_reader(root, partitions={"enabled": True, "layout": "owner_quarter"})
      for ids, embeds_fp32, norms_fp32 in reader.iter_blocks(batch=1024):
          ...
    """

    def __init__(self, shards: List[_ShardInfo], *, layout: str = "none"):
        if not shards:
            raise FileNotFoundError("No shards discovered (meta.json not found)")
        self._shards = shards
        self._layout = layout
        # Aggregate meta
        self.meta = {
            "embed_store_dtype": _dominant_dtype(shards),
            "dim": _assert_same([s.dim for s in shards], name="dim"),
            "count": sum(s.count for s in shards),
            "norms": all(s.has_norms for s in shards),
            "shards": len(shards),
            "partition_layout": layout,
            "schema": SCHEMA_VERSION,
        }

    # Deterministic iterator over blocks across shards
    def iter_blocks(
        self, batch: int
    ) -> Iterator[Tuple[List[str], np.ndarray, Optional[np.ndarray]]]:
        if batch <= 0:
            raise ValueError("batch must be >= 1")
        for sh in self._shards:  # already sorted by path
            ids_path = sh.path / "ids.tsv"
            emb_path = sh.path / "embeddings.bin"
            norms_path = sh.path / "norms.bin"

            # Load ids once
            with ids_path.open("r", encoding="utf-8") as f:
                ids_all = [line.rstrip("\n") for line in f]
            if len(ids_all) != sh.count:
                # Be tolerant; use the shorter length deterministically
                m = min(len(ids_all), sh.count)
            else:
                m = sh.count

            # Memory-map embeddings for chunked reading
            dt = np.float16 if sh.dtype == "fp16" else np.float32
            mm = np.memmap(emb_path, mode="r", dtype=dt, shape=(sh.count, sh.dim))
            norms_mm = None
            if sh.has_norms and norms_path.exists():
                norms_mm = np.memmap(norms_path, mode="r", dtype=np.float32, shape=(sh.count,))

            # Yield batches
            i = 0
            while i < m:
                j = min(i + batch, m)
                ids_batch = ids_all[i:j]
                vecs = np.asarray(mm[i:j], dtype=np.float32)  # cast to fp32 for math parity
                if norms_mm is not None:
                    norms = np.asarray(norms_mm[i:j], dtype=np.float32)
                else:
                    # Compute norms on the fly (fp32)
                    norms = np.linalg.norm(vecs, ord=2, axis=1).astype(np.float32, copy=False)
                yield ids_batch, vecs, norms
                i = j

    # Convenience: iterate as a single block (use carefully)
    def load_all(self) -> Tuple[List[str], np.ndarray, Optional[np.ndarray]]:
        ids: List[str] = []
        chunks: List[np.ndarray] = []
        norms_chunks: List[np.ndarray] = []
        for i_ids, i_vecs, i_norms in self.iter_blocks(batch=1_000_000_000):
            ids.extend(i_ids)
            chunks.append(i_vecs)
            if i_norms is not None:
                norms_chunks.append(i_norms)
        vecs = np.vstack(chunks) if chunks else np.zeros((0, 0), dtype=np.float32)
        norms = np.concatenate(norms_chunks) if norms_chunks else None
        return ids, vecs, norms


# ---------------------------- Discovery API --------------------------------
def open_reader(root: str | Path, *, partitions: Optional[dict] = None) -> EmbedReader:
    """
    Discover shards under `root` deterministically and build an EmbedReader.

    partitions example:
      {"enabled": True, "layout": "owner_quarter", "path": "/data/t2"}
    When disabled or None, we look for either a single shard at `root` or immediate
    subdirectories containing `meta.json`.
    """
    root = Path((partitions or {}).get("path", root))
    layout = (
        (partitions or {}).get("layout", "none")
        if (partitions or {}).get("enabled", False)
        else "none"
    )

    shard_dirs: List[Path] = _discover_shards(root, layout)
    shards: List[_ShardInfo] = []
    for sdir in shard_dirs:
        meta_path = sdir / "meta.json"
        if not meta_path.exists():
            # Skip anything that doesn't look like a shard
            continue
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception as e:
            raise RuntimeError(f"Failed to read meta.json at {meta_path}: {e}")
        dtype = str(meta.get("embed_dtype", "fp32")).lower()
        if dtype not in {"fp16", "fp32"}:
            dtype = "fp32"
        dim = int(meta.get("dim", 0))
        count = int(meta.get("count", 0))
        has_norms = bool(meta.get("norms", False)) and (sdir / "norms.bin").exists()
        # Be strict about required files
        if not (sdir / "embeddings.bin").exists() or not (sdir / "ids.tsv").exists():
            raise FileNotFoundError(f"Shard {sdir} missing embeddings.bin or ids.tsv")
        shards.append(_ShardInfo(path=sdir, dtype=dtype, dim=dim, count=count, has_norms=has_norms))

    # Deterministic order by path
    shards.sort(key=lambda s: str(s.path))
    return EmbedReader(shards, layout=layout)


# ---------------------------- Helpers --------------------------------------
def _discover_shards(root: Path, layout: str) -> List[Path]:
    """Return a sorted list of shard directories under root for the given layout."""
    root = Path(root)
    if (root / "meta.json").exists():
        return [root]

    candidates: List[Path] = []
    if layout == "owner_quarter":
        # Expect root/OWNER/QUARTER/<shard-dir> or meta.json directly under QUARTER
        if not root.exists():
            return []
        for owner in sorted([p for p in root.iterdir() if p.is_dir()], key=lambda p: p.name):
            for quarter in sorted([p for p in owner.iterdir() if p.is_dir()], key=lambda p: p.name):
                # shard directories under quarter
                subdirs = [p for p in quarter.iterdir() if p.is_dir()]
                shard_like = [p for p in subdirs if (p / "meta.json").exists()]
                if shard_like:
                    candidates.extend(shard_like)
                elif (quarter / "meta.json").exists():
                    candidates.append(quarter)
    else:
        # Generic: immediate shard directories under root or root itself
        if not root.exists():
            return []
        subdirs = [p for p in root.iterdir() if p.is_dir()]
        for p in subdirs:
            if (p / "meta.json").exists():
                candidates.append(p)
        # Single-shard case where files are directly under root
        if (root / "meta.json").exists():
            candidates.append(root)

    # Deduplicate & sort
    out: List[Path] = []
    seen = set()
    for p in sorted(candidates, key=lambda x: str(x)):
        s = str(p)
        if s not in seen:
            seen.add(s)
            out.append(p)
    return out


def _dominant_dtype(shards: List[_ShardInfo]) -> str:
    # If any shard is fp16, report fp16; else fp32
    for s in shards:
        if s.dtype == "fp16":
            return "fp16"
    return "fp32"


def _assert_same(vals: Iterable[int], name: str = "value") -> int:
    vals = list(vals)
    if not vals:
        return 0
    base = vals[0]
    for v in vals[1:]:
        if int(v) != int(base):
            raise ValueError(f"Shard meta mismatch: {name} differs across shards ({base} vs {v})")
    return int(base)
