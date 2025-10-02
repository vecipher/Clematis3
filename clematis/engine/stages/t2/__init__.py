from __future__ import annotations

from .core import run_t2, t2_pipeline, t2_semantic
from .config import (
    cfg_get as _cfg_get,
    ensure_dict as _ensure_dict,
    metrics_gate_on as _metrics_gate_on,
    quality_cfg_snapshot as _quality_cfg_snapshot,
)
from .helpers import (
    EpRefShim as _EpRefShim,
    build_label_map as _build_label_map,
    gather_changed_labels as _gather_changed_labels,
    items_for_fusion as _items_for_fusion,
    owner_for_query as _owner_for_query,
    parse_iso as _parse_iso,
    quality_digest as _quality_digest,
)
from .metrics import (
    emit_t2_metrics as _emit_t2_metrics,
    estimate_result_cost as _estimate_result_cost,
)
from .parallel import (
    collect_shard_hits as _collect_shard_hits,
    t2_parallel_enabled as _t2_parallel_enabled,
)
from .cache import get_cache as _get_cache
from ..t2_shard import merge_tier_hits_across_shards_dict as _merge_tier_hits_across_shards, _qscore

__all__ = [
    "t2_semantic",
    "run_t2",
    "t2_pipeline",
    "_cfg_get",
    "_ensure_dict",
    "_metrics_gate_on",
    "_quality_cfg_snapshot",
    "_items_for_fusion",
    "_emit_t2_metrics",
    "_get_cache",
    "_collect_shard_hits",
    "_t2_parallel_enabled",
    "_merge_tier_hits_across_shards",
    "_qscore",
    "_gather_changed_labels",
    "_build_label_map",
    "_parse_iso",
    "_owner_for_query",
    "_EpRefShim",
    "_quality_digest",
    "_estimate_result_cost",
]
