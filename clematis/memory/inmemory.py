def _iter_shards_for_t2(self, tier: str, suggested: int | None = None):
    """
    Private: yield lightweight shard views for PR68 parallel T2.

    Invariants:
    - Deterministic iteration order by episode index.
    - No copying of vectors; uses `_view(...)` to produce lightweight shard views.
    - If `suggested` is None/<=1 OR there are <=1 episodes, yields `self` only.
    """
    episodes = self._episodes_for_tier(tier)  # existing internal accessor
    n = len(episodes)
    # Fallbacks that keep the disabled path identical and avoid zero-step slicing.
    if suggested is None or suggested <= 1 or n <= 1:
        yield self  # treat full index as a single shard
        return

    try:
        parts = int(suggested)
    except Exception:
        parts = 1

    # Clamp parts into [2, n] to avoid degenerate chunking.
    parts = max(2, min(parts, n))
    # Ceil-divide to compute chunk size; guard against 0.
    size = (n + parts - 1) // parts
    if size <= 0:
        yield self
        return

    for i in range(0, n, size):
        # _view should return a lightweight subset index with search_tiered()
        view = self._view(episodes[i:i + size])
        if view is None:
            # Defensive: if view construction fails, collapse to single shard.
            yield self
            return
        yield view
