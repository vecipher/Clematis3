

import pytest

from clematis.engine.util.lru_bytes import LRUBytes
import clematis.engine.stages.t2 as t2mod


class _Ctx:
  def __init__(self, cfg):
    self.cfg = cfg


def _ctx_with_perf_cache(enabled: bool, max_entries: int, max_bytes: int):
  return _Ctx({
    "perf": {
      "enabled": enabled,
      "t2": {
        "cache": {
          "max_entries": max_entries,
          "max_bytes": max_bytes,
        }
      },
      "metrics": {"report_memory": True},
    },
    # Legacy t2.cache off (fallback not used in this smoke)
    "t2": {"cache": {"enabled": False}},
  })


def test_t2_cache_bytes_selected_when_perf_enabled():
  ctx = _ctx_with_perf_cache(True, 2, 64)
  cache, kind = t2mod._get_cache(ctx, ctx.cfg.get("t2", {}))
  assert kind == "bytes"
  assert isinstance(cache, LRUBytes)


def test_t2_cache_none_when_perf_disabled():
  ctx = _ctx_with_perf_cache(False, 2, 64)
  cache, kind = t2mod._get_cache(ctx, ctx.cfg.get("t2", {}))
  assert cache is None and kind is None


def test_t2_cache_eviction_and_sizes_bytes_mode():
  ctx = _ctx_with_perf_cache(True, 2, 32)
  cache, kind = t2mod._get_cache(ctx, ctx.cfg.get("t2", {}))
  assert kind == "bytes"
  
  # Insert two small items
  ev = cache.put("k1", {"v": 1}, 10)
  assert ev == (0, 0)
  ev = cache.put("k2", {"v": 2}, 10)
  assert ev == (0, 0)
  assert cache.size_entries() == 2
  assert cache.size_bytes() == 20
  
  # Insert third that forces eviction by entries cap
  ev_n, ev_b = cache.put("k3", {"v": 3}, 10)
  assert ev_n == 1
  assert cache.size_entries() == 2
  assert cache.size_bytes() <= 32
  
  # Insert an item that would exceed byte cap -> may trigger another eviction
  ev_n2, ev_b2 = cache.put("k4", {"v": 4}, 20)
  assert ev_n2 >= 0
  assert cache.size_entries() <= 2
  assert cache.size_bytes() <= 32


def test_t2_cache_rejects_oversize_item():
  ctx = _ctx_with_perf_cache(True, 10, 25)
  cache, kind = t2mod._get_cache(ctx, ctx.cfg.get("t2", {}))
  assert kind == "bytes"
  cache.put("a", 1, 10)
  cache.put("b", 2, 10)
  assert cache.size_bytes() == 20
  # Oversize (> max_bytes) should be rejected
  ev = cache.put("BIG", object(), 100)
  assert ev == (0, 0)
  assert cache.size_entries() == 2
  assert cache.size_bytes() == 20
import inspect


@pytest.mark.skip(reason="Enable when t2_semantic fixture/signature is exposed for tests.")
def test_t2_cache_gated_metrics_placeholder():
  """
  Placeholder: When enabled, this should:
    1) Build a minimal ctx with perf.enabled=true and metrics.report_memory=true,
       and perf.t2.cache caps > 0 to select bytes-mode.
    2) Call t2mod.t2_semantic(...) twice with identical inputs to exercise cache miss then hit.
    3) Assert that result.metrics contains gated counters when bytes cache is used:
       - 't2.cache_evictions' >= 0
       - 't2.cache_bytes'    >= 0
    4) Assert no new keys appear when perf.enabled=false (identity path).
  """
  # Implementation to be provided once test harness exposes a stable t2_semantic fixture.
  pass