

import pytest

import clematis.engine.stages.t2 as t2mod

HAVE_T2 = hasattr(t2mod, "t2_semantic")


@pytest.mark.skipif(not HAVE_T2, reason="t2_semantic not available for direct invocation in tests")
def test_metrics_present_when_perf_and_reporting_enabled():
    """
    Enable once a stable t2_semantic fixture/signature is exposed.
    Expectations when perf.enabled=true AND metrics.report_memory=true:
      - metrics include: 't2.embed_dtype' == 'fp32'
      - metrics include: 't2.embed_store_dtype' in {'fp16','fp32'} (from reader meta or config)
      - metrics include: 't2.precompute_norms' in {True, False}
      - metrics may include: 't2.reader_shards' (>=0), 't2.partition_layout' ('none'|'owner_quarter')
      - if perf.t2.cache.{max_entries|max_bytes} > 0, bytes-cache counters appear:
          't2.cache_evictions' >= 0, 't2.cache_bytes' >= 0
    """
    raise pytest.skip("Enable when test harness can call t2_semantic with a minimal ctx/state")


@pytest.mark.skipif(not HAVE_T2, reason="t2_semantic not available for direct invocation in tests")
def test_no_new_metrics_on_identity_path():
    """
    With perf.enabled=false OR metrics.report_memory=false, **no new PR32/PR33 metrics** should be added.
    Expectations:
      - keys starting with 't2.' (cache_evictions, cache_bytes, embed_dtype, embed_store_dtype, precompute_norms,
        reader_shards, partition_layout) are **absent**.
    """
    raise pytest.skip("Enable when test harness can call t2_semantic with a minimal ctx/state")