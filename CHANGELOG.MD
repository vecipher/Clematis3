# Changelog
All notable changes to this project will be documented in this file.

## [Unreleased]

## [0.6.0] - 2025-09-21
### Added
- **CI:** `disabled-path-identity` workflow at `.github/workflows/disabled_identity.yml`:
  - Validates repo config (non-strict), runs strict quiet validator, unit test, CLI smoke, and normalized log diff.
- **Configs:** `.ci/validator_strict_quiet.yaml` and `.ci/disabled_path_config.yaml` to support the workflow.
- **Tests:** Identity guard & validator smoke tests (e.g. `tests/test_identity_disabled_path.py`, `tests/test_config_perf_quality.py`).
- **Goldens:** `tests/goldens/pr29/.logs/` with normalized JSONL golden logs for disabled-path demo.

### Changed
- CI now normalizes logs (via `jq`) to ignore volatile fields (`now`, `*_ms`, `version_etag`) before diffing.

### Added (PR31 — T1 compaction; defaults OFF)
- **Engine (T1):** Deterministic compaction controls under `perf.t1`:
  - `caps.frontier` — bounds the T1 priority queue (effective limit is `min(t1.queue_budget, caps.frontier)`).
  - `caps.visited` — bounds the visited set using a FIFO-on-first-visit structure.
  - `dedupe_window` — short-horizon push deduplication to reduce thrash.
  - All features are gated; **disabled-path identity** is preserved when `perf.enabled=false` or keys are absent.
- **Utilities:** 
  - `engine/util/ring.py` — `DedupeRing` (O(1) membership via refcounts) for push dedupe.
  - `engine/util/lru_det.py` — `DeterministicLRUSet` (visited bound) and `DeterministicLRU` (map cache for future PRs).
- **Validator:** Support for `perf.t1.caps.{frontier,visited}` and `perf.t1.dedupe_window` (values must be ≥1). 
  - Legacy `perf.t1.queue_cap` is accepted and mapped to `caps.frontier` if not provided.
  - Emits a warning when caps/dedupe are configured while `perf.enabled=false` (identity path).
- **Docs:** README section “M6 — Perf & Compaction (PR31: T1 caps + dedupe; OFF by default)” with enablement example and metrics.

### Tests (PR31)
- `tests/util/test_ring.py` — deterministic coverage for `DedupeRing`.
- `tests/util/test_lru_det.py` — deterministic coverage for `DeterministicLRUSet` / `DeterministicLRU`.
- `tests/config/test_validate_perf_t1_caps.py` — CLI strict/invalid/warning scenarios.
- `tests/stages/test_t1_compaction.py` — placeholder (skipped) until a stable T1 graph fixture is exposed.


### Changed (PR31)
- `engine/stages/t1.py` wired to use the new caps and dedupe **behind gates**; default behavior/logs remain identical to PR29 goldens (identity CI).
- Metrics only emit when `perf.enabled=true` **and** `perf.metrics.report_memory=true`:
  - `t1_frontier_evicted`, `t1_dedup_hits`, `t1_visited_evicted`.

### Added (PR32 — Size‑aware caches for T1/T2; defaults OFF)
- **Utility:** `engine/util/lru_bytes.py` — `LRUBytes` (deterministic LRU‑by‑bytes with `(max_entries, max_bytes)`, oversize insert rejection, optional `on_evict`).
- **Stages:**
  - `engine/stages/t1.py` — PR32‑aware cache selector prefers `perf.t1.cache` (bytes) when `perf.enabled=true` and caps > 0; legacy LRU fallback preserved.
  - `engine/stages/t2.py` — wired to `perf.t2.cache` (bytes) with deterministic cost estimation; legacy LRU+TTL fallback preserved.
- **Validator:** accepts `perf.t1.cache.{max_entries,max_bytes}` and `perf.t2.cache.{max_entries,max_bytes}` (≥ 0) and warns when configured while `perf.enabled=false` (identity path).
- **Docs:** README section “M6 — Perf & Compaction (PR32: Size‑aware caches for T1/T2; OFF by default)”.

### Tests (PR32)
- `tests/util/test_lru_bytes.py` — sizes, multi‑eviction, MRU on get/put, oversize reject, clear.
- `tests/config/test_validate_perf_cache_pr32.py` — strict acceptance, disabled‑perf warnings, zero‑caps disabled.
- `tests/stages/test_t2_cache_smoke.py` — bytes‑mode selection and eviction math; metrics smoke is skipped pending a stable fixture.

### Changed (PR32)
- **Identity preserved:** with `perf.enabled=false` or zero/absent caps, behavior and logs remain identical to PR29/PR31 disabled path.
- **Metrics (gated):** new counters only when `perf.enabled=true` **and** `perf.metrics.report_memory=true`:
  - `t1.cache_evictions`, `t1.cache_bytes`
  - `t2.cache_evictions`, `t2.cache_bytes`

### Added (PR33 — T2 fp16 store + fp32 math + partition‑friendly reader; opt‑in runtime)
- **Utility:** `engine/util/embed_store.py` — shard writer/reader with deterministic iteration:
  - `write_shard(dir, ids, embeds, *, dtype="fp16|fp32", precompute_norms: bool)`
  - `open_reader(root, partitions={enabled, layout, path}) -> EmbedReader`
  - `EmbedReader.iter_blocks(batch)` yields `(ids, embeds_fp32, norms_fp32)`; no RNG/clocks.
- **Stage (T2):** `engine/stages/t2.py` wired for **opt‑in** runtime retrieval from the on‑disk embed store (cosine in fp32) when `perf.enabled=true` **and** `perf.t2.reader.partitions.enabled=true`; otherwise legacy in‑memory path is used. Emits gated metrics about store dtype, partition layout, and shard count.
- **Validator:** accepts `perf.t2.embed_store_dtype: {fp32, fp16}`, `perf.t2.precompute_norms: bool`, and `perf.t2.reader.partitions.{enabled, layout, path}` (`layout ∈ {owner_quarter, none}`); accepts stage knobs `t2.reader_batch` (≥ 1) and `t2.embed_root` (non‑empty string). Warns when `embed_store_dtype=fp16` with `precompute_norms=false`.
- **Docs:** README section “M6 — Perf & Compaction (PR33: T2 fp16 store + fp32 math + partition‑friendly reader; opt‑in runtime)”.

### Tests (PR33)
- `tests/t2/test_fp16_parity.py` — fp16 store + fp32 math parity vs fp32 baseline (Top‑K identical; ≤ 1e‑6 score drift).
- `tests/t2/test_partition_reader.py` — owner/quarter partitions parity (unpartitioned vs partitioned).
- `tests/t2/test_reader_backcompat.py` — fp32 shards without norms compute norms on the fly; missing meta rejected.
- `tests/stages/test_t2_reader_integration.py` — **runtime flip** path returns identical Top‑K and scores; `tier_sequence == ["embed_store"]`.
- `tests/t2/test_metrics_gated.py` — placeholders documenting gated metric expectations (skipped until fixture is exposed).
- `tests/config/test_validate_perf_t2_fp16.py` — validator accepts fp16+norms and partitions; rejects bad layouts/paths; warns when norms off.
- `tests/config/test_validate_t2_reader_batch.py` — validator accepts `t2.reader_batch` & `t2.embed_root`; rejects invalid; warns when partitions configured with `perf.enabled=false`.

### Changed (PR33)
- **Identity preserved by default:** with `perf.enabled=false` (or reader disabled), behavior/logs remain identical to PR29/PR31/PR32 disabled path.
- **Opt‑in runtime reader:** enabling `perf.t2.reader.partitions.enabled=true` retrieves from the embed store deterministically; legacy tiers are bypassed.
- **Metrics (gated):** new keys emitted only when `perf.enabled=true` **and** `perf.metrics.report_memory=true`:
  - `t2.embed_dtype`, `t2.embed_store_dtype`, `t2.precompute_norms`, `t2.reader_shards`, `t2.partition_layout`, and `t2.tier_sequence` (=`["embed_store"]` when active).

- Local smoke:
  ```bash
  pytest -q
  python3 scripts/validate_config.py .ci/validator_strict_quiet.yaml --strict
  python3 scripts/run_demo.py --config .ci/disabled_path_config.yaml
  ```
  
![disabled-path-identity](https://github.com/vecipher/Clematis3/actions/workflows/disabled_identity.yml/badge.svg)