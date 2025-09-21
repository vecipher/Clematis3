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

- Local smoke:
  ```bash
  pytest -q
  python3 scripts/validate_config.py .ci/validator_strict_quiet.yaml --strict
  python3 scripts/run_demo.py --config .ci/disabled_path_config.yaml
  ```
  
![disabled-path-identity](https://github.com/vecipher/Clematis3/actions/workflows/disabled_identity.yml/badge.svg)