# Changelog
All notable changes to this project will be documented in this file.

## [Unreleased]
## [0.7.0-M7] - 2025-09-29 (planned)

### Added (PR36 — Quality scaffold & shadow mode; defaults OFF)
- **Engine (T2):** `clematis/engine/stages/t2_quality_trace.py` — shadow tracing emitter that writes `rq_traces.jsonl` **without** mutating rankings or metrics. Deterministic headers: `trace_schema_version=1`, `git_sha` (from `CLEMATIS_GIT_SHA` or `"unknown"`), **semantic-only** `config_digest` (excludes `trace_dir`), `query_id` (Unicode NFKC + collapsed whitespace, lowercased), `clock=0`, `seed=0`.
- **Stage wiring:** `clematis/engine/stages/t2.py` emits a **write-only** shadow trace **behind a triple gate**: `perf.enabled && perf.metrics.report_memory && t2.quality.shadow && !t2.quality.enabled`. Ordering and scores are unchanged.
- **Configs (validator):** `configs/validate.py` adds `t2.quality.{enabled,shadow,trace_dir,redact}` with defaults (`enabled=false`, `shadow=false`, `trace_dir="logs/quality"`, `redact=true`). **PR36 guard:** `t2.quality.enabled=true` is rejected until PR37 lands. Verbose warning when the triple gate is incomplete.
- **Examples:** `examples/quality/shadow.yaml` — runnable example exercising the triple gate (ON) with redaction.
- **CLIs:** 
  - `scripts/rq_trace_dump.py` — tail/inspect recent trace entries.
  - `scripts/validate_noop_shadow.py` — Gate D comparator; asserts **no diffs** between baseline vs shadow runs **except** for `rq_traces.jsonl`.
- **Configs (defaults):** `configs/config.yaml` now includes a minimal `t2.quality` block (defaults OFF) so the identity path is explicit.
- **CI (Gate D — Quality shadow is a no‑op):** `.github/workflows/gate_d_quality_shadow.yml` (recommended as **required**). Runs baseline vs shadow and calls the comparator script; fails on any non-trace diffs.

### Changed (PR36)
- **Trace schema stability:** `config_digest` now hashes only **behavioral knobs** (excludes env-specific paths like `trace_dir`); `query_id` normalization tightened (NFKC + whitespace collapse + lowercase) to make A/B diffs mechanical.
- **Docs:** README gains “M7 — Retrieval Quality: shadow tracing (PR36, defaults OFF)” with quickstart, triple‑gate description, and rollback notes.


### Tests (PR36)
- `tests/quality/test_t2_quality_shadow.py` — shadow is a no‑op; triple gate required; validator guard for `enabled=true`.
- `tests/quality/test_trace_emitter_format.py` — redaction on/off; deterministic headers; config digest invariance; query normalization; item order preservation.
- `tests/scripts/test_validate_noop_shadow.py` — comparator ignores `rq_traces.jsonl` but fails on any other diff.
- `tests/examples/test_examples_quality_shadow_yaml.py` — example exists and validates; triple gate is satisfied.
- `tests/integration/test_t2_shadow_noop_integration.py` — end‑to‑end: baseline vs shadow produce identical rankings; trace appears only under shadow (separate dirs).


### Added (PR37 — Lexical BM25 + fusion; opt‑in)
- **Engine (T2):** `clematis/engine/stages/t2_quality.py` — deterministic lexical BM25 scorer over the current candidate set and rank‑based fusion; ties break by `lex(id)`.
- **Stage wiring:** `clematis/engine/stages/t2.py` calls fusion when `t2.quality.enabled=true`; emits quality metrics under the triple gate; **enabled‑path tracing** writes `rq_traces.jsonl` with `meta.reason="enabled"` and per‑item `rank_sem`, `rank_fused`, `score_fused`.
- **Configs (validator):** `configs/validate.py` now accepts `t2.quality.enabled=true` and adds `t2.quality.lexical.{bm25_k1,bm25_b,stopwords}` and `t2.quality.fusion.{mode,alpha_semantic}` (defaults + bounds). Warnings emitted if perf gates are off.
- **Examples:** `examples/quality/lexical_fusion.yaml` — runnable example for PR37.
- **CI (optional smoke):** `.github/workflows/pr37_quality_fusion_smoke.yml` — runs the test suite with PR37 enabled (`CLEMATIS_CONFIG=examples/quality/lexical_fusion.yaml`).

### Changed (PR37)
- **Traces:** When `t2.quality.enabled=true` and the triple gate is satisfied, the pipeline now writes **enabled‑path** traces (`meta.reason="enabled"`) in addition to PR36 shadow traces. Shadow behavior unchanged.
- **Docs:** README updated with a PR37 section, quickstart, and Logs details (enabled‑path fields).

### Tests (PR37)
- `tests/quality/test_t2_quality_fusion.py` — determinism of fused ordering, alpha extremes, and metrics presence only when enabled.
- `tests/integration/test_t2_enabled_traces_integration.py` — verifies that enabled‑path traces are produced with `meta.reason="enabled"`.

### Added (PR38 — MMR diversification; opt-in)
- **Engine (T2):** `clematis/engine/stages/t2_quality_mmr.py` — deterministic MMR utilities: `MMRItem`, `jaccard_distance`, `mmr_select`, `mmr_reorder_full`, and `avg_pairwise_distance` (pure, no side effects).
- **Stage wiring:** `clematis/engine/stages/t2_quality.py` — adds `maybe_apply_mmr(...)` glue that builds token sets (mirroring PR37) and reorders the fused list; ties break by `lex(id)`.
- **Stage wiring:** `clematis/engine/stages/t2.py` — invokes MMR after PR37 fusion when `t2.quality.mmr.enabled=true`; emits gated metrics (`t2q.mmr.lambda`, `t2q.mmr.selected`, `t2q.diversity_avg_pairwise`).
- **Configs (validator):** `configs/validate.py` accepts `t2.quality.mmr.{enabled,lambda,k}` with bounds; supports legacy aliases `lambda_relevance` and `k_final`; truncation warning uses the key provided (e.g., warns with `k_final` when that was set). Warns when `mmr.enabled=true` while `t2.quality.enabled=false` (no effect).
- **Examples:** `examples/quality/mmr.yaml` — minimal runnable example.

### Changed (PR38)
- **Determinism & invariants:** No RNG or wall-clock; ties break by `lex(id)`. When `t2.quality.enabled=false` or `mmr.enabled=false`, results match PR37 fused order. With MMR on, final order is deterministic.
- **Traces:** Enabled-path traces (PR37) now reflect the MMR-reordered list when MMR is engaged; shadow behavior unchanged. Triple gate unchanged.


### Tests (PR38)
- `tests/scripts/test_t2_quality_mmr.py` — selection extremes, tie-breaks, permutation properties, lambda validation.
- `tests/integration/test_t2_mmr_enabled_traces.py` — MMR reorders when enabled; no-op when disabled; text-only tokenization path; asserts gated metrics `t2q.mmr.lambda`, `t2q.mmr.selected`, and `t2q.diversity_avg_pairwise`.

### Added (PR39 — Normalizer & aliasing; quality paths only)
- **Engine (T2):** `clematis/engine/stages/t2_quality_norm.py` — deterministic helpers: `normalize_text`, `tokenize`, `load_alias_map` (cached), `apply_aliases` (exact/phrase; single‑pass, idempotent).
- **Stage wiring:** `clematis/engine/stages/t2_quality.py` — BM25 scorer and MMR glue now use PR39 normalizer/aliasing when `t2.quality.enabled=true`; disabled path unchanged.
- **Configs (validator):** `configs/validate.py` — adds `t2.quality.normalizer.{enabled,case,unicode}` and `t2.quality.aliasing.map_path`; warns when normalizer/aliasing are configured while `t2.quality.enabled=false`; warns when `map_path` is missing/unreadable (no‑op at runtime).
- **Examples:** `tests/examples/quality/normalizer_aliases.yaml`, `tests/examples/quality/aliases.yaml` — minimal runnable example + alias map.

### Changed (PR39)
- **Determinism & invariants:** No RNG/wall‑clock; ties via `lex(id)`; disabled‑path identity preserved. With no alias map, BM25 behavior remains parity with PR37; MMR token sets mirror the same normalization/aliasing policy.
- **Traces & gates:** Enabled‑path traces reflect any normalization/aliasing effects via the fused/MMR outputs; triple gate unchanged.

### Tests (PR39)
- `tests/quality/test_t2_quality_normalizer_aliases.py` — normalization (NFKC/lower/whitespace), tokenization (stops/minlen/stemmer), aliasing rewrite/expansion, alias map I/O; idempotency.
- `tests/integration/test_t2_quality_normalizer_aliases_integration.py` — alias map deterministically changes the MMR head (λ=1.0, k=2); repeatability.

## [0.6.0] - 2025-09-22
### Added
- **CI:** `disabled-path-identity` workflow at `.github/workflows/disabled_identity.yml`:
  - Validates repo config (non-strict), runs strict quiet validator, unit test, CLI smoke, and normalized log diff.
- **Configs:** `.ci/validator_strict_quiet.yaml` and `.ci/disabled_path_config.yaml` to support the workflow.
- **Tests:** Identity guard & validator smoke tests (e.g. `tests/test_identity_disabled_path.py`, `tests/test_config_perf_quality.py`).
- **Goldens:** `tests/goldens/pr29/.logs/` with normalized JSONL golden logs for disabled-path demo.

### Changed
- CI now normalizes logs using the shared Python normalizer (`tests/helpers/identity.py`) and sorts before diffing; volatile fields (`now`, `*_ms`, `*_bytes`, `version_etag`, `rss_bytes`) are dropped.

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

### Added (PR33.5 — Gate C: T2 reader parity & runtime‑flip invariants; defaults OFF)
- **CI:** Required workflow `.github/workflows/t2_reader_parity.yml`:
  - Parity matrix over `{embed_store_dtype∈{fp32,fp16}} × {precompute_norms∈{true,false}}` running parity + integration tests.
  - Identity sub‑job asserts reader does **not** engage when `perf.enabled=false`.
- **Scripts:** `scripts/seed_tiny_shard.py` — deterministic tiny store for parity tests.
- **Tests:** `tests/t2/test_t2_reader_gate_identity.py` — reader must not engage in disabled mode.
- **Helpers:** `tests/helpers/identity.py` — shared normalizer used by CI and tests (replaces `jq` pipeline).
- **Config loader:** `clematis/io/config.py` applies safe env overrides used by CI (`PERF_T2_DTYPE`, `PERF_T2_PRECOMPUTE_NORMS`) without flipping `perf.enabled`.

### Changed (PR33.5)
- **Engine (T2):** `engine/stages/t2.py` now:
  - Emits `metrics.tier_sequence` consistently and a **gated** `metrics.reader` **only** when the reader is engaged with `perf.enabled=true`.
  - Enforces deterministic final sort by `(-score, lex(id))` in all scoring paths.
  - Exposes a stable `run_t2(cfg, **kwargs)` adapter for tests/CI (delegates to stage entrypoint).
- **Validator:** `configs/validate.py`:
  - Accepts `perf.t2.reader.partitions.by` (strictly limited to `["owner","quarter"]`).
  - Emits a warning if any reader partitions config is present while `perf.enabled=false` (identity path).
- **Identity workflow:** `.github/workflows/disabled_identity.yml` tightened:
  - Editable dev/test install, deterministic locale/UTF‑8, shared Python normalizer for diffs, artifact upload on failure.
- **Demo:** `scripts/run_demo.py` adds deterministic clock by default (`--fixed-now-ms`, `--wall-clock` to opt‑in).

### Notes (PR33.5)
- **No behavior change by default:** with `perf.enabled=false`, outputs/logs remain identical to PR29 goldens.
- **Branch protection:** mark both jobs in `t2_reader_parity.yml` as required:
  - “Gate C — T2 Reader Parity / Parity matrix (dtype × norms)”
  - “Gate C — T2 Reader Parity / Reader must not engage (perf=OFF)”

### Added (PR34 — Snapshots: zstd compression + delta mode; defaults OFF)
- **Util:** `clematis/engine/util/snapshot_delta.py` — deterministic path-based delta codec (`compute_delta`, `apply_delta`).
- **Tests:**
  - `tests/snapshots/test_snapshot_delta_roundtrip.py` — delta round-trip identity.
  - `tests/snapshots/test_snapshot_fallback.py` — baseline missing ⇒ warning + full fallback.
  - `tests/config/test_validate_perf_snapshots.py` — validator bounds/warnings (aligned with current schema).
- **Examples:** `examples/perf/snapshots.yaml` — safe example config (used by PR35 docs).
- **CI:** `.github/workflows/snapshots_smoke.yml` — fast smoke for delta/fallback tests on relevant PRs.

### Changed (PR34)
- **Snapshot reader:** `clematis/engine/snapshot.py` adds `read_snapshot(...)` that understands two-line `header\npayload` format for **full** and **delta** files, supports optional `.json.zst` (when `zstandard` is installed), and performs a **safe fallback** to the matching full snapshot when the baseline is missing/mismatched, logging `SNAPSHOT_BASELINE_MISSING`.
- **IO shim:** `clematis/engine/io/snapshot.py` re-exports `read_snapshot` to avoid optional import errors at import time.
- **Validator:** `configs/validate.py` now emits a warning when any `perf.snapshots.*` keys are configured while `perf.enabled=false` (identity path; features remain disabled). No API shape changes.
- **Dev/Test deps:** add `zstandard>=0.22` to dev/test extras (`pyproject.toml`) and `requirements-dev.txt`.

### Notes (PR34)
- **No behavior change by default:** with `perf.enabled=false`, outputs/logs remain identical to PR29 goldens.
- **Semantics unchanged:** enabling compression/delta changes persistence cost/size only; reconstructed snapshot equals the full payload.
- **Metrics deferred:** no new snapshot metrics in PR34; future counters must be gated by `perf.enabled && perf.metrics.report_memory`.
- **Determinism:** ETags computed from canonical JSON; ordering and filenames are stable.

### Added (PR34.1 — Snapshot writer; read/write parity; defaults OFF)
- **Snapshot writer:** `clematis/engine/snapshot.py::write_snapshot_auto(...)` emits `snapshot-{etag}.full.json[.zst]` or `snapshot-{etag_to}.delta.json[.zst]` using the same two-line canonical `header\npayload` format as the reader.
- **Tests:** `tests/snapshots/test_snapshot_writer_roundtrip.py` — writes a baseline full + a delta from that baseline and verifies reconstruction equals the current payload.

### Changed (PR34.1)
- **Reader helpers:** `_read_text` and `_read_header_payload` now accept Path-like inputs for robustness in CLIs/tests; `.zst` handling remains optional and guarded.
- **README:** PR34 section updated to reflect writer availability in PR34.1 (read/write support; defaults unchanged).


### Added (Gate B — Metrics-only guard; required)
- **CI:** `.github/workflows/gate_b.yml` — asserts **no metrics** (including `tier_sequence`) appear when either:
  - `perf.enabled=false`, or
  - `perf.enabled=true` and `perf.metrics.report_memory=false`.
- **Scripts:** `scripts/ci_gate_b_assert.py` — scans normalized logs for disallowed keys; used by the workflow.
- **Configs:** `.ci/gate_b_off.yaml`, `.ci/gate_b_on_nometrics.yaml` — minimal configs to exercise the two OFF modes.

### Changed (Gate B)
- **Stages:**
  - `engine/stages/t2.py` — all metric emissions (including `tier_sequence`) are **gated** behind `perf.enabled && perf.metrics.report_memory`.
  - `engine/stages/t3.py` — no longer synthesizes `tier_sequence` when absent; passes through only what T2 emits under gate.
- **Tests:** updated `tests/test_t2_stage.py` to explicitly enable the metrics gate only in tests that assert on metrics.
- **Identity:** no change to disabled-path outputs; Gate B ensures accidental leaks are caught pre-merge.

### Added (Nice-to-haves — config-only / helper)
- **Validator:** optional config surface for `t2.lancedb.partitions` (e.g., `{by: ["owner","quarter"], shard_order: "lex|score"}`) with warnings when `perf.enabled=false`. **No runtime changes in M6.**
- **Helper:** `engine/util/metrics.py` — central `emit/emit_many/gate_on` helper to prevent future counters from bypassing the gate (not yet widely adopted in M6).

### Changed (Validator & examples smalls)
- **API compatibility:** `configs/validate.py` now supports both call styles:
  - `validate_config(cfg) -> normalized dict` (raises on errors), and
  - `validate_config(cfg, strict=..., verbose=...) -> (errors, warnings)`.
- **Examples:** `examples/perf/caches.yaml` now scopes caches under `perf.t1.cache` and `perf.t2.cache` (replacing the deprecated top-level `perf.cache`).

- **Docs:**
  - README gains an “M6 — Docs & CLIs (PR35)” section with quickstart, CLI usage, and links to examples.
  - CONTRIBUTING adds a rollback playbook (how to force disabled‑path identity) and a quick ref for Gates A/B/C.
- **Examples:**
  - `examples/perf/caps.yaml` — PR31 T1 caps + dedupe ring.
  - `examples/perf/caches.yaml` — PR32 LRU‑by‑bytes caches.
  - `examples/perf/fp16_reader.yaml` — PR33/33.5 fp16 + partition reader (opt‑in).
  - `examples/perf/snapshots.yaml` — PR34 compression + delta (safe fallback) referenced by docs.
- **CLIs (offline only):**
  - `scripts/mem_inspect.py` — lists snapshot files and basic stats (works with `.json` and optional `.json.zst`).
  - `scripts/mem_compact.py` — offline rewrite of **full** snapshots to a new directory with optional zstd compression; never in‑place.
- **Tests:**
  - `tests/cli/test_mem_inspect.py`, `tests/cli/test_mem_compact.py` — smoke tests (exit 0, expected keys/files).
- **CI:**
  - `.github/workflows/cli_smoke.yml` — runs CLI smoke tests on PRs touching `scripts/`.

### Changed (PR35)
- README updated with M6 quickstart, examples, and CLI docs. No changes to runtime code paths.
- CHANGELOG updated accordingly.

### Notes (PR35)
- **No runtime behavior change:** CLIs are offline and do not alter runtime semantics or logs.
- **Identity preserved:** With `perf.enabled=false`, behavior/logs remain identical to PR29 goldens (guarded by the identity CI gate).
- **Safety:** `mem_compact.py` writes to a new directory, never in‑place; deterministic ordering and headers.
- **Optional dependency:** `zstandard` is only required when reading/writing `.zst` files; included in dev/test deps.

### Added (PR40 — Lance partition reader parity; optional, default flat)
- **Engine (T2):** `clematis/engine/stages/t2_lance_reader.py` — dependency-light, deterministic partition wrapper over a flat reader; preserves content and within-partition order; never raises during availability checks.
- **Stage wiring:** `clematis/engine/stages/t2.py` — accepts `t2.reader.mode ∈ {flat, auto, partition}`; selects partition mode only when available; otherwise deterministically falls back to flat. Emits gated metric `t2.reader_mode` ("flat"|"partition"). No scoring/order changes.
- **Configs (validator):** `configs/validate.py` — adds `t2.reader.mode` enum and warning when `mode ∈ {partition, auto}` is set while `perf.enabled=false` (no effect).
- **Examples:** `tests/examples/quality/reader_parity.yaml` — minimal config to exercise mode selection and metrics.
- **CI (smoke):** `.github/workflows/pr40_reader_parity_smoke.yml` — matrix over `{flat, auto, partition}`; runs parity smoke tests; uploads artifacts on failure.

### Changed (PR40)
- **Observability only:** Default remains `flat`; when `partition` is unavailable, code falls back to `flat` and reports it via `t2.reader_mode` under the triple gate. Disabled-path identity and trace behavior remain unchanged.

### Tests (PR40)
- `tests/integration/test_t2_reader_parity.py` — emits `t2.reader_mode=flat` under gate; `partition`/`auto` fall back to `flat` when unavailable; metric suppressed when `report_memory=false`.