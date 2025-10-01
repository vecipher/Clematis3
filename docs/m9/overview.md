# Milestone 9 — Deterministic Parallelism (PR63 surface)

This document describes the configuration **surface only** added in PR63. It does **not** change runtime behavior. All defaults keep parallel execution **OFF** and preserve byte‑for‑byte identity with prior milestones.

## Purpose
Provide a stable, validated config contract for later PRs (PR64+). Teams can start wiring flags in their local branches without risking behavior drift on `main`.

## Config & identity rules

### Keys (under `perf.parallel`)
- `enabled` (bool, default `false`)
- `max_workers` (int, default `0`) — `0/1` means **sequential**; negative values normalize to `0`.
- `t1` (bool, default `false`) — gate for T1 stage parallelism (to be implemented in later PRs).
- `t2` (bool, default `false`) — gate for T2 stage parallelism (later PRs).
- `agents` (bool, default `false`) — gate for agent driver parallelism (later PRs).

### Normalization & validation
- Unknown keys under `perf.parallel` fail validation with a clear path, e.g. `perf.parallel.foo unknown key`.
- `max_workers <= 0` normalizes to `0` (sequential). Tests assert this normalization.
- The validator **does not materialize** a `perf.parallel` block when the user does not provide one, avoiding golden churn.
- No other parts of the config change semantics in PR63.

### Identity guarantee (PR63)
- With defaults (`enabled=false`, all gates false, `max_workers=0`), outputs/logs remain **byte‑for‑byte** identical to pre‑M9.
- CI and golden tests must pass unchanged. There are no new required CI jobs in this PR.

## Examples

### Default (identity‑preserving)
```yaml
# configs/config.yaml (excerpt)
perf:
  parallel:
    enabled: false
    max_workers: 0   # 0/1 = sequential
    t1: false
    t2: false
    agents: false
```

### Opt‑in (no effect yet; surface only)
```yaml
perf:
  parallel:
    enabled: true
    max_workers: 4
    t1: true
    t2: true
    agents: false
```
*Result in PR63:* validation passes and values normalize as described, but runtime remains sequential because later PRs implement execution changes.

### CLI validation
Use the umbrella CLI `validate` command:
```bash
clematis validate --in configs/config.yaml
```
If an unknown key is present, validation fails with a message like:
```
perf.parallel.foo unknown key
```

## Forward‑looking determinism requirements (for PR64+)
These rules guide the upcoming implementation and are called out here so reviewers can align on expectations:
- **Deterministic ordering** after parallel work: use explicit, stable sort keys (e.g., `graph_id`, `shard_id`, `(score,id)` with a documented tie‑break).
- **Use the helper introduced in PR64.** Adopt `run_parallel` (see [`parallel_helper.md`](./parallel_helper.md)) to enforce deterministic aggregation via `order_key`, a sequential identity path when `max_workers<=1`, and a sorted error model for exceptions.
- **No nondeterministic sources** in parallel paths: no random seeds, time‑based ordering, or hidden global state.
- **Thread‑safe caches**: default to lock‑wrapped single cache; optional per‑worker caches must have deterministic post‑merge. See [cache_safety.md](./cache_safety.md).
- **Logging**: stage outputs buffered and written through a centralized, ordered writer to guarantee JSONL line order.
- **Scheduler/driver**: execution model may run tasks concurrently, but **fairness and policy** semantics must be unchanged.

## Troubleshooting
- **Why is nothing faster when I flip the flags in PR63?** Because PR63 is the schema only. Speedups begin in later PRs (T2 shards and optional readers can benefit; T1 is Python‑heavy and GIL‑limited).
- **I set `max_workers: -3`.** It normalizes to `0` (sequential) by design; increase to `2+` once later PRs land.
- **Validation fails with `perf.parallel.foo unknown key`.** Remove the stray key or rename to a supported one.

## Compatibility
- No runtime behavior changes.
- No API changes.
- Tested on Python 3.11–3.13 in CI.

## Hand‑off to PR64/PR65

**PR64 (helper; no wiring)**
- Adds `clematis/engine/util/parallel.py` with `run_parallel(tasks, *, max_workers, merge_fn, order_key)`.
- Pure Python; no RNG/time/global state; no logging.
- **Acceptance:** ordering stable across runs; exceptions deterministic; `max_workers<=1` is identical to sequential.
- **Scope:** unit tests only; no runtime behavior change.

**PR65 (initial wiring; gated)**
- Choose one fanout site (likely T2 shard workers) with a stable key (e.g., `shard_id`/`graph_id`).
- Build zero‑arg thunks per task; define a pure `merge_fn` consuming `(key, result)` **in sorted order**.
- Honor `perf.parallel` gates (`enabled`, `max_workers`, and feature gates `t1|t2|agents`).
- **Cache safety:** wrap shared caches with `ThreadSafeCache` / `ThreadSafeBytesCache`; use per‑worker isolates + deterministic `merge_caches_deterministic` only where contention warrants it (details in [cache_safety.md](./cache_safety.md)).
- Add identity tests (ON/OFF) and stress tests for caches/snapshots under fixed seeds.
- Keep logs ordered (buffer and flush through a single writer if needed).

## PR66 — T1 parallel fan‑out across graphs (flag‑gated)

**What changed.** T1 now constructs one task **per active graph** and (when enabled) executes them via the deterministic runner introduced in PR64. Aggregation is fully deterministic:
- Task keys are `(submit_index, graph_id)` and merged in that order.
- Per‑graph deltas are concatenated in graph order; node IDs remain locally sorted.
- Metrics are reduced deterministically (sums / max). Identity is preserved when parallel is OFF or `max_workers<=1`.

**How to enable (safe defaults remain OFF):**
```yaml
perf:
  parallel:
    enabled: true
    max_workers: 4
    t1: true
```
With the gate OFF (or `max_workers<=1`), behavior and logs remain byte‑for‑byte identical to sequential.

**Notes.** Cache safety from PR65 applies: shared caches are lock‑wrapped by default. No isolate‑merge is used in T1 unless explicitly configured in a later PR.

---

## PR67 — Minimal observability & microbench

**New metrics (gated):** When `perf.enabled=true` and `perf.metrics.report_memory=true`, the T1 stage now includes two fields in its metrics map to help you reason about parallelism:
- `task_count` — number of per‑graph tasks (i.e., `len(active_graphs)`).
- `parallel_workers` — effective workers used, i.e. `min(max_workers, task_count)`.

These fields are **observational only** and do not alter execution. They are helpful to confirm whether adding more workers can help for a given workload (e.g., if `parallel_workers < max_workers`, increasing workers alone will not speed up T1 unless `task_count` also increases).

**Microbench (local only).** `clematis/scripts/bench_t1.py` produces a tiny, deterministic run to inspect shape and metrics:
```bash
python clematis/scripts/bench_t1.py \
  --graphs 32 --iters 3 --workers 8 --parallel --json
```
Output includes elapsed time and the stage metrics. For convenience, the bench mirrors
`parallel_workers` and `task_count` at the top level and will backfill those two keys in the
printed metrics when the metrics gate is on, so they are visible in JSON runs even if the stage
wasn’t configured to emit them in your environment.

**Troubleshooting.**
- Not seeing the two fields? Ensure `perf.enabled=true` and `perf.metrics.report_memory=true` in your config.
- `parallel_workers` smaller than `workers`? That is expected when `task_count < workers`.
- No speedups? T1’s per‑task work is small in this microbench and Python‑bound; the goal here is observability, not perf claims.

---

## PR68 — T2 parallel fan‑out across in‑memory shards (flag‑gated)

**What changed.** T2 can now fan‑out semantic retrieval across **in‑memory** shards and deterministically merge results. This preserves the sequential policy: walk tiers in order, sort within a tier (score desc, id asc), de‑duplicate by `id`, and stop at `k` (post‑merge clamp). When parallel is **OFF** (or `max_workers<=1`) behavior/logs remain byte‑for‑byte identical to sequential.

**Gate & backend.**
- Enabled only when **all** are true:
  - `perf.parallel.enabled: true`
  - `perf.parallel.t2: true`
  - `perf.parallel.max_workers > 1`
  - T2 backend is `"inmemory"`
  - The index exposes a private shard iterator `_iter_shards_for_t2(...)` and yields >1 shard
- Other backends (e.g., Lance) fall back to the sequential path.

**How to enable (safe defaults remain OFF):**
```yaml
perf:
  parallel:
    enabled: true
    max_workers: 4
    t2: true
t2:
  backend: inmemory   # required for PR68
```

**Deterministic merge semantics.**
- **Tier‑ordered walk**: finish `exact_semantic` before `cluster_semantic` (then `archive`).
- **Stable sort key** within a tier: primary = score (quantized) **descending**, secondary = `id` **ascending**.
- **De‑duplicate** across shards by `id` (keep the best by the same key).
- **Clamp after merge**: `k` (aka `t2_k`) applied once globally, preserving sequential results and order.

**Caches.** Uses the **shared, lock‑wrapped** caches from PR65. Isolated per‑worker caches are **not** used in PR68.

**Metrics (gated).** When `perf.enabled=true` and `perf.metrics.report_memory=true`, T2 emits:
- `t2.task_count` — number of shard tasks.
- `t2.parallel_workers` — `min(max_workers, task_count)`.

**Troubleshooting.**
- Not seeing any parallelism? Check the five gate conditions above (most commonly: backend not `inmemory`, or only one shard).
- Expecting cross‑tier interleaving by score? PR68 preserves **tier‑ordered** semantics (matches sequential). High‑score items in later tiers will not displace earlier‑tier items once `k` is satisfied.
- Seeing identical behavior with flags on? That’s expected if `task_count <= 1` or `max_workers <= 1`.

**CI/tests.** Functional tests assert:
- Gate‑OFF identity; gate‑ON equality (results and order) vs sequential on fixed fixtures.
- Deterministic tie‑break for equal scores (id‑ascending).
- Clamp is applied **after** merge.

## Cache safety (PR65 summary)

- Default mode: lock‑wrapped shared caches in T1/T2 (already applied; no behavior change when parallel is OFF or `max_workers<=1`).
- Optional mode: per‑worker isolated caches with deterministic post‑merge using `merge_caches_deterministic` (use the same `order_key` as in `run_parallel`).
- Determinism: worker order = `order_key(worker_key)` (stable); keys within worker sorted; conflict policy explicit (`first_wins` or `assert_equal`).

See **[cache_safety.md](./cache_safety.md)** for API, wiring patterns, and tests.

---
*This page will be expanded in PR75/PR74 with bench notes, expected shapes, and realistic performance guidance.*

## Cross‑references
- PR63 surface (this page): config keys, normalization, and identity guarantees.
- PR64 helper: [`parallel_helper.md`](./parallel_helper.md) — deterministic runner API and usage.
- PR65 cache safety: [`cache_safety.md`](./cache_safety.md) — wrappers + deterministic merge policies.
- Future wiring: see milestone notes in `docs/m9/` as they land (PR65+).
- PR66 wiring: T1 across graphs (this page) — deterministic fan‑out & merge.
- PR67 observability: minimal metrics + `bench_t1.py` (local).

---

## PR69 — M9-07: LanceDB parallel T2 + deterministic T2 microbench (flag‑gated)

**What changed.**
- Extend PR68’s parallel T2 to the **LanceDB** backend using the same deterministic fan‑out/merge semantics. No score/policy changes.
- Add a **deterministic T2 microbench** (`clematis/scripts/bench_t2.py`) for local inspection. Bench is read‑only and does not alter runtime behavior.

**Gate & backend.** Parallel T2 activates only when **all** are true:
- `perf.parallel.enabled: true`
- `perf.parallel.t2: true`
- `perf.parallel.max_workers > 1`
- T2 backend is `"inmemory"` **or** `"lancedb"`
- The index exposes `_iter_shards_for_t2(tier, suggested)` and yields **>1** shards/partitions

**Deterministic partitioning (LanceDB).**
- `LanceIndex._iter_shards_for_t2(...)` enumerates shards in a **stable** order.
  - Prefer one shard per **quarter** when multiple quarters exist.
  - Otherwise create **2–4** stable hash buckets by `id`.
  - Enumeration never truncates by `suggested`; the caller controls concurrency.
- Each shard is a read‑only view mirroring `search_tiered(...)` scoring and tie‑breaks.

**Deterministic merge semantics (unchanged from PR68).**
- **Tier‑ordered walk**: `exact_semantic` → `cluster_semantic` → `archive`.
- **Stable sort within tier**: score (quantized) **desc**, then `id` **asc**.
- **De‑duplicate** by `id` across shards; **global K‑clamp** after merge.

**Metrics (gated).** When `perf.enabled=true` and `perf.metrics.report_memory=true`:
- `t2.task_count` — number of shard tasks (both backends)
- `t2.parallel_workers` — effective workers used (both backends)
- `t2.partition_count` — **Lance‑only**, number of partitions observed when parallel path is active

**Microbench.** Deterministic, read‑only T2 bench:
```bash
python -m clematis.scripts.bench_t2 --iters 3 --workers 4 --backend inmemory --parallel --json
python -m clematis.scripts.bench_t2 --iters 3 --workers 4 --backend lancedb --parallel --json  # falls back if extras missing
```
Output JSON always includes:
`queries, shards, workers, effective_workers, backend, parallel, elapsed_ms, t2_task_count, t2_parallel_workers, t2_partition_count`.
Backfilled `t2_*` fields are provided even if the runtime metrics gate is off.

**Invariants.**
- **OFF / ≤1‑shard** path: outputs/logs remain **byte‑identical** to PR68 baseline.
- **ON** (in‑memory or Lance): result **set and order** identical to sequential under fixed seed/time.

**CI/tests.**
- Default matrix does **not** require Lance extras; import/gate hygiene tests run without them.
- Optional `extras‑lancedb` job may install `.[lancedb]` and run equivalence tests (non‑required).

**Troubleshooting.**
- Not seeing parallelism? Check all five gate conditions; most misses are single‑shard datasets or `max_workers<=1`.
- Expecting cross‑tier interleaving by score? By design, results are **tier‑ordered** to match sequential.
- Seeing identical behavior with flags on? That’s expected when shard count ≤ 1 or flags don’t all hold.

---


## PR70 — M9-08: Agent-level parallel driver (flag‑gated)

**What changed.**
- Allow multiple agents’ turns to **compute concurrently** while keeping writes/logs **deterministic and identical** to sequential.
- No changes to scheduler fairness/policy; only the execution model is extended.

**Gate (all must be true).**
- `perf.parallel.enabled: true`
- `perf.parallel.agents: true`
- `perf.parallel.max_workers > 1`

**Batch eligibility (independence).**
- A batch consists of agents whose graph sets are **pairwise disjoint**.
- Graph sets are inferred from state (e.g., `state["graphs_by_agent"][agent_id]` or `state["agents"][agent_id]["graphs"]`).
- If agents overlap, the driver falls back to sequential for those turns; identity is preserved.

**Execution model (two‑phase, deterministic).**
1) **Compute (parallel):**
   - Run T1→T4 per agent on a **read‑only state snapshot**.
   - **Do not apply** changes yet; stages emit their usual logs, which are **captured** via a LogMux instead of being written immediately.
   - Artifacts captured per turn: `approved_deltas`, `dialogue`, `graphs_touched`, and buffered `(stream, obj)` log pairs.
2) **Commit (sequential):**
   - Sort buffers by `(turn_id, slice_idx)` and then:
     - **Flush logs** via the centralized writer (exact order as sequential).
     - **Apply deltas** to the real state in that same order.

**Logging & determinism.**
- `clematis/io/log.append_jsonl` is **ctx‑aware**; when a LogMux is active, lines are buffered and later flushed—ensuring JSONL **line order** matches the sequential run.
- Because commit order is fixed, outputs and logs are **byte‑identical** to sequential for eligible batches.

**Config snippet (enable agent driver):**
```yaml
perf:
  parallel:
    enabled: true
    max_workers: 4
    agents: true
```

**Metrics (optional, gated).**
- When `perf.enabled=true` and `perf.metrics.report_memory=true`, drivers may emit `driver.agents_parallel_batch_size` to observe effective batch sizes. No schema changes when the gate is off.

**Invariants.**
- **OFF path:** outputs/logs remain **byte‑identical** to PR69 baseline.
- **ON path (disjoint agents):** outputs/logs remain **identical** to sequential; commit order is `(turn_id, slice_idx)`.

**CI/tests.**
- Functional tests only (no perf thresholds).
- Cases covered: gate‑off identity; gate‑on with **disjoint** agents (parallel compute + ordered commit); **overlap** fallback.

**Troubleshooting.**
- Not seeing parallelism? Check the three gate flags and ensure agents’ graph sets are disjoint; also verify `max_workers > 1`.
- Missing or re‑ordered logs? Ensure the ctx‑aware logger is in use; commit sorts and flushes by `(turn_id, slice_idx)`.


## PR71 — M9-09: Log staging & ordered writes (flag‑gated)

**Goal.** Guarantee deterministic on-disk JSONL order under parallel execution; keep the disabled path byte-identical.

### What changed
- **Central staging layer** (`clematis/engine/util/io_logging.py`):
  - Buffers log records with a stable composite key `(turn_id, stage_ord, slice_idx, seq)`.
  - Canonical stream order via `stage_ord`:
    `t1.jsonl` → `t2.jsonl` → `t3_plan.jsonl` → `t3_dialogue.jsonl` → `t4.jsonl` → `apply.jsonl` → `health.jsonl` → `turn.jsonl` → `scheduler.jsonl`.
  - Bounded memory (`byte_limit`, default 32 MB). On limit, raise `LOG_STAGING_BACKPRESSURE` to trigger a deterministic drain→flush→retry cycle.
- **Orchestrator (parallel path)** now stages *all* compute-phase logs and the commit-phase `apply.jsonl` records, then performs a single ordered flush.
- **Unbuffered writer** (`clematis/io/log.py::_append_jsonl_unbuffered`) mirrors the exact JSON serialization of `append_jsonl(...)` to preserve byte parity.

### Gate & invariants
- **Enabled only** when all are true:
  - `perf.parallel.enabled: true`
  - `perf.parallel.agents: true`
  - `perf.parallel.max_workers > 1`
- **Disabled path:** byte-for-byte identical to PR70.
- **Enabled path (disjoint agents):** log **content and line order** are identical to a sequential run.

### Tests
- `tests/engine/test_io_logging_staging.py` — unit tests for ordering and back-pressure.
- `tests/integration/test_agents_parallel_driver.py` — asserts deterministic ordering for `t1.jsonl`, `t4.jsonl`, `apply.jsonl` (agents `A, B`; fixed `turn`).
- `tests/test_golden_path.py` — golden check: `turn.jsonl` and `scheduler.jsonl` are byte-for-byte equal vs sequential baseline.

### Troubleshooting
- **Mismatch vs sequential?** Ensure `_append_jsonl_unbuffered(...)` uses the **same** JSON formatting as `append_jsonl(...)` (e.g., `ensure_ascii=False`, separators).
- **No ordering effect?** Confirm the three gate conditions above; staging is only active in the agent-parallel path.
