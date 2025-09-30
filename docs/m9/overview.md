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
