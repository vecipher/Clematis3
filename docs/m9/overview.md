

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
- **No nondeterministic sources** in parallel paths: no random seeds, time‑based ordering, or hidden global state.
- **Thread‑safe caches**: default to lock‑wrapped single cache; optional per‑worker caches must have deterministic post‑merge.
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

---
*This page will be expanded in PR75/PR74 with bench notes, expected shapes, and realistic performance guidance.*
