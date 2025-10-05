# Clematis v3 — deterministic, turn‑based agent engine

Clematis is a deterministic, turn‑based scaffold for agential AI. It models agents with concept graphs and tiered reasoning (T1→T4), uses small LLMs where needed, and keeps runtime behavior reproducible (no hidden network calls in tests/CI).

> **Status:** **Unreleased (v0.9.0a3–a5)** — **M12 in progress** (Native T1 opt‑in, caps parity, optional GEL tick). Defaults unchanged; all new native paths are **gated and OFF by default**; identity path preserved. M11 remains complete; M10 complete; M9 deterministic parallelism stays flag‑gated and OFF by default.
>
> **M10 — Reflection Sessions (complete):** Finalized across **PR77** and **PR80–PR90** (config, writer/budgets/tests, fixtures‑only LLM backend, planner flag, telemetry/trace, microbench, optional CI smoke, docs, goldens/identity maintenance). See **[docs/m10/reflection.md](docs/m10/reflection.md)**.
>

> **M12 — Native T1 (in progress):** PR97–PR100 shipped (config+stubs, FFI parity, Rust kernel perf‑OFF, wheels/CI). **PR101 (bench & docs)** is advisory and Linux‑only. **PR102 (hardening & diagnostics)** adds counters + once‑only logs and a nightly strict‑parity sweep. **PR104 (perf‑ON caps parity)** enables the native T1 path when `caps.frontier`, `caps.visited`, or `dedupe_window` are ON, with deterministic semantics. **PR105 (GEL micro‑kernel)** adds an optional decay kernel behind `perf.native.gel.enabled` (or `PERF_NATIVE_GEL_ENABLED`) with parity to the Python reference. Enable native T1 via `CLEMATIS_NATIVE_T1=1`; opt‑in strict parity with `CLEMATIS_STRICT_PARITY=1`. Diagnostics stream: `t1_native_diag.jsonl` (see "Where to see them"). Wheels are **abi3 (cp311+)**; avoid source shadowing by installing from wheel. See **[docs/m12/native_t1.md](docs/m12/native_t1.md)**.

---

## Goals
- Universalisable scaffold for simulating characters/agents.
- Turn‑based core with deterministic logs and budgets.
- Modular stages: T1 (propagation) → T2 (retrieval) → T3 (planning) → T4 (meta‑filter/apply).
- Identity persistence via vector memories (BGE) + LanceDB; optional LLM planning path.
- Separation of planner/utterance phases to support future MoE.
- Practical latency targets on modest hardware; scalable to larger models later.

## Architecture (high level)
- **Memories:** LanceDB vector store (BGE small); tiered retrieval; deterministic scoring.
- **Concept graph:** nodes/edges with decay and relations; surface views for I/O.
- **Stages:**
  T1 keyword/seeded propagation → T2 semantic retrieval (+ residual) → T3 bounded policy (rule‑based by default; fixtures‑only LLM backend available) → T4 meta‑filter & apply/persist.
  **Reflection (M10):** gated and deterministic. Default OFF; when enabled it runs post‑Apply, never mutates T1/T2/T4/apply artifacts for the current turn. Rule‑based backend is pure/deterministic; LLM backend is **fixtures‑only** for determinism.
  **GEL (M11):** optional field‑control substrate (co‑activation update + half‑life decay; merge/split/promotion available), **default OFF**. See **[docs/m11/overview.md](docs/m11/overview.md)**.
- **Determinism:** golden logs, identity path when gates are OFF; shadow/quality traces never affect results.

## Quick start
```bash
# install (editable)
python -m pip install -e .

# check umbrella CLI is wired
python -m clematis --version

# try a wrapper (both orders work; single leading -- is stripped by the wrapper)
python -m clematis rotate-logs -- --dir ./.logs --dry-run
# or
python -m clematis --dir ./.logs rotate-logs -- --dry-run

# Some scripts need optional extras. See docs/m8/packaging_cli.md (e.g., pip install "clematis[zstd]" or "clematis[lancedb]").
```


### GEL (HS1) examples

Ready-to-run configs:

- Enabled (observe + decay only; ops OFF): `examples/gel/enabled.yaml`
- Disabled (identity path): `examples/gel/disabled.yaml`

Run:
```bash
python scripts/examples_smoke.py --examples examples/gel/enabled.yaml
python scripts/examples_smoke.py --examples examples/gel/disabled.yaml
# or the bundled set
python scripts/examples_smoke.py --all
```

### M10: reflection sessions (deterministic, gated)

Reflection is OFF by default. To enable the **rule‑based** deterministic backend:

```yaml
t3:
  allow_reflection: true
  reflection:
    backend: "rulebased"   # deterministic, no network
    summary_tokens: 128
    embed: true
    log: true
    topk_snippets: 3
scheduler:
  budgets:
    time_ms_reflection: 6000
    ops_reflection: 5
```

To enable the **fixtures‑only LLM** backend (deterministic via fixtures):

```yaml
t3:
  allow_reflection: true
  reflection:
    backend: "llm"         # fixtures-only
  llm:
    fixtures:
      enabled: true
      path: tests/fixtures/reflection_llm.jsonl  # must be a non-empty string
scheduler:
  budgets:
    time_ms_reflection: 6000
    ops_reflection: 5
```

**Planner requirement:** reflection runs only when **all** are true:
1) `t3.allow_reflection: true`, and
2) the planner sets `reflection: true` in its output (PR85). The LLM planner path carries this flag via the policy state; the orchestrator honors either the explicit plan flag or the stashed value.
3) not in dry‑run mode (the orchestrator’s `_dry_run` is **false**).

**Determinism invariants (current):**
- No network; CI uses `CLEMATIS_NETWORK_BAN=1`.
- Rule‑based summary is normalization + token clamp; embeddings use `DeterministicEmbeddingAdapter(dim=32)`.
- Budgets enforced: wall‑clock timeout (`time_ms_reflection`) and entry cap (`ops_reflection`).
- Fail‑soft: reflection errors never break the turn; on error/timeout, no writes are persisted.
- Writer (PR80) fixes `ts` from `ctx.now_iso` and produces stable IDs; ops‑cap is double‑enforced.

**Logging/telemetry (PR86):** writes a `t3_reflection.jsonl` stream with schema `{turn, agent, summary_len, ops_written, embed, backend, ms, reason[, fixture_key]}`. In CI, only the `ms` field is normalized to `0.0`. This stream is staged with `STAGE_ORD["t3_reflection.jsonl"]=10` and is **not** part of the identity log set.

**Troubleshooting:**
- *“Nothing happens”*: ensure `t3.allow_reflection: true` **and** planner `reflection: true`. Dry‑run modes skip reflection.
- *LLM backend rejected*: set `t3.llm.fixtures.enabled: true` and provide a non‑empty `path`. The validator rejects empty or missing paths.
- *Missing fixture at runtime*: seed a fixture for the canonical prompt JSON (see `FixtureLLMAdapter` docs).

**Microbench & optional CI smoke**

Local, deterministic microbench (prints one stable JSON line):
```bash
python -m clematis.scripts.bench_reflection -c examples/reflection/enabled.yaml
python -m clematis.scripts.bench_reflection -c examples/reflection/llm_fixture.yaml
```

Optional CI workflow: `.github/workflows/reflection_smoke.yml`.
- Trigger manually via **Actions → Reflection Smoke (optional)** with `run=true`.
- To auto‑run on pushes temporarily, set `RUN_REFLECTION_SMOKE: "true"` in that workflow’s top‑level `env:` and revert before merging.

### M9: deterministic parallelism (flag‑gated)
The PR63 surface adds a validated config for deterministic parallelism. Defaults keep behavior identical to previous milestones. As of PR66, T1 can fan‑out **per active graph** via the deterministic runner, with stable merge ordering; as of PR67, minimal observability metrics are available when enabled. As of **PR68**, T2 can fan‑out across **in‑memory** shards with a deterministic, tier‑ordered merge (score‑desc, id‑asc; de‑dupe by id); as of **PR69**, T2 can also fan‑out across **LanceDB** partitions behind the same gate; OFF path remains byte‑identical. As of **PR70**, an **agent‑level parallel driver** allows multiple agents’ turns to compute concurrently while preserving deterministic logs; it is gated separately via `perf.parallel.agents`.
As of **PR71**, log staging guarantees deterministic on‑disk JSONL order under parallel execution; the disabled path remains byte‑identical.
As of **PR72**, a dedicated identity & race test suite proves byte‑identical artifacts (t1/t2/t4/apply/turn/scheduler) between sequential and parallel runs and under contention/back‑pressure; CI runs this across Python 3.11–3.13 (see `.github/workflows/identity_parallel.yml`).

As of **PR73**, an **opt‑in CI parallel smoke** exists (not required for branch protection). Trigger via workflow dispatch or environment `RUN_PERF=1`; it prints a simple “parallel smoke OK”.

As of **PR74**, **microbench scripts & docs** are shipped for T1/T2. They emit stable one‑line JSON and are for local comparison only (no CI gating). See **[docs/m9/benchmarks.md](docs/m9/benchmarks.md)**.

```yaml
perf:
  parallel:
    enabled: true
    max_workers: 4   # >1 to enable parallel path
    t1: true
```

**Enable T2 fan‑out (in‑memory or LanceDB backend):**

```yaml
perf:
  parallel:
    enabled: true
    max_workers: 4   # >1 to enable parallel path
    t2: true
t2:
  backend: inmemory  # or "lancedb" (requires extras)
```

**Enable agent‑level parallel driver (agents):**

```yaml
perf:
  parallel:
    enabled: true
    max_workers: 4   # >1 to enable parallel path
    agents: true
```

**Observability (optional):** To see minimal parallel metrics (`task_count`, `parallel_workers`) in T1 outputs and in the microbench, also set:

```yaml
perf:
  enabled: true
  metrics:
    report_memory: true
```

When enabled, T2 emits `t2.task_count` and `t2.parallel_workers`; on Lance it also emits `t2.partition_count` (number of partitions).
When the agent driver is enabled, the system may also emit `driver.agents_parallel_batch_size` under the same metrics gate.

**Microbenches:**

See **[docs/m9/benchmarks.md](docs/m9/benchmarks.md)** for usage, flags, and output shapes.

```bash
# T1 microbench
python -m clematis.scripts.bench_t1 --graphs 32 --iters 3 --workers 8 --parallel --json

# T2 microbench (in‑memory)
python -m clematis.scripts.bench_t2 --iters 3 --workers 4 --backend inmemory --parallel --json

# T2 microbench (LanceDB; falls back if extras missing)
python -m clematis.scripts.bench_t2 --iters 3 --workers 4 --backend lancedb --parallel --json
```

#### M9 knobs — quick reference

Use these example configs to toggle deterministic parallelism. With all knobs **OFF**, outputs are byte‑identical to sequential.

```yaml
# examples/perf/parallel_off.yaml  (sequential baseline)
perf:
  parallel:
    enabled: false
    max_workers: 1
    t1: false
    t2: false
    agents: false
  metrics:
    enabled: false
    report_memory: false
```

```yaml
# examples/perf/parallel_on.yaml  (opt‑in parallel)
perf:
  parallel:
    enabled: true
    max_workers: 2
    t1: true
    t2: true
    agents: true
  metrics:
    enabled: true
    report_memory: true
```

See **docs/m9/overview.md** for determinism rules, identity guarantees, and troubleshooting.

*Cache safety:* PR65 adds thread-safe wrappers for shared caches and an optional isolate+merge strategy. See **[docs/m9/cache_safety.md](docs/m9/cache_safety.md)**.

## Repository layout (brief)
- `clematis/engine/` — core stages (T1–T4), scheduler stubs, persistence, logs.
- `clematis/engine/util/parallel.py` — deterministic thread-pool helper (`run_parallel`), unit tests only.
- `clematis/engine/util/logmux.py` — ctx‑aware buffered logging (PR70 driver capture & deterministic flush).
- `clematis/engine/util/io_logging.py` — deterministic log staging and ordered flush (PR71).
- `clematis/engine/stages/state_clone.py` — read‑only state snapshot utilities for the compute phase.
- `clematis/engine/stages/t2/` — T2 retrieval stack (post‑PR76 refactor):
  - `core.py` — lean orchestrator (retrieval + calls quality/metrics)
  - `quality.py` — quality orchestration (hybrid→fusion→MMR) + shadow trace (triple‑gated)
  - `state.py` — index/labels helpers
  - `metrics.py` — assemble/finalize metrics (side‑effect free)
  - `lance_reader.py`, `quality_ops.py`, `quality_mmr.py`, `quality_norm.py`, `quality_trace.py`, `shard.py`
- `clematis/engine/stages/t3/reflect.py` — reflection backends (`rulebased`, `llm` fixtures‑only); deterministic summary + optional embedding.
- `clematis/engine/stages/t3/policy.py` — planner prompt + policy glue; surfaces the planner’s `reflection` flag (PR85).
- `clematis/engine/orchestrator/reflection.py` — deterministic write path for reflection entries (stable `ts` and IDs).
- `tests/t2/test_t2_parallel_merge.py` — gate semantics, tie‑break, tier‑ordered K‑clamp, normalization.
- `clematis/cli/` — umbrella + wrapper subcommands (delegates to `clematis.scripts.*`).
- `scripts/` — direct script shims (`*_hint.py`, tolerant import, single stderr hint).
- `clematis/scripts/` — local microbenches and helpers (e.g., `bench_t1.py`, `bench_t2.py`).
- `examples/gel/` — HS1/GEL substrate example configs (enabled vs disabled).
- `docs/` — milestone docs and updates (see `docs/m9/overview.md`, `docs/m9/parallel_helper.md`, `docs/m9/cache_safety.md`).
- `tests/` — deterministic tests, golden comparisons, CLI checks.

## Environment flags
- `CLEMATIS_NATIVE_T1=1` — force‑enable the native T1 path (overrides config).
- `CLEMATIS_STRICT_PARITY=1` — compute native & python and fail on mismatch (CI/advisory).
-
`PERF_NATIVE_GEL_ENABLED=1` — enable the optional GEL tick native path (engine fast‑path). Equivalent config gate: `perf.native.gel.enabled: true`. When disabled (default), behavior/logs are unchanged; counters appear under `metrics["native_gel"]` when enabled.
- `CLEMATIS_NETWORK_BAN=1` — enforce no network (recommended in CI).
- `CLEMATIS_DEBUG=1` — enable a single stderr breadcrumb for wrapper delegation.
  Exit codes and stdout remain identical.
- `CLEMATIS_LOG_DIR` / `CLEMATIS_LOGS_DIR` — override the logs output directory.
  If both are set, `CLEMATIS_LOG_DIR` wins; otherwise we fall back to `<repo>/.logs`.
  The directory is created on demand so wrappers/scripts can log immediately.


When `CI=true`, log writes route through `clematis/engine/orchestrator/logging.append_jsonl`, which applies `clematis/engine/util/io_logging.normalize_for_identity`. Identity logs keep their existing rules (e.g., drop `now`, clamp times) to ensure byte identity. For the reflection stream `t3_reflection.jsonl`, only the `ms` field is normalized to `0.0`; this stream is **not** part of the identity set.

### CI workflows (advisory)
- **Perf smoke (Linux)** — `.github/workflows/perf-smoke.yml`: builds & installs the wheel, asserts `_t1_rs` import + `t1.available()` and runs the bounded perf smoke. Trigger on push to default branches or label PRs `perf-smoke`. Non‑required.
- **Strict parity nightly** — `.github/workflows/strict-parity-nightly.yml`: daily at 20:00 UTC (or manual). Runs perf smoke and diagnostics under `CLEMATIS_NATIVE_T1=1`, `CLEMATIS_STRICT_PARITY=1`, `PERF_SMOKE=1`. Non‑required.
- **Strict parity (caps advisory):** The nightly strict‑parity job includes a small caps matrix for T1 under advisory status; uploads `t1_native_diag.jsonl` artifacts.
- **GEL parity (optional):** A tiny GEL parity test (native or fallback) runs in the nightly advisory job when enabled.

## Milestones snapshot
- **M1–M4:** core stages + apply/persist + logs.
- **M5:** scheduler config and groundwork (feature‑gated; identity path when disabled).
- **M6:** memory/perf scaffolding; caches and snapshot hygiene (default‑off quality toggles).
- **M7:** observability/dev‑ex; shadow quality traces; golden tests; gate hardening.
- **M8 (finished):** packaging & CLI docs/CI polish.
  – README trimmed; canonical CLI doc split to `docs/m8/cli.md`.
  – Add fast CLI smokes to CI (help phrase, arg‑order, sentinel strip, shim hint).
  – pre-commit + Ruff/Mypy configs; dual Ruff CI gates (repo safety + CLI strict).
  – declare NumPy as a runtime dependency (examples smoke).
- **M9 (complete):** deterministic parallelism — PR63–PR76 shipped (config + deterministic runner + cache safety + T1/T2/agents gates + ordered logs + identity & race tests + optional CI smoke and benches). Defaults keep parallelism OFF; identity path preserved.
- **M10 (complete):** reflection sessions — PR77 (config surface), PR80–PR83 (deterministic writer + budgets + identity tests), PR84 (fixtures‑only LLM backend), PR85 (planner flag + wiring), PR86 (telemetry & trace), PR87 (microbench & examples), PR88 (optional smoke), PR89 (docs), PR90 (goldens/identity maintenance). Defaults keep reflection OFF; identity path preserved.
- **M11 (complete):** HS1/GEL substrate — contracts + plumbing present; observe/update + decay enabled only when `graph.enabled=true`; merge/split/promotion documented but **OFF** by default; disabled path is byte‑identical. See **docs/m11/overview.md**.

Pre‑M8 hardening notes: **`Changelog/PreM8Hardening.txt`**.
LLM adapter + fixtures: **`docs/m3/llm_adapter.md`**.

## Contributing
- Keep changes deterministic. If a gate is OFF, results must be byte‑for‑byte identical.
- Tests should run offline; prefer fixtures and golden logs.
- Include small, focused PRs with a clear scope and a short DoD checklist.

---
_Read the milestone docs under `docs/` for deeper details. This README stays lean and stable._
