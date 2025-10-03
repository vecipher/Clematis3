# Clematis v3 — deterministic, turn‑based agent engine

Clematis is a deterministic, turn‑based scaffold for agential AI. It models agents with concept graphs and tiered reasoning (T1→T4), uses small LLMs where needed, and keeps runtime behavior reproducible (no hidden network calls in tests/CI).

> **Status (Milestone 9):** In progress ⚙️ — PR63 (config schema; defaults **OFF**), PR64 (deterministic runner), PR65 (cache safety wrappers), PR66 (flag‑gated T1 fan‑out across graphs), PR67 (parallel metrics + microbench), **PR68 (flag‑gated T2 fan‑out across in‑memory shards)**, **PR69 (LanceDB parallel T2 + deterministic T2 microbench)**, **PR70 (agent‑level parallel driver)**, **PR71 (log staging & ordered writes)**, **PR72 (identity & race test suite)**, **PR73 (opt‑in CI parallel smoke; non‑required)**, **PR74 (bench kits + docs)**, and **PR76 (T2 + orchestrator module refactor)** have landed. Parallelism remains **OFF by default**; identity is unchanged unless explicitly enabled. See **[docs/m9/overview.md](docs/m9/overview.md)**, **[docs/m9/parallel_helper.md](docs/m9/parallel_helper.md)**, **[docs/refactors/PR76](docs/refactors/PR76)**, and **[docs/m9/migration.md](docs/m9/migration.md)**.

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
  T1 keyword/seeded propagation → T2 semantic retrieval (+ residual) → T3 bounded policy (rule‑based now; LLM adapter gated) → T4 meta‑filter & apply/persist.
  Reflection and scheduler features are gated for determinism.
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

CLI details, delegation rules, and recipes live in **[docs/m8/cli.md](docs/m8/cli.md)**. Packaging/extras and quality gates: **[docs/m8/packaging_cli.md](docs/m8/packaging_cli.md)** · **[CONTRIBUTING.md](CONTRIBUTING.md)**.

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
- `tests/t2/test_t2_parallel_merge.py` — gate semantics, tie‑break, tier‑ordered K‑clamp, normalization.
- `clematis/cli/` — umbrella + wrapper subcommands (delegates to `clematis.scripts.*`).
- `scripts/` — direct script shims (`*_hint.py`, tolerant import, single stderr hint).
- `clematis/scripts/` — local microbenches and helpers (e.g., `bench_t1.py`, `bench_t2.py`).
- `docs/` — milestone docs and updates (see `docs/m9/overview.md`, `docs/m9/parallel_helper.md`, `docs/m9/cache_safety.md`).
- `tests/` — deterministic tests, golden comparisons, CLI checks.

## Environment flags
- `CLEMATIS_NETWORK_BAN=1` — enforce no network (recommended in CI).
- `CLEMATIS_DEBUG=1` — enable a single stderr breadcrumb for wrapper delegation.
  Exit codes and stdout remain identical.
- `CLEMATIS_LOG_DIR` / `CLEMATIS_LOGS_DIR` — override the logs output directory.
  If both are set, `CLEMATIS_LOG_DIR` wins; otherwise we fall back to `<repo>/.logs`.
  The directory is created on demand so wrappers/scripts can log immediately.

When `CI=true`, all log sinks route through `clematis/io/log.append_jsonl`, which
normalizes payloads via `clematis/engine/util/io_logging.normalize_for_identity`
to zero `ms`, drop `now`, and keep byte identity across runs.

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
- **M9 (WIP):** deterministic parallelism — schema landed in PR63 (defaults OFF, identity preserved); runtime work continues in PR64–PR76.

Pre‑M8 hardening notes: **`Changelog/PreM8Hardening.txt`**.
LLM adapter + fixtures: **`docs/m3/llm_adapter.md`**.

## Contributing
- Keep changes deterministic. If a gate is OFF, results must be byte‑for‑byte identical.
- Tests should run offline; prefer fixtures and golden logs.
- Include small, focused PRs with a clear scope and a short DoD checklist.

---
_Read the milestone docs under `docs/` for deeper details. This README stays lean and stable._
