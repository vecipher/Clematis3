# Clematis v3 — deterministic, turn‑based agent engine

Clematis is a deterministic, turn‑based scaffold for agential AI. It models agents with concept graphs and tiered reasoning (T1→T4), uses small LLMs where needed, and keeps runtime behavior reproducible (no hidden network calls in tests/CI).

> **Status:** **v0.10.0** (2025‑10‑08) — **M13 Hardening & Freeze (frozen)**. See **[docs/m13/overview.md](docs/m13/overview.md)**. **M12 skipped** for v3. **M11 complete** ✅ (HS1/GEL substrate). Defaults unchanged; all GEL paths are **gated and OFF by default**; identity path preserved. M10 remains complete; M9 deterministic parallelism remains flag‑gated and OFF by default.
>
> **License:** Apache‑2.0 — see [LICENSE](./LICENSE) & [NOTICE](./NOTICE).
> **Support matrix:** Python **3.11–3.13**; Ubuntu, macOS, Windows. Cross‑OS identity and reproducible builds (SBOM/SLSA) enforced in CI.
> **Changelog:** see [CHANGELOG.md](CHANGELOG.md) for **v0.10.0**.
>
> **M13 — Hardening & Freeze (v3):** See **[docs/m13/overview.md](docs/m13/overview.md)**.
>
> **M10 — Reflection (complete):** Deterministic, gated; defaults OFF. See **[docs/m10/reflection.md](docs/m10/reflection.md)**.
>
> **M11 — HS1/GEL (complete):** Substrate landed; defaults OFF; identity preserved. See **[docs/m11/overview.md](docs/m11/overview.md)**.
>
> **Identity & perf:** Shadow/perf diagnostics are written under `logs/perf/` and are ignored by identity; canonical logs remain `t1.jsonl`, `t2.jsonl`, `t4.jsonl`, `apply.jsonl`, `turn.jsonl` (and `scheduler.jsonl` where applicable). CLI help text is deterministic (Linux + Python 3.13, `COLUMNS=80`).

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
- **Determinism:** golden logs, identity path when gates are OFF; shadow/quality traces never affect results. Shadow/perf diagnostics are written under `logs/perf/` and ignored by identity.
- **Config freeze:** v3 config schema is frozen at `version: "v1"`. Unknown top‑level keys are rejected. See [docs/m13/config_freeze.md](docs/m13/config_freeze.md).
- **Snapshot freeze:** v3 snapshots include a header field `schema_version: "v1"`; the inspector validates the header and **fails by default** (exit 2). Use `--no-strict` to only warn. See [docs/m13/snapshot_freeze.md](docs/m13/snapshot_freeze.md).
- **Typed errors:** operator‑facing failures use `clematis.errors.*`. See [docs/m13/error_taxonomy.md](docs/m13/error_taxonomy.md).

> 🔒 **M13 – Hardening & Freeze (v3):** v3 is frozen as of 2025‑10‑08 SGT.
> See **[docs/m13/overview.md](docs/m13/overview.md)** for what’s locked (Config v1, Snapshot v1), identity guarantees, support matrix, and EOL stance.

## Quick start

> **Operator Guide (single page):** see [docs/operator-guide.md](docs/operator-guide.md)
> **Public API (v3):** see [docs/api_reference.md](docs/api_reference.md)
```bash
# install (editable)
python -m pip install -e .

# check umbrella CLI is wired
python -m clematis --version

# try a wrapper (both orders work; single leading -- is stripped by the wrapper)
python -m clematis rotate-logs -- --dir ./.logs --dry-run
# or
python -m clematis --dir ./.logs rotate-logs -- --dry-run

# Some scripts need optional extras. See [docs/m8/packaging_cli.md](docs/m8/packaging_cli.md) (e.g., pip install "clematis[zstd]" or "clematis[lancedb]").
```


### Operator‑facing errors (typed)

CLIs print a single, typed line to **stdout** and exit with **code 2** on user errors (e.g., invalid config), keeping logs quiet and machine‑parseable.

Example message:

```text
ConfigError: unknown top-level key: foo
```

From Python:

```python
from clematis.errors import format_error, ConfigError
print(format_error(ConfigError("unknown top-level key: foo")))
# -> "ConfigError: unknown top-level key: foo"
```

### Reproducible builds (local)

Build artifacts deterministically and verify hashes:

```bash
scripts/repro_check_local.sh            # build sdist+wheel, print SHA256
scripts/repro_check_local.sh --twice    # build twice and assert byte‑identical artifacts
```

CI also enforces cross‑OS reproducibility; see `.github/workflows/pkg_build.yml`.
For SBOM and **SLSA provenance** verification, see [docs/m8/packaging_cli.md#supply-chain-sbom--provenance](docs/m8/packaging_cli.md#supply-chain-sbom--provenance).


### Perf/diagnostic logs (non‑canonical)

- Non‑canonical diagnostics are routed to `logs/perf/` (or files ending with `-perf.jsonl`).
- Identity/golden comparisons **ignore** these files.
- Example: enabling the hybrid reranker in T2 may emit `logs/perf/t2_hybrid.jsonl`.
- To toggle features locally without editing configs, you can supply a JSON overrides file:

```bash
python -m clematis.scripts.demo --config examples/perf/parallel_on.yaml --config-overrides overrides.json
```

Where `overrides.json` could be:

```json
{"t2": {"hybrid": {"enabled": true}}}
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

Deterministic parallelism is available for **T1**, **T2** (in‑memory or LanceDB), and **agent‑level compute**. Defaults keep parallelism **OFF**; the disabled path is byte‑identical to sequential. See **[docs/m9/overview.md](docs/m9/overview.md)** for design, invariants, and troubleshooting.

**Quick enable (pick one or more):**
```yaml
perf:
  parallel:
    enabled: true
    max_workers: 4
    t1: true       # or t2: true / agents: true
```

LanceDB backend (optional):
```yaml
t2:
  backend: lancedb
```

Optional metrics in JSON logs require:
```yaml
perf:
  enabled: true
  metrics:
    report_memory: true
```

Microbenches and the optional CI smoke are documented in **[docs/m9/benchmarks.md](docs/m9/benchmarks.md)**.

## Repository layout (brief)
- `clematis/engine/` — core stages (T1–T4), scheduler stubs, persistence, logs.
- `clematis/engine/util/parallel.py` — deterministic thread-pool helper (`run_parallel`), unit tests only.
- `clematis/engine/util/logmux.py` — ctx‑aware buffered logging (PR70 driver capture & deterministic flush).
- `clematis/engine/util/io_logging.py` — deterministic log staging and ordered flush (PR71).
- `clematis/engine/observability_perf.py` — non‑canonical diagnostics writer (`logs/perf/*.jsonl`).
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
- `CLEMATIS_NETWORK_BAN=1` — enforce no network (recommended in CI).
- `CLEMATIS_DEBUG=1` — enable a single stderr breadcrumb for wrapper delegation.
  Exit codes and stdout remain identical.
- `CLEMATIS_LOG_DIR` / `CLEMATIS_LOGS_DIR` — override the logs output directory.
  If both are set, `CLEMATIS_LOG_DIR` wins; otherwise we fall back to `<repo>/.logs`.
  The directory is created on demand so wrappers/scripts can log immediately.

When `CI=true`, log writes route through `clematis/engine/orchestrator/logging.append_jsonl`, which applies `clematis/engine/util/io_logging.normalize_for_identity`. Identity logs keep their existing rules (e.g., drop `now`, clamp times) to ensure byte identity. For the reflection stream `t3_reflection.jsonl`, only the `ms` field is normalized to `0.0`; this stream is **not** part of the identity set.

## Milestones snapshot
- **M13 (complete; frozen 2025‑10‑08):** Hardening & Freeze — cross‑OS identity (PR106), LF/CRLF & path normalization (PR107), config v1 lock (PR108), snapshot v1 header + strict inspector (PR109), reproducible builds (PR110). **M12 skipped** for v3.
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

## License

**Apache-2.0** — see [LICENSE](./LICENSE) and [NOTICE](./NOTICE).
Copyright © 2025 vecipher

Pre‑M8 hardening notes: **`Changelog/PreM8Hardening.txt`**.
LLM adapter + fixtures: **`docs/m3/llm_adapter.md`**.

## Contributing
- Keep changes deterministic. If a gate is OFF, results must be byte‑for‑byte identical.
- Tests should run offline; prefer fixtures and golden logs.
- Include small, focused PRs with a clear scope and a short DoD checklist.

---
_Read the milestone docs under `docs/` for deeper details. This README stays lean and stable._
