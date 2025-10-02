

# M9 Microbenches — T1 / T2 (PR74)

**Intent:** tiny, reproducible yardsticks for inner‑loop costs. No thresholds, no CI gating, no runtime behavior change. These scripts measure local wall‑time and emit **stable JSON** so you can diff before/after.

> Both benches are wrappers that **delegate to `scripts/...`** and rely only on public engine surfaces.

---

## Quick start

**Prereqs**
- Python 3.11–3.13 (same range we test elsewhere)
- `clematis` installed in editable or wheel form
- `numpy` required only for `bench_t2`
- Network is **blocked** via `CLEMATIS_NETWORK_BAN=1` (the scripts set this defensively)

**Determinism**
- Fixed internal seeds
- Warmup is outside timed region
- `--json` prints **one line** with `sort_keys=True`, fixed separators, `ensure_ascii=True`

---

## T1 bench

Runs repeated T1 propagations over tiny synthetic graphs. Single‑threaded by design. `--parallel` only toggles the perf flags for observability; it does **not** spawn workers (use PR73 smoke for end‑to‑end parallel timing).

```bash
# Compact, stable JSON
python -m clematis.scripts.bench_t1 --graphs 4 --iters 3 --json

# Human‑readable
python -m clematis.scripts.bench_t1 --graphs 4 --iters 3
```

**Output shape (example, values will differ):**
```json
{"effective_workers":1,"elapsed_ms":12.345,"graphs":4,"iters":3,"metrics":{"cache_hits":0,"cache_misses":0,"graphs_touched":4,"iters":12,"layer_cap_hits":0,"max_delta":0.0,"node_budget_hits":0,"pops":24,"propagations":12,"radius_cap_hits":0},"parallel":false,"task_count":4,"workers":1}
```

**Notes**
- Metrics are **accumulated** across `--iters` (except `max_delta` which is max‑aggregated).
- Scales roughly with `graphs × iters`.
- For actual parallel measurement, use **PR73** (`scripts/run_demo.py` + `examples/perf/parallel_on.yaml`).

---

## T2 bench

Builds a deterministic synthetic corpus and runs semantic retrieval with a simple, import-robust index:
- **Default backend:** `inmemory` (no extras)
- **Optional:** `--backend lancedb` if Lance extras are present
- Parallel path (`--parallel`) searches shards concurrently and merges results deterministically.
- **Index selection:** When available we import `clematis.memory.index.InMemoryIndex`, which now exposes `_iter_shards_for_t2(tier, suggested)` so the bench reflects real shard behavior. If the import fails (or extras are missing) we fall back to the bundled bench index. The Lance path likewise prefers `clematis.memory.lance_index.LanceIndex` and only falls back when extras are absent.

```bash
# Compact, stable JSON (default: inmemory)
python -m clematis.scripts.bench_t2 --rows 256 --iters 3 --json

# Human‑readable, with parallel across shards
python -m clematis.scripts.bench_t2 --rows 512 --iters 5 --parallel --workers 3
```

**Output shape (example, values will differ):**
```json
{"backend":"inmemory","parallel":false,"queries":3,"k":32,"dim":32,"rows":256,"shards":2,"workers":4,"effective_workers":1,"elapsed_ms":34.210,"metrics":{"total_k_returned":96,"avg_returned_per_query":32.0,"t2_task_count":0,"t2_parallel_workers":0,"t2_partition_count":0}}
```

**Flags**
- `--rows` corpus size (default 256); `--dim` embedding dim (default 32); `--k` top‑K (default 32)
- `--parallel/--workers` enable per‑shard concurrency (deterministic merge by `(-score, id)`)

**Caveats**
- Micro sizes are GIL/allocator sensitive; compare **ratios** on the same machine.
- Bench index partitions by quarter timestamps; if only one bucket, it falls back to two hash buckets to keep shard semantics.
- Real engine indices create shards deterministically; the fallback index mirrors the same ordering so repeated runs stay reproducible.

---

## When to run

- After touching T1/T2 internals, caches, or embedding adapters
- Before/after toggling perf flags locally
- As a quick smoke after merges that might influence inner‑loop costs

---

## CI & repo policy

- No new CI jobs; these benches do **not** gate merges.
- If you really want to run them in CI, piggyback PR73’s opt‑in path with `RUN_PERF=1` — but keep them non‑required.

---

## Troubleshooting

- `ModuleNotFoundError: numpy` → `pip install numpy` (only needed for `bench_t2`)
- “weirdly low/high ms” → ensure you’re not on a heavily throttled VM; prefer multiple runs and compare deltas
- “parallel didn’t speed up T1” → expected; T1 bench is single‑threaded by design. Use PR73 smoke for end‑to‑end.

---

## CLI reference (summary)

`bench_t1`
```
--graphs INT        Number of tiny graphs (default: 24)
--iters INT         Repetitions (default: 5)
--warmup INT        Warmup iterations (default: 1; not timed)
--parallel          Advisory flag; sets perf.parallel for observability
--workers INT       Worker hint when --parallel (default: 4)
--json              Emit stable one‑line JSON
```

`bench_t2`
```
--rows INT          Corpus size (default: 256)
--dim INT           Embedding dimension (default: 32)
--k INT             Top‑K (default: 32)
--iters INT         Query iterations (default: 3)
--warmup INT        Warmup runs (default: 1; not timed)
--backend {inmemory,lancedb}
--parallel          Enable per‑shard concurrency
--workers INT       Max parallel shard workers (default: half cores)
--json              Emit stable one‑line JSON
```

---

## Guarantees & non‑goals

- ✅ Deterministic seeds, stable JSON output
- ✅ No network, no changes to orchestrator/engine behavior
- ✅ Import‑robust: falls back to an internal in‑memory index for T2
- ❌ Not a formal perf suite, no thresholds, no required checks
