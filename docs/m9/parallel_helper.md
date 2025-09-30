

# M9: Deterministic Parallel Helper (`run_parallel`)

**Status:** Introduced in PR64. **Unused by default** (no runtime wiring). Intended for PR65+ to safely enable parallelism without breaking identity-path determinism.

## Design goals

- **Deterministic aggregation.** Final results are ordered by a caller-provided `order_key(key)`; completion order is irrelevant.
- **Sequential identity path.** `max_workers <= 1` executes tasks in-process, then applies the same deterministic ordering. This must be byte-for-byte identical to a simple loop + stable sort.
- **Pure utility.** No RNG, no time sources, no global state, no logging. Exceptions are aggregated deterministically.
- **Narrow surface.** Keep parallel concerns localized so orchestrator/stages can adopt it incrementally.

## API (stable)

```py
run_parallel(
    tasks: Sequence[Tuple[K, Callable[[], R]]],
    *,
    max_workers: int,
    merge_fn: Callable[[List[Tuple[K, R]]], A],
    order_key: Callable[[K], Any],
) -> A
```

- **`tasks`** — sequence of `(key, thunk)` where `thunk` is a zero-arg callable producing `R`. The `key` is a stable identifier (e.g., `graph_id`, `shard_id`, `(agent_id, phase)`).
- **`max_workers`** — `<= 1` ⇒ sequential path; `> 1` ⇒ bounded `ThreadPoolExecutor` with `min(max_workers, len(tasks))`.
- **`merge_fn`** — receives a **sorted** list of `(key, result)` (sorted by `order_key(key)`, tie‑broken by submit index) and returns the aggregated `A`. **Must be pure.**
- **`order_key`** — maps a task `key` to a total‑orderable value. Use tuples for multi-field keys.

### Deterministic error model

If any task raises, `run_parallel` raises `ParallelError` with:

- `errors: List[TaskError]` where each `TaskError = { key, exc_type, message }`
- `errors` are sorted by `order_key(key)`, then submit index (deterministic tie‑break)
- No partial `merge_fn` call occurs when there are errors

### Empty input

`tasks == []` calls `merge_fn([])` and returns its value. This lets callers define empty-aggregation semantics explicitly.

---

## Usage patterns

### 1) Sharded work with stable keys

```py
from clematis.engine.util.parallel import run_parallel

def _work_on_shard(shard_id: int) -> Result:
    # pure function over the shard; avoid global mutation
    ...

shards = [0, 1, 2, 3]
tasks = [(sid, (lambda s=sid: _work_on_shard(s))) for sid in shards]  # freeze sid

def _merge(pairs):
    # pairs is sorted by sid; build a single object
    out = []
    for sid, res in pairs:
        out.append((sid, res))
    return out

results = run_parallel(
    tasks,
    max_workers=cfg.perf.parallel.max_workers,   # already normalized in PR63
    merge_fn=_merge,
    order_key=lambda sid: (sid,),                # total order
)
```

**Why the `lambda s=sid`?** To avoid Python late-binding in loops and to supply a zero-arg thunk, matching the contract.

### 2) Graph-scoped fanout

```py
tasks = []
for g in graphs:
    gid = g.graph_id
    tasks.append((gid, (lambda _gid=gid: do_t1_for_graph(_gid))))

def merge_graphs(pairs):
    # deterministic append in gid order
    return [res for _, res in pairs]

out = run_parallel(tasks, max_workers=workers, merge_fn=merge_graphs, order_key=lambda k: (k,))
```

### 3) Deterministic failure surfaces

```py
try:
    run_parallel(tasks, max_workers=workers, merge_fn=_merge, order_key=lambda k: (k,))
except ParallelError as e:
    # e.errors is stable; suitable for logs or test assertions
    for item in e.errors:
        log.error("shard=%s type=%s msg=%s", item.key, item.exc_type, item.message)
    raise
```

---

## What **not** to do (anti‑patterns)

- ❌ **Non-zero-arg callables**: `tasks=[(k, fn)]` where `fn` expects parameters will raise `TypeError`. Wrap with `lambda`.
- ❌ **Non-deterministic `merge_fn`**: no randomization, no time-based ordering, no dict iteration that depends on hash seed.
- ❌ **Shared mutable state without synchronization**: mutate per-task globals or caches; instead, return data and merge deterministically.
- ❌ **Rely on completion order**: results are explicitly re-ordered; do not infer timing from the sequence passed to `merge_fn`.
- ❌ **Unstable keys**: using memory addresses or transient counters as `key` will break ordering across runs.

---

## Performance notes (GIL reality check)

- The helper uses `ThreadPoolExecutor`. It benefits most for **I/O-bound** or **C-extension** heavy tasks. For pure Python CPU-bound work, consider keeping `max_workers<=1` or moving to process pools in a future milestone (not in scope for M9).
- `max_workers` is bounded by task count, so small fanouts won’t oversubscribe.
- If tasks produce large payloads, prefer merging lightweight descriptors and lazy-load heavy artifacts in the merge step.

---

## Determinism & identity expectations

- With `max_workers <= 1`, output must be identical (byte-for-byte) to a sequential loop + stable sort by `order_key`.
- With `max_workers > 1`, output must be identical to the sequential path under fixed inputs. The helper enforces this by re-sorting.
- Exception sets are deterministic: same tasks → same `ParallelError.errors` ordering.

---

## Testing contract (mirrors `tests/helpers/test_parallel_helper.py`)

- **Ordering independent of completion**: artificial skew in task runtimes must not affect final order.
- **Sequential equivalence**: `max_workers=1` and `max_workers>1` return identical aggregates.
- **Exceptions deterministic**: multiple failures yield a stable, sorted error list.
- **Zero/negative workers normalization**: treated as sequential.
- **Empty input**: `merge_fn([])` is invoked exactly once.

---

## Wiring guidance (for PR65+; not done in PR64)

1. Identify a fanout site with a natural stable key (e.g., `graph_id`, `shard_id`).
2. Extract work into a **pure thunk**: `lambda key=...: do_work(key)`.
3. Define a **pure** `merge_fn` that consumes `(key, result)` in **sorted** order and assembles the final object.
4. Feed `cfg.perf.parallel.max_workers` and keep **boolean gates** (`t1`, `t2`, `agents`) to decide which fanouts are allowed.
5. Add **identity tests** (ON/OFF) under a fixed seed; add stress tests for shared caches/snapshots.

---

## FAQ

- **Why not return results as a dict?** Dict iteration order depends on insertion; we want explicit ordering via `order_key` and stable tie‑breaks.
- **What about process pools?** Out of scope for M9. Threads suffice for safe adoption; processes may be considered later if justified by profiling.
- **Can I log from inside tasks?** Prefer returning structured data and logging in the merge phase to avoid interleaved output that might hide races.

---

## Cross‑refs

- **PR63** — Introduced `perf.parallel` config surface (defaults OFF; normalizes non‑positive workers to sequential; unknown-key rejection).
- **PR64** — This helper (tests only; no runtime wiring).
- **PR65+** — Wire stages/orchestrator to honor `perf.parallel` using `run_parallel`.
