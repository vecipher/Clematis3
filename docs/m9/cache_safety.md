

# M9: Cache safety under parallelism

**Status:** Introduced in PR65. Default mode keeps parallelism OFF and uses lock‑wrapped shared caches when enabled later. No behavior change in the OFF path; outputs remain byte‑identical.

## Why this exists

When we fan out T1/T2 work, shared caches can be touched by multiple threads. Without guardrails, this risks:

- Data races / iterator invalidation
- Non‑deterministic eviction ordering
- Heisenbugs from “read‑modify‑write” sequences on shared entries

PR65 introduces **two deterministic strategies**:

1) **Lock‑wrapped single cache** (default): serialize access to an existing cache instance.
2) **Per‑worker isolates + deterministic post‑merge**: each task uses its own ephemeral cache; results and cache entries are merged back in a fixed order.

Both paths preserve **identity** when parallelism is OFF and when `max_workers<=1`.

---

## Modes & trade‑offs

| Mode | What it does | Pros | Cons | Typical use |
|---|---|---|---|---|
| `lock` (default) | Wrap a single cache in a thin lock | Minimal memory, zero semantic drift | Contention on hot paths | Good first step; keeps behavior identical |
| `isolate` | One cache per worker; merge back deterministically | Avoids contention; clearer ownership | Slightly higher memory and a merge pass | Hot fanouts with many lookups |

> If a config knob is surfaced, it will be `perf.parallel.cache_mode: lock|isolate` (default `lock`). Until then, wiring may select internally per stage.

---

## API surface (added in PR65)

```py
# clematis/engine/cache.py

class ThreadSafeCache(Generic[K, V]):
    def __init__(self, inner: CacheProtocol[K, V], lock: Optional[threading.RLock] = None) -> None: ...
    def get(self, key: K) -> Optional[V]: ...
    def put(self, key: K, value: V) -> None: ...
    def __contains__(self, key: K) -> bool: ...
    def items(self) -> Iterable[Tuple[K, V]]: ...

class ThreadSafeBytesCache(Generic[K, V]):
    def __init__(self, inner: Any, lock: Optional[threading.RLock] = None) -> None: ...
    def get(self, key: K) -> Optional[V]: ...
    def put(self, key: K, value: V, cost: int): ...
    def __contains__(self, key: K) -> bool: ...
    def items(self) -> Iterable[Tuple[K, V]]: ...

def merge_caches_deterministic(
    target: CacheProtocol[K, V],
    worker_caches: List[Tuple[Any, CacheProtocol[K, V]]],
    *,
    worker_order_key: Callable[[Any], Any],
    key_order_key: Callable[[K], Any],
    on_conflict: str = "first_wins",  # or "assert_equal"
) -> None: ...
```

Support stubs:

- `LRUCache` now supports `__contains__` and `items()` snapshots with TTL pruning and stable ordering.
- `LRUBytes` exposes a deterministic `items()` iterator (LRU→MRU) without mutating recency.

---

## Wiring patterns

### 1) Lock‑wrapped shared cache (default)

```py
# t1.py / t2.py (already applied)
from clematis.engine.cache import ThreadSafeCache, ThreadSafeBytesCache, LRUCache
from clematis.engine.util.lru_bytes import LRUBytes

# Legacy LRU (entries count + TTL)
shared_cache = ThreadSafeCache(LRUCache(max_entries=..., ttl_s=...))

# Size‑aware cache (entries/bytes)
shared_bytes = ThreadSafeBytesCache(LRUBytes(max_entries=..., max_bytes=...))
```

This path is **semantics‑preserving** and sufficient unless we hit contention hotspots.

### 2) Per‑worker isolates + deterministic merge

Combine with `run_parallel` (PR64). Each task returns its result **and** its worker‑local cache. Merge in `merge_fn` using the **same** `order_key` you gave to `run_parallel`.

```py
from clematis.engine.util.parallel import run_parallel
from clematis.engine.cache import ThreadSafeCache, merge_caches_deterministic, LRUCache

shared = ThreadSafeCache(LRUCache(max_entries=1024, ttl_s=600))

def thunk_for(shard_id: int):
    def _():
        wc = LRUCache(max_entries=128, ttl_s=600)     # isolated worker cache
        res = do_work(shard_id, cache=wc)
        return (res, wc)
    return _

tasks = [(sid, thunk_for(sid)) for sid in shards]

def merge_fn(pairs):
    out = []
    worker_caches = []
    for key, (res, wc) in pairs:   # pairs sorted by order_key(key)
        out.append(res)
        worker_caches.append((key, wc))
    merge_caches_deterministic(
        target=shared,
        worker_caches=worker_caches,
        worker_order_key=lambda k: (k,),
        key_order_key=lambda k: (k,),
        on_conflict="first_wins",  # or "assert_equal" in tests
    )
    return out

out = run_parallel(
    tasks,
    max_workers=cfg.perf.parallel.max_workers,
    merge_fn=merge_fn,
    order_key=lambda shard_id: (shard_id,),
)
```

**Determinism rules:**
- Worker order is `order_key(worker_key)` with stable tie‑breaks (submit index).
- Within each worker, `items()` are sorted by `key_order_key`.
- Conflict policy is explicit:
  - `"first_wins"` (default): first worker’s value wins, later ones skipped.
  - `"assert_equal"`: raises if a later worker produces a different value (useful in tests).

---

## Testing contract (what CI asserts)

- **Guardrails** (`tests/test_cache_guardrails.py`)
  - `ThreadSafeCache` and `ThreadSafeBytesCache` tolerate concurrent writers/readers.
  - `items()` is a stable snapshot and does not mutate recency.
- **Coherency** (`tests/test_cache_coherency.py`)
  - T2 cache keying is version‑ and query‑sensitive.
  - Disabling the cache bypasses hits.
  - Deterministic merge policies behave as documented (first‑wins vs assert‑equal).

All tests avoid timing asserts and sleeps; they are deterministic and fast.

---

## Performance notes

- `lock` mode adds only the cost of an `RLock` critical section. For I/O‑bound or C‑extension heavy work, this is usually negligible.
- `isolate` mode avoids lock contention at the cost of per‑worker memory and one merge pass.
- For pure Python CPU‑bound work, parallel gains may be limited by the GIL; prefer smaller fanouts or keep `max_workers<=1`.

---

## Acceptance & invariants

- With parallel **OFF**: byte‑for‑byte identical outputs vs prior milestones.
- With parallel **ON** and `max_workers<=1`: identical outputs to sequential.
- Merge order and conflict handling are explicit and test‑verified.

---

## Cross‑references

- **PR63** — parallel config surface & identity rules: [`docs/m9/overview.md`](./overview.md)
- **PR64** — deterministic parallel helper: [`docs/m9/parallel_helper.md`](./parallel_helper.md)
- **PR65** — this page (cache safety wrappers + deterministic merge)
