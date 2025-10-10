## Appendix — LanceDB optional integration (T2 reader)

**Status:** Shipped (optional, defaults OFF). The in-memory reader remains the default; LanceDB is an opt-in backend for T2 retrieval.

### Why

Use LanceDB to persist and query the memory index with vector search while keeping CI deterministic (CI still uses the in-memory path).

### Enable locally

Minimal config sketch (adjust keys/paths to your environment; validation enforces exact names and bounds):

```yaml
t2:
  reader: "auto"        # "inmemory" | "lance" | "auto"
lance:
  enabled: true
  # Choose one URI/path style depending on your setup:
  uri: ".lancedb"       # local directory store (example)
  table: "mem_index"    # logical table name
  partitions:
    owner_quarter: true  # optional partitioning (validated in tests)
  precompute_norms: true # required when using fp16 indexes
```

> Tip: With `reader: "auto"`, the system tries Lance first and **falls back** to the in-memory reader when Lance is unavailable or misconfigured (emits a single warning; behavior remains deterministic).

### Deterministic recency filters

- The exact-semantic tier applies a rolling `recent_days` cutoff (default 30 days). When LanceDB is in use, the index now honors the orchestrator-supplied `hints["now"]` timestamp before falling back to `datetime.now()`.
- Identity paths already pass `now` (e.g., the console sets `--now-ms`); keep doing so in bespoke scripts to avoid drift when replaying logs or comparing bundles.
- When `hints["now"]` is absent, LanceDB and the in-memory index both fall back to wall-clock UTC, which is acceptable for ad-hoc runs but not identity comparisons.

### Quick verification

Run the existing reader and Lance optional tests:

```bash
# Reader parity / integration
pytest -q tests/integration/test_t2_reader_parity.py
# Lance optional index behaviors (requires lancedb extras)
pytest -q tests/test_lance_index_optional.py
```

### Optional: seed a tiny demo table

If you add a helper script like `scripts/seed_lance_demo.py`, you can populate a tiny Lance table and verify parity end-to-end:

```bash
python scripts/seed_lance_demo.py  # creates a local .lancedb with a small table
pytest -q tests/integration/test_t2_reader_parity.py
```

### Failure modes & fallbacks

- Missing Lance URI/table → falls back to in-memory; warns once.
- Invalid partition spec → config validator rejects with a clear message.
- FP16 without precomputed norms → tests warn (performance) or the validator flags the config, depending on whether the fast path is requested.

CI remains unaffected: workflows continue to run **offline** and **in-memory** to guarantee deterministic results.
