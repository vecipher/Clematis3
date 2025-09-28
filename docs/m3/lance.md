## Appendix — LanceDB optional integration (T2 reader)

**Status:** Shipped (optional, defaults OFF). The in‑memory reader remains the default; LanceDB is an opt‑in backend for T2 retrieval.

### Why
Use LanceDB to persist and query the memory index with vector search while keeping CI deterministic (CI still uses the in‑memory path).

### Enable locally
Minimal config sketch (adjust keys/paths to your environment; validation will enforce exact names and bounds):

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
> Tip: With `reader: auto`, the system will try Lance first and **fall back** to the in‑memory reader when Lance is unavailable or misconfigured (emits a single warning).
### Quick verification
Run the existing reader and Lance optional tests:
# Reader parity / integration
pytest -q tests/integration/test_t2_reader_parity.py
# Lance optional index behaviors
pytest -q tests/test_lance_index_optional.py
### Optional: seed a tiny demo table
If you add a helper script like `scripts/seed_lance_demo.py`, you can populate a tiny Lance table and verify parity end‑to‑end:
python scripts/seed_lance_demo.py  # creates a local .lancedb with a small table
pytest -q tests/integration/test_t2_reader_parity.py
### Failure modes & fallbacks
- Missing Lance URI/table → falls back to in‑memory; warns once.
- Invalid partition spec → config validator rejects with a clear message.
- FP16 without precomputed norms → test warns (performance) or validator flags depending on config.
CI remains unaffected: workflows continue to run **offline** and **in‑memory** to guarantee deterministic results.
