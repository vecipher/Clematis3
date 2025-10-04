## Tests & Identity (how we verify this)

We assert the disabled path is byte‑for‑byte identical and produces no GEL artifacts:

- **PR92 (config + runtime log identity):**
  - `tests/test_identity_disabled_path.py::test_disabled_path_identity_config_roundtrip_graph_subtree` — proves a full `graph.*` subtree with `enabled:false` is inert after validation (compared to a config without `graph`), using `_strip_perf_and_quality_and_graph(...)`.
  - `tests/test_identity_disabled_path.py::test_disabled_path_runtime_no_gel_and_log_identity` — runs a tiny smoke turn twice (baseline vs. `graph.enabled:false`) via `run_smoke_turn`; asserts **no `gel.jsonl`** and **byte‑identical** logs.
- **PR93 (runtime identity + snapshots):**
  - `tests/test_gel_disabled_path_runtime.py::test_gel_disabled_path_runtime_logs_and_snapshots` — repeats the smoke run and additionally asserts **snapshot parity** (same files; identical hashes when present).

Shared helpers live in **`tests/helpers/identity.py`**:

- Normalization utilities for deterministic diffing: `normalize_json_lines(...)`, `normalize_logs_dir(...)`.
- Config strippers: `_strip_perf_and_quality(...)`, `_strip_perf_and_quality_and_graph(...)`.
- Snapshot helpers: `collect_snapshots_from_apply(...)`, `hash_snapshots(...)`.

These tests run with `CI=true` and `CLEMATIS_NETWORK_BAN=1` to keep outputs stable; if `run_smoke_turn` is unavailable on a branch, the runtime tests **skip** rather than fail.

## Pointers

- Follow‑ups for M11:
  - PR92: Identity assertions w/ `graph.*` in config (disabled path).
  - PR93: Runtime smoke asserting no `gel.jsonl` when disabled.
  - PR94: Example configs (`examples/gel/{enabled,disabled}.yaml`) + doc polish.
- See also: **Reflection (M10)** → `docs/m10/reflection.md` (section “Next: HS1/GEL”).
