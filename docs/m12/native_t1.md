# M12 — Native Kernel Acceleration (T1)

**Status**: PR98 — Python FFI surface & strict‑parity harness (defaults OFF; identity‑safe). Still inert by default (
`available() -> False`). The compiled native kernel lands in PR99; this page documents the target design and how to use it once available.

---

## What this is
An optional, strictly-parity **native inner loop** for the T1 propagation stage. When enabled and eligible, the Python inner loop is replaced by a compiled kernel (initially Rust via PyO3) that produces **identical deltas and metrics** under the default perf‑OFF semantics.

### Goals
- **Determinism first**: byte-identical disabled path; native path returns exactly what the Python loop would (same order, same tie‑breaks).
- **Opt‑in**: behind `perf.native.t1.enabled` and additional safety gates.
- **Cross‑platform**: prebuilt wheels for Linux/macOS/Windows (Py 3.11–3.13) once shipped.
- **Frictionless fallback**: if any gate fails, we silently use the Python path and optionally log a gated metric.

### Non-goals (M12)
- Changing semantics, ranking, or T1 defaults.
- Implementing perf‑ON features in native (dedupe ring, frontier/visited caps) — these are planned for a follow‑up.

---

## Invariants
- `perf.native.t1.enabled: false` by default. Disabled path is **byte‑identical** to pre‑native builds.
- The native kernel is only used when all **gates** (see below) are satisfied; otherwise the Python loop runs.
- No RNG, no wall‑clock dependence. All tie‑breaks are lexicographic and mirrored from Python.

---

## Configuration
Add the `perf.native.t1` block to your config. In PR97 this is accepted but inert; later PRs activate the kernel.

```yaml
perf:
  native:
    t1:
      enabled: false        # default; set true to opt‑in once available
      backend: rust         # rust|python (python = debug fallback)
      strict_parity: false  # when true, compute both native & python and assert equality
```

> **Note:** The `perf.native` subtree appears in the normalized config **only** if you provide it. We do not inject this block by default.

---

## When does the native path run?
All of the following must hold:

1. `perf.native.t1.enabled == true`.
2. The native backend is available (e.g., the `_t1_rs` extension is importable in a future PR).
3. **Perf‑ON features are OFF** (M12 scope):
   - No `perf.t1.caps.frontier` / `visited` gating.
   - No `perf.t1.dedupe_window`.
4. Optional: `backend: rust` (default) or `python` (forces legacy path even if enabled).

> **Tip (PR98):**
> `available()` deliberately returns **False** until PR99 ships the compiled extension. Enabling `perf.native.t1` in PR98 therefore does not activate the native path outside of tests; use strict‑parity tests with a monkeypatched `available()` to validate the harness.

If any of these fail, the engine uses the Python loop. A gated counter may be incremented for observability in later PRs (not in PR97).

---

## Data contract (kernel API)
This is the stable interface between Python and the native kernel.

**Inputs** (NumPy arrays unless noted):
- `indptr: int32[n_nodes+1]`, `indices: int32[n_edges]` — CSR over *outgoing* edges.
- `weights: float32[n_edges]` — edge weights.
- `rel_code: int32[n_edges]` — per‑edge relation code (reserved for future use; ignored by the Python reference).
- `rel_mult: float32[n_edges]` — per‑edge relation multiplier. *(Note: a compact table form `float32[3]` may be added later; both are Rust‑compatible.)*
- `seed_nodes: int32[k]`, `seed_weights: float32[k]` — initial wavefront.
- `key_rank: int32[n_nodes]` — stable lexicographic rank for tie‑breaks.
- Scalars: `rate: float32`, `floor: float32`, `radius_cap: int32`, `iter_cap_layers: int32`, `node_budget: float32`.

**Outputs**:
- `d_nodes: int32[m]`, `d_vals: float32[m]` — non‑zero node deltas.
- `metrics: dict` — at least:
  - `pops: int`
  - `iters: int` (layers processed)
  - `propagations: int`
  - `radius_cap_hits: int`
  - `layer_cap_hits: int`
  - `node_budget_hits: int`
  - `_max_delta_local: float`

**Deterministic ordering**: the kernel uses a heap ordered by `(priority=-abs(w), key_rank[u], node_id)` to replicate Python’s processing order.

### PR98 parity harness (dispatcher)
The engine exposes a test‑visible dispatcher that selects the native path when eligible:

```python
_t1_one_graph_dispatch(cfg, indptr, indices, weights, rel_code, rel_mult, key_rank, seeds, params)
```

Logic:

```
if cfg.perf.native.t1.enabled and native.available() and _native_t1_allowed(cfg):
    if strict_parity: compute both & assert exact equality
    else:             return native result
else:
    return python result
```

In PR98, `native.available()` returns **False** by default; tests can monkeypatch it to `True` to exercise strict‑parity without requiring the Rust kernel.

---

## Parity modes
- `strict_parity: false` — use native when eligible; else fall back.
- `strict_parity: true` — compute **both** paths and assert that deltas & metrics are identical. Intended for CI/nightly sweeps and local debugging.

---

## How to enable (once shipped)
Minimal example (YAML):

```yaml
perf:
  native:
    t1:
      enabled: true
      backend: rust
```

Programmatic smoke (Python):

```python
from clematis.engine.orchestrator.core import run_smoke_turn
cfg = {
    "perf": {"native": {"t1": {"enabled": True, "backend": "rust"}}}
}
res = run_smoke_turn(cfg, log_dir="/tmp/clematis-native-smoke")
print("ok", res is not None)
```

If the extension is unavailable or a gate is not met, the run still completes via the Python path.

---

## Building from source (developers)
> Not required when installing official wheels; this is for contributors.

1. Install toolchain:
   ```bash
   python -m pip install -U pip build setuptools setuptools-rust
   # Rust toolchain (rustup) is required once the kernel lands
   ```
2. Build artifacts:
   ```bash
   python -m build   # produces sdist and wheel under dist/
   ```
3. Test installed wheel:
   ```bash
   python -m pip install dist/*.whl
   pytest -q tests/native tests/config
   ```

---

## Troubleshooting (once enabled)
- **Native path not active (PR98)** — expected: `available()` returns False until PR99 ships the compiled extension. The engine will fall back to the Python loop even if `perf.native.t1.enabled=true`.
- **Strict‑parity assertion error** — capture logs and open an issue; include config & a minimal repro. In PR98 this means the Python stub and Python inner disagree (should not happen).
- **Extension import fails** — ensure you installed a wheel that matches your Python/OS/arch, or build from source with Rust.
- **Parity assertion error (strict mode)** — capture logs and open an issue; include the config and a minimal repro.

---

## Roadmap & scope notes
- **PR97**: Config surface + stubs.
- **PR98**: (shipped) Python FFI & strict‑parity harness (still Python only; default OFF; identity‑safe).
- **PR99**: Rust kernel (perf‑OFF semantics), parity tests.
- **PR100+**: Wheels/CI, docs, hardening.
- **Post‑M12**: perf‑ON semantics in kernel (dedupe & caps), GEL micro‑kernels.

---

## License & reproducibility
- The native backend follows the project’s license.
- Wheels are built with locked dependencies and reproducibility flags where feasible; functional identity is CI‑gated.
