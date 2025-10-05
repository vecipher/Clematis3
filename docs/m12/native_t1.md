# M12 — Native Kernel Acceleration (T1)

**Status**: PR99 — Rust kernel (perf‑OFF semantics) shipped; PR98 — Python FFI & strict‑parity harness included; **PR100 — Wheels & CI matrix (abi3, cp311) shipped**; **PR101 — Bench & Docs (advisory) shipped**; **PR102 — Hardening & Diagnostics (in progress)**. Default behavior unchanged unless enabled. `available()` returns **True** when the compiled extension (`clematis.native._t1_rs`) is importable; otherwise we fall back to Python.

---

## What this is
An optional, strictly-parity **native inner loop** for the T1 propagation stage. When enabled and eligible, the Python inner loop is replaced by a compiled kernel (initially Rust via PyO3) that produces **identical deltas and metrics** under the default perf‑OFF semantics.

### Goals
- **Determinism first**: byte-identical disabled path; native path returns exactly what the Python loop would (same order, same tie‑breaks).
- **Opt‑in**: behind `perf.native.t1.enabled` and additional safety gates.
- **Cross-platform**: prebuilt wheels **shipped in PR100** (Linux/macOS/Windows; Py 3.11–3.13).
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

### Env overrides (PR102)
- `CLEMATIS_NATIVE_T1=1` — enables the native path even if config sets `enabled: false`.
- `CLEMATIS_STRICT_PARITY=1` — turns on strict parity without touching config.

---

## When does the native path run?
All of the following must hold:

1. `perf.native.t1.enabled == true` **or** environment `CLEMATIS_NATIVE_T1=1`.
2. The native backend is available (e.g., the `_t1_rs` extension is importable in a future PR).
3. **Perf‑ON features are OFF** (M12 scope):
   - No `perf.t1.caps.frontier` / `visited` gating.
   - No `perf.t1.dedupe_window`.
4. Optional: `backend: rust` (default) or `python` (forces legacy path even if enabled).

> **Note (PR99):**
> `available()` now reflects whether the compiled extension can be imported. If it’s missing, the engine silently uses the Python path—even when `perf.native.t1.enabled=true`.

Counters are incremented (PR102) for observability when native is gated or disabled.

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

**Deterministic ordering**: processing is ordered by a heap key equivalent to `(priority desc by |Δ|, key_rank asc, node_id asc)`; this mirrors the Python reference.

### Parity harness & dispatcher (PR98 + PR99)
The engine exposes a dispatcher that selects the native path when eligible and supports strict‑parity checks:

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

## Wheels & CI Matrix (PR100)

**Status:** Shipped in PR100.

**Targets**
- Platforms: **Linux (manylinux x86_64), macOS (x86_64 on macOS‑13; arm64 on macOS‑14), Windows (AMD64)**.
- Python: **abi3 tagged at cp311** → one wheel per OS/arch, tested on **3.11 / 3.12 / 3.13**.

**Key choices**
- **PyO3 abi3** is enabled via Cargo features: `pyo3 = { features = ["extension-module", "abi3-py311"] }`.
- `pyproject.toml` uses `setuptools-rust` with `binding = "PyO3"` and the extension marked `optional = true` (source installs succeed without Rust).
- CI builds wheels with **cibuildwheel**:
  - macOS builds use `CIBW_ARCHS_MACOS=native` on `macos-13` (x86_64) and `macos-14` (arm64); no cross‑compile.
  - Linux sets `RUSTFLAGS=-C link-arg=-Wl,--build-id=none` for lean ELF.
  - sdist is produced **once on Linux**; other OS jobs skip sdist.
- CI test job **installs the built wheels** (not the repo) and runs a focused subset: `tests/native`, `tests/config`, `tests/helpers`.
- Packaging smoke: `tests/packaging/test_wheel_has_native.py` asserts `_t1_rs` is present and `t1.available() is True`.

**Artifacts & tags**
- Wheels are tagged **cp311‑abi3** per platform:
  - `manylinux_x86_64`, `macosx_13_x86_64`, `macosx_14_arm64`, `win_amd64`.
- The sdist **includes Rust sources** (`clematis/native/Cargo.toml` and `clematis/native/src/*.rs`).

**Local wheel smoke (quick)**
Use this when developing locally (no cibuildwheel required):

```bash
export PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1  # Python 3.13 + PyO3 ≤ 0.21
python -m pip install -U build setuptools-rust
python -m build --wheel -o wheelhouse
WHL=$(ls wheelhouse/*.whl | head -n1)
python -m pip install --force-reinstall "$WHL"
# verify from outside the repo to avoid source shadowing
TMP=$(mktemp -d)
python - <<'PY'
import os, sys, importlib.util
os.chdir("$TMP"); sys.path = [p for p in sys.path if p and p != os.getcwd()]
from clematis.native import t1
print("native?", t1.available())
print("path:", importlib.util.find_spec("clematis.native._t1_rs").origin)
PY
```

**Common pitfall**
- Importing from the **repo root** shadows the installed wheel, so `_t1_rs` seems missing. Run Python from outside the repo (as above) or remove the repo path from `sys.path`.

---

## Perf Smoke (PR101 — advisory)

**Goal**
Validate a modest speedup for the native kernel with a tiny, deterministic smoke test. This job is informational and **not required** for merge.

**Test**
- Location: `tests/perf/test_bench_t1_smoke.py`
- Deterministic graph: ~8k nodes, ~50k directed edges (seed=1337)
- Mode: prefers parity helpers; otherwise toggles backend via `CLEMATIS_NATIVE_T1`
- Assertion (Linux CI): native ≥ **1.8×** faster than Python
- Gating: `@pytest.mark.slow`; **Linux-only** by default; opt-in via `PERF_SMOKE=1`
- Local override (optional): set `PERF_SMOKE_ANY_OS=1` to run on macOS/Windows

**Local run (outside the repo to avoid source shadowing)**
```bash
PERF_SMOKE=1 pytest -q /path/to/Clematis3/tests/perf/test_bench_t1_smoke.py -m slow
```

**CI job (GitHub Actions, runs on push or when PR labeled `perf-smoke`)**
```yaml
name: perf-smoke-linux
on:
  push:
    branches: [ main, master, trunk, develop ]
  pull_request:
    types: [ opened, synchronize, labeled, reopened ]
jobs:
  perf-smoke:
    if: github.event_name == 'push' || contains(join(fromJson(toJson(github.event.pull_request.labels)).*.name, ','), 'perf-smoke')
    runs-on: ubuntu-22.04
    timeout-minutes: 20
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.12'
      - uses: dtolnay/rust-toolchain@stable
      - name: Install deps
        run: |
          python -m pip install --upgrade pip wheel setuptools
          python -m pip install pytest pytest-timeout pyyaml
      - name: Build wheel (abi3-capable)
        run: |
          python -m pip wheel . -w wheelhouse --no-deps
      - name: Install wheel
        run: |
          python -m pip install --find-links=wheelhouse clematis
      - name: Sanity check native import (outside checkout to avoid shadowing)
        working-directory: /tmp
        env:
          PYTHONNOUSERSITE: "1"
        run: |
          python - <<'PY'
          import importlib.util
          spec = importlib.util.find_spec('clematis.native._t1_rs')
          print('ext spec:', spec)
          import clematis.native.t1 as n
          print('available():', n.available())
          assert n.available(), 'native module should be available from installed wheel'
          PY
```

> **Why run from `/tmp`?** Importing from the repo root can **shadow** the installed wheel so `_t1_rs` appears missing. Running from a clean working directory ensures the compiled extension is used. See **Troubleshooting** below.

---

## Diagnostics & Hardening (PR102 — in progress)

**What’s new** (observable behavior):
- **Metrics counters** under `metrics["native_t1"]`:
  - `used_native` — native kernel executed successfully.
  - `fallback_import_failed` — native was enabled+allowed but `available()`/import failed; Python fallback used.
  - `fallback_gated_caps` — native gated off due to `perf.t1.caps`.
  - `fallback_gated_dedupe` — native gated off due to `perf.t1.dedupe_window`.
  - `fallback_gated_other` — native gated off for other reasons.
  - `fallback_runtime_exc` — native raised a runtime exception; Python fallback used.
  - `strict_parity_mismatch` — strict parity found differences (value = count of mismatched nodes).
  - `strict_parity_native_exc` — native raised while in strict parity mode.
- **Once‑only logs** (no spam):
  - `gate_off` (INFO) — native gated (caps/dedupe) → Python path.
  - `import_failed` (WARNING) — enabled but native import/`available()` failed.
  - `runtime_exc` (ERROR) — native raised; fell back to Python.
  - `runtime_exc_strict` (ERROR) — native raised in strict parity.
  - `strict_parity_mismatch` (ERROR) — first mismatch summary.
  - `pyo3_runtime_exc` (ERROR) — exception surfaced out of the Rust/PyO3 layer.

**Where to see them**
- Stage metrics carry these under `metrics["native_t1"]`. They also bubble up into the top-level turn log under `t1.native_t1` (see `logs/turn.jsonl`). Example:

```json
{
  "turn_id": 42,
  "t1": {
    "pops": 1234,
    "iters": 17,
    "graphs_touched": 1,
    "native_t1": {
      "used_native": 1,
      "fallback_import_failed": 0,
      "fallback_gated_caps": 0,
      "fallback_gated_dedupe": 0,
      "fallback_runtime_exc": 0
    }
  }
}

**Strict parity (mode)**
- Toggle via config (`perf.native.t1.strict_parity: true`) or `CLEMATIS_STRICT_PARITY=1`.
- Engine computes **both** Python and native results and compares with tolerances `atol=1e-6`, `rtol=1e-5`.
- On mismatch → raises with a concise report (first few nodes):
  ```
  strict parity mismatch: 1 node(s) differ. node=123 py=0.532 rs=0.531 Δ=0.001
  ```

**Failure modes & fallbacks**
- **Import unavailable**: if native is enabled but `_t1_rs` cannot be imported, the engine logs `import_failed`, bumps `fallback_import_failed`, and uses the Python path.
- **Runtime exception** (`MemoryError`, `TypeError`, `ValueError`, `OverflowError`): logs `runtime_exc`, bumps `fallback_runtime_exc`, and uses the Python path (unless strict parity, in which case it re‑raises).

**Local toggle examples**
```bash
# Force native and strict parity locally (useful for debugging)
export CLEMATIS_NATIVE_T1=1
export CLEMATIS_STRICT_PARITY=1
```

## Nightly strict parity (PR102)
A nightly workflow validates that native matches Python under strict parity on Linux.

- **Workflow**: `.github/workflows/strict-parity-nightly.yml` (scheduled daily at 20:00 UTC; also manual).
- **Env**: `CLEMATIS_NATIVE_T1=1`, `CLEMATIS_STRICT_PARITY=1`, `PERF_SMOKE=1`.
- **Runs**:
  - Perf smoke: `tests/perf/test_bench_t1_smoke.py -m slow` (from `/tmp` to avoid shadowing).
  - Diagnostics/parity: `tests/native`.

```yaml
# excerpt
- name: Run strict-parity suites
  working-directory: /tmp
  env:
    CI: "true"
    CLEMATIS_NETWORK_BAN: "1"
    PERF_SMOKE: "1"
    CLEMATIS_NATIVE_T1: "1"
    CLEMATIS_STRICT_PARITY: "1"
  run: |
    pytest --maxfail=1 -q $GITHUB_WORKSPACE/tests/perf/test_bench_t1_smoke.py -m slow
    pytest --maxfail=1 -q $GITHUB_WORKSPACE/tests/native
```

---

## Troubleshooting (once enabled)
- **Perf smoke skipped locally** — The perf test is Linux-only by default. This is expected on macOS/Windows. Use CI, or set

  `PERF_SMOKE_ANY_OS=1` to override locally (advisory only).
- **`ModuleNotFoundError: clematis.native._t1_rs`** — You’re likely importing from the source tree which shadows the installed package. Run Python from outside the repo, or remove the repo path from `sys.path`. Ensure you installed a platform wheel.
- **`error: can't find Rust compiler`** — Install Rust via `rustup` and ensure `cargo` is on `PATH`.
- **`the configured Python interpreter version (3.13) is newer than PyO3's maximum supported`** — set `PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1` in the build environment (or upgrade PyO3 in a follow‑up).
- **Strict‑parity assertion** — capture config + minimal repro and file an issue. The native and Python paths are expected to be bit‑identical under perf‑OFF semantics.

---

### Repo hygiene
- Do **not** commit compiled artifacts (e.g., `clematis/native/_t1_rs*.so`). Add them to `.gitignore`. CI builds wheels when needed.

---

## Roadmap & scope notes
- **PR97**: Config surface + stubs. ✅
- **PR98**: Python FFI & strict‑parity harness. ✅
- **PR99**: Rust kernel (perf‑OFF semantics), parity tests. ✅
- **PR100**: Wheels/CI matrix, prebuilt artifacts, docs polish, hardening. ✅
- **PR101**: Bench & docs (advisory perf smoke on Linux). ✅
- **PR102**: Hardening & diagnostics (counters, once‑logs, strict‑parity nightly). _In progress_
- **Post‑M12**: perf‑ON semantics in kernel (dedupe & caps), GEL micro‑kernels.

---

## License & reproducibility
- The native backend follows the project’s license.
- Wheels are built with locked dependencies and reproducibility flags where feasible; functional identity is CI‑gated.
