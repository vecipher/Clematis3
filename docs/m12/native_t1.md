# M12 — Native Kernel Acceleration (T1)

**Status**: PR99 — Rust kernel (perf‑OFF semantics) shipped; PR98 Python FFI & strict‑parity harness included; **PR100 — Wheels & CI matrix (abi3, cp311) in progress**. Default behavior unchanged unless enabled. `available()` returns **True** when the compiled extension (`clematis.native._t1_rs`) is importable; otherwise we fall back to Python.

---

## What this is
An optional, strictly-parity **native inner loop** for the T1 propagation stage. When enabled and eligible, the Python inner loop is replaced by a compiled kernel (initially Rust via PyO3) that produces **identical deltas and metrics** under the default perf‑OFF semantics.

### Goals
- **Determinism first**: byte-identical disabled path; native path returns exactly what the Python loop would (same order, same tie‑breaks).
- **Opt‑in**: behind `perf.native.t1.enabled` and additional safety gates.
- **Cross‑platform**: prebuilt wheels planned for PR100+ (Linux/macOS/Windows; Py 3.11–3.13).
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

> **Note (PR99):**
> `available()` now reflects whether the compiled extension can be imported. If it’s missing, the engine silently uses the Python path—even when `perf.native.t1.enabled=true`.

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

**Targets**
- Platforms: **Linux (manylinux x86_64), macOS (x86_64 on macOS‑13; arm64 on macOS‑14), Windows (AMD64)**.
- Python: **abi3 tagged at cp311** → one wheel per OS/arch, tested on **3.11 / 3.12 / 3.13**.

**Key choices**
- **PyO3 abi3** is enabled via Cargo features: `pyo3 = { features = ["extension-module", "abi3-py311"] }`.
- `pyproject.toml` uses `setuptools-rust` with `binding = "PyO3"` and the extension marked `optional = true` (source installs succeed without Rust).
- CI builds wheels with **cibuildwheel**:
  - macOS builds with `CIBW_ARCHS_MACOS=native` (each runner builds its own arch; no cross‑compile).
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

## Building from source (developers)
> Not required when installing official wheels (planned for PR100+); this is for contributors.

1. Install toolchains:
   ```bash
   xcode-select --install               # macOS SDK/linker (macOS only)
   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
   source "$HOME/.cargo/env"
   rustup default stable
   ```
2. (Python 3.13 only) enable ABI3 forward‑compat for PyO3 ≤ 0.21:
   ```bash
   export PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1
   ```
3. Build the wheel with the extension:
   ```bash
   python -m pip install -U pip build wheel setuptools-rust
   rm -rf build dist *.egg-info
   python -m build -w
   ```
4. Install & verify:
   ```bash
   python -m pip uninstall -y clematis || true
   python -m pip install dist/clematis-*.whl
   python - <<'PY'
   import clematis.native.t1 as t
   print('native available:', t.available())
   PY
   ```

---

## Troubleshooting (once enabled)
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
- **PR100+**: Wheels/CI matrix, prebuilt artifacts, docs polish, hardening.
- **Post‑M12**: perf‑ON semantics in kernel (dedupe & caps), GEL micro‑kernels.

---

## License & reproducibility
- The native backend follows the project’s license.
- Wheels are built with locked dependencies and reproducibility flags where feasible; functional identity is CI‑gated.
