# Changelog
All notable changes to this project will be documented in this file.










## [Unreleased]
**Summary:** PR98–PR102 land the Python FFI surface, strict‑parity harness, Rust kernel, **abi3 wheels + CI matrix**, advisory **bench + docs**, and **hardening/diagnostics** (counters, once‑only logs, strict‑parity nightly). Defaults remain identity‑safe unless explicitly enabled.


### Breaking changes
None.

-### Upgrade notes
- **Native T1 diagnostics:** Counters moved to `t1_native_diag.jsonl` (non-identity). `turn.jsonl` schema unchanged.
- **pytest discovery scope:** `pytest.ini` now sets `testpaths = tests`. If you normally run `pytest` from a parent directory (e.g., your Desktop or a monorepo root), only this repo’s `tests/` will be collected. Run from the project root or pass an explicit path (e.g., `pytest tests/`).
- **abi3 wheels/local build:** Wheels use abi3 (cp311). When building locally on Python 3.13, set `PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1` (or upgrade PyO3) before building to avoid version probes.
- **Nightly strict parity:** New nightly job (`strict-parity-nightly.yml`) is advisory (non‑required). It compares native vs Python; failures won’t block PRs but should be triaged on `main`.


### Added (PR98 — M12-02: Python FFI surface & strict-parity harness; identity-safe)
- **Native stub (FFI shape):** `clematis/native/t1.py`
  - Implements `propagate_one_graph_rs(...)` (Python) with canonical dtypes (`int32`/`float32`) and forward-compatible `rel_code`.
  - Packs decay/limits from positional args: `{rate, floor, radius_cap, iter_cap_layers, node_budget}`; lazy-imports `_t1_one_graph_python_inner(...)`.
  - `available() -> False` by default; tests may monkeypatch to `True`.
- **Engine wiring (gated, lazy):** `clematis/engine/stages/t1.py`
  - Factors legacy inner loop into `_t1_one_graph_python_inner(...)` (no behavior change).
  - Adds `_t1_one_graph_native(...)` (calls the Python stub), `_native_t1_allowed(cfg)`, and `_t1_one_graph_dispatch(...)`.
  - Dispatcher logic:
    ```
    if cfg.perf.native.t1.enabled and native.available() and _native_t1_allowed(cfg):
        if strict_parity: compute both & assert exact equality
        else:             return native result
    else:
        return python result
    ```
  - All imports of `clematis.native` are **lazy**; default path untouched.
- **Tests:** (pure Python)
  - `tests/native/test_contract_shapes.py` — exercises the FFI call shape; asserts dtypes/shapes; metrics dict present.
  - `tests/native/test_strict_parity_stub.py` — monkeypatches `available()->True`; strict-parity compares stub vs python inner; must match exactly.
- **Invariants**
  - Native **OFF by default**; disabled-path identity stays byte-identical to `0.9.0a2`.
  - No Rust code yet; no wheels/CI matrix changes.
  - Any attempt to import the native kernel when unavailable remains inert (stub).
- **How to verify (quick):**
  ```
  pytest -q tests/native/test_contract_shapes.py
  pytest -q tests/native/test_strict_parity_stub.py
  pytest -q              # full sweep; identity unchanged
  rg "^from clematis\\.native" -n clematis/engine/stages/t1.py || true  # only lazy imports inside functions
  ```
- **Risks / notes**
  - Parity mode computes both paths; failures raise `AssertionError` with short messages.
  - Keep `_t1_one_graph_python_inner(...)` a verbatim move of the legacy inner loop to ensure a trustworthy baseline for PR99.

#### Roadmap
- **PR99:** (shipped in Unreleased) Rust kernel (perf‑OFF semantics) + parity tests.

### Added (PR99 — M12-03: Rust kernel (perf‑OFF semantics); optional, identity‑safe)
- **Native extension:** `clematis/native/src/lib.rs` via PyO3 exposes `t1_propagate_one_graph(...)` and returns `(d_nodes:int32, d_vals:float32, metrics:dict)`; outputs only non‑zeros. Heap ordering is deterministic: `(priority: f32, key_rank: i64, node: i64)`.
- **Packaging:** `pyproject.toml` integrates setuptools‑rust and builds extension module `clematis.native._t1_rs`. Wheels include the compiled `.so` when built on a platform toolchain.
- **Python bridge:** `clematis/native/t1.py` now:
  - `available()` returns `True` iff `from clematis.native import _t1_rs` succeeds.
  - `propagate_one_graph_rs(...)` delegates to `_t1_rs.t1_propagate_one_graph(...)` (same positional arg order as PR98 stub), returning canonical dtypes.
- **Engine gate:** `clematis/engine/stages/t1.py` flips `_native_t1_allowed(cfg)` to **disallow** the native path when any T1 compaction/caps/dedupe knobs are set (`perf.t1.caps.*` or `perf.t1.dedupe_window`). With `strict_parity=true` the stage computes both paths and asserts equality; otherwise it uses the native result.
- **Tests:**
  - `tests/native/test_import_extension_optional.py` — project remains usable when the extension is missing.
  - `tests/native/test_t1_parity_small.py` / `..._random.py` — exact parity on hand‑built and randomized tiny graphs.
  - `tests/native/test_t1_fallback_when_perf_caps_on.py` — enabling any cap/dedupe forces fallback to Python.
  - Hygiene: `tests/native/test_t1_parity_random.py` now uses proper Hypothesis `@given(...)` instead of `.example()` loops (less flake, faster).
- **How to verify (local):**
  ```
  # 1) Install Rust toolchain (one time)
  curl https://sh.rustup.rs -sSf | sh
  rustup target add aarch64-apple-darwin  # on Apple Silicon

  # 2) Build a wheel with the extension
  export PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1   # Python 3.13 workaround for PyO3<0.22
  python -m pip install -U pip build wheel setuptools-rust
  rm -rf build dist *.egg-info
  python -m build -w

  # 3) Install the wheel into a clean venv and verify
  python -m pip uninstall -y clematis || true
  python -m pip install dist/clematis-0.9.0a2-*.whl
  python - <<'PY'
  import clematis.native.t1 as t; print("native available:", t.available())
  PY

  # 4) Run the native tests
  pytest -q tests/native
  ```
- **Troubleshooting:**
  - `ModuleNotFoundError: clematis.native._t1_rs` after install usually means you're importing from the **source tree** (editable import shadows the installed wheel). Ensure the current working directory is **not** the repo root when testing the installed wheel, or run `python -c 'import sys;print(sys.path[0])'` to confirm.
  - `error: can't find Rust compiler` → install with `rustup` and ensure `cargo` is on `PATH`.
  - `the configured Python interpreter version (3.13) is newer than PyO3's maximum supported` → set `PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1` or upgrade PyO3 in a follow‑up.
- **Invariants:**
  - Native path is **OFF by default** and additionally gated by `_native_t1_allowed(cfg)`.
  - Disabled‑path identity remains **byte‑identical** to `0.9.0a2`.


### Added (PR100 — M12-04: Wheels & CI Matrix; abi3 cp311)
- **Goal:** Produce prebuilt wheels for Linux (manylinux x86_64), macOS (x86_64 on macOS‑13; arm64 on macOS‑14), and Windows (AMD64) using **abi3 (cp311)** so one wheel per OS/arch covers Py3.11–3.13. Test the **installed** wheels.

- **Cargo (abi3):** `clematis/native/Cargo.toml` sets PyO3 features to `{"extension-module", "abi3-py311"}`.
- **Pyproject (setuptools-rust):** `[[tool.setuptools-rust.ext-modules]]` configured with `target = "clematis.native._t1_rs"`, `path = "clematis/native/Cargo.toml"`, `binding = "PyO3"`, and `optional = true`.
- **Workflow:** new `.github/workflows/wheels.yml` using **cibuildwheel**.
  - **Build job:** matrix over `{ubuntu-22.04, macos-13, macos-14, windows-2022}`; builds **cp311-abi3** wheels with `CIBW_ARCHS_MACOS=native` (runner-native arch), sets `PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1`, and on Linux sets `RUSTFLAGS="-C link-arg=-Wl,--build-id=none"`. sdist is built **only on Linux**.
  - **Test job:** installs the produced wheel on Python **3.11 / 3.12 / 3.13** and runs a focused subset: `tests/native`, `tests/config`, `tests/helpers`. Tests are executed from the `tests/` directory to avoid importing the repo tree.

- **Packaging test:** `tests/packaging/test_wheel_has_native.py` — asserts the compiled module exists and that `from clematis.native import t1; t1.available() is True`; also checks the bridge exposes `propagate_one_graph_rs`.

- **.gitignore hygiene:** ignore local wheel outputs and native binaries: `wheelhouse/`, `*.whl`, `*.pyd`, `*.dll`, `*.dylib`, plus `.benchmarks/`.

- **Invariants:** native path remains **OFF by default** and additionally gated by `_native_t1_allowed(cfg)`; disabled path stays byte‑identical to `0.9.0a2`.

- **How to verify (local smoke):**
  ```bash
  export PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1
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
  print("native available:", t1.available())
  print("module:", importlib.util.find_spec("clematis.native._t1_rs").origin)
  PY
  pytest -q tests/native tests/config tests/helpers
  ```

- **CI acceptance:**
  - Each OS/arch produces **one** `cp311-abi3` wheel; Linux also uploads the sdist.
  - Installing those wheels on **3.11 / 3.12 / 3.13** passes the focused test subset and `t1.available() == True`.
- **PR101–102:** Bench doc + diagnostics/hardening; optional nightly strict-parity job.
- **PR103:** Version bump (0.9.0a3) + release.


### Added (PR101 — M12-05: Bench & Docs; advisory)
- **Goal:** bounded perf smoke and complete operator docs (no change to defaults).
- **Bench:** `tests/perf/test_bench_t1_smoke.py`
  - Builds a deterministic synthetic graph (~50k edges) with a fixed seed.
  - Measures Python vs native (when enabled/allowed) and asserts native ≥ **1.8×** speedup on Linux.
  - Markers: `@pytest.mark.slow`; runs only when `PERF_SMOKE=1` and on Linux in CI.
- **Docs:** `docs/m12/native_t1.md` updated — scope, enablement, parity, fallback conditions, build‑from‑source, bench notes.
- **README:** status/badge updated for M12 Native T1.
- **CI:** `perf-smoke-linux` (advisory, non‑required)
  - `ubuntu-22.04`, Python 3.12, native enabled.
  - Triggers on push to default branch and when PRs are labelled `perf-smoke`.
  - Does **not** gate merges.
- **Invariants:** identity path unchanged when native is disabled; bench is observational only.

### Changed (PR102 — M12-06: Hardening & Diagnostics; finalized)
- **Diagnostics sink (Option C):**
  - Counters now emit to a dedicated **non-identity** stream `t1_native_diag.jsonl` via the helper `engine/orchestrator/logging.log_t1_native_diag(...)`.
  - `turn.jsonl` schema is **unchanged**; no `native_t1` appears there anymore.
  - Staging: `engine/util/io_logging.py` adds `STAGE_ORD["t1_native_diag.jsonl"] = 10`; the stream is **excluded** from identity normalization.
- **Engine (T1 dispatcher):** `clematis/engine/stages/t1.py`
  - Once-only logging helper and metrics counter helper.
  - Fallback counters under `metrics["native_t1"]`:
    - `used_native`, `fallback_import_failed`,
    - `fallback_gated_caps`, `fallback_gated_dedupe`, `fallback_gated_other`,
    - `fallback_runtime_exc`, `strict_parity_mismatch`, `strict_parity_native_exc`.
  - Gated-off behavior (caps/dedupe) logs once and never calls native.
  - **Strict parity mode** (`perf.native.t1.strict_parity=true` or `CLEMATIS_STRICT_PARITY=1`): compute both, compare (`atol=1e-6`, `rtol=1e-5`), raise with concise diff on mismatch.
  - Runtime exceptions from native (MemoryError/TypeError/ValueError/OverflowError) cause a **soft fallback** (unless strict parity, which re-raises).
- **Native bridge:** `clematis/native/t1.py`
  - Wraps the PyO3 call in try/except; logs once (`pyo3_runtime_exc`) then re-raises for the dispatcher to handle.
- **Orchestrator logging:** `clematis/engine/orchestrator/logging.py` & `core.py`
  - New `log_t1_native_diag(ctx, agent, native_t1)` helper; `core.py` refactored to call it after `t1.jsonl` write when `metrics["native_t1"]` is truthy.
  - Fail-soft semantics (logging never crashes the turn). `turn` is serialized as a string for consistency.
- **Tests:**
  - `tests/native/test_diagnostics.py` — import-failure fallback & logging (once-only), gated-off counters, runtime-exception fallback, strict parity mismatch raising, and once-only log semantics via `caplog`.
  - `tests/native/test_diag_stream_jsonl.py` — unit coverage for `log_t1_native_diag(...)`: emits exactly one record when diagnostics exist; no record when empty/None.
- **CI:** `.github/workflows/strict-parity-nightly.yml`
  - Nightly (20:00 UTC) and manual dispatch.
  - Installs the wheel, validates `_t1_rs` import from `/tmp` (no source shadowing), runs perf smoke and diagnostics under `CLEMATIS_NATIVE_T1=1`, `CLEMATIS_STRICT_PARITY=1`, `PERF_SMOKE=1`.
  - Optional artifact upload of `logs/t1_native_diag.jsonl` recommended (non-required).
- **Repo hygiene:** `pytest.ini`
  - `testpaths = tests` to constrain discovery; corrected `filterwarnings` placement; markers documented.
- **Invariants:** defaults remain identity-safe; diagnostics are observational; parity mode affects tests/CI behavior only; **identity tests stay green** because `turn.jsonl` is untouched.


### Added (PR97 — M12-01: Native T1 scaffold; config + stubs; identity-safe)
- **Config:** Validator now accepts `perf.native.t1` with keys `{enabled, backend, strict_parity}`. Unknown‑key detection added for `perf.native` and `perf.native.t1`. `backend` is normalized to `{"rust","python"}` (defaults to `rust` when the block is present). The `native` subtree only appears in the normalized config **when provided**; defaults remain unchanged.
- **Scaffolds:** `clematis/native/__init__.py` (docstring‑only) and `clematis/native/t1.py` (PR97 stub). `available() -> False`; `propagate_one_graph_rs(...)` raises a clear stub error. **No runtime wiring** yet — the engine never calls the stub in PR97.
- **Docs:** `docs/m12/native_t1.md` — scope, invariants, gates, API/contract, parity modes, build/troubleshooting. Marked **PR97 scaffold**; enablement occurs in later PRs.

#### Invariants
- Native is **OFF by default**; disabled‑path identity stays **byte‑identical** to `0.9.0a2`.
- No imports or branches were added to T1; behavior and logs remain unchanged.

- **PR98:** (shipped) Python FFI & strict‑parity harness — see Unreleased PR98 above.
- **PR99:** Rust kernel (perf‑OFF semantics) + parity tests.
- **PR100+**: Wheels/CI matrix, diagnostics/hardening, release.


### Added (PR80 — M10-04: Reflection write path; deterministic)
- **Writer module:** `clematis/engine/orchestrator/reflection.py` with `write_reflection_entries(ctx, state, cfg_root, result) -> WriteReport`.
- **Determinism:** fixed `ts_iso` from `ctx.now_iso`; stable episode IDs `refl-<turn>-<agent>-<slot>-<sha12>`.
- **Schema clamp:** `{id, owner, ts, kind, tags, text[, vec_full]}`; double-enforces `ops_reflection` cap.
- **Fail‑soft:** index missing/add‑error logs `memory_write_error` and never crash the turn.
- **Hook:** orchestrator sets `ctx.now_iso` once and stashes `ctx._reflection_write_report` after writes.
- **Tests:** `tests/reflection/test_reflection_write_path.py` — determinism (IDs/ts), ops‑cap reporting, error resilience.

### Added (PR81 — M10-05: Log staging & ordering for reflection)
- **Staging:** `engine/util/io_logging.py` adds `STAGE_ORD["t3_reflection.jsonl"] = 10` without renumbering prior ordinals.
- **Identity:** `_IDENTITY_LOGS` unchanged (reflection logs are **not** part of identity).
- **Test:** `tests/reflection/test_reflection_stage_ord.py` asserts ordinal and identity exclusion.

### Added (PR82 — M10-06: Disabled‑path identity for reflection)
- **Test:** `tests/reflection/test_reflection_disabled_identity.py` runs a demo twice with reflection OFF; byte‑compares `t1.jsonl`, `t2.jsonl`, `t4.jsonl`, `apply.jsonl`, `turn.jsonl`; asserts **absence** of `t3_reflection.jsonl`.
- **Invariant:** M9 identity remains intact when `t3.allow_reflection=false`.

### Added (PR83 — M10-07: Reflection determinism, budget, ops cap)
- **Determinism:** identical inputs ⇒ byte‑identical `ReflectionResult`, memory rows (id/ts/text/tags/vec), and `WriteReport`.
- **Budgets:** wall‑clock timeout (`time_ms_reflection`) sets `metrics.reason="reflection_timeout"`; produces **no writes**.
- **Ops cap:** writer enforces `ops_reflection` cap even if the backend returns more entries.
- **Tests:** `tests/reflection/test_reflection_determinism.py`, `..._budget_timeout.py`, `..._ops_cap.py`.

### Added (PR84 — M10-08: LLM backend (fixtures‑only); default OFF)
- **Backend switch:** `t3/reflect.py` keeps rule‑based path unchanged and adds an `llm` path via `FixtureLLMAdapter`.
- **Canonical prompt:** JSON with sorted keys and whitespace‑collapsed fields; `fixture_key = sha256(prompt_json)[:12]`.
- **Determinism:** `generate(..., temperature=0.0)`; output clamped via existing token truncator; deterministic embeddings when `embed=true`.
- **Validation:** enabling `backend=llm` **requires** fixtures; validator rejects empty/missing `t3.llm.fixtures.path`.
- **Exports:** `FixtureMissingError` re‑exported from `t3/__init__.py`; orchestrator wraps failures as `metrics.reason="reflect_error:<Type>"`.
- **Tests:** `tests/reflection/test_reflection_llm_*.py` — fixture determinism and missing‑fixture error.


### Added (PR85 — M10-09: Policy glue — planner flag & schema wiring)
- **Schema:** `engine/policy/json_schemas.py` allows top‑level `reflection: boolean` (default `false`), `additionalProperties=false` preserved.
- **Sanitizer:** `engine/policy/sanitize.py` coerces boolean‑likes (`"true"/"false"`, `1/0`); `sanitize_plan(...)` validates `ops` and returns normalized `{"reflection": bool}` for tests.
- **Planner prompt:** `t3/policy.py` requires `reflection` in the JSON and **stashes** the LLM planner’s flag on `state._planner_reflection_flag` (keeps return shape stable).
- **Orchestrator:** `core._run_reflection_if_enabled(...)` honors `Plan.reflection` **or** `state._planner_reflection_flag` for gate/log/dry‑run.
- **Tests:** `tests/policy/test_reflection_flag.py` and `tests/reflection/test_reflection_gate_plumbing.py` cover coercion and gate plumbing.

### Added (PR86 — M10-10: Reflection telemetry & trace; identity-safe)
- **New stream:** `t3_reflection.jsonl` with stable schema
  `{turn, agent, summary_len, ops_written, embed, backend, ms, reason[, fixture_key]}`.
  Ordering is staged with `STAGE_ORD["t3_reflection.jsonl"]=10`; **not** part of identity logs.
- **Normalization (CI-only):** `engine/util/io_logging.normalize_for_identity(...)` zeroes `"ms"` for this stream when `CI=true`. No other fields are touched; identity logs remain unchanged.
- **Orchestrator helper:** `engine/orchestrator/logging.log_t3_reflection(...)`
  Emits exactly one record per reflected turn; fail-soft. Picks `ops_written` from the write report when present, falls back to entry count. Propagates `backend`, `embed`, `reason`, and (LLM fixtures) `fixture_key`. `summary_len` is computed as whitespace-token count.
- **Core wiring:** `engine/orchestrator/core.py` calls the helper after the deterministic write path (PR80), only when the reflection gate is open. Reflection never alters T1/T2/T4/Apply outputs.
- **Writer:** `engine/orchestrator/logging.append_jsonl(...)` now applies identity normalization before delegating to the unbuffered writer. **On-disk serialization remains unchanged** for identity logs; we did **not** change `io/log.py` formatting to preserve Gate A goldens.
- **Utility:** Added `stable_json_dumps(obj)` (sorted keys, compact) in `engine/util/io_logging.py`. Used for deterministic CLI JSON in `clematis/cli/_io.py` (polish; unrelated to identity logs).

#### Tests
- `tests/reflection/test_reflection_logging_shape.py` — rule-based path: CI normalizes `ms` to `0.0`; schema stable; deterministic payload across runs.
- `tests/reflection/test_reflection_logging_llm_fixture_extra.py` — LLM fixtures path includes `fixture_key`; deterministic under CI normalization.
- `tests/reflection/test_reflection_logging_timeout_reason.py` — `reason="reflection_timeout"` propagated; CI normalizes `ms`.

#### Notes / invariants
- Default OFF path unchanged; `t3_reflection.jsonl` is written **only** when reflection runs.
- Reflection logs are **not** in `_IDENTITY_LOGS`; Gate A identity remains byte-for-byte.
- CI normalization targets **only** this stream; identity streams are unaffected.

#### Heads‑up (compat shim)
- A thin compatibility shim still exists at `clematis/engine/util/io_logging.py::write_jsonl(...)`. It is **not** used by production call sites (which call `engine/orchestrator/logging.append_jsonl`).
  If you want to remove it later, search for usages with:
  ```
  grep -R "io_logging.write_jsonl" -n
  ```

  and delete the function/export once no references remain.

### Added (PR87 — M10-11: Reflection microbench & examples; deterministic, identity‑safe)
- **Script:** `clematis/scripts/bench_reflection.py` — runs the reflection unit in isolation and prints one stable JSON object (sorted keys, compact separators). When `CI=true`, the microbench normalizes `"ms": 0.0` to eliminate flakes.
- **Examples:** `examples/reflection/enabled.yaml` (rule‑based, deterministic) and `examples/reflection/llm_fixture.yaml` (fixtures‑only LLM backend; no network).
- **Fixtures:** `tests/fixtures/reflection_llm.jsonl` — SHA256‑based `prompt_hash → completion` mapping; deterministic; used only by the fixtures path.
- **Tests:** `tests/test_bench_reflection.py` — zero‑flake: forces CI‑style timing via `monkeypatch`; asserts stable fields; optional fixtures test runs only when the fixture file is present.
- **Invariants:** Does **not** write identity logs; default path unchanged when reflection is disabled; embeddings remain deterministic when `embed=true`.

### CI (PR88 — M10-12: Optional Reflection Smoke; non‑required)
- **Workflow:** `.github/workflows/reflection_smoke.yml` — **opt‑in** CI that validates the microbench paths without touching identity logs.
  - Trigger via **workflow_dispatch** (set `run=true`) or temporarily set `RUN_REFLECTION_SMOKE: "true"` in the workflow `env:` while iterating.
  - Uses a small `gate` job to compute an `enabled` output (robust to context access rules); `smoke` runs only when enabled.
- **What it runs:**
  - Microbench (rule‑based) and microbench (LLM fixtures) — each validated with tiny Python assertions.
  - `pytest -q tests/test_bench_reflection.py` only.
  - Uploads `bench_rule.json` and `bench_llm.json` as artifacts.
- **Determinism:** Sets `CI=true` and `CLEMATIS_NETWORK_BAN=1`; no network; `"ms"` normalized to `0.0`; identity logs untouched.

#### Updates (polish)
- **Action summary:** bench JSON is appended to the GitHub job summary via `$GITHUB_STEP_SUMMARY` for quick inspection (no artifact download required).
- **Fixtures guard:** the LLM fixtures step is conditional on `hashFiles('tests/fixtures/reflection_llm.jsonl')`; when absent, a friendly skipped step is shown and summarized.

### Docs
- **README:** M10 status updated; added “Reflection sessions (deterministic, gated)” quick‑start (rule‑based & fixtures‑only).
- **docs/m10/reflection.md:** gate conditions, LLM fixtures example, `ReflectionResult`/metrics schema, fixture key notes, troubleshooting.
- **README (PR87/PR88):** adds a “Reflection Microbench (PR87)” quick‑start (commands for rule‑based & fixtures) and links to the optional “Reflection Smoke (optional)” workflow.
- **docs/m10/reflection.md (PR87/PR88):** adds §12 “Microbench (PR87)” and §13 “Optional CI Smoke (PR88)”, and corrects the fixtures path to `tests/fixtures/reflection_llm.jsonl`.


### Docs (PR89 — M10-13: Reflection docs finalize & cross-links)
- **docs/m10/reflection.md:** finalized with (a) explicit dry‑run gate in the conditions, (b) §3.1 **Configuration reference** table (keys, types, defaults), (c) telemetry schema now documents optional `fixture_key` for the LLM fixtures path, (d) **fixture key helper** snippet to recompute the SHA‑256 prompt hash, (e) enablement requires the **planner** to set `reflection: true`, plus links to `examples/reflection/{enabled.yaml,llm_fixture.yaml}`.
- **README:** tightened M10 status line; corrected CI logging path to `engine/orchestrator/logging.append_jsonl`; clarified that the orchestrator honors either the plan flag or the stashed policy flag; added “no dry‑run” as a gate condition.
- **docs/m9/overview.md:** short banner pointing forward to M10 dossier.

### Changed
- **.gitignore:** deduplicated entries; clarified JSONL ignore patterns for local logs while keeping identity goldens intact.


## [0.9.0a2] - 2025-10-04

### M11 — Field‑Control GEL (HS1): contracts + safe plumbing (no planner)

#### Added (PR91 — M11-01: Docs overview)
- **Docs:** `docs/m11/overview.md` — what GEL is in v3 (contracts + optional ops), what it is not (no nudge planner), call sites & gates, algorithm summaries (update/decay/merge/split/promotion), determinism, defaults.
- **README:** short pointer under Architecture/Examples (GEL substrate).

#### Added (PR92 — M11-02: Disabled‑path identity for graph.*)
- **Helpers:** `tests/helpers/identity.py` — `_strip_perf_and_quality(...)`, `_strip_perf_and_quality_and_graph(...)`; shared normalizers consolidated here.
- **Engine test hook:** `clematis/engine/orchestrator/core.py` adds `run_smoke_turn(cfg, log_dir=None, input_text="")` (stable smoke entrypoint; no behavior change).
- **Tests:** `tests/test_identity_disabled_path.py` — config round‑trip equality with `graph.enabled=false` (inert subtree); runtime smoke asserts **no `gel.jsonl`** and **byte‑identical logs** vs baseline.

#### Added (PR93 — M11-03: Runtime identity + snapshots)
- **Tests:** `tests/test_gel_disabled_path_runtime.py` — repeats smoke run; asserts log identity and **snapshot parity** (same files; identical content hashes when present).

#### Added (PR94 — M11-04: Examples + doc polish)
- **Examples:** `examples/gel/enabled.yaml` (observe+decay only; ops OFF) and `examples/gel/disabled.yaml` (full subtree, inert with `enabled:false`).
- **Script:** `scripts/examples_smoke.py` — includes the two GEL examples in the default set; help text mentions HS1/GEL.
- **Docs:** `docs/m11/overview.md` — Examples section and status updated; **README.md** adds a “GEL (HS1) examples” block with run commands.

#### Packaging / CI (PR94)
- **Packaging:** `MANIFEST.in` allow‑lists `examples/gel/*.yaml` into sdists; `pyproject.toml` adds `data-files` entry (`share/examples/clematis/gel`) so wheels ship the examples; registers `pytest` mark `identity`.
- **CI:** `.github/workflows/ci.yml` runs the GEL example smoke (offline) and verifies examples are present in both sdist and wheel (robust path patterns).

#### Docs (PR95 — M11-05: CLI inspector note)
- **Docs:** `docs/m11/overview.md` — “Inspecting GEL state (CLI)” shows `clematis cli inspect-snapshot` usage and `jq` one‑liners for `state.graph.{meta,nodes,edges}`.

#### Milestone (PR96 — M11-06: Close)
- **Footer:** `docs/m11/overview.md` adds a milestone note: **v3 HS1 (GEL substrate) complete; nudge planner deferred to v4**.

#### Invariants
- Defaults keep `graph.enabled=false`; disabled‑path identity stays **byte‑identical**; **no GEL files** when disabled.
- Hybrid reranker remains OFF by default (shadow‑only hook present).
- CI green across suites; no runtime behavior changes outside explicitly enabled examples.

## [0.9.0a1] - 2025-10-03

### Added (PR77 — M10-01: Reflection gate & budgets; defaults OFF)
- **Config surface (t3):** Introduce `t3.allow_reflection: false` and `t3.reflection.{backend,summary_tokens,embed,log,topk_snippets}`. Defaults keep identity path unchanged; `backend: "rulebased"` is deterministic. `"llm"` is reserved for a fixtures-only path to land in PR84.
- **Budgets (scheduler):** Add `scheduler.budgets.ops_reflection: 5` and `scheduler.budgets.time_ms_reflection: 6000` with strict non‑negative bounds (time budget ≥1). Coercion follows existing budget semantics.
- **Validator:** Unknown‑key rejection for `t3.reflection.*`; strict type/bounds; backend enum restricted to `{"rulebased","llm"}`. Normalizes booleans/ints and writes back a canonical `t3.reflection` block.
- **Examples/Defaults:** `configs/config.yaml` now surfaces the `t3.reflection` block (gate remains OFF) and comments the two new scheduler budget knobs in the example.
- **Tests:** `tests/config/test_validate_reflection.py` covers defaults injection, typing/bounds, backend enum, unknown‑keys, and budget coercion vs negatives.
- **Invariants:** **Disabled‑path identity preserved.** No runtime behavior change until reflection is enabled in later PRs.


### Refactor (PR76 — M7/PR37–38: T2 extraction & file moves; **no shim re-exports**)
- **Lean core:** `t2/core.py` no longer embeds quality/fusion/MMR or trace assembly. Core now calls:
  - `clematis.engine.stages.t2.quality.apply_quality(...)`
  - `clematis.engine.stages.t2.metrics.finalize(...)`
- **New modules:**
  - `clematis/engine/stages/t2/quality.py` — orchestrates GEL hybrid → fusion → MMR; owns **shadow trace** (PR36) under the triple gate; never mutates rankings in shadow mode.
  - `clematis/engine/stages/t2/state.py` — `_init_index_from_cfg`, `gather_changed_labels`, `build_label_map`.
  - `clematis/engine/stages/t2/metrics.py` — `assemble_metrics(...)`, `finalize(...)` (pure assembly; side‑effect free).
- **Behavioral invariants:** identity path unchanged; shadow traces still require `perf.enabled && perf.metrics.report_memory && t2.quality.shadow && !t2.quality.enabled`.
- **File relocations into `t2/` package (update imports):**
  - `engine/stages/t2_lance_reader.py`  → `engine/stages/t2/lance_reader.py`
  - `engine/stages/t2_quality_mmr.py`   → `engine/stages/t2/quality_mmr.py`
  - `engine/stages/t2_quality_norm.py`  → `engine/stages/t2/quality_norm.py`
  - `engine/stages/t2_quality_trace.py` → `engine/stages/t2/quality_trace.py`
  - `engine/stages/t2_quality.py`       → `engine/stages/t2/quality_ops.py`
  - `engine/stages/t2_shard.py`         → `engine/stages/t2/shard.py`
- **Breaking import change:** no compatibility shims were added. If you import these modules directly, update paths to the new locations (e.g., `from clematis.engine.stages.t2.quality_mmr import ...`).
- **Tests/docs:** updated to reflect new module layout and gating; all suites green.
- **T3 Phase A/B:** Introduced façade modules under `clematis.engine.stages.t3` (with `legacy.py` retaining the monolithic entry points) and delegated `make_plan_bundle(...)` through the bundle façade. Behavior remains unchanged; identity path preserved while the package split lands.
- **Bundle helpers:** `_iso_now`, config/agent readers, and the T1/T2 extractors now live in `t3/bundle.py`, with `legacy.py` forwarding to the shared helpers.
- **Trace wiring:** `clematis.engine.stages.t3.trace.emit_trace(...)` is now invoked from the orchestrator, honoring the same triple-gate as T2 and writing into `state.logs` for downstream consumers.
- **Scaffolds:** Added `t3.policy.select_policy`, `t3.policy.run_policy`, `t3.metrics.finalize`, and helpers for future policy/metrics extraction (not yet used by orchestrator).


### Added (PR74 — M9-12: Bench kits (T1/T2) + docs)
- **Scripts:** `clematis/scripts/bench_t1.py` and `clematis/scripts/bench_t2.py`.
  - Stable one-line JSON via `sort_keys=True`, fixed separators, `ensure_ascii=True`.
  - Deterministic seeds and warmup **outside** the timed region.
  - `bench_t1`: single-threaded by design; accumulates metrics across `--iters` (uses `max` for `max_delta`). `--parallel/--workers` are advisory (observability only).
  - `bench_t2`: builds a deterministic synthetic corpus; supports `--backend {inmemory,lancedb}` and `--parallel` shard fan-out with deterministic merge (by `(-score, id)`); **import-robust** fallback to an internal in-memory shard adapter when extras are missing.
  - Both set `CLEMATIS_NETWORK_BAN=1` defensively.
- **Docs:** `docs/m9/benchmarks.md` — quick start, flags, output shapes, caveats; reproducibility guidance.
- **README:** microbenches section with module-form invocations and a pointer to the docs.

### CI (PR74)
- No new workflows. These benches **do not gate merges**.
- Optional: may be triggered manually in CI via the existing PR73 smoke path (`RUN_PERF=1`), but keep non-required.

### Invariants (PR74)
- No runtime behavior change; **disabled-path identity remains unchanged**.
- Metrics gating rules unchanged; benches backfill convenience fields in their own JSON only.
- No network or external I/O introduced; LanceDB path remains optional and guarded.

### Added (PR73 — M9-11: Parallel smoke in CI; opt-in, non-required)
- **Workflow:** `.github/workflows/parallel-smoke.yml` — opt-in via `RUN_PERF=1` (or `workflow_dispatch` input); defaults **OFF** (`RUN_PERF=0`). Runs a tiny end-to-end execution with `perf.parallel.enabled=true` (agents/T1/T2 ON, `max_workers=2`) and prints a success line. **No perf thresholds.**
- **Example config:** `examples/perf/parallel_on.yaml` — enables parallel gates with quiet metrics (no extra emissions).
- **Docs:** README and `docs/m9/overview.md` note that the job is non-required and how to run it locally / in CI.

### CI (PR73)
- New, **non-required** job `parallel-smoke` (Linux, Python 3.12). Skips cleanly when `RUN_PERF=0`; contributors can opt-in per-run without affecting branch protection.

### Invariants (PR73)
- No behavior or threshold changes to required CI jobs; runtime behavior and disabled-path identity remain unchanged.

### Added (PR72 — M9-10: Identity & race test suite)
- **Identity (seq vs par):** Proves byte-for-byte equality for `t1.jsonl`, `t2.jsonl`, `t4.jsonl`, `apply.jsonl`, `turn.jsonl`, and `scheduler.jsonl` between sequential (disabled path) and parallel (enabled path; `max_workers=2`) runs.
- **Race regression (contention):** Deterministically perturbs pool scheduling (reverses job order) while keeping inputs fixed; artifacts remain identical.
- **Back-pressure path:** Forces a tiny stager `byte_limit` to trigger drain→ordered-flush→retry; artifacts remain identical to baseline.
- **Tests:** `tests/integration/test_identity_parallel.py`, `tests/integration/test_race_regression.py`.
- **Helpers:** `tests/helpers/identity.py`, `tests/helpers/world.py`, `tests/helpers/configs.py`, `tests/helpers/memory.py`.

### CI (PR72)
- New workflow: `.github/workflows/identity_parallel.yml` — matrix over Python **3.11–3.13** and modes `{off,on}`; sets deterministic env (`SOURCE_DATE_EPOCH=0`, `PYTHONHASHSEED=0`, `CLEMATIS_NETWORK_BAN=1`) and runs the identity + race/back-pressure tests.

### Invariants (PR72)
- **Disabled path:** outputs/logs remain **byte-identical** to prior baseline.
- **Enabled path:** with fixed inputs and flags, artifacts are **byte-identical** regardless of internal scheduling or stager back-pressure.

### Added (PR71 — M9-09: Log staging & ordered writes; flag-gated, defaults OFF)
- **Central staging layer:** `engine/util/io_logging.py` buffers log records with a stable composite key `(turn_id, stage_ord, slice_idx, seq)`; canonical stream order via `stage_ord`: `t1.jsonl → t2.jsonl → t3_plan.jsonl → t3_dialogue.jsonl → t4.jsonl → apply.jsonl → health.jsonl → turn.jsonl → scheduler.jsonl`.
- **Orchestrator refactor:** split orchestration into `core.py` (sequential loop), `logging.py` (mux/stager + override-aware writers), and `parallel.py` (agent fan-out helpers `_run_turn_compute`, `_run_agents_parallel_batch`).
- **Unbuffered writer:** `io/log._append_jsonl_unbuffered(...)` mirrors `append_jsonl(...)` serialization to preserve byte parity while bypassing LogMux during staged flush.
- **Tests:** `tests/engine/test_io_logging_staging.py` (order & back-pressure), `tests/integration/test_agents_parallel_driver.py` (A→B ordering for `t1.jsonl`, `t4.jsonl`, `apply.jsonl` with fixed `turn`), `tests/test_golden_path.py` (byte-for-byte equality of `turn.jsonl` & `scheduler.jsonl` vs sequential baseline).
- **Docs:** `docs/m9/overview.md` adds a PR71 section (design, gate, invariants, troubleshooting).

### Invariants (PR71)
- **Disabled path:** with gate OFF or `max_workers<=1`, behavior/logs are **byte-identical** to PR70.
- **Enabled (disjoint agents):** with agent parallelism ON, final **content and line order** on disk are identical to sequential (ordered flush by composite key).

### Added (PR70 — M9-08: Agent-level parallel driver; flag‑gated, defaults OFF)
- **Orchestrator/Driver:** Optional agent-level **compute→commit** path. Compute runs T1→T4 per agent on a **read‑only snapshot** (no Apply); stage logs are **captured** via the ctx‑aware logger (`io.log.append_jsonl` + `engine/util/logmux.py`). Commit phase sorts buffers by `(turn_id, slice_idx)` and then **flushes** the buffered logs and **applies** deltas sequentially; outputs/logs remain deterministic.
- **Gate:** Activates only when `perf.parallel.enabled=true`, `perf.parallel.agents=true`, and `perf.parallel.max_workers>1`.
- **Independence:** Batches are selected greedily with **pairwise disjoint** graph sets; overlapping agents fall back to sequential (identity preserved).
- **State snapshot:** `engine/stages/state_clone.py` provides shallow, **read‑only** snapshots (frozen containers; identity preserved for heavy objects).
- **Logging:** `io/log.append_jsonl` is now **ctx‑aware**; when a `LogMux` is active, lines are buffered and flushed later in deterministic order; otherwise write‑through behavior is unchanged.
- **Helpers:** Orchestrator helpers `_select_independent_batch`, `_run_turn_compute`, `_run_agents_parallel_batch`; centralized `engine/util/logmux.py` (buffer/flush; `ContextVar`).
- **Tests:** `tests/test_config_validate_parallel_agents.py` (validator warnings), `tests/engine/test_logmux_capture.py` (buffer/flush), `tests/integration/test_agents_parallel_driver.py` (gate‑off identity; disjoint/overlap behaviors).
- **Docs:** `docs/m9/overview.md` adds a PR70 section; README updated with status and enablement snippet.

### Invariants (PR70)
- **Disabled path:** with gate OFF or `max_workers<=1`, behavior/logs are **byte‑identical** to PR69 baseline.
- **Enabled (eligible batches):** with disjoint agents, outputs/logs are **identical to sequential**; logs emitted in `(turn_id, slice_idx)` order.
- **Scheduler fairness/policy:** unchanged.


### Added (PR69 — M9-07: LanceDB parallel T2 + deterministic T2 microbench; flag‑gated, defaults OFF)
- **Stage (T2):** Gate now allows `inmemory` **and** `lancedb` backends when `perf.parallel.enabled=true`, `perf.parallel.t2=true`, and `perf.parallel.max_workers>1`, and when the index exposes `_iter_shards_for_t2(...)` yielding >1 shards. Parallel fan‑out uses the existing deterministic merge (tier‑ordered; sort by score desc (quantized) then `id` asc; de‑dupe; global K‑clamp). Disabled or single‑shard paths remain sequential.
- **Lance index:** `LanceIndex._iter_shards_for_t2(tier, suggested)` deterministically enumerates logical shards — prefer per‑quarter shards when multiple quarters exist; otherwise 2–4 stable hash buckets by `id`. New `_LancePartition` provides a read‑only shard view that mirrors `search_tiered(...)` scoring and tie‑breaks.
- **Metrics (gated):** When `perf.enabled && perf.metrics.report_memory`, T2 now emits `t2.task_count` and `t2.parallel_workers` (both backends) and **Lance‑only** `t2.partition_count`. No metrics are emitted when the gate is off.
- **Bench:** `clematis/scripts/bench_t2.py` — deterministic T2 microbench with `--iters/--workers/--backend {inmemory,lancedb}/--parallel/--json`. JSON always includes `queries, shards, workers, effective_workers, backend, parallel, elapsed_ms` and backfilled `t2_task_count, t2_parallel_workers, t2_partition_count` even when the runtime metrics gate is off.
- **Tests:** Added `tests/t2/test_t2_lance_gate_import.py` (no‑extras import hygiene + gate truth table) and `tests/perf/test_bench_t2_smoke.py` (bench smoke & CLI JSON). Updated `tests/t2/test_t2_parallel_merge.py` to reflect PR69 gate semantics for `lancedb`.
- **Docs:** `docs/m9/overview.md` notes PR69 enablement/invariants; README adds bench usage and gating snippet.

### Invariants (PR69)
- **Disabled/≤1‑shard path:** outputs and logs are **byte‑identical** to PR68 baseline (identity preserved).
- **Enabled path (inmemory/Lance):** result **set and order** identical to sequential under fixed seed/time (tier‑ordered; score desc then `id` asc; de‑dupe; global K‑clamp).


### Changed
- Prerelease 0.8.0a5 (python_requires <3.14; Homebrew formula now on python@3.13).
- **Engine (orchestrator):** `run_turn` supports a **dry‑run** mode (`_dry_run_until_t4`) used by the PR70 compute phase; GEL observe is skipped during dry‑run; default sequential path is unchanged.
- **IO (logs):** `clematis/io/log.append_jsonl` is **context‑aware**; when a `LogMux` is active it buffers lines for deterministic flush; otherwise it writes through exactly as before.


### Added (PR63 — M9: deterministic parallelism config schema; defaults OFF)
- **Configs:** Introduce `perf.parallel` in `configs/config.yaml` (identity path by default):
  ```yaml
  perf:
    parallel:
      enabled: false
      max_workers: 0    # 0/1 ⇒ sequential; >1 reserved for future PRs
      t1: false         # stage toggles; schema only in PR63
      t2: false
      agents: false
  ```
- **Validator:** `configs/validate.py` accepts the new block and enforces:
  - unknown-key checks for `perf.parallel.*` (helpful suggestion when near‑miss),
  - coercions/bounds (`enabled: bool`, `max_workers: int≥0`; negatives normalized to `0`),
  - stage toggles as strict booleans. No runtime wiring yet.
- **Types:** `clematis/engine/types.py` gains an optional `PerfParallelCfg` surface (used by the validator/tests; not consulted by runtime).
- **Tests:**
  - `tests/test_config_perf_defaults.py` — asserts new defaults are present and identity‑safe (OFF/seq),
  - `tests/test_config_validate_parallel.py` — strict acceptance, bounds/normalization, and unknown‑key rejection for `perf.parallel`.
- **Docs:** `docs/m9/overview.md` summarized the schema and identity rules; README adds an M9 stub that points to the doc.
- **Identity & CI:** defaults keep outputs **byte‑for‑byte** unchanged (goldens unaffected); no new metrics; piggybacks existing config tests.

### Added (PR64 — M9: deterministic parallel helper; tests‑only, not wired)
- **Engine util:** `clematis/engine/util/parallel.py` — `run_parallel(tasks, *, max_workers, merge_fn, order_key)` enforces deterministic aggregation regardless of completion order. `max_workers<=1` executes sequentially; exceptions aggregate deterministically via `ParallelError` with ordered `errors`.
- **Tests:** `tests/helpers/test_parallel_helper.py` — ordering independent of completion, sequential equivalence, deterministic exceptions, empty/normalization cases.
- **Docs:** `docs/m9/parallel_helper.md` (API, error model, examples, anti‑patterns, wiring guidance); README M9 section links and a small key/thunk example; `docs/m9/overview.md` references PR64 and outlines the hand‑off.
- **Identity:** helper is **unused by default** (tests only); no runtime behavior change.

### Added (PR65 — M9: Cache safety wrappers + deterministic merge; defaults OFF)
- **Cache API:** `clematis/engine/cache.py` adds:
  - `CacheProtocol` (minimal `get/put/__contains__/items`),
  - `ThreadSafeCache` (lock wrapper for count/TTL caches),
  - `ThreadSafeBytesCache` (lock wrapper for size‑aware caches with `put(key, value, cost)`), and
  - `merge_caches_deterministic(...)` (deterministic worker/key‑sorted merge with explicit conflict policy: `"first_wins"` or `"assert_equal"`).
- **LRU bytes conformance:** `clematis/engine/util/lru_bytes.py` exposes a deterministic `items()` iterator (LRU→MRU) without mutating recency; no behavior change.
- **Stages (safety only):** `clematis/engine/stages/t1.py` and `.../t2.py` now wrap legacy caches with `ThreadSafeCache` / `ThreadSafeBytesCache` to guard optional parallel paths. **No behavior change** when parallel is OFF or when `max_workers<=1`.
- **Docs:** `docs/m9/cache_safety.md` — modes (`lock` vs `isolate+merge`), determinism rules, wiring patterns; `docs/m9/overview.md` updated with a PR65 summary and links.
- **Tests:**
  - `tests/test_cache_guardrails.py` — concurrency smoke for both wrappers; stable `items()` snapshot,
  - `tests/test_cache_coherency.py` — cache disable path, deterministic merge policies (first‑wins/assert‑equal) parity with sequential shared cache.
- **Identity:** outputs remain **byte‑identical** with parallel OFF; with parallel ON + `max_workers=1`, results match sequential.

### Added (PR66 — M9: Parallel T1 across graphs; flag‑gated, defaults OFF)
- **Stage wiring:** `clematis/engine/stages/t1.py` fans out per **active graph** via `run_parallel`. Uses a stable key `(submit_index, graph_id)` with `order_key=lambda k: (k[0], k[1])` to preserve the original iteration order.
- **Deterministic aggregation:**
  - Deltas are concatenated strictly in graph order.
  - Logs are **buffered per graph** in tasks and flushed in merge order (no interleaving).
  - Metrics are reduced deterministically (sums/max) and now include `radius_cap_hits` and `node_budget_hits` (with zeros on early paths/cache hits for back‑compat).
- **Cache safety:** Reuses PR65 lock‑wrapped caches; no isolate+merge in T1 yet (can be added later if contention warrants).
- **Tests:** `tests/stages/test_t1_parallel_off_on.py` verifies gate semantics (defaults OFF) and functional equivalence between sequential (`max_workers=1`) and parallel (`max_workers>1`) runs on a tiny in‑memory multi‑graph fixture; cache is stubbed for clean equivalence.
- **Identity:** With parallel **OFF** and when `max_workers<=1`, outputs/logs are byte‑identical to sequential. With parallel **ON** (`t1=true`, `enabled=true`, `max_workers>1`), final outputs/logs match sequential. No CI perf thresholds.
- **Docs:** `docs/m9/overview.md` mentions PR66 under the hand‑off; README unchanged.


### Added (PR68 — M9: Parallel T2 across in‑memory shards; flag‑gated, defaults OFF)
- **Stage wiring:** `clematis/engine/stages/t2.py` can now fan‑out semantic retrieval across **in‑memory** shards via `run_parallel`. Uses the submit index as the deterministic key; falls back to sequential when the gate is OFF.
- **Deterministic merge semantics:**
  - **Tier‑ordered walk**: finish `exact_semantic` before `cluster_semantic` (then `archive`).
  - Within a tier, **stable sort** by score **descending** (quantized for portability) then `id` **ascending**.
  - **De‑duplicate** by `id` across shards; keep the best by the same key.
  - **Clamp after merge**: apply `t2_k` once globally. Results (set **and** order) match sequential under fixed seed/time.
- **Backend affordance:** `clematis/memory/inmemory.py` adds a private `_iter_shards_for_t2(tier, suggested)` yielding lightweight shard views; when `suggested<=1` or there’s ≤1 episode, yields `self` only (sequential behavior).
- **Helpers:** `clematis/engine/stages/t2_shard.py` provides `RawHit`, `_qscore`, and `merge_tier_hits_across_shards_dict(...)`; imported by `t2.py` so determinism comes from a single place.
- **Metrics (gated):** When `perf.enabled=true` **and** `perf.metrics.report_memory=true` and the parallel T2 gate is ON, T2 emits:
  - `t2.task_count` — number of shard tasks,
  - `t2.parallel_workers` — `min(max_workers, task_count)`.
- **Tests:** `tests/t2/test_t2_parallel_merge.py` — gate semantics; tie‑break on equal scores; K‑clamp after tier‑walk; equivalence to a sequential emulation; normalization of episode objects.
- **Docs:** `docs/m9/overview.md` documents enablement and invariants; `README.md` notes PR68 and shows the config snippet. Default remains **OFF**.
- **Identity & rollback:** With parallel **OFF** (or `max_workers<=1`), outputs/logs remain **byte‑identical**. With parallel **ON** (`t2=true`, `enabled=true`, `max_workers>1`, in‑memory backend), results equal sequential. Rollback by turning `perf.parallel.t2=false` or disabling `perf.parallel.enabled`.

### Added (PR67 — M9: T1 parallel metrics + microbench; gated)
- **Stage (T1):** Emits minimal observability when gates are satisfied:
  - `parallel_workers` — effective workers used (`min(max_workers, task_count)`).
  - `task_count` — number of per‑graph tasks (`len(active_graphs)`).
  - Emitted only when `perf.enabled=true` **and** `perf.metrics.report_memory=true` **and** the T1 parallel gate is ON (`perf.parallel.enabled && t1 && max_workers>1`).
- **Bench:** `clematis/scripts/bench_t1.py` (deterministic, local‑only; `--json`) prints elapsed time and mirrors the two fields at top level. For convenience, the bench **backfills** these fields from its own task/worker counts when the metrics gate is on, so they always show up in JSON runs.
- **Tests:** `tests/test_perf_guardrails.py` adds fast gate‑only checks (always run) and a bench JSON smoke (opt‑in via `RUN_PERF=1`).
- **Identity:** No behavior change; metrics are observational only and fully gated. No CI perf thresholds.

### Added (PR55 — CLI output contract & exit codes)
- **Std flags across wrappers:** `--json` / `--table` (mutually exclusive) and `--quiet` / `--verbose` parsed by all CLI wrappers. Defaults unchanged when flags are omitted.
- **Helpers:** `clematis/cli/_io.py` (plain ASCII `print_table`, compact `print_json`, `set_verbosity`) and `clematis/cli/_exit.py` (standard exit codes).
- **Quiet contract:** Wrapper diagnostics respect `--quiet` (stderr suppressed) and `--verbose` (extra breadcrumbs). Stdout remains machine‑readable; no color.
- **validate:** `--json` output preserved **byte‑identical** to previous behavior (pass‑through formatting).
- **inspect-snapshot:** supports `--json` (wrapper forces `--format json`) and `--table` (ASCII table). Non‑format default output unchanged.
- **rotate-logs:** supports `--json` and `--table` summaries (e.g., `dir`, `pattern`, `dry_run`). Non‑format default output unchanged.
- **Other wrappers:** parse the flags but reject structured formats for now (exit 1; message suppressed by `--quiet`).

### Tests (PR55)
- `tests/cli/test_output_contract.py` — JSON identity for `validate`; `inspect-snapshot` JSON/table paths; mutual‑exclusion guard.
- `tests/cli/test_exit_codes.py` — standardized exit codes (0 OK, 1 user/validation, 2 I/O/parse, 3 internal) and quiet/verbose gates.

### CI (PR55)
- **pkg_build.yml:** Linux‑only CLI contract probes in the pkg‑smoke matrix:
  - `python -m clematis validate --json` must succeed.
  - `python -m clematis inspect-snapshot -- --dir does_not_exist_dir_12345` must exit **2**.
  - `python -m clematis demo -- --steps 1 --json --table` must exit **1**.
  - `python -m clematis rotate-logs -- --dry-run --json` and `--table` must succeed.


### Docs (PR55)
- `docs/m8/packaging_cli.md` updated with: structured output quickstart; support matrix; exit‑code table; `--quiet/--verbose` semantics. Defaults and human output remain unchanged when no format flag is provided.

### Docs (PR56 — Supply chain: SBOM & provenance)
- Added **Supply chain** section in [`docs/m8/packaging_cli.md`](docs/m8/packaging_cli.md#supply-chain-sbom--provenance-m8-14): how SBOMs are generated on tag builds, how SLSA attestations are created, and how to verify them with `gh attestation` (online/offline) and `jq` sanity checks.

### Added (PR52 — Cross-platform packaging smoke)
- **CI:** Cross-platform matrix over Linux (`ubuntu-latest`), macOS (`macos-latest`, arm64), and Windows (`windows-latest`) on Python 3.11 & 3.12. Builds an axis-local wheel, installs into a fresh venv, and runs offline CLI smokes with `CLEMATIS_NETWORK_BAN=1`:
  - `python -m clematis --version`
  - `python -m clematis validate --json` (or `--` passthrough)
  - `python -m clematis demo -- --steps 1` (or direct)

  Uploads the wheel as an artifact. **No runtime behavior changes.**

### Added (PR53 — CLI hardening & CI hygiene)
- **CLI:** Added wrappers `validate` and `demo` (delegates to scripts/, behavior‑neutral). Their `-h/--help` output includes the literal "Delegates to scripts/" for deterministic testing.
- **Tests:** Expanded help‑matrix to include `validate` and `demo`; assert the literal string.
- **Docs:** Note in `docs/m8/packaging_cli.md` that CI installs NumPy solely for the demo smoke.

### Packaging (PR53 — optional extras)
- **Extras:** New optional dependency group `cli-demo = ["numpy>=1.24"]`.
- **CI usage:** Package smoke installs with extras: `pip install "$WHL[cli-demo]"` (works with wheels and file URLs).

### CI (PR53)
- **Cache:** Enable `actions/setup-python@v5` pip cache in the package smoke job to reduce latency.
- **Reproducible build:** Add `twine check` step (metadata sanity) — no upload.
- **Branch protection:** Recommend marking `pkg-smoke-matrix` as a required status check.

```
### M8 — Packaging & CLI (PR46 — pass-through wrappers & merge-order fix)

#### Added
- **CLI wrappers** under `clematis/cli/`: `rotate-logs`, `inspect-snapshot`, `bench-t4`, `seed-lance-demo`. Each delegates to `scripts/<name>.py`, accepts `REMAINDER` arguments, and strips a leading `--` when present to preserve parity with direct script invocation.
- **Entry-point**: `clematis/__main__.py` trampoline enables `python -m clematis …`.
- **Docs**: README gains a short subcommands table and example invocations for the four wrappers.

#### Fixed
- **Argparse merge-order** in `clematis/cli/main.py`: `parse_known_args` can split a `--flag value` pair across `extras` (flag) and the subparser’s `REMAINDER` (value). We now **prepend** `extras` to `ns.args` (`ns.args = extras + ns.args`) so flags remain adjacent to their values. Fixes failures like:
  - `python -m clematis rotate-logs --dir ./.logs --dry-run`
  - `python -m clematis bench-t4 --num 1 --runs 1 --json`
  - `python -m clematis seed-lance-demo --uri ./.data/lance-demo --overwrite`

#### CI
- **CLI help smokes**: lightweight offline checks to guard wrapper wiring and argument merging. Add as a step in the main test workflow or as a tiny dedicated workflow (`.github/workflows/cli-smokes.yml`) that runs:
  ```
  python -m clematis --help
  python -m clematis rotate-logs --help
  python -m clematis inspect-snapshot --help
  python -m clematis bench-t4 --help
  python -m clematis seed-lance-demo --help
  ```
  Environment: `CI=true`, `CLEMATIS_NETWORK_BAN=1`.

#### Tests
- `tests/cli/test_pr46_merge_order.py`: regression coverage for (a) preserving flag→value pairing via **extras + ns.args** merge and (b) stripping a leading `--` in wrapper pass-through. Fully offline and deterministic.
```
## [0.8.0-M3] - 2025-09-25

### M3 — T3 Policy Tier (LLM backend; opt-in, defaults OFF)

#### Added (PR8 — LLM adapter scaffolding)
- **Adapters:** `clematis/adapters/llm.py` — minimal interface with two concrete adapters:
  - `FixtureLLMAdapter`: deterministic lookup via `fixtures/llm/*.jsonl` (maps `prompt_hash → completion`).
  - `QwenLLMAdapter`: local runtime via Ollama (no streaming; simple POST to `/api/generate`).
- **Config surface (defaults OFF):**
  ```yaml
  t3:
    backend: "rulebased"               # "rulebased" | "llm"
    llm:
      provider: "fixture"              # "fixture" | "ollama"
      model: "qwen3:4b-instruct"
      endpoint: "http://localhost:11434/api/generate"
      max_tokens: 256
      temp: 0.2
      timeout_ms: 10000
      fixtures:
        enabled: true
        path: "fixtures/llm/qwen_small.jsonl"
  ```
- **Validator:** `configs/validate.py` gains `validate_config_api(...)` and strict checks for `t3.backend` and `t3.llm.*` (enums, bounds, unknown-key detection).

#### Added (PR9 — Qwen runtime wiring; opt-in, non‑CI)
- **Stage wiring:** `clematis/engine/stages/t3.py` — `_get_llm_adapter_from_cfg(...)` constructs either fixture or Ollama adapter; rule‑based path remains the default.
- **Prompt builder:** `make_planner_prompt(ctx)` produces a deterministic prompt bundle.
- **Local smoke:** `scripts/llm_smoke.py` — runs Qwen3 via Ollama and prints planner JSON. **CI never calls this.**
- **Docs:** `docs/m3/llm_adapter.md` — quickstart, config, and smoke instructions.

#### Added (PR10 — LLM smoke runner (manual) + CI skip)
- **Manual marker:** `tests/llm/test_manual_smoke_marker.py` (decorated `@pytest.mark.manual`).
- **Pytest gating:** `tests.yml` runs `pytest -q -m "not manual"`; manual smokes only run when explicitly requested locally.
- **CI guards:** `tests/llm/test_ci_guard.py` ensures CI sets `CLEMATIS_NETWORK_BAN=1` and that the LLM provider is `fixture` with `fixtures.enabled=true`. When mis‑configured, T3 returns a safe fallback and logs the reason.
- **Workflows:** standardized pytest steps to run offline and skip manual tests by default:
  - `ci.yml`, `cli_smoke.yml`, `disabled_identity.yml`, `gate_d_quality_shadow.yml`, `pr37_quality_fusion_smoke.yml`, `pr40_reader_parity_smoke.yml`, `pr41_rq_eval_smoke.yml`, `snapshots_smoke.yml`, `t2_reader_parity.yml` now set `CLEMATIS_NETWORK_BAN=1`, `CI=true`, and append `-m "not manual"`.

#### Added (PR11 — Prompt & safety hardening)
- **Schema:** `clematis/engine/policy/json_schemas.py` — `PLANNER_V1` (strict JSON: `{plan: list[str], rationale: str}`, caps: ≤16 items, ≤200 chars/item, rationale ≤2000; `additionalProperties=false`).
- **Sanitizer:** `clematis/engine/policy/sanitize.py` — `parse_and_validate(text, PLANNER_V1)` enforces:
  - JSON‑only (no prose; code fences rejected),
  - type/shape/length caps, and
  - size guards to prevent oversized outputs.
- **Enforcement:** `clematis/engine/stages/t3.py` calls `parse_and_validate(...)` on LLM output. **On failure:** returns `{"plan": [], "rationale": "fallback: invalid llm output"}` and appends a `llm_validation_failed` log entry; **no graph edits** occur.

#### Tests (PR8–PR11)
- `tests/llm/test_adapter_fixture.py`, `tests/llm/test_adapter_errors.py` — fixture determinism and failure modes.
- `tests/llm/test_config_validation.py` — strict config validation.
- `tests/llm/test_ci_guard.py` — CI offline/fixture guardrails.
- `tests/llm/test_prompt_safety.py` — sanitizer unit coverage (non‑JSON, fences, size caps, whitespace‑only items).
- `tests/llm/test_t3_validation_enforcement.py` — end‑to‑end: invalid → fallback+log; valid → pass‑through.

#### CI
- All workflows running pytest set `CLEMATIS_NETWORK_BAN=1` and use `-m "not manual"` to maintain determinism and avoid network use. Manual smokes are opt‑in locally.

#### Identity & rollback
- **Disabled‑path identity preserved (Gate A):** with `t3.backend=rulebased` (default), outputs and logs are byte‑for‑byte identical to M7.
- **Fallback safety:** any adapter or validation error yields an empty plan with a logged reason; state is never mutated by invalid completions.
## [0.7.0-M7] - 2025-09-24

### M7 roll-up summary

#### Added
- `t2_pipeline(cfg, query, emit_metric=None, ctx=None)` shim for integration tests.
- Quality trace writer: deterministic JSONL; honors `perf.metrics.trace_dir`; persists `meta.reason` (from `ctx.trace_reason` or `perf.metrics.trace_reason`).

#### Changed
- Comparator ignore narrowed to `**/logs/quality/rq_traces.jsonl`.
- MMR metrics now always emitted under the quality gate: `t2q.mmr.selected`, `t2q.mmr.lambda`, `t2q.diversity_avg_pairwise`.
- T2 cache key now includes a quality-config digest when `t2.quality.enabled=true`.
- Reader parity: one-time availability warning; `t2.reader_mode` always recorded under the triple gate.
- Examples smoke: added `--examples-glob` and `--fail-fast`; trace check precedence matches writer.
- `scripts/validate_config.py`: added `--json` output flag.

#### Deprecated
- `t2.quality.mmr.k_final` (accepted with warning; normalized to `k`).

#### Fixed
- Trace reason passthrough in both enabled and shadow traces (fallback to `perf.metrics.trace_reason`).


### Added (PR36 — Quality scaffold & shadow mode; defaults OFF)
- **Engine (T2):** `clematis/engine/stages/t2_quality_trace.py` — shadow tracing emitter that writes `rq_traces.jsonl` **without** mutating rankings or metrics. Deterministic headers: `trace_schema_version=1`, `git_sha` (from `CLEMATIS_GIT_SHA` or `"unknown"`), **semantic-only** `config_digest` (excludes `trace_dir`), `query_id` (Unicode NFKC + collapsed whitespace, lowercased), `clock=0`, `seed=0`.
- **Stage wiring:** `clematis/engine/stages/t2.py` emits a **write-only** shadow trace **behind a triple gate**: `perf.enabled && perf.metrics.report_memory && t2.quality.shadow && !t2.quality.enabled`. Ordering and scores are unchanged.
- **Configs (validator):** `configs/validate.py` adds `t2.quality.{enabled,shadow,trace_dir,redact}` with defaults (`enabled=false`, `shadow=false`, `trace_dir="logs/quality"`, `redact=true`). **PR36 guard:** `t2.quality.enabled=true` is rejected until PR37 lands. Verbose warning when the triple gate is incomplete.
- **Examples:** `examples/quality/shadow.yaml` — runnable example exercising the triple gate (ON) with redaction.
- **CLIs:**
  - `scripts/rq_trace_dump.py` — tail/inspect recent trace entries.
  - `scripts/validate_noop_shadow.py` — Gate D comparator; asserts **no diffs** between baseline vs shadow runs **except** for `rq_traces.jsonl`.
- **Configs (defaults):** `configs/config.yaml` now includes a minimal `t2.quality` block (defaults OFF) so the identity path is explicit.
- **CI (Gate D — Quality shadow is a no‑op):** `.github/workflows/gate_d_quality_shadow.yml` (recommended as **required**). Runs baseline vs shadow and calls the comparator script; fails on any non-trace diffs.

### Changed (PR36)
- **Trace schema stability:** `config_digest` now hashes only **behavioral knobs** (excludes env-specific paths like `trace_dir`); `query_id` normalization tightened (NFKC + whitespace collapse + lowercase) to make A/B diffs mechanical.
- **Docs:** README gains “M7 — Retrieval Quality: shadow tracing (PR36, defaults OFF)” with quickstart, triple‑gate description, and rollback notes.


### Tests (PR36)
- `tests/quality/test_t2_quality_shadow.py` — shadow is a no‑op; triple gate required; validator guard for `enabled=true`.
- `tests/quality/test_trace_emitter_format.py` — redaction on/off; deterministic headers; config digest invariance; query normalization; item order preservation.
- `tests/scripts/test_validate_noop_shadow.py` — comparator ignores `rq_traces.jsonl` but fails on any other diff.
- `tests/examples/test_examples_quality_shadow_yaml.py` — example exists and validates; triple gate is satisfied.
- `tests/integration/test_t2_shadow_noop_integration.py` — end‑to‑end: baseline vs shadow produce identical rankings; trace appears only under shadow (separate dirs).


### Added (PR37 — Lexical BM25 + fusion; opt‑in)
- **Engine (T2):** `clematis/engine/stages/t2_quality.py` — deterministic lexical BM25 scorer over the current candidate set and rank‑based fusion; ties break by `lex(id)`.
- **Stage wiring:** `clematis/engine/stages/t2.py` calls fusion when `t2.quality.enabled=true`; emits quality metrics under the triple gate; **enabled‑path tracing** writes `rq_traces.jsonl` with `meta.reason="enabled"` and per‑item `rank_sem`, `rank_fused`, `score_fused`.
- **Configs (validator):** `configs/validate.py` now accepts `t2.quality.enabled=true` and adds `t2.quality.lexical.{bm25_k1,bm25_b,stopwords}` and `t2.quality.fusion.{mode,alpha_semantic}` (defaults + bounds). Warnings emitted if perf gates are off.
- **Examples:** `examples/quality/lexical_fusion.yaml` — runnable example for PR37.
- **CI (optional smoke):** `.github/workflows/pr37_quality_fusion_smoke.yml` — runs the test suite with PR37 enabled (`CLEMATIS_CONFIG=examples/quality/lexical_fusion.yaml`).

### Changed (PR37)
- **Traces:** When `t2.quality.enabled=true` and the triple gate is satisfied, the pipeline now writes **enabled‑path** traces (`meta.reason="enabled"`) in addition to PR36 shadow traces. Shadow behavior unchanged.
- **Docs:** README updated with a PR37 section, quickstart, and Logs details (enabled‑path fields).

### Tests (PR37)
- `tests/quality/test_t2_quality_fusion.py` — determinism of fused ordering, alpha extremes, and metrics presence only when enabled.
- `tests/integration/test_t2_enabled_traces_integration.py` — verifies that enabled‑path traces are produced with `meta.reason="enabled"`.

### Added (PR38 — MMR diversification; opt-in)
- **Engine (T2):** `clematis/engine/stages/t2_quality_mmr.py` — deterministic MMR utilities: `MMRItem`, `jaccard_distance`, `mmr_select`, `mmr_reorder_full`, and `avg_pairwise_distance` (pure, no side effects).
- **Stage wiring:** `clematis/engine/stages/t2_quality.py` — adds `maybe_apply_mmr(...)` glue that builds token sets (mirroring PR37) and reorders the fused list; ties break by `lex(id)`.
- **Stage wiring:** `clematis/engine/stages/t2.py` — invokes MMR after PR37 fusion when `t2.quality.mmr.enabled=true`; emits gated metrics (`t2q.mmr.lambda`, `t2q.mmr.selected`, `t2q.diversity_avg_pairwise`).
- **Configs (validator):** `configs/validate.py` accepts `t2.quality.mmr.{enabled,lambda,k}` with bounds; supports legacy aliases `lambda_relevance` and `k_final`; truncation warning uses the key provided (e.g., warns with `k_final` when that was set). Warns when `mmr.enabled=true` while `t2.quality.enabled=false` (no effect).
- **Examples:** `examples/quality/mmr.yaml` — minimal runnable example.

### Changed (PR38)
- **Determinism & invariants:** No RNG or wall-clock; ties break by `lex(id)`. When `t2.quality.enabled=false` or `mmr.enabled=false`, results match PR37 fused order. With MMR on, final order is deterministic.
- **Traces:** Enabled-path traces (PR37) now reflect the MMR-reordered list when MMR is engaged; shadow behavior unchanged. Triple gate unchanged.


### Tests (PR38)
- `tests/scripts/test_t2_quality_mmr.py` — selection extremes, tie-breaks, permutation properties, lambda validation.
- `tests/integration/test_t2_mmr_enabled_traces.py` — MMR reorders when enabled; no-op when disabled; text-only tokenization path; asserts gated metrics `t2q.mmr.lambda`, `t2q.mmr.selected`, and `t2q.diversity_avg_pairwise`.

### Added (PR39 — Normalizer & aliasing; quality paths only)
- **Engine (T2):** `clematis/engine/stages/t2_quality_norm.py` — deterministic helpers: `normalize_text`, `tokenize`, `load_alias_map` (cached), `apply_aliases` (exact/phrase; single‑pass, idempotent).
- **Stage wiring:** `clematis/engine/stages/t2_quality.py` — BM25 scorer and MMR glue now use PR39 normalizer/aliasing when `t2.quality.enabled=true`; disabled path unchanged.
- **Configs (validator):** `configs/validate.py` — adds `t2.quality.normalizer.{enabled,case,unicode}` and `t2.quality.aliasing.map_path`; warns when normalizer/aliasing are configured while `t2.quality.enabled=false`; warns when `map_path` is missing/unreadable (no‑op at runtime).
- **Examples:** `tests/examples/quality/normalizer_aliases.yaml`, `tests/examples/quality/aliases.yaml` — minimal runnable example + alias map.

### Changed (PR39)
- **Determinism & invariants:** No RNG/wall‑clock; ties via `lex(id)`; disabled‑path identity preserved. With no alias map, BM25 behavior remains parity with PR37; MMR token sets mirror the same normalization/aliasing policy.
- **Traces & gates:** Enabled‑path traces reflect any normalization/aliasing effects via the fused/MMR outputs; triple gate unchanged.

### Tests (PR39)
- `tests/quality/test_t2_quality_normalizer_aliases.py` — normalization (NFKC/lower/whitespace), tokenization (stops/minlen/stemmer), aliasing rewrite/expansion, alias map I/O; idempotency.
- `tests/integration/test_t2_quality_normalizer_aliases_integration.py` — alias map deterministically changes the MMR head (λ=1.0, k=2); repeatability.

### Added (PR42 — Docs & Examples Round-Up)
- **Docs:**
  - `docs/m7/overview.md` — milestone invariants, feature summaries (PR36–PR41), enablement path.
  - `docs/m7/troubleshooting.md` — triple-gate checklist; symptom→cause→fix for shadow/fusion/MMR/aliases/reader parity; rollback.
  - `docs/m7/rq_eval.md` — CLI quickstart; input formats; metric definitions; JSON/CSV schema; determinism notes.
- **Examples:**
  - `examples/quality/normalizer_aliases.yaml` — enabled path with PR39 normalizer & PR38 MMR; `aliasing.map_path` wired.
  - `examples/quality/aliases.yaml` — example alias map used by the above.
  - `examples/quality/reader_parity.yaml` — PR40 parity example; uses shadow to satisfy triple gate without result changes.
- **Tooling:** `scripts/examples_smoke.py` — validates example YAMLs, runs short queries via `run_t2`, checks determinism, asserts gated metrics (`t2.reader_mode`, and `t2q.mmr.*` when MMR enabled). Caches disabled per run to avoid cross-config pollution.

### CI (PR42)
- `.github/workflows/pr42_docs_examples_smoke.yml` — runs `examples_smoke.py` on changes to docs/examples; uploads logs on failure.

### Notes (PR42)
- Comparator ignore pattern narrowing (from any `rq_traces.jsonl` to `**/logs/quality/rq_traces.jsonl`) remains an optional improvement and is **not** applied in this PR.

## [0.6.0] - 2025-09-22
### Added
- **CI:** `disabled-path-identity` workflow at `.github/workflows/disabled_identity.yml`:
  - Validates repo config (non-strict), runs strict quiet validator, unit test, CLI smoke, and normalized log diff.
- **Configs:** `.ci/validator_strict_quiet.yaml` and `.ci/disabled_path_config.yaml` to support the workflow.
- **Tests:** Identity guard & validator smoke tests (e.g. `tests/test_identity_disabled_path.py`, `tests/test_config_perf_quality.py`).
- **Goldens:** `tests/goldens/pr29/.logs/` with normalized JSONL golden logs for disabled-path demo.

### Changed
- CI now normalizes logs using the shared Python normalizer (`tests/helpers/identity.py`) and sorts before diffing; volatile fields (`now`, `*_ms`, `*_bytes`, `version_etag`, `rss_bytes`) are dropped.

### Added (PR31 — T1 compaction; defaults OFF)
- **Engine (T1):** Deterministic compaction controls under `perf.t1`:
  - `caps.frontier` — bounds the T1 priority queue (effective limit is `min(t1.queue_budget, caps.frontier)`).
  - `caps.visited` — bounds the visited set using a FIFO-on-first-visit structure.
  - `dedupe_window` — short-horizon push deduplication to reduce thrash.
  - All features are gated; **disabled-path identity** is preserved when `perf.enabled=false` or keys are absent.
- **Utilities:**
  - `engine/util/ring.py` — `DedupeRing` (O(1) membership via refcounts) for push dedupe.
  - `engine/util/lru_det.py` — `DeterministicLRUSet` (visited bound) and `DeterministicLRU` (map cache for future PRs).
- **Validator:** Support for `perf.t1.caps.{frontier,visited}` and `perf.t1.dedupe_window` (values must be ≥1).
  - Legacy `perf.t1.queue_cap` is accepted and mapped to `caps.frontier` if not provided.
  - Emits a warning when caps/dedupe are configured while `perf.enabled=false` (identity path).
- **Docs:** README section “M6 — Perf & Compaction (PR31: T1 caps + dedupe; OFF by default)” with enablement example and metrics.

### Tests (PR31)
- `tests/util/test_ring.py` — deterministic coverage for `DedupeRing`.
- `tests/util/test_lru_det.py` — deterministic coverage for `DeterministicLRUSet` / `DeterministicLRU`.
- `tests/config/test_validate_perf_t1_caps.py` — CLI strict/invalid/warning scenarios.
- `tests/stages/test_t1_compaction.py` — placeholder (skipped) until a stable T1 graph fixture is exposed.


### Changed (PR31)
- `engine/stages/t1.py` wired to use the new caps and dedupe **behind gates**; default behavior/logs remain identical to PR29 goldens (identity CI).
- Metrics only emit when `perf.enabled=true` **and** `perf.metrics.report_memory=true`:
  - `t1_frontier_evicted`, `t1_dedup_hits`, `t1_visited_evicted`.

### Added (PR32 — Size‑aware caches for T1/T2; defaults OFF)
- **Utility:** `engine/util/lru_bytes.py` — `LRUBytes` (deterministic LRU‑by‑bytes with `(max_entries, max_bytes)`, oversize insert rejection, optional `on_evict`).
- **Stages:**
  - `engine/stages/t1.py` — PR32‑aware cache selector prefers `perf.t1.cache` (bytes) when `perf.enabled=true` and caps > 0; legacy LRU fallback preserved.
  - `engine/stages/t2.py` — wired to `perf.t2.cache` (bytes) with deterministic cost estimation; legacy LRU+TTL fallback preserved.
- **Validator:** accepts `perf.t1.cache.{max_entries,max_bytes}` and `perf.t2.cache.{max_entries,max_bytes}` (≥ 0) and warns when configured while `perf.enabled=false` (identity path).
- **Docs:** README section “M6 — Perf & Compaction (PR32: Size‑aware caches for T1/T2; OFF by default)”.

### Tests (PR32)
- `tests/util/test_lru_bytes.py` — sizes, multi‑eviction, MRU on get/put, oversize reject, clear.
- `tests/config/test_validate_perf_cache_pr32.py` — strict acceptance, disabled‑perf warnings, zero‑caps disabled.
- `tests/stages/test_t2_cache_smoke.py` — bytes‑mode selection and eviction math; metrics smoke is skipped pending a stable fixture.

### Changed (PR32)
- **Identity preserved:** with `perf.enabled=false` or zero/absent caps, behavior and logs remain identical to PR29/PR31 disabled path.
- **Metrics (gated):** new counters only when `perf.enabled=true` **and** `perf.metrics.report_memory=true`:
  - `t1.cache_evictions`, `t1.cache_bytes`
  - `t2.cache_evictions`, `t2.cache_bytes`

### Added (PR33 — T2 fp16 store + fp32 math + partition‑friendly reader; opt‑in runtime)
- **Utility:** `engine/util/embed_store.py` — shard writer/reader with deterministic iteration:
  - `write_shard(dir, ids, embeds, *, dtype="fp16|fp32", precompute_norms: bool)`
  - `open_reader(root, partitions={enabled, layout, path}) -> EmbedReader`
  - `EmbedReader.iter_blocks(batch)` yields `(ids, embeds_fp32, norms_fp32)`; no RNG/clocks.
- **Stage (T2):** `engine/stages/t2.py` wired for **opt‑in** runtime retrieval from the on‑disk embed store (cosine in fp32) when `perf.enabled=true` **and** `perf.t2.reader.partitions.enabled=true`; otherwise legacy in‑memory path is used. Emits gated metrics about store dtype, partition layout, and shard count.
- **Validator:** accepts `perf.t2.embed_store_dtype: {fp32, fp16}`, `perf.t2.precompute_norms: bool`, and `perf.t2.reader.partitions.{enabled, layout, path}` (`layout ∈ {owner_quarter, none}`); accepts stage knobs `t2.reader_batch` (≥ 1) and `t2.embed_root` (non‑empty string). Warns when `embed_store_dtype=fp16` with `precompute_norms=false`.
- **Docs:** README section “M6 — Perf & Compaction (PR33: T2 fp16 store + fp32 math + partition‑friendly reader; opt‑in runtime)”.

### Tests (PR33)
- `tests/t2/test_fp16_parity.py` — fp16 store + fp32 math parity vs fp32 baseline (Top‑K identical; ≤ 1e‑6 score drift).
- `tests/t2/test_partition_reader.py` — owner/quarter partitions parity (unpartitioned vs partitioned).
- `tests/t2/test_reader_backcompat.py` — fp32 shards without norms compute norms on the fly; missing meta rejected.
- `tests/stages/test_t2_reader_integration.py` — **runtime flip** path returns identical Top‑K and scores; `tier_sequence == ["embed_store"]`.
- `tests/t2/test_metrics_gated.py` — placeholders documenting gated metric expectations (skipped until fixture is exposed).
- `tests/config/test_validate_perf_t2_fp16.py` — validator accepts fp16+norms and partitions; rejects bad layouts/paths; warns when norms off.
- `tests/config/test_validate_t2_reader_batch.py` — validator accepts `t2.reader_batch` & `t2.embed_root`; rejects invalid; warns when partitions configured with `perf.enabled=false`.

### Changed (PR33)
- **Identity preserved by default:** with `perf.enabled=false` (or reader disabled), behavior/logs remain identical to PR29/PR31/PR32 disabled path.
- **Opt‑in runtime reader:** enabling `perf.t2.reader.partitions.enabled=true` retrieves from the embed store deterministically; legacy tiers are bypassed.
- **Metrics (gated):** new keys emitted only when `perf.enabled=true` **and** `perf.metrics.report_memory=true`:
  - `t2.embed_dtype`, `t2.embed_store_dtype`, `t2.precompute_norms`, `t2.reader_shards`, `t2.partition_layout`, and `t2.tier_sequence` (=`["embed_store"]` when active).

- Local smoke:
  ```bash
  pytest -q
  python3 scripts/validate_config.py .ci/validator_strict_quiet.yaml --strict
  python3 scripts/run_demo.py --config .ci/disabled_path_config.yaml
  ```

![disabled-path-identity](https://github.com/vecipher/Clematis3/actions/workflows/disabled_identity.yml/badge.svg)

### Added (PR33.5 — Gate C: T2 reader parity & runtime‑flip invariants; defaults OFF)
- **CI:** Required workflow `.github/workflows/t2_reader_parity.yml`:
  - Parity matrix over `{embed_store_dtype∈{fp32,fp16}} × {precompute_norms∈{true,false}}` running parity + integration tests.
  - Identity sub‑job asserts reader does **not** engage when `perf.enabled=false`.
- **Scripts:** `scripts/seed_tiny_shard.py` — deterministic tiny store for parity tests.
- **Tests:** `tests/t2/test_t2_reader_gate_identity.py` — reader must not engage in disabled mode.
- **Helpers:** `tests/helpers/identity.py` — shared normalizer used by CI and tests (replaces `jq` pipeline).
- **Config loader:** `clematis/io/config.py` applies safe env overrides used by CI (`PERF_T2_DTYPE`, `PERF_T2_PRECOMPUTE_NORMS`) without flipping `perf.enabled`.

### Changed (PR33.5)
- **Engine (T2):** `engine/stages/t2.py` now:
  - Emits `metrics.tier_sequence` consistently and a **gated** `metrics.reader` **only** when the reader is engaged with `perf.enabled=true`.
  - Enforces deterministic final sort by `(-score, lex(id))` in all scoring paths.
  - Exposes a stable `run_t2(cfg, **kwargs)` adapter for tests/CI (delegates to stage entrypoint).
- **Validator:** `configs/validate.py`:
  - Accepts `perf.t2.reader.partitions.by` (strictly limited to `["owner","quarter"]`).
  - Emits a warning if any reader partitions config is present while `perf.enabled=false` (identity path).
- **Identity workflow:** `.github/workflows/disabled_identity.yml` tightened:
  - Editable dev/test install, deterministic locale/UTF‑8, shared Python normalizer for diffs, artifact upload on failure.
- **Demo:** `scripts/run_demo.py` adds deterministic clock by default (`--fixed-now-ms`, `--wall-clock` to opt‑in).

### Notes (PR33.5)
- **No behavior change by default:** with `perf.enabled=false`, outputs/logs remain identical to PR29 goldens.
- **Branch protection:** mark both jobs in `t2_reader_parity.yml` as required:
  - “Gate C — T2 Reader Parity / Parity matrix (dtype × norms)”
  - “Gate C — T2 Reader Parity / Reader must not engage (perf=OFF)”

### Added (PR34 — Snapshots: zstd compression + delta mode; defaults OFF)
- **Util:** `clematis/engine/util/snapshot_delta.py` — deterministic path-based delta codec (`compute_delta`, `apply_delta`).
- **Tests:**
  - `tests/snapshots/test_snapshot_delta_roundtrip.py` — delta round-trip identity.
  - `tests/snapshots/test_snapshot_fallback.py` — baseline missing ⇒ warning + full fallback.
  - `tests/config/test_validate_perf_snapshots.py` — validator bounds/warnings (aligned with current schema).
- **Examples:** `examples/perf/snapshots.yaml` — safe example config (used by PR35 docs).
- **CI:** `.github/workflows/snapshots_smoke.yml` — fast smoke for delta/fallback tests on relevant PRs.

### Changed (PR34)
- **Snapshot reader:** `clematis/engine/snapshot.py` adds `read_snapshot(...)` that understands two-line `header\npayload` format for **full** and **delta** files, supports optional `.json.zst` (when `zstandard` is installed), and performs a **safe fallback** to the matching full snapshot when the baseline is missing/mismatched, logging `SNAPSHOT_BASELINE_MISSING`.
- **IO shim:** `clematis/engine/io/snapshot.py` re-exports `read_snapshot` to avoid optional import errors at import time.
- **Validator:** `configs/validate.py` now emits a warning when any `perf.snapshots.*` keys are configured while `perf.enabled=false` (identity path; features remain disabled). No API shape changes.
- **Dev/Test deps:** add `zstandard>=0.22` to dev/test extras (`pyproject.toml`) and `requirements-dev.txt`.

### Notes (PR34)
- **No behavior change by default:** with `perf.enabled=false`, outputs/logs remain identical to PR29 goldens.
- **Semantics unchanged:** enabling compression/delta changes persistence cost/size only; reconstructed snapshot equals the full payload.
- **Metrics deferred:** no new snapshot metrics in PR34; future counters must be gated by `perf.enabled && perf.metrics.report_memory`.
- **Determinism:** ETags computed from canonical JSON; ordering and filenames are stable.

### Added (PR34.1 — Snapshot writer; read/write parity; defaults OFF)
- **Snapshot writer:** `clematis/engine/snapshot.py::write_snapshot_auto(...)` emits `snapshot-{etag}.full.json[.zst]` or `snapshot-{etag_to}.delta.json[.zst]` using the same two-line canonical `header\npayload` format as the reader.
- **Tests:** `tests/snapshots/test_snapshot_writer_roundtrip.py` — writes a baseline full + a delta from that baseline and verifies reconstruction equals the current payload.

### Changed (PR34.1)
- **Reader helpers:** `_read_text` and `_read_header_payload` now accept Path-like inputs for robustness in CLIs/tests; `.zst` handling remains optional and guarded.
- **README:** PR34 section updated to reflect writer availability in PR34.1 (read/write support; defaults unchanged).


### Added (Gate B — Metrics-only guard; required)
- **CI:** `.github/workflows/gate_b.yml` — asserts **no metrics** (including `tier_sequence`) appear when either:
  - `perf.enabled=false`, or
  - `perf.enabled=true` and `perf.metrics.report_memory=false`.
- **Scripts:** `scripts/ci_gate_b_assert.py` — scans normalized logs for disallowed keys; used by the workflow.
- **Configs:** `.ci/gate_b_off.yaml`, `.ci/gate_b_on_nometrics.yaml` — minimal configs to exercise the two OFF modes.

### Changed (Gate B)
- **Stages:**
  - `engine/stages/t2.py` — all metric emissions (including `tier_sequence`) are **gated** behind `perf.enabled && perf.metrics.report_memory`.
  - `engine/stages/t3.py` — no longer synthesizes `tier_sequence` when absent; passes through only what T2 emits under gate.
- **Tests:** updated `tests/test_t2_stage.py` to explicitly enable the metrics gate only in tests that assert on metrics.
- **Identity:** no change to disabled-path outputs; Gate B ensures accidental leaks are caught pre-merge.

### Added (Nice-to-haves — config-only / helper)
- **Validator:** optional config surface for `t2.lancedb.partitions` (e.g., `{by: ["owner","quarter"], shard_order: "lex|score"}`) with warnings when `perf.enabled=false`. **No runtime changes in M6.**
- **Helper:** `engine/util/metrics.py` — central `emit/emit_many/gate_on` helper to prevent future counters from bypassing the gate (not yet widely adopted in M6).

### Changed (Validator & examples smalls)
- **API compatibility:** `configs/validate.py` now supports both call styles:
  - `validate_config(cfg) -> normalized dict` (raises on errors), and
  - `validate_config(cfg, strict=..., verbose=...) -> (errors, warnings)`.
- **Examples:** `examples/perf/caches.yaml` now scopes caches under `perf.t1.cache` and `perf.t2.cache` (replacing the deprecated top-level `perf.cache`).

- **Docs:**
  - README gains an “M6 — Docs & CLIs (PR35)” section with quickstart, CLI usage, and links to examples.
  - CONTRIBUTING adds a rollback playbook (how to force disabled‑path identity) and a quick ref for Gates A/B/C.
- **Examples:**
  - `examples/perf/caps.yaml` — PR31 T1 caps + dedupe ring.
  - `examples/perf/caches.yaml` — PR32 LRU‑by‑bytes caches.
  - `examples/perf/fp16_reader.yaml` — PR33/33.5 fp16 + partition reader (opt‑in).
  - `examples/perf/snapshots.yaml` — PR34 compression + delta (safe fallback) referenced by docs.
- **CLIs (offline only):**
  - `scripts/mem_inspect.py` — lists snapshot files and basic stats (works with `.json` and optional `.json.zst`).
  - `scripts/mem_compact.py` — offline rewrite of **full** snapshots to a new directory with optional zstd compression; never in‑place.
- **Tests:**
  - `tests/cli/test_mem_inspect.py`, `tests/cli/test_mem_compact.py` — smoke tests (exit 0, expected keys/files).
- **CI:**
  - `.github/workflows/cli_smoke.yml` — runs CLI smoke tests on PRs touching `scripts/`.

### Changed (PR35)
- README updated with M6 quickstart, examples, and CLI docs. No changes to runtime code paths.
- CHANGELOG updated accordingly.

### Notes (PR35)
- **No runtime behavior change:** CLIs are offline and do not alter runtime semantics or logs.
- **Identity preserved:** With `perf.enabled=false`, behavior/logs remain identical to PR29 goldens (guarded by the identity CI gate).
- **Safety:** `mem_compact.py` writes to a new directory, never in‑place; deterministic ordering and headers.
- **Optional dependency:** `zstandard` is only required when reading/writing `.zst` files; included in dev/test deps.

## [0.6.1] - 2025-09-24
### i make changes on a whim

### Added (PR40 — Lance partition reader parity; optional, default flat)
- **Engine (T2):** `clematis/engine/stages/t2_lance_reader.py` — dependency-light, deterministic partition wrapper over a flat reader; preserves content and within-partition order; never raises during availability checks.
- **Stage wiring:** `clematis/engine/stages/t2.py` — accepts `t2.reader.mode ∈ {flat, auto, partition}`; selects partition mode only when available; otherwise deterministically falls back to flat. Emits gated metric `t2.reader_mode` ("flat"|"partition"). No scoring/order changes.
- **Configs (validator):** `configs/validate.py` — adds `t2.reader.mode` enum and warning when `mode ∈ {partition, auto}` is set while `perf.enabled=false` (no effect).
- **Examples:** `tests/examples/quality/reader_parity.yaml` — minimal config to exercise mode selection and metrics.
- **CI (smoke):** `.github/workflows/pr40_reader_parity_smoke.yml` — matrix over `{flat, auto, partition}`; runs parity smoke tests; uploads artifacts on failure.

### Changed (PR40)
- **Observability only:** Default remains `flat`; when `partition` is unavailable, code falls back to `flat` and reports it via `t2.reader_mode` under the triple gate. Disabled-path identity and trace behavior remain unchanged.

### Tests (PR40)
- `tests/integration/test_t2_reader_parity.py` — emits `t2.reader_mode=flat` under gate; `partition`/`auto` fall back to `flat` when unavailable; metric suppressed when `report_memory=false`.

### Added (PR41 — Retrieval Quality evaluation harness; offline, deterministic)
- **CLI:** `scripts/rq_eval.py` — A/B evaluator that computes `recall@k`, `MRR@k`, and `nDCG@k` with stable JSON (and optional CSV). Fully local (no network), no RNG/wall-clock. Optional `--emit-traces` respects the existing triple gate; trace `meta.reason` may remain `"enabled"` depending on runtime.
- **Fixtures:** `tests/fixtures/rq/{corpus.jsonl,queries.tsv,qrels.tsv}` — tiny toy set for stable tests.
- **CI (smoke):** `.github/workflows/pr41_rq_eval_smoke.yml` — runs metric unit + CLI E2E smoke on relevant changes; uploads artifacts on failure.

### Changed (PR41)
- **Observability:** The evaluator records digests of inputs (`corpus/queries/qrels`) and configuration for reproducibility. No runtime behavior changes.

### Tests (PR41)
- `tests/scripts/test_rq_eval_cli.py` — unit tests for metric functions and CLI schema/CSV; skips gracefully if runtime entrypoint is unavailable.
- `tests/integration/test_rq_eval_end_to_end.py` — subprocess smoke; verifies outputs; trace presence is optional and sanity-checked only when present.

---

## CI guardrails !!!

### 4.1 `.github/workflows/cli-smokes.yml` (new)
```yaml
name: CLI Smokes
on:
  push:
    paths:
      - 'clematis/cli/**'
      - 'scripts/**'
      - 'docs/cli.md'
      - 'tests/cli/**'
      - '.github/workflows/cli-smokes.yml'
  pull_request:

jobs:
  cli:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: '3.11' }
      - run: pip install -e .[dev]
      - name: Help matrix
        run: |
          python -m clematis --help >/dev/null
          python -m clematis rotate-logs --help >/dev/null
          python -m clematis inspect-snapshot --help >/dev/null
          python -m clematis bench-t4 --help >/dev/null
          python -m clematis seed-lance-demo --help >/dev/null
      - name: Pass-through with leading --
        run: |
          python -m clematis --debug rotate-logs -- --dir ./.logs --dry-run 2>stderr.txt || true
          head -n1 stderr.txt | grep "delegate -> scripts.rotate_logs"
          head -n1 stderr.txt | grep "argv=\['--dir', './.logs', '--dry-run'\]"
      - name: Script shim hint (deprecation stderr)
        run: |
          python scripts/rotate_logs.py --dry-run 1>out.txt 2>err.txt || true
          head -n1 err.txt | grep "direct script is deprecated"
      - name: CLI tests
        run: pytest -q tests/cli

### PR54 — Man pages & packaging
- Add `scripts/gen_manpages.py` (stdlib-only, deterministic via `SOURCE_DATE_EPOCH`)
- Package manpages with `tool.setuptools.data-files` → `share/man/man1`
- Include manpages in sdist via `MANIFEST.in`
- CI: generate before build, upload artifact, verify presence in sdist/wheel, best-effort render check
