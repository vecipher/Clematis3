

# M8 Canvas — Packaging & CLI Polish (Clematis v3) 27/09/25

**Objective**: Harden the umbrella CLI and packaging so post‑install use is deterministic and zero‑surprise. Keep **zero behavior change** while improving docs, tests, and CI.

---

## Context / Goals
- Preserve PR47 invariants: first‑subcommand anchoring, single‑sentinel strip, wrapper help phrase, stderr‑only breadcrumbs, no stdout/exit code changes.
- Make default experiences work **after wheel install** (no repo checkout): packaged examples discoverable via `importlib.resources`.
- Keep CI lean (no heavy deps) but still guard core CLI invariants.

---

## Delivered in PR48
- **Docs**
  - Split canonical CLI doc → `docs/m8/cli.md` (delegation diagram, rules, recipes).
  - Trimmed `README.md` to a lean pointer.
- **Wrappers (no behavior change)**
  - `rotate-logs`: inject `--dir` only when absent → packaged `clematis/examples/logs` else `./.logs`.
  - `inspect-snapshot`: inject `--dir` only when absent → packaged `clematis/examples/snapshots` else `./snapshots`.
- **Resources / Packaging**
  - Wheel‑safe helpers: `clematis/cli/_resources.py` (`importlib.resources`).
  - Common util: `inject_default_from_packaged_or_cwd(...)` in `_wrapper_common.py`.
  - Packaged demo snapshot: `clematis/examples/snapshots/snap_000001.json` (schema‑valid JSON) to enable post‑install default.
  - Packaging knobs: `include-package-data = true` (pyproject), `MANIFEST.in` includes `clematis/examples/*`.
  - Optional extras (metadata only):
    - `bench = ["numpy>=1.26,<2"]`
    - `cli-extras = ["numpy>=1.26,<2","lancedb>=0.8,<1","pyarrow>=15,<19"]`
- **Tests** (focused, deterministic)
  - Delegation + sentinel: `tests/cli/test_cli_delegation.py`.
  - Help phrase + umbrella `--version`: `tests/cli/test_cli_help_phrase.py`.
  - Debug breadcrumb parity (stderr‑only): `tests/cli/test_cli_debug_breadcrumb.py`.
  - Direct script shim hint: `tests/cli/test_cli_shim_hint.py` (tolerates missing heavy deps; still asserts hint when deps exist).
  - Default path works: `tests/cli/test_cli_inspect_snapshot_default.py`.
- **CI**
  - `.github/workflows/cli-smokes.yml` runs the above tests with `CLEMATIS_NETWORK_BAN=1`.

---

## Acceptance signals (for this step)
- `python -m clematis --version` works post‑install.
- `python -m clematis inspect-snapshot -- --format json` succeeds post‑install using the packaged demo snapshot.
- `python -m clematis rotate-logs -- --dry-run` runs (packaged logs dir optional; falls back to `./.logs`).
- CLI smokes pass in CI without installing heavy extras.

---

## Guardrails kept (no behavior change)
- Explicit user flags **always win** over defaults.
- Only one leading `--` is stripped; pass‑through order preserved.
- Wrapper help is intercepted; phrase **“Delegates to scripts/…”** is stable.
- Breadcrumbs are **stderr only** via `--debug` or `CLEMATIS_DEBUG=1`.

---

## Next (PR49 candidates)
1. **Wheel‑smoke CI** (post‑install acceptance): build → install in clean venv → run three commands:
   - `python -m clematis --version`
   - `python -m clematis inspect-snapshot -- --format json`
   - `python -m clematis rotate-logs -- --dry-run`
2. **Examples coverage (optional)**: add `clematis/examples/logs/.placeholder` to exercise packaged‑first path for `rotate-logs` in wheels.
3. **Docs note**: short paragraph in `docs/m8/cli.md` on packaged defaults (done), verify anchors/links.
4. **Explicit‑flag bypass test**: tiny smoke asserting defaults are *not* injected when `--dir` is provided.
5. **(Optional) Import timing for heavy scripts**: if desired, move `hint_once()` before heavy imports to guarantee shim line even when extras missing (would be a behavior change; keep for later).

---

## References
- CLI doc: `docs/m8/cli.md`
- Wrappers: `clematis/cli/rotate_logs.py`, `clematis/cli/inspect_snapshot.py`
- Common/resources: `clematis/cli/_wrapper_common.py`, `clematis/cli/_resources.py`
- Packaged data: `clematis/examples/snapshots/snap_000001.json`
- CI: `.github/workflows/cli-smokes.yml`