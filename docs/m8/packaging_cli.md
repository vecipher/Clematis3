# Packaging & CLI (Clematis v3)

> Canonical doc for **post‑install** usage and CLI invariants. Zero behavior change. If you need wrapper internals, see **`docs/m8/cli.md`**.

---

## Quickstart (post‑install)
These work after installing the wheel (no repo checkout):

```bash
python -m clematis --version
python -m clematis validate --json
python -m clematis demo -- --steps 1
python -m clematis inspect-snapshot -- --format json
python -m clematis rotate-logs -- --dry-run
```

### Structured output (PR55)

Optional flags for machine‑friendly output:

```bash
# Inspector (JSON or TABLE)
python -m clematis inspect-snapshot --json
python -m clematis inspect-snapshot -- --table | head

# Rotate-logs summaries (JSON or TABLE)
python -m clematis rotate-logs -- --dry-run --json
python -m clematis rotate-logs -- --dry-run --table
```

Notes: flags are mutually exclusive (`--json` XOR `--table`). Defaults remain unchanged when you omit them.

**Note:** If `demo` or `validate` complain about missing NumPy/PyYAML on a base install, add the minimal extra:

```bash
pip install "clematis[cli-demo]"
```

**Rules:**
- One leading `--` (the sentinel) is stripped by the umbrella CLI; everything after is passed to the script.
- Explicit user flags **always win** over defaults.

---

## What’s packaged (defaults that “just work”)
We ship minimal data so the two demos run out‑of‑the‑box after `pip install`:

- `clematis/examples/snapshots/snap_000001.json` — a schema‑valid demo snapshot used by `inspect-snapshot` **only when** you don’t pass `--dir`.
- `clematis/examples/logs/.placeholder` — ensures the logs directory exists in the wheel so `rotate-logs` has a packaged‑first path when you don’t pass `--dir`.

Wrappers locate these via `importlib.resources`. See implementation helpers:
- `clematis/cli/_resources.py`
- `clematis/cli/_wrapper_common.py` → `inject_default_from_packaged_or_cwd(...)`

---

## CLI invariants (guardrails)
- **First‑subcommand anchoring**: the first positional token selects the wrapper.
- **Single sentinel strip**: only the first leading `--` is removed; argument order is preserved.
- **Help phrase is stable**: wrapper `--help` includes **“Delegates to scripts/…”** (non‑semantic breadcrumb).
- **Breadcrumbs to stderr** only: enabled by `--debug` or `CLEMATIS_DEBUG=1`; stdout remains clean.
- **No behavior change** vs PR47: exit codes and stdout are unchanged.
- **Mutually exclusive formats**: `--json` and `--table` cannot be combined; choosing both exits with code 1 and prints a short reason (suppressed by `--quiet`).

---

## Default injection rules
Wrappers supply defaults **only if you didn’t** provide an explicit `--dir`.

- `inspect-snapshot`:
  - Default: packaged `clematis/examples/snapshots`
  - Otherwise: `./snapshots` (when present)
- `rotate-logs`:
  - Default: packaged `clematis/examples/logs`
  - Otherwise: `./.logs` (fallback)

**Bypass example (explicit beats default):**
```bash
python -m clematis inspect-snapshot -- --dir ./my_snaps --format json
python -m clematis rotate-logs     -- --dir ./my_logs  --dry-run
```

---

## Config discovery & env (PR60)

When you **do not** pass `--config` (or `-c`), Clematis selects a config file via a deterministic search order:

1. `$CLEMATIS_CONFIG` — file path **or** directory containing `config.yaml`.
2. `./configs/config.yaml` — relative to the **current working directory**.
3. `${XDG_CONFIG_HOME:-$HOME/.config}/clematis/config.yaml`.

**Rules**
- Explicit `--config`/**`-c`** **always wins** and bypasses discovery.
- In `--verbose` mode, the CLI prints the selected path and source (e.g., `env:CLEMATIS_CONFIG`, `cwd:configs/config.yaml`, `xdg`).

**Example (implicit discovery):**
```bash
# Will use ./configs/config.yaml if present; otherwise falls back to XDG path
python -m clematis validate -- --json --verbose
```

### Environment overrides (consolidated)
The CLI and tests recognize these environment variables:

| Variable                | Effect                                                                 | Notes |
|-------------------------|-------------------------------------------------------------------------|-------|
| `CLEMATIS_CONFIG`       | Points to a config file **or** a directory containing `config.yaml`.   | Used only when `--config` is absent. |
| `XDG_CONFIG_HOME`       | Base for XDG lookup (`$XDG_CONFIG_HOME/clematis/config.yaml`).          | Defaults to `$HOME/.config` when unset. |
| `CLEMATIS_DEBUG`        | `1` enables stderr breadcrumbs (wrapper debug).                         | Does not affect stdout or exit codes. |
| `CLEMATIS_NETWORK_BAN`  | `1` forbids network usage during CLI smokes/tests.                      | Runtime ban only; install/build may still use network. |
| `CLEMATIS_LLM_SMOKE`    | Enables lightweight LLM smoke paths where applicable.                   | Off by default; not used in basic CLI smokes. |
| `CLEMATIS_GIT_SHA`      | Injects git SHA into trace headers/logs for determinism.               | Used in trace files and CI checks. |

> Tip: prefer `--config` for reproducibility when invoking tools in scripts; discovery is for convenience.

---

## Environment knobs
- `CLEMATIS_DEBUG=1` — enable stderr breadcrumbs (does not alter stdout or exit codes).
- `CLEMATIS_NETWORK_BAN=1` — CI/offline guard; networking is not required for the CLI smokes.
- `--quiet` — suppress wrapper diagnostics on stderr (structured outputs remain on stdout).
- `--verbose` — increase wrapper breadcrumbs on stderr; does not alter stdout.

---

## Exit codes & I/O expectations (PR55)

Exit codes are standardized across wrappers:

| Code | Meaning                                   | Typical causes                                 |
|-----:|-------------------------------------------|------------------------------------------------|
| 0    | OK                                        | Happy path                                     |
| 1    | User/validation error                     | Bad flags (e.g., `--json --table`), misuse     |
| 2    | I/O or parse error                        | Missing/invalid `--dir`, file parse errors     |
| 3    | Internal error                            | Unexpected exception                            |

I/O rules:
- Wrapper help prints to **stdout**; diagnostics print to **stderr**.
- `--quiet` suppresses non‑essential wrapper diagnostics on **stderr**; exit codes are unchanged.
- `--verbose` increases stderr breadcrumbs only (no color). Stdout stays clean/machine‑readable.

---

## Structured output support matrix

| Command             | `--json`                         | `--table`           | Notes |
|---------------------|----------------------------------|---------------------|-------|
| `validate`          | ✓ (pass‑through; format unchanged) | ✗                   | Output is preserved verbatim for compatibility; may not be strict JSON. |
| `inspect-snapshot`  | ✓ (JSON; wrapper forces `--format json`) | ✓ (ASCII table)     | Defaults inject packaged demo dir if `--dir` not provided. |
| `rotate-logs`       | ✓ (summary JSON)                 | ✓ (ASCII table)     | Summaries include fields like `dry_run`, `dir`, `pattern`. |
| `demo`              | ✗                                | ✗                   | Flags parsed but currently rejected (exit 1). |
| `bench-t4`          | ✗                                | ✗                   | Flags parsed but currently rejected (exit 1). |
| `seed-lance-demo`   | ✗                                | ✗                   | Flags parsed but currently rejected (exit 1). |

**Formatting:** Tables are plain ASCII (no color). JSON uses compact separators. When no format flag is provided, the human‑readable defaults are unchanged.

---

## Optional extras (pip groups)
These are **metadata-only** groups. For offline `demo`/`validate`, the minimal `cli-demo` extra is recommended.

| Extra name     | Purpose / typical libs                                           |
|----------------|------------------------------------------------------------------|
| `cli-demo`     | Minimal deps for offline demo/validate (e.g., `numpy`, `PyYAML`) |
| `bench`        | Lightweight benchmarking helpers (e.g., `numpy`)                 |
| `cli-extras`   | CLI demos for snapshots/IO (e.g., `numpy`, `pyarrow`, `lancedb`) |

Install with e.g.:
```bash
pip install "clematis[cli-demo]"
```
(Adjust name to your published package.)

---

## CI: Wheel‑smoke (post‑install acceptance)

CI builds the wheel, installs into a fresh venv, then runs the three quickstart commands above with `CLEMATIS_NETWORK_BAN=1`. No heavy extras are installed. This guards the packaged defaults and CLI invariants without changing runtime behavior.

### Cross‑platform smoke matrix (PR52)
We build a wheel per axis and run offline CLI smokes in a fresh venv.

**Axes**
- OS: Linux (`ubuntu-latest`), macOS (`macos-latest`, arm64), Windows (`windows-latest`)
- Python: 3.11, 3.12

**Commands (runtime offline via `CLEMATIS_NETWORK_BAN=1`):**
```bash
python -m clematis --version
python -m clematis validate --json   # or: python -m clematis validate -- --json
python -m clematis demo -- --steps 1 # or: python -m clematis demo --steps 1
```

Notes:
- No LLM smokes in this job.
- The job installs the minimal extra via: `pip install "$WHL[cli-demo]"` to satisfy demo/validate imports.
- Install/build may use network; only the **runtime** of the smokes is network‑banned (`CLEMATIS_NETWORK_BAN=1`).
- Windows enables long paths with `git config --global core.longpaths true` to avoid path length issues.

---

## Troubleshooting
- **`ModuleNotFoundError` for optional libs**: expected if you run heavier scripts without installing extras; the two demos above do not require them.
- **No breadcrumbs shown**: add `--debug` or export `CLEMATIS_DEBUG=1`.
- **Default directories not used**: passing `--dir` disables default injection by design.

---

## Pointers
- Wrapper internals & delegation rules: **`docs/m8/cli.md`**
- Tests asserting the invariants: `tests/cli/` (help phrase, sentinel strip, default injection, bypass)

## Man pages

We ship minimal man(1) pages generated from argparse `--help` (help2man‑style, offline, deterministic).

Generate locally (deterministic):
```bash
SOURCE_DATE_EPOCH=1704067200 python scripts/gen_manpages.py --outdir man
man -l man/clematis.1
```

Installed location (POSIX):
```
$VENV/share/man/man1/clematis.1
```

Note: Windows runners don’t have `man(1)`, but pages ship inside the wheel for parity.

---

## Supply chain: SBOM + provenance (M8‑14)

On tag builds (`push` tags matching `v*` or when a Release is published), CI produces:

- **Deterministic artifacts**: sdist and wheel (respecting `SOURCE_DATE_EPOCH`).
- **CycloneDX SBOM** at `dist/sbom.cdx.json`, generated from a clean virtualenv where the built wheel (optionally with `[cli-demo]`) is installed, using the CycloneDX Python v4 CLI:

  ```bash
  python -m cyclonedx_py environment \
    --output-format JSON \
    --schema-version 1.5 \
    --outfile dist/sbom.cdx.json
  ```

- **SLSA provenance attestations** for `dist/*.whl` and `dist/*.tar.gz` via GitHub `actions/attest-build-provenance`.
- **SBOM publication**: uploaded as a workflow artifact; when triggered by a Release, attached to the Release page.

### Verify provenance

With GitHub CLI (online):

```bash
gh attestation verify dist/clematis-<ver>-py3-none-any.whl -R <OWNER>/<REPO>
```

Offline flow:

```bash
gh attestation download dist/clematis-<ver>-py3-none-any.whl -R <OWNER>/<REPO> > attestation.intoto.jsonl
gh attestation verify dist/clematis-<ver>-py3-none-any.whl -R <OWNER>/<REPO> --bundle attestation.intoto.jsonl
```

### Verify SBOM

Quick sanity:

```bash
jq -e . dist/sbom.cdx.json >/dev/null
```

### Notes

- CycloneDX Python v4 uses the `cyclonedx_py` module entrypoint. We invoke it with `python -m cyclonedx_py` to avoid PATH issues across runners.
- The SBOM job runs only on tags/releases; regular PR/branch jobs remain unchanged (disabled‑path identity preserved).
- SBOM generation installs the built wheel into a temporary venv to resolve dependencies for an **environment SBOM** (more useful than manifest-only).

## Optional extras (PR61)

Some features are optional and installed via extras:

| Extra     | Installs     | Enables                                  |
|-----------|--------------|------------------------------------------|
| `zstd`    | zstandard    | .zst helpers/tests                       |
| `lancedb` | lancedb      | LanceDB import smoke                      |
| `dev`     | test+linters | Local dev setup (`.[dev]`)               |

Install examples:
python -m pip install 'clematis[zstd]'
python -m pip install 'clematis[lancedb]'
python -m pip install 'clematis[dev]'

Tests are skip-aware: if an extra isn’t installed, its tests are skipped.