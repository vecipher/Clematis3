

# Packaging & CLI (Clematis v3)

> Canonical doc for **post‑install** usage and CLI invariants. Zero behavior change. If you need wrapper internals, see **`docs/m8/cli.md`**.

---

## Quickstart (post‑install)
These work after installing the wheel (no repo checkout):

```bash
python -m clematis --version
python -m clematis inspect-snapshot -- --format json
python -m clematis rotate-logs -- --dry-run
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

## Environment knobs
- `CLEMATIS_DEBUG=1` — enable stderr breadcrumbs (does not alter stdout or exit codes).
- `CLEMATIS_NETWORK_BAN=1` — CI/offline guard; networking is not required for the CLI smokes.

---

## Exit codes & I/O expectations
- Successful demo runs exit **0**.
- Wrapper help prints to **stdout**; debug breadcrumbs (when enabled) print to **stderr**.
- The demos’ outputs are deterministic; expect stable field sets (subject to documented schema fields in the inspector output).

---

## Optional extras (pip groups)
These are **metadata only**; the core CLI smokes don’t require them.

| Extra name     | Purpose / typical libs                                  |
|----------------|----------------------------------------------------------|
| `bench`        | lightweight benchmarking helpers (e.g., `numpy`)         |
| `cli-extras`   | CLI demos for snapshots/IO (e.g., `numpy`, `pyarrow`, `lancedb`) |

Install with e.g.:
```bash
pip install "clematis3[cli-extras]"
```
(Adjust name to your published package.)

---

## CI: Wheel‑smoke (post‑install acceptance)
CI builds the wheel, installs into a fresh venv, then runs the three quickstart commands above with `CLEMATIS_NETWORK_BAN=1`. No heavy extras are installed. This guards the packaged defaults and CLI invariants without changing runtime behavior.

---

## Troubleshooting
- **`ModuleNotFoundError` for optional libs**: expected if you run heavier scripts without installing extras; the two demos above do not require them.
- **No breadcrumbs shown**: add `--debug` or export `CLEMATIS_DEBUG=1`.
- **Default directories not used**: passing `--dir` disables default injection by design.

---

## Pointers
- Wrapper internals & delegation rules: **`docs/m8/cli.md`**
- Tests asserting the invariants: `tests/cli/` (help phrase, sentinel strip, default injection, bypass)
