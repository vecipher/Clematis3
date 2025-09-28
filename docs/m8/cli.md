# Clematis CLI — Umbrella, Wrappers, and Script Delegation

Use `python -m clematis <subcommand> -- [args]`. The umbrella CLI delegates to wrapper subcommands, which in turn delegate to `clematis.scripts.*` (via `scripts/*.py` shims). **Zero behavior change** is guaranteed by design.

## TL;DR (no behavior change)
- **First‑subcommand anchoring:** the umbrella locates the first subcommand and **prepends any top‑level extras** to the delegated `argv`.
- **Single leading `--`:** if present immediately before delegated args, the wrapper strips **exactly one** sentinel; nothing else is touched.
- **Help interception:** wrapper `-h/--help` is intercepted by the wrapper and **always shows wrapper help** (not the delegated script’s) and includes the phrase **"Delegates to scripts/"**.
- **Debug breadcrumbs:** enabled by umbrella `--debug` **or** `CLEMATIS_DEBUG=1`; they go to **stderr only** and never alter stdout or exit codes.
- **Top‑level `--version`:** exposed at the umbrella for deterministic help UX.

## Delegation diagram (ASCII)
```
python -m clematis  ──►  wrapper (e.g., rotate-logs)
                        ├─ intercepts -h/--help → prints wrapper help ("Delegates to scripts/")
                        ├─ strips single leading -- (if present)
                        └─ delegates to clematis.scripts.rotate_logs.main(argv)
                                                ▲
 scripts/rotate_logs.py (shim)  ── hint_once() ┘  (single stderr hint; tolerant import)
```

## Subcommands (delegates)
- `rotate-logs` → `scripts/rotate_logs.py`
- `inspect-snapshot` → `scripts/inspect_snapshot.py`
- `bench-t4` → `scripts/bench_t4.py`
- `seed-lance-demo` → `scripts/seed_lance_demo.py`
- `validate` → `scripts/validate_config.py`
- `demo` → `scripts/run_demo.py`

## Invocation patterns (both orders supported)
```bash
# extras after subcommand with sentinel:
python -m clematis rotate-logs -- --dir ./.logs --dry-run

# extras before subcommand:
python -m clematis --dir ./.logs rotate-logs -- --dry-run
```
> The *single* leading `--` immediately before delegated args is removed by the wrapper.

## Packaged defaults & post‑install behavior
Wrappers only inject defaults when a required flag is **omitted**. They first look for a **packaged** example via `importlib.resources`, then fall back deterministically to a local path.

- **rotate-logs** — default `--dir`:
  1. packaged: `clematis/examples/logs/`
  2. fallback: `./.logs`

- **inspect-snapshot** — default `--dir`:
  1. packaged: `clematis/examples/snapshots/`
  2. fallback: `./snapshots`

  Snapshot discovery (script behavior): prefers `snap_*.json` with the highest numeric suffix; else latest `state_*.json` by mtime; else latest `*.json` by mtime.

> Post‑install smoke:
> ```bash
> python -m clematis --version
> python -m clematis inspect-snapshot -- --format json
> python -m clematis rotate-logs -- --dry-run
> ```

## Environment flags
- `CLEMATIS_DEBUG=1` — enable stderr breadcrumb:
  ```
  [clematis] delegate -> scripts.rotate_logs argv=['--dir','./.logs','--dry-run']
  ```
  (No stdout/exit changes.)
- `CLEMATIS_NETWORK_BAN=1` — recommended in CI to prevent accidental network.

- Wrapper help **always** includes the phrase: **"Delegates to scripts/"**.
- To view delegated script help directly, call the script/module itself:
  - `python -m clematis.scripts.rotate_logs --help`
  - or `python scripts/rotate_logs.py --help`
- Umbrella `--version` is available and stable.

## Recipes
**Rotate logs (dry run)**
```bash
python -m clematis rotate-logs -- --dir ./.logs --dry-run
```

**Inspect snapshot (dir-based)**
```bash
# default (wrapper injects a packaged dir or ./snapshots)
python -m clematis inspect-snapshot --

# explicit dir + format
python -m clematis inspect-snapshot -- --dir ./snapshots --format pretty
# or
python -m clematis inspect-snapshot -- --dir ./snapshots --format json
```

**Bench T4 (JSON)**
```bash
# Heavy extras optional; install for local use: pip install "clematis[cli-extras]"
python -m clematis bench-t4 -- --json --iterations 3 --seed 0
```

**Seed Lance demo (idempotent)**
```bash
# Heavy extras optional; install for local use: pip install "clematis[cli-extras]"
python -m clematis seed-lance-demo -- --dry-run
# Running twice should not duplicate entries when executed for real.
```

**Validate (JSON report)**
```bash
python -m clematis validate --json
# Optionally verify packaged resources too
python -m clematis validate -- --check-resource examples/logs/.placeholder --json
```

**Demo (minimal run)**
```bash
python -m clematis demo -- --steps 1
```

## Troubleshooting
- **I passed `--` and it still reached the script.**
  The wrapper strips **one** sentinel by design. If you need a literal `--` for the script, quote it or provide another `--` where appropriate.
- **Direct scripts show a hint.**
  Calling `scripts/*.py` prints a **single stderr hint** via the shim; behavior is unchanged. For heavy scripts without optional deps installed, you may see an import error before the hint — use the umbrella wrappers for CI.
- **“No snapshot found …”**
  Ensure the directory contains `*.json` snapshots. A demo file `clematis/examples/snapshots/snap_000001.json` is packaged for quick checks.

## Guarantees (from PR47)
- **Zero behavior change** codified: order‑agnostic extras, first‑subcommand anchoring, single‑sentinel strip, wrapper help phrase, stderr‑only breadcrumbs. No stdout or exit code changes.

### Packaged defaults (post-install)

When installed from a wheel, wrappers will inject defaults **only if** you did not pass an explicit `--dir`:
- `inspect-snapshot` → packaged `clematis/examples/snapshots` (contains `snap_000001.json`)
- `rotate-logs` → packaged `clematis/examples/logs` (placeholder directory)

This is a convenience for first-run; explicit flags always win.

## Reproducible builds

We aim for byte-identical sdists/wheels given identical inputs.

**Local smoke:**
```bash
export SOURCE_DATE_EPOCH=1704067200 PYTHONHASHSEED=0 TZ=UTC
python -m pip install -U build
rm -rf build dist *.egg-info && python -m build && shasum -a256 dist/*
rm -rf build dist *.egg-info && python -m build && shasum -a256 dist/*
# hashes must match

CI: see .github/workflows/pkg_build.yml — it builds twice with
SOURCE_DATE_EPOCH set and diffs the checksums.

Matches the issue’s “Document the reproducibility recipe.”  [oai_citation:2‡GitHub](https://github.com/vecipher/Clematis3/issues/82)

---

## Why this satisfies PR51

- **SOURCE_DATE_EPOCH honored** and fixed in CI.  [oai_citation:3‡GitHub](https://github.com/vecipher/Clematis3/issues/82)
- **Deterministic outputs**: two back-to-back builds compared via SHA256.  [oai_citation:4‡GitHub](https://github.com/vecipher/Clematis3/issues/82)
- **Docs updated** with the local smoke and CI pointer.  [oai_citation:5‡GitHub](https://github.com/vecipher/Clematis3/issues/82)
- **Zero behavior change** for runtime/CLI.
