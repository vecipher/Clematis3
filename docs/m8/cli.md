# Clematis CLI — Umbrella, Wrappers, and Script Delegation

Use `python -m clematis <subcommand> -- [args]`. The umbrella CLI delegates to wrapper subcommands, which in turn delegate to `clematis.scripts.*` (via `scripts/*.py` shims). **Zero behavior change** is guaranteed by design.

## TL;DR (no behavior change)
- **First‑subcommand anchoring:** the umbrella locates the first subcommand and **prepends any top‑level extras** to the delegated `argv`.
- **Single leading `--`:** if present immediately before delegated args, the wrapper strips **exactly one** sentinel; nothing else is touched.
- **Help interception:** wrapper `-h/--help` is intercepted by the wrapper and **always shows wrapper help** (not the delegated script’s) and includes the phrase **“Delegates to scripts/…”**.
- **Debug breadcrumbs:** enabled by umbrella `--debug` **or** `CLEMATIS_DEBUG=1`; they go to **stderr only** and never alter stdout or exit codes.
- **Top‑level `--version`:** exposed at the umbrella for deterministic help UX.

## Delegation diagram (ASCII)
```
python -m clematis  ──►  wrapper (e.g., rotate-logs)
                        ├─ intercepts -h/--help → prints wrapper help ("Delegates to scripts/…")
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

## Invocation patterns (both orders supported)
```bash
# extras after subcommand with sentinel:
python -m clematis rotate-logs -- --dir ./.logs --dry-run

# extras before subcommand:
python -m clematis --dir ./.logs rotate-logs -- --dry-run
```
> The *single* leading `--` immediately before delegated args is removed by the wrapper.

## Environment flags
- `CLEMATIS_DEBUG=1` — enable stderr breadcrumb:
  ```
  [clematis] delegate -> scripts.rotate_logs argv=['--dir','./.logs','--dry-run']
  ```
  (No stdout/exit changes.)
- `CLEMATIS_NETWORK_BAN=1` — recommended in CI to prevent accidental network.

## Help determinism
- Wrapper help **always** includes the phrase: **“Delegates to scripts/…”**.
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
python -m clematis bench-t4 -- --json --iterations 3 --seed 0
```

**Seed Lance demo (idempotent)**
```bash
python -m clematis seed-lance-demo -- --dry-run
# Running twice should not duplicate entries when executed for real.
```

## Troubleshooting
- **I passed `--` and it still reached the script.**  
  The wrapper strips **one** sentinel by design. If you need a literal `--` for the script, quote it or provide another `--` where appropriate.
- **Direct scripts show a hint.**  
  Calling `scripts/*.py` prints a **single stderr hint** via the shim; behavior is unchanged.

## Guarantees (from PR47)
- **Zero behavior change** codified: order‑agnostic extras, first‑subcommand anchoring, single‑sentinel strip, wrapper help phrase, stderr‑only breadcrumbs. No stdout or exit code changes.