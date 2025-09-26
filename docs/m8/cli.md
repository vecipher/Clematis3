# Clematis Umbrella CLI

Use `python -m clematis <subcommand> -- [args]`. The CLI delegates to `scripts/*.py` without changing behavior.

## Subcommands (delegates)
- `rotate-logs` → `scripts/rotate_logs.py`
- `inspect-snapshot` → `scripts/inspect_snapshot.py`
- `bench-t4` → `scripts/bench_t4.py`
- `seed-lance-demo` → `scripts/seed_lance_demo.py`

### Pass-through rules
- **Order preserved:** `extras + ns.args`
- **Leading `--`:** strip exactly one; nothing else is touched.
- **`--debug`:** stderr-only breadcrumb; never forwarded. Or set `CLEMATIS_DEBUG=1`.

### Cookbook :tongue:
- Rotate logs (dry-run):  
  `python -m clematis rotate-logs -- --dir ./.logs --dry-run`
- Inspect snapshot (table):  
  `python -m clematis inspect-snapshot -- ./snapshots/last.snap`
- Bench T4 (JSON):  
  `python -m clematis bench-t4 -- --json`
- Seed Lance demo (idempotent):  
  `python -m clematis seed-lance-demo -- --dir ./data/lance_demo`

### Troubleshooting
- Prefer `python -m clematis`; direct scripts print a deprecation hint to stderr (behavior unchanged).  [oai_citation:2‡GitHub](https://github.com/vecipher/Clematis3/issues/78)
- If needed, set `CLEMATIS_DEBUG=1` to see delegation breadcrumbs.
- If needed, bother vecipher
- If needed, cry