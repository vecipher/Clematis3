

# Error Taxonomy (v3)

Operator-facing errors are standardized under `clematis.errors`. These classes give short, consistent messages and stable program behavior (typed exceptions + exit codes).

## Classes & guidance

| Class                | Typical cause                                                      | Operator action                                                                 |
|----------------------|--------------------------------------------------------------------|---------------------------------------------------------------------------------|
| `ConfigError`        | Wrong `version`; unknown top‑level keys; bad types/shape          | Fix the config (see **Config v1 Freeze**). Remove unknown keys; set `version: "v1"`. |
| `SnapshotError`      | Missing/mismatched `schema_version`; corrupt/malformed header     | Re‑generate snapshot or run inspector with `--no-strict` to view details. See **Snapshot Schema v1**. |
| `IdentityError`      | Golden/identity mismatch; path/line‑ending divergence             | Re‑run identity; inspect normalized diffs; refresh goldens only after root cause is fixed. |
| `QualityShadowError` | Shadow tracing/adapter failure (feature is off by default in v3)  | Keep feature off in v3; file an issue with logs if reproducible.               |
| `ReflectionError`    | Reflection session/setup failure (feature is off by default)      | Keep reflection off in v3; file an issue with minimal repro.                   |
| `ParallelError`      | Parallel task(s) raised; error list aggregated deterministically  | Read aggregated messages; identify failing task key(s); rerun minimal repro.   |
| `CLIError`           | Generic CLI wrapper for unexpected failures                        | Re‑run with `-v`/logs; file an issue if persistent.                             |

## Message format

Use `format_error(e)` to produce uniform, operator‑friendly text:

```py
from clematis.errors import format_error, ConfigError
print(format_error(ConfigError("unknown top-level key: foo")))
# -> "ConfigError: unknown top-level key: foo"
```

When an error has an empty message, `format_error(e)` renders only the class name (e.g., `SnapshotError`).

## Where these appear

- **Config load** (`clematis/cli/_config.py`): raises `ConfigError` and exits **2** on invalid configs.
- **Snapshot inspector** (`clematis/scripts/inspect_snapshot.py`): raises `SnapshotError` on schema issues; `--no-strict` downgrades to a warning.
- **Parallel execution** (`clematis.engine.util.parallel`): `ParallelError` aggregates task failures and now subclasses `ClematisError`.

## References

- **Config v1 Freeze:** `docs/m13/config_freeze.md`
- **Snapshot Schema v1:** `docs/m13/snapshot_freeze.md`

---

**Notes**
- Error messages are deterministic and concise; avoid stack traces in operator paths.
- Identity is preserved in v3; enabling quality/perf/reflection features may change behavior and should remain off by default.
- Typed errors inherit only from `ClematisError`; update callers to catch the specific subclasses instead of `ValueError`.
