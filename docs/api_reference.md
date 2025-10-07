

# Clematis v3 — Public API Reference

This page freezes the **v3** public surface. Anything not listed here is **internal and subject to change**.
The only supported import roots in v3 are:

- `clematis`
- `clematis.errors`

The surface below is locked for all **v3.x** releases. Additions (new names) may occur in **v3** only if they are strictly additive and deterministic; removals/renames are deferred to **v4**.

---

## Top‑level module: `clematis`

```py
from clematis import (
    __version__,          # e.g. "0.9.0"
    CONFIG_VERSION,       # "v1"
    SCHEMA_VERSION,       # "v1"
    validate_config,
    validate_config_verbose,
    validate_config_api,
    errors,               # submodule: clematis.errors
)
```

### Symbols

- `__version__: str`
  Package version string (semver). Stable across platforms for a given build.

- `CONFIG_VERSION: Literal["v1"]`
  Frozen configuration schema version for v3. See the Config v1 freeze notes.

- `SCHEMA_VERSION: Literal["v1"]`
  Frozen snapshot schema version for v3 (embedded in snapshots and enforced by CLI tools).

- `validate_config(config: Mapping | str | bytes) -> dict`
  Validates a config (object or YAML/JSON text). Returns a **normalized** dict.
  **Raises**: `clematis.errors.ConfigError` on invalid input (unknown keys, bad types, out‑of‑range values).
  **Determinism**: error messages and normalization are stable under the v1 contract.

- `validate_config_verbose(config: Mapping | str | bytes) -> tuple[dict, list[str]]`
  Like `validate_config`, but also returns a list of human‑readable notes/warnings.

- `validate_config_api(config: Mapping) -> dict`
  A minimal variant intended for programmatic callers. Same validation rules; input must already be a mapping.

- `errors: ModuleType`
  Convenience re‑export of the `clematis.errors` submodule.

> ⚠️ **Star imports**: `from clematis import *` is supported only to expose the names above. It does **not** expose internal modules.

---

## Submodule: `clematis.errors`

Typed error hierarchy used by public entry points and CLIs.

```py
from clematis.errors import (
    ClematisError,       # base class
    ConfigError,
    SnapshotError,
    IdentityError,
    QualityShadowError,
    ReflectionError,
    ParallelError,       # alias preserved for backward compatibility
    CLIError,
    format_error,        # helper to render user‑facing error lines
)
```

### Semantics

- All public validators and CLIs raise subclasses of `ClematisError`.
- `ConfigError` is used for configuration parsing/validation failures.
- `SnapshotError` covers snapshot header/schema/version issues.
- `IdentityError` covers identity/normalization failures.
- `QualityShadowError`, `ReflectionError` cover their respective features.
- `ParallelError` remains as a named subclass for compatibility.
- `CLIError` is used by script entry points for consistent non‑zero exits.
- `format_error(e: BaseException) -> str` formats a single deterministic line suitable for operator logs/CLI output.

---

## Stability & compatibility policy (v3)

- The names listed above are **frozen** for v3.x.
- Function signatures, return types, and raised exception types are stable for v3.x.
- Anything **not** listed (including subpackages like `clematis.engine`, `clematis.io`, etc.) is internal and may change without notice.
- Breaking changes, renames, or removals will be scheduled for **v4**.

---

## Usage examples

Validate a config and handle typed errors:

```py
from clematis import validate_config, errors

try:
    cfg = validate_config({"perf": {"enabled": False}, "t2": {"quality": {"enabled": False}}, "t4": {"cache": {"namespaces": []}}})
except errors.ConfigError as e:
    print(errors.format_error(e))  # deterministic, operator‑friendly
    raise
```

Surface‑stable metadata:

```py
from clematis import __version__, CONFIG_VERSION, SCHEMA_VERSION
print(__version__, CONFIG_VERSION, SCHEMA_VERSION)  # e.g. "0.9.0 v1 v1"
```

---

### Change log discipline

If we ever extend this surface in v3, the addition will be:
1) documented here,
2) covered by tests, and
3) called out in the CHANGELOG.

For removals/renames, expect a deprecation period in late v3, with the actual change in v4.
