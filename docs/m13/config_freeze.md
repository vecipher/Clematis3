

# Config v1 Freeze (v3)

The v3 configuration schema is **frozen at `version: "v1"`**. This locks the surface so v3 stays stable and identity‑preserving across OSes and Python versions.

---

## TL;DR
- All v3 configs **must** be `version: "v1"` (or omit `version` and it will be injected as `"v1"`).
- **Unknown top‑level keys are rejected** with a clear error.
- The CLI **validates on startup** and exits with code **2** on invalid configs.

---

## What “v1” covers
Allowed top‑level keys in v3 (current list mirrors `configs/validate.py:ALLOWED_TOP`):

- `version` — must be `"v1"`
- `t1`, `t2`, `t3`, `t4`
- `graph`
- `k_surface`, `surface_method`
- `budgets`
- `flags`
- `scheduler`
- `perf`

Anything else at the top level is rejected.

> Note: This list reflects v3’s **stable** surface. Adding/removing/renaming top‑level keys would be a **schema change** and is not permitted in v3.

---

## Operator notes
- The CLI loads your YAML, then calls the validator. If invalid:
  - You’ll see `Config error: ...` on stderr.
  - Process exits with code **2**.
- If `version` is **missing**, the validator injects `"v1"` so you don’t have to write it explicitly.
- If `version` is present and **not** `"v1"`, validation fails with a precise message.

---

## Minimal example (valid)
```yaml
# configs/config.yaml
version: "v1"
t1: {}
t2: {}
t3: {}
t4: {}
# other allowed sections may appear here, e.g. flags, perf, scheduler, graph, budgets, etc.
```

This is also valid (version will be injected as `"v1"`):
```yaml
t1: {}
t2: {}
t3: {}
t4: {}
```

---

## Typical failures
**Wrong version**
```yaml
version: "v999"
t1: {}
t2: {}
t3: {}
t4: {}
```
Error (example):
```
Config error: config.version: must be 'v1', got: 'v999'
```

**Unknown top‑level keys**
```yaml
version: "v1"
t1: {}
t2: {}
t3: {}
t4: {}
experimental: true   # <-- not allowed
```
Error (example):
```
Config error: root: unknown top-level key(s): experimental
```

---

## Migration checklist
1. Ensure `version: "v1"` at the top (or omit it and rely on injection).
2. Remove any keys that are not in the **Allowed top‑level keys** list above.
3. Keep section names and value types consistent with v3 docs and defaults.

---

## Rationale (why freeze now?)
- **Determinism & identity:** A fixed schema avoids accidental drift that breaks golden tests or cross‑OS parity.
- **Operator ergonomics:** Fail fast with explicit messages is better than silent partial acceptance.
- **Upgrade path:** v4+ can introduce `version: "v2"` (or higher) with a deliberate migration guide.

---

## FAQ
**Can v1 add new fields inside existing sections?**
Minor additive fields **within** existing sections may be tolerated if they don’t change defaults/behavior on omission. Any change that affects identity or removes/renames fields should be treated as a schema bump.

**Do I need to set `version: "v1"` explicitly?**
No. If omitted, the validator injects it. Setting it explicitly can be clearer for audits.

**Where is the validator?**
See `configs/validate.py` — exported constant `CONFIG_VERSION = "v1"` and strict checks are enforced there.
