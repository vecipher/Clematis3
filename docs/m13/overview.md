# M13 — Hardening & Freeze (v3)

**Status:** Frozen on **2025-10-06 SGT**. v3 is a stable, reproducible baseline. No breaking changes will be made to the v1 configuration or snapshot schemas before v4.

> TL;DR: Config **v1** and Snapshot **v1** are locked. Identity is cross-OS. Builds are reproducible. Windows file I/O is atomic. Errors are typed. Shadow/perf diagnostics are segregated under `logs/perf/` and excluded from identity. Everything else is conservative and default-OFF.

---

## 1) What is locked in M13

- **Config schema: v1 (frozen).**
  - `version: "v1"` is required (or injected).
  - Unknown top-level keys are rejected.
  - See: `docs/m13/config_freeze.md`.

- **Snapshot schema: v1 (frozen).**
  - Sidecar `.meta` carries `schema_version: "v1"` and `created_at` (honors `SOURCE_DATE_EPOCH`).
  - Inspector is strict with non-zero exit on mismatch (opt-out flag available).
  - See: `docs/m13/snapshot_freeze.md`.

- **Identity guarantees (cross-OS).**
  - Deterministic env (required for identity & CI):
    ```
    TZ=UTC
    PYTHONUTF8=1
    PYTHONHASHSEED=0
    LC_ALL=C.UTF-8
    SOURCE_DATE_EPOCH=315532800
    ```
  - **Canonicalization:** LF newlines on all text; forward-slash path separators in logs; canonical JSON (`sort_keys=True`, compact separators).
  - **Shadow diagnostics segregation:** anything under `logs/perf/` or files ending with `-perf.jsonl` are ignored by identity comparisons; canonical logs remain the stage logs (`t1.jsonl`, `t2.jsonl`, `t4.jsonl`, `apply.jsonl`, `turn.jsonl`; `scheduler.jsonl` where applicable).
  - Goldens normalized once post-M13; thereafter byte-stable across OS.

- **Reproducible builds.**
  - Wheels/sdists built on Ubuntu/macOS/Windows are byte-identical; hashes compared in CI.
  - SBOM/attestations attached on tag builds.

- **Windows-safe file I/O.**
  - All temp→final writes use `NamedTemporaryFile(delete=False)` + `os.replace` with retry and best-effort fsync.
  - Text writes enforce `encoding="utf-8", newline="\n"`.

- **Typed error taxonomy (operator-facing).**
  - `ConfigError`, `SnapshotError`, `IdentityError`, `ParallelError`, `QualityShadowError`, `ReflectionError`, `CLIError`.
  - One-line CLI messages with deterministic wording.
  - See: `docs/m13/error_taxonomy.md`.

---

## 2) What is **out of scope** for v3

- **M12 native kernel acceleration**: **OFF** (deferred; no performance changes in v3 stable).
- **Quality/Perf “shadow” paths**: shipped but **OFF by default**; do not affect identity when off.
- **Reflection**: available but **OFF by default** for operators; keep disabled unless explicitly testing.
- **No breaking schema/API changes**: no renames, no required new config keys, no CLI contract changes that affect help or exit codes.

---

## 3) Support matrix (identity path)

- **OS**: `ubuntu-latest`, `macos-latest`, `windows-latest`
- **Python**: `3.11`, `3.12`, `3.13`
- **Identity**: byte-for-byte stable outputs across the above matrix, with newline and path normalization.

---

## 4) Compatibility & EOL stance for v3

- **Compatibility promise (v3.x):**
  - No changes to **Config v1** and **Snapshot v1** semantics.
  - Public CLI and error contracts remain stable (**deterministic help text is locked**; goldens are recorded on Linux + Python 3.13 with `COLUMNS=80`).
  - Any optional features must default to **OFF** and must not change identity when OFF.

- **EOL policy for v3:**
  - v3 receives **critical fixes only** until **v4 GA + 90 days**.
  - After that window, the v3 branch closes to feature work; security updates may be cherry-picked at maintainer discretion.

---

## 5) Operator quicklinks

- **Operator Guide (single page):** `docs/operator-guide.md`
- **Config freeze details:** `docs/m13/config_freeze.md`
- **Snapshot freeze details:** `docs/m13/snapshot_freeze.md`
- **Error taxonomy (typed):** `docs/m13/error_taxonomy.md`
- **Packaging & CLI:** `docs/m8/packaging_cli.md`, `docs/m8/cli.md`

---

## 6) Upgrade notes (post-M13)

- If you have goldens recorded before M13’s newline/path normalization, re-record once using the provided normalization script; lock after that.
- Ensure CI uses the **deterministic env** block above (notably `SOURCE_DATE_EPOCH=315532800`).
- If you enable perf/quality flags, expect extra logs under `logs/perf/`. These are non-canonical diagnostics and are excluded from identity gates.

---

## 7) Rationale

M13 makes v3 a **boring**, dependable base: identity stays byte-stable; operators have crisp runbooks; and any future work (v4+) can innovate without regressing determinism or tooling.
