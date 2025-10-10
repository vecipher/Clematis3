

# Operator Guide (v3)

Single-page, task‑centric runbook for Clematis v3 operators. Use this for day‑to‑day usage; deeper rationale is linked.

> **Status:** M13 — Hardening & Freeze. Config and snapshot schemas are frozen at **v1**. Identity is enforced across **Ubuntu/macOS/Windows** on Python **3.11–3.13**.

---

## 1) Install & sanity check

```bash
python -m pip install -e .          # dev install
python -m clematis --version        # umbrella CLI works
python -m clematis demo -- --help   # wrapper round-trip
```

If something fails, the CLI prints a one‑line, typed error (see §8).

---

## 2) Deterministic environment (identity path)

Set these for identity tests and reproducible runs:

```
TZ=UTC
PYTHONUTF8=1
PYTHONHASHSEED=0
LC_ALL=C.UTF-8
# ZIP requires timestamps >= 1980; use the deterministic minimum (1980-01-01)
SOURCE_DATE_EPOCH=315532800
CLEMATIS_NETWORK_BAN=1
```


CI sets these in identity/build jobs. Add them to your shell or workflow as needed.

**Optional overrides for local runs:**
- `CLEMATIS_LOG_DIR` — directory where stage JSONL logs are written during a step (default: a temp dir chosen by the console if unset).
- `CLEMATIS_SNAPSHOTS_DIR` — directory to scan for the latest snapshot when `--snapshot` is not provided (default: `./.data/snapshots`).

---

## 3) Config v1 (frozen)

- All configs must declare `version: "v1"` (or omit it; the loader injects `"v1"`).
- Unknown **top‑level** keys are rejected with a `ConfigError`.
- Validation runs before subcommands execute.

See **docs/m13/config_freeze.md** for the allowed keys and examples.

**Minimal example:**
```yaml
version: "v1"
t1: { decay: { mode: "exp_floor", rate: 0.6, floor: 0.05 } }
t2: { owner_scope: "any" }
t3: { max_rag_loops: 1 }
t4: { churn_cap: 64 }
flags: { enable_world_memory: false, allow_reflection: false }
```

**Validate:**
```bash
python -m clematis validate -- --config ./configs/config.yaml
```

---

## 4) Snapshots (schema v1)

- Snapshot JSON embeds `schema_version: "v1"` and `created_at` (respects `SOURCE_DATE_EPOCH`).
- `inspect-snapshot` is strict with `--strict` (exit **2** on schema violations). Without `--strict`, it warns but prints JSON and exits **0**.

**Examples:**
```bash
python -m clematis inspect-snapshot -- --path ./snapshots/foo.full.json --strict
python -m clematis inspect-snapshot -- --path ./snapshots/foo.delta.json
```

Details: **docs/m13/snapshot_freeze.md**.

---

## 5) Logs & rotation

- Logs are JSONL with **LF** newlines on all platforms.
- Identity compares with normalized paths (forward slashes) and LF line endings.

**Rotate logs:**
```bash
python -m clematis rotate-logs -- --dir ./.logs --keep 30 --dry-run
```

---

## 6) Reproducible builds (local & CI)

**Local check (Unix/macOS):**
```bash
python -m pip wheel . -w dist_local
shasum -a 256 dist_local/* | sort
```

**Local check (Windows PowerShell):**
```powershell
python -m pip wheel . -w dist_local
Get-ChildItem dist_local | ForEach-Object { certutil -hashfile $_.FullName SHA256 } | Out-String
```

CI runs a cross‑OS matrix and asserts identical hashes. See `.github/workflows/pkg_build.yml`.

Packaging/CLI details: **docs/m8/packaging_cli.md**, **docs/m8/cli.md**.

---

## 7) Support matrix

- **OS:** Ubuntu‑latest, macOS‑latest, Windows‑latest
- **Python:** 3.11, 3.12, 3.13
- **Identity:** byte‑stable outputs across the above (with normalized path/newline handling)

---

## 8) Troubleshooting (typed errors)

Clematis CLIs emit a single, typed line and non‑zero exit on failure. Common cases:

- **`ConfigError`** — Wrong/missing `version`, unknown top‑level key, invalid type.
  *Fix:* set `version: "v1"`, remove unknown keys. See **docs/m13/config_freeze.md**.

- **`SnapshotError`** — Missing/mismatched `schema_version` or corrupt body.
  *Fix:* re‑write snapshot; use `--strict`; see **docs/m13/snapshot_freeze.md**.

- **`IdentityError`** — Golden/identity mismatch (often CRLF or backslashes), unpinned env.
  *Fix:* use env in §2; refresh goldens **only after** fixing root cause.

- **`ParallelError`** — One or more parallel tasks failed; messages include task keys.
  *Fix:* rerun the failing task in isolation; file a minimal repro.

- **`QualityShadowError`** — Shadow/quality path failed (feature off by default).
  *Fix:* keep shadow/quality **off** in v3; file an issue with logs.

- **`ReflectionError`** — Reflection misconfiguration (off by default).
  *Fix:* keep reflection **off** unless explicitly testing.

API & examples: **docs/m13/error_taxonomy.md**.

---

## 9) Windows notes

- Atomic writes use `NamedTemporaryFile(delete=False)` + `os.replace` to avoid “file in use”.
- All file opens use `encoding="utf-8"`; identity normalizes paths to forward slashes.
- Unicode paths are supported; see `tests/io/` on Windows in CI.

---

## 10) FAQ

**Q: Which CLI entrypoint?**
Use the umbrella wrapper: `python -m clematis <subcmd> -- <args>`. Subcommands delegate to `scripts/*.py`.

**Q: Can I enable networked operations?**
CI forbids network (`CLEMATIS_NETWORK_BAN=1`). For local experiments, document deviations; identity guarantees only apply under §2.


**Q: Manpages?**
Generated deterministically by `scripts/gen_manpages.py` into `man/`. Install via your package manager or point `MANPATH` to the repo.

**Q: How do I step a run locally and compare outputs?**
Use the local console (see §11):
```bash
# Status against latest snapshot (or set CLEMATIS_SNAPSHOTS_DIR)
python -m clematis console -- status

# Reset to a specific snapshot
python -m clematis console -- reset --snapshot ./.data/snapshots/snap_000001.json

# Deterministic single-step (defaults to now_ms = SOURCE_DATE_EPOCH*1000 if omitted)
TZ=UTC PYTHONHASHSEED=0 SOURCE_DATE_EPOCH=315532800 CLEMATIS_NETWORK_BAN=1 \
python -m clematis console -- step --now-ms 315532800000 --out ./.data/run_step.json

# Compare two bundles (exit 1 on differences)
python -m clematis console -- compare --a runA.json --b runB.json
```
The console prints a warning if the env in §2 is not set as recommended.


---

## 11) Local console (deterministic stepping & compare)

A minimal, deterministic console for operators to drive the orchestrator locally and inspect/compare results. It is offline by default and preserves identity when the environment in §2 is set.

**Subcommands**
- `status` — prints scheduler/budgets summary (uses latest snapshot by default).
- `reset [--snapshot PATH]` — loads a snapshot into state. If `--snapshot` is omitted, the console uses the latest from `CLEMATIS_SNAPSHOTS_DIR` (or `./.data/snapshots`).
- `step [--now-ms N] [--out BUNDLE.json]` — runs exactly one turn. If `--now-ms` is omitted, it defaults to `SOURCE_DATE_EPOCH * 1000`.
- `compare --a A.json --b B.json` — diffs two run bundles (counts per stage, snapshot/meta keys). **Exit 1** when differences are detected.

**Determinism**
- Set the env in §2 (`TZ=UTC`, `PYTHONHASHSEED=0`, `SOURCE_DATE_EPOCH=315532800`, `CLEMATIS_NETWORK_BAN=1`).
- The console warns when these do not match.
- Logs are written to `CLEMATIS_LOG_DIR` if set; otherwise the console uses a temporary directory and cleans it up.
- T2’s exact-semantic recency window (controlled by `t2.exact_recent_days`) uses the orchestrator’s notion of “now”. Supplying `--now-ms` (or exporting `SOURCE_DATE_EPOCH`) keeps the in-memory and LanceDB backends aligned for log replays and golden comparisons.

**Examples**
```bash
# Status (latest snapshot)
python -m clematis console -- status

# One turn with fixed time; write run bundle
TZ=UTC PYTHONHASHSEED=0 SOURCE_DATE_EPOCH=315532800 CLEMATIS_NETWORK_BAN=1 \
python -m clematis console -- step --now-ms 315532800000 --out ./.data/run_step.json

# Diff bundles
python -m clematis console -- compare --a runA.json --b runB.json
```

**Exit codes**
- `0` — success / bundles equal.
- `1` — `compare` found differences.
- `2` — adapter or usage error (e.g., missing snapshot, orchestrator not available).
