

# Snapshot Schema v1 (v3 Freeze)

**Status:** Frozen for v3
**Schema tag:** `schema_version: "v1"`
**Applies to:** All snapshots written by Clematis3 (legacy single‑JSON and PR34 full/delta)

## TL;DR
- Writer emits a **sidecar** file `<snapshot>.meta` alongside each snapshot body.
- The sidecar carries:
  - `schema_version: "v1"` (frozen)
  - `created_at: <UTC-ISO>Z` (honors `SOURCE_DATE_EPOCH` for reproducible builds)
- The **body** of PR34 snapshots is unchanged (keeps delta math stable).
  Legacy single‑JSON bodies include `graph_schema_version` and a `gel` mirror.
- `inspect_snapshot` validates the schema:
  - Default: **warns to stderr**, prints JSON to stdout, **exit 0**.
  - `--strict`: **fails** on schema issues, **exit 2**.
- Loader understands PR34 full/delta and legacy bodies; latest‑snapshot picker **excludes sidecars**.

---

## What gets written
For a snapshot body at path `…/snapshot-<etag>.full.json` (or `.delta.json`) Clematis3 also writes a sidecar:
- **Sidecar path:** `…/snapshot-<etag>.full.json.meta` (or `.delta.json.meta`)
- **Sidecar JSON** (example):
  ```json
  {
    "schema_version": "v1",
    "created_at": "2025-01-01T00:00:00Z"
  }
  ```

**Determinism:** `created_at` honors `SOURCE_DATE_EPOCH` if set; otherwise current UTC time. Files end with a single `\n`.

### Body invariants
- **PR34 full/delta bodies** do **not** include `schema_version` (intentionally). Identity/delta math remains untouched.
- **Legacy single‑JSON bodies** (e.g., `state_<Agent>.json`) include:
  - `graph_schema_version` ∈ `{ "v1", "v1.0", "v1.1" }`
  - `gel` mirror (graph nodes/edges) so loader can round‑trip state.

---

## Inspecting snapshots
Command:
```bash
python -m clematis.scripts.inspect_snapshot --dir ./.data/snaps --format json
```
Behavior:
- **Default (non‑strict):** If the body lacks `schema_version`, the tool falls back to the sidecar. On mismatch/missing it **prints a warning to stderr** and still prints JSON to stdout; returns **0**.
- **Strict mode:**
  ```bash
  python -m clematis.scripts.inspect_snapshot --dir ./.data/snaps --format json --strict
  ```
  If the body+sidecar do not provide `schema_version == "v1"`, the tool prints an **error to stderr** and returns **2**.

**Exit codes**
| Code | Meaning                                                 |
|-----:|---------------------------------------------------------|
|   0  | Snapshot found; JSON printed; warned if non‑strict.     |
|   2  | Not found / unreadable / schema invalid in `--strict`.  |

---

## Loader & discovery semantics
- **Discovery:** Latest‑snapshot selection ignores sidecars; prefers PR34 `snapshot-*.full.json`/`snapshot-*.delta.json`, then legacy `state_*.json`.
- **PR34 delta:** Loader reconstructs current state from the baseline full + delta (uses writer‑compatible naming).
- **Legacy single‑JSON:** Loader reads graph + `gel` (or synthesizes empty maps) and sets `loaded=True` when successful.

---

## Operator notes
- For **reproducible** archives, set:
  ```bash
  export SOURCE_DATE_EPOCH=1735689600  # 2025-01-01T00:00:00Z
  ```
- For **older/legacy** snapshots that lack sidecars, either re‑write snapshots or use the inspector without `--strict`.
- Warnings and errors from the inspector appear on **stderr**; JSON is printed on **stdout** (useful in scripts).

---

## Compatibility matrix
| Format                   | Body contains `schema_version` | Sidecar used | Inspector (default) | Inspector (`--strict`) |
|--------------------------|-------------------------------:|-------------:|---------------------|------------------------|
| PR34 full/delta          | No                             | Yes          | Warns to stderr     | Must match `"v1"`      |
| Legacy single‑JSON       | No (schema); **Yes** (`graph_schema_version`) | Optional | OK                  | Accepts if sidecar v1  |

---

## FAQ
**Q: Why not put `schema_version` in the PR34 body?**
A: To avoid changing identity and delta payload semantics in v3. The sidecar provides a stable, external schema marker.

**Q: Why `.meta` (no `.json`)?**
A: To avoid globbing collisions with `state_*.json` and similar discovery patterns.

**Q: How do I check the schema quickly?**
```bash
jq -r .schema_version /path/to/snapshot.json.meta
# or
python -m clematis.scripts.inspect_snapshot --dir /path/to/dir --strict --format json
```
