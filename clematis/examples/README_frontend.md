

# Frontend demo bundle (offline)

This directory ships a tiny, deterministic **run bundle** for the offline viewer.

The goal: give operators a known‑good bundle that always loads in the viewer without network, and a one‑command recipe to regenerate it deterministically.

---

## Quick start

You have two ways to open the viewer:

**A) From a repo checkout (build once, then open)**
```bash
npm ci --prefix frontend
npm run --prefix frontend build
make frontend-build
# Then open in a browser:
#   file://…/frontend/dist/index.html
```
Now use **Load** to select the demo bundle below.

**B) From an installed wheel (no Node required)**
Print the viewer and demo‑bundle paths from the installed package:
```bash
python - <<'PY'
from importlib.resources import files
print("viewer:", files("clematis").joinpath("frontend/dist/index.html"))
print("bundle:", files("clematis").joinpath("examples/run_bundles/run_demo_bundle.json"))
PY
```
Open the printed `viewer` path in your browser and **Load** the printed `bundle`.

> The viewer is fully static and offline; there are **no http(s)** requests on load (enforced by tests).

---

## What’s in the demo bundle?

- A single, tiny turn recorded with fixed clocks and a minimal config.
- Canonical JSON (sorted keys, stable separators, LF newlines).
- Small enough to be committed and shipped in both sdist and wheel under:
  ```
  clematis/examples/run_bundles/run_demo_bundle.json
  ```

---

## Regenerate deterministically (maintainers)

Use the console with fixed environment and a fixed clock. This will **overwrite** the committed demo:

```bash
export TZ=UTC PYTHONUTF8=1 PYTHONHASHSEED=0 LC_ALL=C.UTF-8
export SOURCE_DATE_EPOCH=315532800 CLEMATIS_NETWORK_BAN=1

# Produce a tiny one‑turn bundle at a fixed epoch time (1980‑01‑01)
python -m clematis console -- step \
    --now-ms 315532800000 \
    --out clematis/examples/run_bundles/run_demo_bundle.json
```

Recommended sanity checks:

```bash
# Canonical JSON and LF newlines are expected
python -m json.tool clematis/examples/run_bundles/run_demo_bundle.json >/dev/null

# Optional: validate with the viewer locally (offline)
file:///…/frontend/dist/index.html  # then Load the bundle
```

---

## Troubleshooting

- **`frontend/dist/index.html` missing** (from repo): run the build steps in *Quick start (A)*.
- **Viewer opens but nothing loads after selecting the bundle**:
  - Ensure you selected the committed file at `clematis/examples/run_bundles/run_demo_bundle.json` (not a stale path).
  - Check browser console for syntax errors (should be none); the viewer is ESM‑only.
- **CI fails “viewer assets present”**:
  - Packaging expects files under `clematis/frontend/dist/**`. Ensure you ran `make frontend-build` (or that compiled assets are committed) before building wheels.

---

## See also

- **docs/m14/frontend.md** — operator‑grade notes for the offline viewer and console.
- **tests/frontend/test_example_bundle.py** — smoke test that loads this demo bundle over `file://` with no network.
