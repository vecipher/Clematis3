# Clematis v3 Healthcheck

## Repository Reality & Dossier Update
- Version metadata now consistently reports **0.10.3** across `pyproject.toml`, `clematis/VERSION`, `clematis.egg-info/PKG-INFO`, CLI man pages, and help goldens (`tests/cli/goldens/help/top.txt`).
- Deterministic turn loop (T1→T4) and staging remain intact; identity CI continues to rely on fixed env vars and normalized LF logs. GEL (`graph.enabled`) and deterministic parallelism (`perf.parallel`) are present but default **OFF**.
- Config schema v1 is enforced via `configs/validate.py`; snapshots remain schema v1 with strict inspector semantics. CLI (`python -m clematis …`) still emits one-line typed errors.
- Default config (`configs/config.yaml`) now mirrors the identity baseline: `t3.allow_reflection=false`, `t3.backend=rulebased`, LLM fixtures disabled, and `flags.allow_reflection=false`. Optional features stay opt-in through overlays/examples.

## Observed Gaps (carried into v4 planning)
- Native kernels, GEL-informed nudge planner, retrieval quality defaults, and CLI/frontend evolution remain unimplemented; existing scaffolding is deterministic but dormant.

## Hardening & Maintenance Steps Completed
- Audited defaults and flipped identity gates OFF by editing `configs/config.yaml` (reflection + LLM fixtures now opt-in; added explicit `version: "v1"` header).
- Updated `clematis/engine/types.Config.flags` to default both `enable_world_memory` and `allow_reflection` to `False` to keep in-process defaults consistent with config files and docs.
- Validated the adjusted config with `python3 -m clematis validate -- --config configs/config.yaml` (passes with reflection/quality gates reported as disabled).
- Bumped project version markers to 0.10.3 and refreshed CLI artifacts: updated `pyproject.toml`, `clematis/VERSION`, `clematis.egg-info/PKG-INFO`, `man/*.1`, and `tests/cli/goldens/help/top.txt`.
- Identity marker suite verified (`pytest -q -m identity`), and full test suite passes locally (`pytest -q`).
- Built wheels into `dist_local/` and captured deterministic hashes via `shasum -a 256 dist_local/* | sort` as part of the packaging parity check.
- Removed `ValueError` inheritance from typed errors, updated validators/scripts/tests to catch `ConfigError`/`SnapshotError`, and refreshed docs to match the cleaned taxonomy.
- Chat CLI `[wipe]` now purges the configured LanceDB store (when enabled) and reinitialises the memory index so subsequent `[seed]` runs repopulate the persistent backend with fresh embeddings.

## Recommended Next Actions
- Re-run the identity and packaging matrices once the version/doc alignment is settled, ensuring the refreshed defaults keep byte-identical outputs.
- Keep overlays/examples (e.g., `examples/reflection/*.yaml`, GEL demos) up to date with the new defaults so operators enable gates explicitly.
- For v4 kickoff, formalize the migration doc covering schema v2, error taxonomy cleanup, native acceleration strategy, and GEL-driven planner roadmap.

## Suggested Verification Commands
- Run in order: `python3 -m clematis validate -- --config configs/config.yaml`
- `pytest -q -m identity` *(sequential vs parallel/disabled-path matches)*
- `pytest -q tests/config/test_validate_reflection.py` *(reflection gate defaults + fixtures contract)*
- `python3 -m pip wheel . -w dist_local` **then** `shasum -a 256 dist_local/* | sort`
- `python3 -m clematis --help > /tmp/help.txt` and compare against `tests/cli/goldens/help/top.txt`
