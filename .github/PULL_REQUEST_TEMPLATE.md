

## Summary
Briefly explain **what** this PR changes and **why**. Include user impact (if any).

## Changes
- Bullet the notable changes (configs, CI, docs, code). Keep it terse.

## Testing / Verification
Explain how you verified the change locally. Paste key logs or attach screenshots if helpful.

## Risk & Rollback
- Risk level: low / medium / high
- Rollback plan: how to revert safely if needed

## Related
Link issues/PRs: e.g. Fixes #<ID>, Ref #<ID>.

---
## Pre‑flight Checklist
_Run locally before opening the PR. Check only the items that apply._

**Hygiene**
- [ ] `pre-commit run -a` is **clean**

**Formatting**
- [ ] `ruff format --check .` is clean
      _(CI may check a subset; keep the repo formatted to avoid drift.)_

**Lint**
- [ ] **Repo safety set**: `ruff check --force-exclude --select E9,F63,F7,F82 .` is clean
- [ ] **CLI strict (if touching CLI)**: `ruff check --force-exclude clematis/cli` is clean

**Types**
- [ ] **CLI only (if touching CLI)**: `mypy --config-file pyproject.toml` is clean

**Build & Smoke** (as relevant)
- [ ] Wheel builds: `python -m build`
- [ ] Examples smoke: `python scripts/examples_smoke.py --all --no-check-traces`

**Docs & Changelog** (as relevant)
- [ ] Updated docs/dev notes (e.g. `docs/m8/packaging_cli.md`, `CONTRIBUTING.md`)
- [ ] Updated `CHANGELOG.MD`

**Runtime behavior**
- [ ] No runtime behavior change (or explicitly documented above)
