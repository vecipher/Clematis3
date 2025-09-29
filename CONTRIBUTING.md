

# Contributing to Clematis

This repo aims for **deterministic**, **offline‑friendly** development and CI. Keep changes reproducible and document any behavior that affects CLI UX.

## Dev setup

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -U pip
# install in editable mode with test extras (provides shtab for completion goldens)
python -m pip install -e '.[test]'
```

## Development hygiene (M8)

Set up lint and type gates locally so your PR passes the CI gates from PR62.

- Install dev tools and hooks:

  ```bash
  python -m pip install -e '.[dev]'
  pre-commit install
  ```

- Before pushing, ensure a clean run:

  ```bash
  pre-commit run -a
  ```

- Local CI parity (what the workflow runs):

  ```bash
  # Formatting gate
  ruff format --check .
  # Repo-wide safety lint (fatal error set only)
  ruff check --force-exclude --select E9,F63,F7,F82 .
  # Strict lint for CLI package
  ruff check --force-exclude clematis/cli
  # Types on CLI only
  mypy --config-file pyproject.toml
  ```

**Policy**
- **Types:** MyPy runs on `clematis/cli/**` only (we’ll expand later).
- **Lint:** Repo-wide “safety” (E9,F63,F7,F82), and **strict** on `clematis/cli/**`.
- **Format:** Ruff format gate is enforced.

## Running tests

Most tests are offline by default:
```bash
pytest -q -m "not manual"
```

## Updating CLI golden snapshots (PR59 and onward)

We lock the CLI UX via golden files for `--help` and shell completions.

**Help output goldens**
```bash
export BLESS=1
pytest -q tests/cli/test_cli_help_golden.py
unset BLESS
```
This generates/refreshes files in `tests/cli/goldens/help/`.

**Completion script goldens**
```bash
python -m pip install -e '.[test]'  # ensures shtab is available
export BLESS=1
pytest -q tests/cli/test_cli_completions_golden.py   # currently bash & zsh only
unset BLESS
```
This writes `tests/cli/goldens/completions/{bash,zsh}.txt`.

Notes:
- Width/line‑wrap, dates, and versions are normalized out; substantive text changes will still fail tests.
- **Fish completions are intentionally omitted** for now (current `shtab` pin lacks fish support). We’ll add fish when supported and then bless new goldens.
- Only bless when you intend a user‑visible CLI change; mention the change and the fixture refresh in your PR description.

## Troubleshooting

- **zsh eats extras**: use `python -m pip install -e '.[test]'` (quotes) or `.\[test]`.
- **Completion tests skipped**: ensure `shtab` is installed (provided by the `[test]` extra).
- **Help diffs vary by terminal**: CI sets `COLUMNS=80`, `LC_ALL=C`, `PYTHONHASHSEED=0`; you can export the same locally if needed.

## Git workflow (house rules)

- **Branch‑first** always; **no direct pushes to `main`**. Open a PR for all changes.
- Main is protected with required checks; prefer **squash merges**.
- Releases are tag‑driven; OCI images/SBOMs attach on tagged releases.
- Keep PR descriptions factual and include any golden fixture updates (what changed and why).

- Before opening a PR: run `pre-commit run -a` locally; PRs must pass **Lint & Types** (Ruff format repo-wide, Ruff safety repo-wide, Ruff strict on CLI, MyPy on CLI).
