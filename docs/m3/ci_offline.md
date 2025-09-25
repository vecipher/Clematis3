# CI Offline Discipline (M3‑09)

**Goal:** Keep CI runs deterministic and offline while allowing local LLM smokes.

---

## CI rules

1. **Manual tests are skipped in CI**
   - We register a `manual` pytest marker and run CI with `-m "not manual"`.
   - Local manual runs are opt‑in via `CLEMATIS_LLM_SMOKE=1`.

2. **Hard network ban in CI**
   - `tests/conftest.py` installs a session fixture that blocks outbound sockets when `CLEMATIS_NETWORK_BAN=1`.
   - The workflow sets `CLEMATIS_NETWORK_BAN=1` and `CI=true`.

3. **Fixture‑only LLM in CI**
   - In CI (`CI=true`), `_get_llm_adapter_from_cfg()` requires `provider == "fixture"` and a valid fixture file.
   - Any other provider (e.g., `ollama`) or disabled fixtures fail fast with a clear error.

---

## Environment flags

- `CI=true` – signals CI context and enables stricter guards.
- `CLEMATIS_NETWORK_BAN=1` – bans sockets in tests.
- `CLEMATIS_LLM_SMOKE=1` – opt‑in to run manual LLM smoke tests **locally**.

---

## Commands

CI‑equivalent local run (offline):
```bash
CI=true CLEMATIS_NETWORK_BAN=1 pytest -q -m "not manual"
```

Manual smoke (local only):
```bash
pytest -q -m manual                            # shows SKIPPED by default
CLEMATIS_LLM_SMOKE=1 pytest -q -m manual       # opt-in to run
```

---

## Files & contracts

- **Workflow:** `.github/workflows/tests.yml` – runs `pytest -m "not manual"` with `CLEMATIS_NETWORK_BAN=1`, `CI=true`.
- **Socket ban:** `tests/conftest.py` – blocks network when `CLEMATIS_NETWORK_BAN=1`.
- **CI guard tests:** `tests/llm/test_ci_guard.py` – asserts safe fallback and explicit error logs in CI.
- **Manual marker test:** `tests/llm/test_manual_smoke_marker.py` – `@pytest.mark.manual`, gated by `CLEMATIS_LLM_SMOKE=1`.
- **Fixture mapping:** `fixtures/llm/qwen_small.jsonl` – deterministic mapping for planner prompt.
- **Runtime guard:** `clematis/engine/stages/t3.py` – in CI, requires `provider=fixture` with a valid fixture path.

---

## Common failures & fixes

- **"Network calls are banned in CI"**
  - A test attempted a socket call. Disable network usage or mark it `manual`.

- **"CI requires fixture provider; got ollama"**
  - In CI, set `t3.llm.provider: fixture` and ensure `fixtures.enabled: true`.

- **"CI fixture path not found: …"**
  - Fix `t3.llm.fixtures.path` or include the JSONL in the repo/package data.

---

## Why this matters

- Guarantees reproducible CI by eliminating nondeterministic LLM/network behavior.
- Keeps the `rulebased` default path byte‑for‑byte identical, while enabling local LLM smokes.
