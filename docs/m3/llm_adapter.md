

# M3 — LLM Adapter (Scaffolding & Fixtures)

**Status:** M3‑07 shipped (adapter scaffolding).  
**Default posture:** OFF (`t3.backend=rulebased`).  
**Goal:** Provide a pluggable, deterministic LLM layer with fixtures for CI. Runtime wiring and prompt contract arrive in M3‑08/M3‑10.

---

## What shipped in M3‑07

Components (paths are relative to repo root):

- **Config surface** — `configs/config.yaml`
  ```yaml
  t3:
    backend: "rulebased"      # "rulebased" | "llm"
    llm:
      provider: "fixture"      # "fixture" | "ollama"
      model: "qwen2:4b-instruct-q4_K_M"
      endpoint: "http://localhost:11434/api/generate"
      max_tokens: 256
      temp: 0.2
      timeout_ms: 10000
      fixtures:
        enabled: true
        path: "fixtures/llm/qwen_small.jsonl"
  ```
- **Validator** — `configs/validate.py`
  - Normalizes and bounds `t3.llm.*`
  - Unknown‑key checks for `t3.llm` and `t3.llm.fixtures`
  - Stable API for tests/tools: `validate_config_api(cfg) -> (ok, errs, cfg_or_none)`
- **Adapter layer** — `clematis/adapters/llm.py`
  - `FixtureLLMAdapter` — offline JSONL fixture adapter
  - `LLMAdapterError` — uniform error surface
  - `_prompt_hash()` / `_canon_prompt()` — deterministic hash helpers
  - (Runtime `QwenLLMAdapter` exists but is **not wired** in M3‑07)
- **Fixtures (data)** — `fixtures/llm/qwen_small.jsonl`
  - JSONL mapping: `prompt_hash → completion` (one JSON object per line)
- **Tests (offline)** — `tests/llm/`
  - `test_adapter_fixture.py` — round‑trip + token clipping
  - `test_adapter_errors.py` — missing file / bad JSONL / missing mapping / CRLF canon
  - `test_config_validation.py` — provider bounds, defaults, key checks

**Identity guarantee:** With defaults, outputs are byte‑for‑byte identical to M7 (no runtime LLM use).

---

## Fixture format (JSONL)

Each line is a standalone JSON object:

```json
{
  "prompt_hash": "<sha256 of canonicalized prompt>",
  "completion": "<raw text returned by adapter>",
  "meta": { "model": "qwen2:4b-instruct-q4_K_M", "note": "optional info" },
  "prompt_preview": "first ~120 chars of the prompt (for humans)"
}
```

Notes:
- `completion` is **raw text**. For planner usage later (M3‑10), prefer a JSON string with keys `plan` and `rationale`, e.g.:
  ```json
  "{\"plan\":[\"step1\",\"step2\"],\"rationale\":\"why\"}"
  ```
- Map is exact: a different prompt (even by whitespace) → different hash.
- We normalize only newlines (`\r\n`/`\r` → `\n`) before hashing.

Example entry:
```json
{"prompt_hash":"6ee0071ff0f58ff850f48672cac37eb24d6cebb3e250f2a6890ff672e1f5073f","completion":"{\"plan\":[\"a\"],\"rationale\":\"r\"}","meta":{"model":"qwen2:4b-instruct-q4_K_M","note":"simple sanity mapping for early tests"},"prompt_preview":"SYSTEM: Return ONLY valid JSON. STATE: {\"turn\":1}"}
```

The repo includes `fixtures/llm/qwen_small.jsonl` with:
- one working sanity mapping; and
- one placeholder line with `prompt_hash: "TBD_UPDATE_ON_M3_08"` to be updated once the real prompt shape lands.

---

## Computing a prompt hash

You can compute the SHA‑256 for any prompt via the helper in `clematis/adapters/llm.py`:

```bash
python - <<'PY'
from clematis.adapters.llm import _prompt_hash
prompt = """SYSTEM: Return ONLY valid JSON matching keys {plan: list[str], rationale: str}. No prose. No markdown. No trailing commas.
STATE: {"turn":1,"agent":"demo"}
SCHEMA: {"plan":["string"], "rationale":"string"}
USER: Propose up to 4 next steps as short strings; include a brief rationale."""
print(_prompt_hash(prompt))
PY
```

### (Optional) small helper script
If desired, add `tools/print_prompt_hash.py`:
```python
#!/usr/bin/env python3
import sys
from clematis.adapters.llm import _prompt_hash
print(_prompt_hash(sys.stdin.read()))
```
Usage:
```bash
python tools/print_prompt_hash.py <<'EOF'
SYSTEM: Return ONLY valid JSON...
EOF
```

---

## Local vs CI behavior

- **CI:** must remain offline & deterministic
  - Use `provider: fixture`
  - Keep network disabled; tests do not import or call network code
- **Local (M3‑08+ only):** switch to `provider: ollama` to smoke Qwen (manual)
  - Requires the M3‑08 wiring (`make_dialog_bundle()` and adapter selection)
  - Example (once M3‑08 lands):
    ```bash
    ollama pull qwen2:4b-instruct-q4_K_M
    python scripts/llm_smoke.py
    ```

---

## Running tests

LLM slice only:
```bash
pytest -q tests/llm
```

Full suite (should remain identical to M7 with backend=rulebased):
```bash
pytest -q
```

---

## Troubleshooting

- **`LLMAdapterError: Fixture file not found`**  
  Check `t3.llm.fixtures.path` or ensure the JSONL exists.
- **`LLMAdapterError: Bad fixture JSONL at line N`**  
  Ensure each line is valid JSON; no trailing commas; UTF‑8 encoding.
- **`LLMAdapterError: No fixture for prompt hash`**  
  The prompt string used at runtime must match the one hashed in the JSONL (after newline normalization).
- **Config validation failures (ValueError)**  
  - `t3.llm.provider` must be one of `{fixture, ollama}`  
  - `t3.llm.temp` in `[0, 1]`  
  - `t3.llm.max_tokens ≥ 1`  
  - `t3.llm.timeout_ms ≥ 1`  
  - `t3.llm.fixtures.path` non‑empty

---

## Roadmap alignment

- **M3‑07 (this doc):** adapter scaffolding + fixtures + tests; defaults OFF.
- **M3‑08:** runtime wiring (`t3.backend="llm"` path), `make_dialog_bundle()` prompt; update fixture hash for real prompt.
- **M3‑09:** CI guards (manual smoke marker, network ban).
- **M3‑10:** JSON schema + sanitizer (PLANNER_V1); reject unsafe output; no state mutation on invalid JSON.