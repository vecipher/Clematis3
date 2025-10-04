# M10 — Reflection Sessions (Deterministic, Gated)

**Status:** complete and gated. The configuration surface has been present since **v0.9.0a1** and remains **disabled by default**; the feature was finalized across PR77 and PR80–PR90. Enabling reflection must not change the identity path for prior milestones when the gate is off. Landed: PR77 (config), PR80–PR83 (writer/budgets/tests), PR84 (fixtures‑only LLM backend), PR85 (planner flag & wiring), PR86 (telemetry & trace), **PR87 (microbench & examples)**, **PR88 (optional CI smoke)**, **PR89 (docs)**, **PR90 (goldens/identity maintenance)**.

## 1) Scope & Invariants
- Add a post-turn reflection step that can **summarize** the turn and **persist** small memory entries for future retrieval.
- **Deterministic only.** No RNG, no network; all outputs must be byte-identical given the same inputs.
- **Disabled-path identity preserved.** If `t3.allow_reflection=false` or the plan does not request reflection, the run produces no new artifacts, and all identity logs are unchanged.
- Reflection must **not influence the current turn**; outputs are only visible to future retrieval.

## 2) Where it runs in the pipeline
`T1 → T2 → T3 (plan/speak) → T4 (gate/apply) → **Reflection** → end-of-turn`

- The orchestrator invokes reflection **after Apply**. Any memory writes happen after T4.
- **Gate conditions:** reflection runs only if **both** are true:
  1. `t3.allow_reflection == true`, and
  2. the planner requests it (`plan.reflection == true`; for the LLM planner this flag is carried via policy state).
  3. not in dry‑run mode (the orchestrator’s `_dry_run` is **false**).
- Under parallel agent batches, reflection logs are staged and flushed in deterministic order (stage ordinal reserved: `STAGE_ORD["t3_reflection.jsonl"]=10`).

## 3) Configuration (added in v0.9.0a1)

```yaml
# configs/config.yaml
t3:
  allow_reflection: false       # gate (keep false for identity)
  reflection:
    backend: rulebased          # deterministic summariser; "llm" requires fixtures (PR84)
    summary_tokens: 128         # whitespace-token cap for output
    embed: true                 # if true, embed summary deterministically
    log: true                   # write t3_reflection.jsonl (not an identity log)
    topk_snippets: 3            # number of retrieval snippets passed to summariser

scheduler:
  budgets:
    time_ms_reflection: 6000    # wall budget for reflection step
    ops_reflection: 5           # max memory entries reflection may write
```

**LLM backend (fixtures-only) example**

```yaml
t3:
  allow_reflection: true
  reflection:
    backend: "llm"     # fixtures-only for determinism
  llm:
    fixtures:
      enabled: true
      path: tests/fixtures/reflection_llm.jsonl
scheduler:
  budgets:
    time_ms_reflection: 6000
    ops_reflection: 5
```

**Validator behavior**
- Unknown keys under `t3.reflection` are rejected.
- `backend` ∈ {`rulebased`, `llm`} (the `llm` backend is fixtures-only; see PR84).
- Non-negative integer bounds on `summary_tokens`, `topk_snippets`, and `ops_reflection`.
- If `backend: llm` and `t3.allow_reflection: true` but `t3.llm.fixtures.enabled` is `false`, validation errors (fixtures-only enforcement).
- `t3.llm.fixtures.path` must be a **non-empty string**; empty string is rejected.

### 3.1 Configuration reference (keys & defaults)

| Key                                   | Type / Enum                 | Default | Notes |
|---------------------------------------|-----------------------------|---------|-------|
| `t3.allow_reflection`                 | bool                        | `false` | Global gate; OFF preserves identity path. |
| `t3.reflection.backend`               | `rulebased` \| `llm`        | `rulebased` | `llm` requires fixtures (no network). |
| `t3.reflection.summary_tokens`        | int ≥ 0                     | `128`   | Whitespace token cap for the summary. |
| `t3.reflection.embed`                 | bool                        | `true`  | When `true`, encodes summary via DeterministicEmbeddingAdapter(dim=32). |
| `t3.reflection.log`                   | bool                        | `true`  | Emits `t3_reflection.jsonl` (not an identity log). |
| `t3.reflection.topk_snippets`         | int ≥ 0                     | `3`     | Pass top‑K snippets from T2 into the summariser. |
| `scheduler.budgets.time_ms_reflection`| int ms ≥ 0                  | `6000`  | Wall‑clock budget; on timeout → `reason="reflection_timeout"` and no writes. |
| `scheduler.budgets.ops_reflection`    | int ≥ 0                     | `5`     | Max number of memory entries the reflection step may write. |
| `t3.llm.fixtures.enabled`             | bool                        | `false` | Must be `true` to use `backend: llm`. |
| `t3.llm.fixtures.path`                | string (non‑empty)          | —       | Required when fixtures are enabled; path to JSONL fixtures. |

## 4) APIs & Data Model
Reflection is implemented inside `clematis/engine/stages/t3/reflect.py`.

### 4.1 `ReflectionBundle`

```python
ReflectionBundle = {
  "ctx": TurnCtx,                # fixed seeds/budgets; stable now
  "state_view": ReadOnlyView,    # read-only selections of graph/memory
  "plan": Plan,                  # executed plan; includes plan.reflection flag
  "utter": str,                  # final user-facing utterance
  "snippets": List[str],         # top-K retrieval snippets from T2
}
```

### 4.2 `reflect(bundle, cfg) -> ReflectionResult`

```python
ReflectionResult = {
  "summary": str,                # normalized + truncated to summary_tokens
  "memory_entries": [            # ≤ ops_reflection
    {
      "owner": agent_id,
      "ts": ctx.now_iso,         # set by writer
      "text": str,
      "tags": ["reflection"],
      "kind": "summary",
      "vec_full": Optional[List[float]]  # present when embed=true (dim=32; deterministic)
    }
  ],
  "metrics": {
    "backend": "rulebased" | "llm",
    "summary_len": int,
    "embed": bool,
    "fixture_key": Optional[str],    # present for llm-fixture
    "ms": float,                     # normalized to 0.0 in CI
    "reason": Optional[str]          # e.g., "reflection_timeout" or "reflect_error:<Type>"
  }
}
```

**LLM backend (fixtures-only)**
- Builds a canonical JSON prompt with sorted keys and whitespace-collapsed fields: `{"task":"reflect_summary","version":1,"agent", "turn", "summary_tokens", "utter", "snippets", "plan_reflection"}`.
- Computes `fixture_key = sha256(prompt_json)[:12]` and queries `FixtureLLMAdapter(path).generate(prompt_json, max_tokens=summary_tokens, temperature=0.0)`.
- Missing fixture or adapter error raises a local `FixtureMissingError`; the orchestrator fail-softs this into `reason="reflect_error:FixtureMissingError"` and writes nothing.

#### 4.3 Fixture key / prompt hash helper

To recompute the prompt hash used by the fixtures path:

```python
import json, hashlib

prompt_obj = {
    "agent": "unknown",
    "plan_reflection": False,
    "snippets": ["example snippet"],
    "summary_tokens": 128,
    "task": "reflect_summary",
    "turn": 0,
    "utter": "assistant: summary: . next: question",
    "version": 1,
}
prompt_json = json.dumps(prompt_obj, sort_keys=True, separators=(",", ":"))
print(prompt_json)
print(hashlib.sha256(prompt_json.encode("utf-8")).hexdigest())
```

The fixtures file uses `{"prompt_hash": "<sha256 hex>", "completion": "<text>"}` lines; see `tests/fixtures/reflection_llm.jsonl`.

## 5) Logging
Reflection uses a dedicated log file, not part of identity logs.

### 5.1 `t3_reflection.jsonl` (schema)

```jsonc
{
  "turn": 1,
  "agent": "AgentA",
  "backend": "rulebased",      // or "llm"
  "summary_len": 87,
  "ops_written": 1,
  "embed": true,
  "ms": 0.0,                   // normalized to 0.0 when CI=true
  "reason": null,              // e.g., "reflection_timeout" or "reflect_error:<Type>"
  "fixture_key": "abc123def456" // present only for LLM fixtures path
}
```

**Implementation notes**
- Writer path: `orchestrator/logging.log_t3_reflection(...) → orchestrator/logging.append_jsonl(...) → util/io_logging.normalize_for_identity(...)`.
- Stage ordering: `STAGE_ORD["t3_reflection.jsonl"]=10` ensures deterministic flush relative to other stages.
- CI normalization: only the `ms` field is normalized (`0.0`) when `CI=true`. No other fields are mutated. Reflection logs are **not** part of identity logs.
- Fail-soft: logging errors are swallowed; they do not affect the turn.

## 6) Budgets & Failure Modes
- `scheduler.budgets.time_ms_reflection`: wall budget; on expiry reflection returns an empty or truncated result and sets `metrics.reason = "reflection_timeout"`.
- `scheduler.budgets.ops_reflection`: hard cap on entries written.
- All errors (embedding/memory/logging) are **fail-soft** and never crash a turn; they are recorded as reasons in metrics/logs.

## 7) Enabling Reflection (local dev)
1. In `configs/config.yaml`, set:
   ```yaml
   t3:
     allow_reflection: true
   ```
2. Ensure the **planner requests reflection**: the plan must include `reflection: true` (the LLM policy branch stashes this flag into policy state; the orchestrator honors either the plan flag or the stashed value).
3. Keep `reflection.backend: rulebased` unless you’ve enabled fixtures for an LLM backend (see example above).
4. Run a small scenario and inspect `t3_reflection.jsonl`. Identity logs must remain identical when the gate is off.

Examples: `examples/reflection/enabled.yaml`, `examples/reflection/llm_fixture.yaml`.

> **Note:** Do not enable reflection in CI identity jobs. It is optional and should be covered by targeted reflection tests.

## 8) Tests (overview)
- **Config validation**: `tests/config/test_validate_reflection.py` (defaults, typing, backend enum, unknown keys, budget behavior).
- **Disabled-path identity** (PR82): reflection OFF → no reflection log; identity logs byte-identical to baseline.
- **Determinism** (PR83): two runs with the same seed → identical `t3_reflection.jsonl` and identical memory entries.
- **Budgets** (PR83): `time_ms_reflection` timeout + `ops_reflection` cap enforced.
- **Policy glue (PR85):** planner output accepts `reflection` (boolean), sanitizer coerces string/int forms; orchestrator gate honors `Plan.reflection` or the stashed planner flag.
- **Gate plumbing:** `_run_reflection_if_enabled` runs when `t3.allow_reflection=true` and the planner requests reflection, including the LLM path that stashes the flag in policy state.

## 9) Compatibility with M9 (deterministic parallelism)
- Reflection runs **after** T4/Apply on the orchestrator, and logs are staged with a fixed ordinal so parallel agent batches flush deterministically.
- Reflection logs are **not** counted among identity logs; they don’t affect M9 identity gates.

## 10) Roadmap & PR links
- **PR77** — Config & validator (this doc’s keys; shipped in v0.9.0a1).
- **PR78** — `reflect()` rulebased implementation and data classes.
- **PR79** — Orchestrator integration (post-Apply call; budgets; gating).
- **PR81** — Log staging ordinal for `t3_reflection.jsonl`.
- **PR82–PR83** — Tests: identity OFF, determinism, budgets/limits.
- **PR84** — Optional LLM backend (fixtures-only); default remains rulebased.
- **PR85** — Planner `reflection` flag (schema + sanitizer + policy/orchestrator wiring).
- **PR86** — Telemetry & trace (`t3_reflection.jsonl`, CI-only `ms` normalization, helper & wiring).
- **PR87** — Microbench & example configs.
- **PR88** — CI: optional reflection smoke.
- **PR89** — Docs (this dossier), README, CHANGELOG.
- **PR90** — Goldens/identity maintenance.

## 11) Troubleshooting
- **Unknown key under `t3.reflection`** → remove/rename; the validator rejects unknowns.
- **`backend: llm` without fixtures** → validator will error (fixtures-only). Set `t3.llm.fixtures.enabled: true` and a non-empty `t3.llm.fixtures.path`.
- **No `t3_reflection.jsonl` written** → ensure both the gate is enabled and the plan requested reflection; also check budgets (timeouts will still create a log with `reason`).
- **Memory write errors** → check LanceDB/in-memory settings; writes should fail-soft and be noted in logs.
- **Planner flag not picked up** → LLM path carries `reflection` via policy state; ensure your policy call ran and the orchestrator reads either `Plan.reflection` or the stashed state flag.
- **`ms` is not 0.0 locally** → expected: normalization to `0.0` happens only when `CI=true`. Local runs will show the measured milliseconds.

## 12) Microbench (PR87)

A deterministic microbench exercises the reflection unit in isolation and prints one stable JSON object.

**Rule-based (deterministic):**
```bash
python -m clematis.scripts.bench_reflection -c examples/reflection/enabled.yaml
```

**LLM fixtures (deterministic, no network):**
```bash
python -m clematis.scripts.bench_reflection -c examples/reflection/llm_fixture.yaml
```

**Output schema (example):**
```jsonc
{
  "backend": "rulebased",           // or "llm"
  "allow_reflection": true,
  "tokens_budget": 128,
  "summary_len": 17,                // whitespace tokens in the summary
  "ops": 1,                         // ≤ ops_cap
  "embed": true,
  "ops_cap": 5,
  "time_budget_ms": 6000,
  "ms": 0.0,                        // normalized to 0.0 when CI=true (default for tests)
  "reason": null,                   // or "reflection_timeout", "reflect_error:<Type>"
  "fixture_key": "abc123def456"     // present only for LLM fixtures path
}
```

Notes:
- The microbench uses stable inputs and does **not** write identity logs.
- Timing is normalized to `0.0` when `CI=true` to eliminate flakes (tests force this).

## 13) Optional CI Smoke (PR88)

An opt‑in GitHub Actions workflow validates the microbench paths without touching identity logs.

**Workflow:** `.github/workflows/reflection_smoke.yml`

**How to run**
- **Manual one‑off:** Actions → *Reflection Smoke (optional)* → Run workflow → set `run=true`.
- **Auto on push (temporary while iterating):** in the workflow’s top‑level `env:` set
  ```yaml
  RUN_REFLECTION_SMOKE: "true"
  ```
  and revert to `"false"` before merging.

**What it does**
- Runs the microbench twice (rule‑based and LLM fixtures) and asserts deterministic fields.
- Runs the dedicated microbench test file only: `tests/test_bench_reflection.py`.
- Uploads artifacts `bench_rule.json` and `bench_llm.json` for inspection.

**Determinism**
- `CI=true` is set in the workflow to normalize `"ms": 0.0`.
- No network access is required; fixtures are local and deterministic.

## Next: HS1 / GEL (M11)

Reflection in v3 is deterministic and strictly gated. The next layer is HS1/GEL (M11), which maintains an optional concept graph via co‑activation updates and half‑life decay, with conservative maintenance passes (merge/split/promotion).
See **[M11 — Field‑Control GEL (HS1) Overview](../m11/overview.md)**.
*(Note: the “field‑control nudge planner” is a v4 feature; M11 documents the substrate only.)*
