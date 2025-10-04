# M10 — Reflection Sessions (Deterministic, Gated)

**Status:** in progress. Starting from **v0.9.0a1** the configuration surface is present but **disabled by default**. Enabling reflection must not change the identity path for prior milestones when the gate is off. Landed to date: PR77 (config), PR80–PR83 (writer/budgets/tests), PR84 (fixtures-only LLM backend), PR85 (planner flag & wiring).

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
- Under parallel agent batches, reflection logs are staged and flushed in deterministic order (stage ordinal reserved for reflection logs).

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
      path: tests/fixtures/llm/reflection.json
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
    "backend": "rulebased" | "llm-fixture",
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

## 5) Logging
Reflection uses a dedicated log file, not part of identity logs.

### 5.1 `t3_reflection.jsonl` (schema)

```jsonc
{
  "turn": 1,
  "agent": "AgentA",
  "backend": "rulebased",
  "summary_len": 87,
  "ops_written": 1,
  "embed": true,
  "ms": 0.0,                     // normalized to 0.0 when CI=true
  "reason": null                 // e.g., "reflection_timeout" on budget abort
}
```
For LLM fixtures, a `fixture_key` may also be logged for debugging.

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
2. Keep `reflection.backend: rulebased` unless you’ve enabled fixtures for an LLM backend (see example above).
   - For LLM: set `t3.llm.fixtures.enabled: true` and provide a non-empty `t3.llm.fixtures.path`.
3. Run a small scenario and inspect `t3_reflection.jsonl`. Identity logs must remain identical when the gate is off.

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
- **PR86–PR90** — Telemetry polish, microbench, CI smoke, goldens, docs.

## 11) Troubleshooting
- **Unknown key under `t3.reflection`** → remove/rename; the validator rejects unknowns.
- **`backend: llm` without fixtures** → validator will error (fixtures-only). Set `t3.llm.fixtures.enabled: true` and a non-empty `t3.llm.fixtures.path`.
- **No `t3_reflection.jsonl` written** → ensure both the gate is enabled and the plan requested reflection; also check budgets (timeouts will still create a log with `reason`).
- **Memory write errors** → check LanceDB/in-memory settings; writes should fail-soft and be noted in logs.
- **Planner flag not picked up** → LLM path carries `reflection` via policy state; ensure your policy call ran and the orchestrator reads either `Plan.reflection` or the stashed state flag.
