

---
# M10 — Reflection Sessions (Deterministic, Gated)

**Status:** in progress. Starting from **v0.9.0a1** the configuration surface is present but **disabled by default**. Enabling reflection must not change the identity path for prior milestones when the gate is off.

## 1) Scope & Invariants
- Add a post-turn reflection step that can **summarize** the turn and **persist** small memory entries for future retrieval.
- **Deterministic only.** No RNG, no network; all outputs must be byte-identical given the same inputs.
- **Disabled-path identity preserved.** If `t3.allow_reflection=false` or the plan does not request reflection, the run produces no new artifacts, and all identity logs are unchanged.
- Reflection must **not influence the current turn**; outputs are only visible to future retrieval.

## 2) Where it runs in the pipeline
`T1 → T2 → T3 (plan/speak) → T4 (gate/apply) → **Reflection** → end-of-turn`

- The orchestrator invokes reflection **after Apply**. Any memory writes happen after T4.
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
**Validator behavior**
- Unknown keys under `t3.reflection` are rejected.
- `backend` ∈ {`rulebased`, `llm`} (the `llm` backend is fixtures-only; see PR84).
- Non-negative integer bounds on `summary_tokens`, `topk_snippets`, and `ops_reflection`.

## 4) APIs & Data Model
Reflection is implemented inside `clematis/engine/stages/t3/reflect.py`.

### 4.1 `ReflectionBundle`
Inputs used to compute a deterministic reflection:
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
A pure function (no I/O) producing a summary and entries:
```python
ReflectionResult = {
  "summary": str,                # lowercased, punctuation-trimmed, whitespace-normalized; truncated to summary_tokens
  "memory_entries": [            # ≤ ops_reflection
    {"owner": agent_id, "ts": ctx.now_iso, "text": str, "tags": ["reflection"], "vec": Optional[List[float]]}
  ],
  "metrics": {"tokens": int, "ms": float, "reason": Optional[str]}
}
```
- **Rulebased backend**: summary is built from `utter` + up to `topk_snippets` snippets, lowercased, punctuation-stripped, whitespace-normalized, then hard-truncated by words to `summary_tokens`.
- **Embedding** (when `embed=true`): via the deterministic embedding adapter (fixed dim, no RNG). Embedding failures are logged; they must not fail the turn.

## 5) Logging
Reflection uses a dedicated log file, not part of identity logs.

### 5.1 `t3_reflection.jsonl` (schema)
```json
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
- In CI identity mode, timing fields are normalized to preserve determinism of golden logs.
- When the gate is off or the plan doesn’t request reflection, this file is **absent**.

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
2. Keep `reflection.backend: rulebased` unless you’ve enabled fixtures for an LLM backend (see PR84).
3. Run a small scenario and inspect `t3_reflection.jsonl`. Identity logs must remain identical when the gate is off.

> **Note:** Do not enable reflection in CI identity jobs. It is optional and should be covered by targeted reflection tests.

## 8) Tests (overview)
- **Config validation**: `tests/config/test_validate_reflection.py` (defaults, typing, backend enum, unknown keys, budget behavior).
- **Disabled-path identity** (PR82): reflection OFF → no reflection log; identity logs byte-identical to baseline.
- **Determinism** (PR83): two runs with the same seed → identical `t3_reflection.jsonl` and identical memory entries.
- **Budgets** (PR83): `time_ms_reflection` timeout + `ops_reflection` cap enforced.

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
- **PR86–PR90** — Telemetry polish, microbench, CI smoke, goldens, docs.

## 11) Troubleshooting
- **Unknown key under `t3.reflection`** → remove/rename; the validator rejects unknowns.
- **`backend: llm` without fixtures** → validator will error (PR84). Use `rulebased` unless fixtures are configured.
- **No `t3_reflection.jsonl` written** → ensure both the gate is enabled and the plan requested reflection; also check budgets (timeouts will still create a log with `reason`).
- **Memory write errors** → check LanceDB/in-memory settings; writes should fail-soft and be noted in logs.

---
