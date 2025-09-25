
# Clematis v3 — Scaffold (M1–M7)

> M7 wrap: This repo now spans M1–M7. To avoid README bloat, detailed notes live in:
> - docs/m7/ (validator shapes, quality tracing, MMR λ semantics)
> - docs/updates/ (progressive PR notes; see template in docs/updates/_template.md)
> - docs/m3/llm_adapter.md (LLM adapter, fixtures, CI guardrails)
>
> For a pre-M8 hardening overview, see Changelog/PreM8Hardening.txt.

### Updates stream (rolling)

See docs/updates/ for progressive notes per PR (lightweight, append-only).

Minimal scaffold matching the Clematis v2 steering capsule. Stages are pure; the orchestrator handles I/O and logging.  
The demo exercises the canonical turn loop and writes structured JSONL logs for first‑class observability.

## Quick start


```bash
# Run the end-to-end demo turn (writes logs under .logs/)
python3 scripts/run_demo.py

# Run unit tests
pytest -q
# PR36 (M7) — run shadow traces (no-op) and inspect
export CLEMATIS_CONFIG=examples/quality/shadow.yaml
python3 scripts/rq_trace_dump.py --trace_dir logs/quality --limit 5

# PR37 (M7) — enable lexical BM25 + fusion (alpha) and inspect enabled-path traces
export CLEMATIS_CONFIG=examples/quality/lexical_fusion.yaml
pytest -q
python3 scripts/rq_trace_dump.py --trace_dir logs/quality --limit 5
```

**Examples smoke (M7):**
```bash
# Run all example configs; stop on first failure
python3 scripts/examples_smoke.py --all --fail-fast

# Or select a subset via globs
python3 scripts/examples_smoke.py --examples-glob "examples/quality/*.yaml"
```

**LLM smoke (M3, optional local):**
```bash
# Ensure Ollama is running and the model is installed
ollama pull qwen3:4b-instruct

# Run a strict-JSON sanity check against the local model
python3 scripts/llm_smoke.py

# Manual tests (only when you explicitly ask for them)
pytest -q -m manual           # runs tests/llm/test_manual_smoke_marker.py
```
> CI never runs these manual tests and never hits the network; it uses fixtures only.

## Validate your config (PR16)

Use the built-in validator to sanity-check `configs/config.yaml` (or any YAML/JSON) before running:

```bash
# Default path (configs/config.yaml)
python3 scripts/validate_config.py

# Explicit file
python3 scripts/validate_config.py path/to/your.yaml

# From STDIN
cat path/to/your.yaml | python3 scripts/validate_config.py -
```
```bash
# Machine-readable
python3 scripts/validate_config.py --json | jq .
```

**Expected success output**:
```text
OK
t4.cache: ttl_sec=600 namespaces=[] cache_bust_mode=on-apply
```

**On error** you get explicit field paths, e.g.:
```text
t4.weight_min/weight_max must satisfy weight_min < weight_max
t2.k_retrieval must be >= 1
```

**TTL keys:**
- Stage caches (T1/T2) use `ttl_s`.
- Orchestrator cache (PR15) uses `t4.cache.ttl_sec` (alias: accepts `ttl_s`; normalized to `ttl_sec`).

## M5 — Scheduler (PR25 core, OFF by default)

PR25 adds a **pure, deterministic scheduler core** and a validated `scheduler` config block. There is **no runtime wiring** yet; with `scheduler.enabled: false` the system behaves exactly like pre‑M5.

- Core module: `clematis/scheduler.py` (init/next_turn/on_yield; RR + fair‑queue aging).
- Config source of truth: `configs/validate.py` (`DEFAULTS["scheduler"]`), with a **commented example** at the end of `configs/config.yaml`.
- Loader passthrough: `clematis/io/config.py` accepts the `scheduler` block.
- Types: `Config.scheduler` (dict) in `clematis/engine/types.py`.

**Try it (tests only; no wiring yet):**
```bash
pytest -q tests/test_scheduler_basic.py tests/test_scheduler_config_validate.py
```

**Determinism:** No RNG; lexicographic tie‑breaks. `fair_queue` priority = `idle_ms // aging_ms` using a fixed test clock.

## What’s implemented

- **T1 — Keyword propagation (deterministic)**  
  Wavefront PQ with decay + budgets, cache keyed by graph `version_etag`. Deterministic seeding, tie‑breaking, and delta emission.
- **T2 — Semantic retrieval + residual (deterministic, tiered, offline)**  
  Deterministic embedding stub (BGEAdapter) + in‑memory index with exact/cluster/archive tiers. Emits small, monotonic residual nudges (never undoes T1). Cached per query.
- **T3 — Bundle, policy, one‑shot RAG, dialogue (deterministic)**  
  PR4: bundle; PR5: rule‑based policy; PR6: one‑shot RAG refinement; PR7: deterministic dialogue via template. Logs: t3_plan.jsonl, t3_dialogue.jsonl.
- **T4 — Meta‑filter & apply**  
  Accepts/filters proposed deltas; Apply persists and logs.
- **Optional — T3 LLM adapter (behind flag)**  Enable `t3.backend: llm` and inject an adapter; falls back to rule-based if absent.

## Determinism guardrails

- `_match_keywords` is **case‑insensitive**; labels iterated **sorted**.
- Priority queue uses **node‑id tie‑break** to stabilize pop order.
- Delta emission is **alphabetically sorted by node id**.
- Configured caps are independent and observable:
  - `pops` = PQ pops
  - `iters` = layers **beyond seeds** explored (depth)
  - `propagations` = edge traversals applied (relaxations)

## Configuration (YAML)

```yaml
# configs/config.yaml
t1:
  decay: {mode: exp_floor, rate: 0.6, floor: 0.05}
  edge_type_mult: {supports: 1.0, associates: 0.6, contradicts: 0.8}
  iter_cap: 50            # legacy; see iter_cap_layers
  iter_cap_layers: 50     # depth cap: layers beyond seeds
  node_budget: 1.5
  queue_budget: 10000
  radius_cap: 4
  relax_cap: null         # optional total relaxation cap
  cache: {enabled: true, max_entries: 512, ttl_s: 300}

t2:
  backend: inmemory        # or "lancedb"
  k_retrieval: 64
  sim_threshold: 0.3
  tiers: [exact_semantic, cluster_semantic, archive]
  exact_recent_days: 30
  ranking: {alpha_sim: 0.75, beta_recency: 0.2, gamma_importance: 0.05}
  clusters_top_m: 3
  residual_cap_per_turn: 32
  cache: {enabled: true, max_entries: 512, ttl_s: 300}
  lancedb:
    uri: ./.data/lancedb
    table: episodes
    meta_table: meta
    index:
      metric: cosine
      ef_search: 64
      m: 16
```

### Switching to LanceDB (optional, guarded)

The T2 retrieval stage can use a LanceDB-backed index with API parity to the in‑memory index. It’s fully optional and guarded: if LanceDB isn’t available or the DB can’t be opened, the system falls back to the in‑memory index and records the reason in metrics.

**Enable LanceDB**
1) Install the extras locally (optional):
```bash
pip install lancedb pyarrow
```
2) In `configs/config.yaml`, set:
```yaml
t2:
  backend: lancedb
  lancedb:
    uri: ./.data/lancedb
    table: episodes
    meta_table: meta
```

**Fallback behavior**
- If import/open fails, T2 uses the in‑memory index automatically.
- Metrics include:
  - `backend`: `inmemory` or `lancedb`
  - `backend_fallback`: true if a fallback occurred
  - `backend_fallback_reason`: short reason string (when present)

**Determinism & parity**
- Cosine scoring is computed on the Python side for parity with in‑memory behavior.
- Ties are broken by `id` ascending.
- Tier semantics (`exact_semantic`, `cluster_semantic`, `archive`) are identical to the in‑memory path.

**Optional tests**
- Lance tests are skipped automatically if `lancedb`/`pyarrow` aren’t installed.
- To run just the Lance adapter test:
```bash
pytest -q tests/test_lance_index_optional.py
```

## M3 — T3 Bundle & Plan Schemas

### Bundle (t3-bundle-v1)
Deterministic, compact snapshot produced by `make_plan_bundle(ctx, state, t1, t2)` (pure; no I/O). Lists are sorted and capped; missing fields have explicit defaults.

```json
{
  "version": "t3-bundle-v1",
  "now": "2025-09-19T00:00:00Z",
  "agent": {"id": "agentA", "style_prefix": "", "caps": {"tokens": 256, "ops": 3}},
  "world": {"hot_labels": [], "k": 0},
  "t1": {
    "touched_nodes": [
      {"id": "n01", "label": "Label1", "delta": 0.42}
    ],
    "metrics": {"pops": 0, "iters": 0, "propagations": 0, "radius_cap_hits": 0, "layer_cap_hits": 0, "node_budget_hits": 0}
  },
  "t2": {
    "retrieved": [
      {"id": "e1", "score": 0.83, "owner": "any", "quarter": ""}
    ],
    "metrics": {"tier_sequence": [], "k_returned": 0, "sim_stats": {"mean": 0.0, "max": 0.0}, "cache_used": false}
  },
  "text": {"input": "…", "labels_from_t1": ["…"]},
  "cfg": {
    "t3": {"max_rag_loops": 1, "tokens": 256, "temp": 0.7},
    "t2": {"owner_scope": "any", "k_retrieval": 64, "sim_threshold": 0.3}
  }
}
```

**Determinism rules**
- `t1.touched_nodes`: choose top‑32 by |delta|, then sort by `id` ascending in the final list.
- `t2.retrieved`: sort by `(-score, id)` and trim to `k_retrieval`.
- `labels_from_t1`: dedupe + sort alphabetically.
- Provide explicit empty defaults (`[]`, `{}`) for absent fields.

### Plan & Ops (t3-plan-v1)
Policy output (added in later PRs) uses a fixed schema. Allowed op kinds only:
`CreateGraph`, `EditGraph`, `SetMetaFilter`, `RequestRetrieve`, `Speak`.

```json
{
  "version": "t3-plan-v1",
  "reflection": false,
  "ops": [
    {"kind": "Speak", "intent": "summary", "topic_labels": ["hot-topic"], "max_tokens": 128}
  ],
  "request_retrieve": null
}
```

### T3 Config (defaults)
Add under `t3:` in `configs/config.yaml` (already present in this repo):

```yaml
t3:
  max_rag_loops: 1
  tokens: 256
  temp: 0.7
  max_ops_per_turn: 3
  allow_reflection: false
  backend: rulebased
  dialogue:
    template: "{style_prefix}| summary: {labels}. next: {intent}"
    include_top_k_snippets: 2
```

## T3 — Rule‑based Policy (PR5)

Deterministic policy that maps the T3 bundle to a `Plan` with a small, whitelisted set of ops. No RAG loop yet (that lands in PR6).

**Inputs:** `make_plan_bundle(ctx, state, t1, t2)` output.

**Outputs:** `Plan{version: "t3-plan-v1", reflection: false, ops: [Op], request_retrieve: null}`

**Decision logic (deterministic):**
- Let `s_max = t2.metrics.sim_stats.max`.
- Thresholds (configurable under `t3.policy`): `tau_high=0.8`, `tau_low=0.4`, `epsilon_edit=0.10`.
- Choose `Speak.intent`:
  - `s_max ≥ tau_high` → `summary`
  - `tau_low ≤ s_max < tau_high` → `assertion` (or `ack` if no labels)
  - `s_max < tau_low` → `question`
- `topic_labels`: from `text.labels_from_t1`, deduped + sorted (fallback to `t1.touched_nodes[].label`).
- Optionally add a small `EditGraph` op when `s_max ≥ tau_low`, including a few `upsert_node` edits for nodes with `|delta| ≥ epsilon_edit` (ids sorted asc).  
- Optionally add a `RequestRetrieve` op **only** when `s_max < tau_low` (owner/k/tier from `cfg.t2`); it’s emitted but not executed until PR6.
- Enforce `len(ops) ≤ t3.max_ops_per_turn`.

**Config (already in `configs/config.yaml`):**
```yaml
t3:
  ...
  policy:
    tau_high: 0.8
    tau_low: 0.4
    epsilon_edit: 0.10
```

**Determinism guardrails:** sort labels and node ids; fixed thresholds; no RNG; pure function.

## T3 — One‑shot RAG refinement (PR6)

A single, optional retrieval refinement step for low‑evidence cases. Stage‑only: no dialogue yet, no orchestrator changes required beyond passing a retrieve function when you wire it later.

**Function:**
```python
# clematis/engine/stages/t3.py
rag_once(bundle: dict, plan: Plan, retrieve_fn: Callable[[dict], dict], already_used: bool=False) -> (Plan, dict)
```

**Inputs**
- `bundle`: output of `make_plan_bundle(...)` (PR4).
- `plan`: output of `deliberate(bundle)` (PR5). If it contains a `RequestRetrieve` op, RAG can be used once.
- `retrieve_fn(payload)` (injected): must return `{ "retrieved": [{id, score, owner, quarter}...], "metrics": {...} }`.
- `already_used`: if `True`, RAG is blocked (idempotent no‑op).

**Payload built deterministically** (from the first `RequestRetrieve` op):
```json
{
  "query": "...",
  "owner": "agent|world|any",
  "k": 8,
  "tier_pref": "exact_semantic|cluster_semantic|archive|null",
  "hints": { "now": ISO8601, "sim_threshold": 0.3 }
}
```

**Refinement policy**
- Compute `pre_s_max = bundle.t2.metrics.sim_stats.max` and `s_max_rag = max(score)` from `retrieve_fn` results.
- `post_s_max = max(pre_s_max, s_max_rag)`.
- Update the first `Speak` op’s intent deterministically using thresholds (`t3.policy`):
  - `post_s_max ≥ tau_high` → `summary`
  - `tau_low ≤ post_s_max < tau_high` → `assertion` (or `ack` if no labels)
  - `post_s_max < tau_low` → `question`
- Optionally add **one** `EditGraph` op if evidence ≥ `tau_low` and none exists yet (ids sorted asc; edits capped).  
- **Never** add a second `RequestRetrieve` op.  
- Enforce `len(ops) ≤ t3.max_ops_per_turn`.

**Outputs**
- `(refined_plan, metrics)` where `metrics` is:
```json
{
  "rag_used": true|false,
  "rag_blocked": true|false,
  "pre_s_max": 0.2,
  "post_s_max": 0.85,
  "k_retrieved": 2,
  "owner": "any",
  "tier_pref": "cluster_semantic"
}
```

**Determinism**
- Pure function: given the same `(bundle, plan)` and a deterministic `retrieve_fn`, the refined plan and metrics are identical.
- Sorted ordering and caps are preserved; no DB or I/O.

**Tests**
- `tests/test_t3_rag.py` covers: one‑shot blocking, low→high refinement, no‑improvement case, optional `EditGraph`, determinism, and payload sanity.

## T3 — Dialogue & Logging (PR7)

Deterministic dialogue synthesis from the plan/bundle. No LLM dependency is required; output is produced via a simple template and hard caps.

**Functions**
```python
# clematis/engine/stages/t3.py
make_dialog_bundle(ctx, state, t1, t2, plan) -> dict
speak(dialog_bundle: dict, plan: Plan) -> tuple[str, dict]
```

**Template variables** (from `t3.dialogue.template`): `{labels}`, `{intent}`, `{snippets}`, `{style_prefix}`.  
- Labels: deduped + sorted from the Plan’s Speak op (fallback to bundle `labels_from_t1`).
- Snippets: top‑K retrieved IDs (cap = `t3.dialogue.include_top_k_snippets`).
- If the template omits `{style_prefix}` but a prefix exists, it is auto‑prefixed as `"{style_prefix}| "`.

**Token budget**
- Enforced deterministically with whitespace tokenization. Budget = `SpeakOp.max_tokens` if present, else `agent.caps.tokens`, else 256.

**Outputs**
- `utterance: str` and `metrics`: `{tokens, truncated, style_prefix_used, snippet_count}`.

**Orchestrator wiring**
- Flow: `make_plan_bundle → deliberate → (optional) rag_once → make_dialog_bundle → speak`.
- JSONL logs written per turn:
  - `t3_plan.jsonl`: `{policy_backend, backend, backend_fallback?, fallback_reason?, ops_counts, requested_retrieve, rag_used, reflection, ms_deliberate, ms_rag}`
  - `t3_dialogue.jsonl`: `{backend, tokens, truncated, style_prefix_used, snippet_count, ms, adapter?, model?, temperature?}`

## T3 — LLM adapter (PR8, optional)

Opt-in LLM-driven dialogue while keeping the default rule-based backend and deterministic CI.

**Enable (config)**
```yaml
# configs/config.yaml
t3:
  backend: rulebased   # or "llm"
  llm:
    provider: fixture   # "fixture" | "ollama"
    model: qwen3:4b-instruct
    endpoint: http://localhost:11434/api/generate
    max_tokens: 256
    temp: 0.2
    timeout_ms: 10000
    fixtures:
      enabled: true
      path: fixtures/llm/qwen_small.jsonl
```

**Provide an adapter at runtime** (no network in CI):
```python
from clematis.adapters.llm import QwenLLMAdapter

# Endpoint-based (Ollama)
state["llm_adapter"] = QwenLLMAdapter(
    endpoint="http://localhost:11434/api/generate",
    model="qwen3:4b-instruct",
    max_tokens=256,
    temperature=0.2,
    timeout_s=30,
)

# Alternative: custom call_fn if you integrate a different runtime
def qwen_chat(prompt: str, *, model: str, max_tokens: int, temperature: float, timeout_s: int) -> str:
    ...  # return raw text
state["llm_adapter"] = QwenLLMAdapter(call_fn=qwen_chat, model="qwen3:4b-instruct", temperature=0.2, timeout_s=30)
```

**Deterministic tests**
- CI uses `DeterministicLLMAdapter` (offline). No network calls. Token caps enforced the same as rule-based `speak()`.

**Fallback behavior**
- If `t3.backend: llm` but no adapter is present, orchestrator falls back to rule-based and logs:
  - `t3_plan.jsonl`: `backend_fallback`, `fallback_reason: "no_adapter"`
  - `t3_dialogue.jsonl`: `backend: "rulebased"`

**CI guardrails (summary)**
- Network ban enforced in CI; providers other than `fixture` are rejected.
- If `t3.backend=llm` in CI, `fixtures.enabled` must be `true` and the `fixtures.path` must exist.
- Every LLM output is parsed and validated against `PLANNER_V1`; invalid JSON or shape ⇒ empty plan with a logged reason.
- See also: `docs/m3/llm_adapter.md` for fixture format and prompt hashing.

## M4 — T4 Meta‑Filter & Apply/Persist

A deterministic safety gate plus a side‑effecting apply step.

### T4 Meta‑Filter (pure)
Approves/clamps/limits proposed deltas from T1/T2/T3 without mutating state.

**Policy (deterministic, order‑stable by `(target_id, attr)`):**
1. **Normalize & combine** duplicate targets (sum deltas; keep stable order).
2. **Cooldowns:** drop ops that violate `t4.cooldowns[kind]` → reason `COOLDOWN_BLOCKED`.
3. **Per‑node novelty cap:** `|Δ_i| ≤ t4.novelty_cap_per_node` (clip magnitudes).
4. **Global L2 cap:** if `‖Δ‖₂ > t4.delta_norm_cap_l2`, scale all remaining deltas proportionally.
5. **Churn cap:** keep top‑K by `|Δ|`, stable tie‑break by id; drop tail → `CHURN_CAP_HIT`.

**Outputs**
- `T4Result{ approved_deltas, rejected_ops, reasons[], metrics }`
- Log: `.logs/t4.jsonl` with `{turn, agent, approved, rejected_ops, reasons, metrics, ms}`.

### Apply/Persist (side effects)
Applies approved deltas to the in‑memory graph store, bumps versions, and snapshots.

**Behavior**
- **Clamp** weights into `[t4.weight_min, t4.weight_max]`; count clamps.
- **Idempotent**: applying the same deltas twice yields no net change.
- **Versioning**: increments `state.version_etag` (monotonic string).
- **Snapshot cadence**: write `t4.snapshot_dir/state_{agent}.json` every `t4.snapshot_every_n_turns`. Snapshot schema is stable and **always** includes a top‑level `"store": {}` object (empty when not exportable).
- **Cache coherency (PR15)**:
  - Orchestrator maintains a **version‑aware** cache around T2 (keyed by `(version_etag or "0", input)`).
  - On apply and when `t4.cache_bust_mode: on-apply`, the orchestrator invalidates configured namespaces (default: `["t2:semantic"]`). `apply.jsonl` records `cache_invalidations`.

**Kill switch**
- `t4.enabled: true|false`. When `false`, T4 and Apply are bypassed; no `t4.jsonl`/`apply.jsonl` entries are written.
Validation is provided via the CLI (above); the turn loop does not auto-validate configs by default.

### T4 configuration (add under `t4:` in `configs/config.yaml`)
```yaml
t4:
  enabled: true
  # Filter caps
  delta_norm_cap_l2: 1.5
  novelty_cap_per_node: 0.3
  churn_cap_edges: 64
  cooldowns:
    EditGraph: 2
    CreateGraph: 10
  # Apply + snapshots
  weight_min: -1.0
  weight_max: 1.0
  snapshot_every_n_turns: 1
  snapshot_dir: ./.data/snapshots
  # Cache coherency (PR15)
  cache_bust_mode: on-apply   # or "none"
  cache:
    enabled: true
    # M4 centralized example; M7 default leaves namespaces empty and uses stage-local t2.cache
    namespaces:
      - "t2:semantic"
    max_entries: 512
    ttl_sec: 600
```


### Caches (two layers in M4)
- **Stage‑level** LRU caches in T1/T2 (legacy, per‑stage knobs; `ttl_s`).
- **Orchestrator‑level** CacheManager around T2 (version‑aware, invalidated on Apply; `ttl_sec`).
This is intentional for M4; consolidation is a later follow‑up.
Use the validator to check TTLs and sizes across both layers quickly.

> **M7 default:** stage-owned T2 cache; \
> set `t4.cache.namespaces: []`. To centralize invalidation instead, disable `t2.cache` and set `t4.cache.namespaces: ["t2:semantic"]`.

#### Cache coherency: end-to-end walkthrough

This shows how the **orchestrator cache** (version‑aware, wraps T2) and **stage caches** (T1/T2 LRUs) behave across turns.

**Pre‑reqs (config):**
- `t2.cache.enabled: true`
- `t4.cache.enabled: true`
- `t4.cache.namespaces: ["t2:semantic"]`
- `t4.cache_bust_mode: on-apply`
- `t4.enabled: true` (so Apply runs and bumps `version_etag`)
- Optional: `t1.cache.ttl_s` and `t2.cache.ttl_s` > 0

> Tip: run the validator first:
> ```bash
> python3 scripts/validate_config.py
> ```
> You should see a line like: `t4.cache: ttl_sec=... namespaces=['t2:semantic'] cache_bust_mode=on-apply`.

**Reset environment (fresh logs & snapshots):**
```bash
rm -rf ./.logs ./.data/snapshots
mkdir -p ./.logs
```

**1) Cold run ⇒ MISS**
```bash
python3 scripts/run_demo.py
# Inspect the latest retrieval log line (MISS expected)
tail -n 1 ./.logs/t2.jsonl | python3 -c 'import sys,json;print(json.loads(sys.stdin.read()).get("cache_hit"))'
# → False
```

**2) Warm run (same input, same version) ⇒ HIT**
```bash
python3 scripts/run_demo.py
# Inspect the latest retrieval log line (HIT expected)
tail -n 1 ./.logs/t2.jsonl | python3 -c 'import sys,json;print(json.loads(sys.stdin.read()).get("cache_hit"))'
# → True
```

**3) Apply occurs (version bump) ⇒ orchestrator invalidates**
When `t4.enabled=true` and `snapshot_every_n_turns ≥ 1`, Apply runs and bumps `version_etag`. Check `apply.jsonl`:
```bash
tail -n 1 ./.logs/apply.jsonl | python3 -c 'import sys,json;d=json.loads(sys.stdin.read());print(d.get("version_etag"), d.get("cache_invalidations"))'
# → v<N+1> <positive_int>
```

**4) Next run (same input, new version) ⇒ MISS again**
```bash
python3 scripts/run_demo.py
tail -n 1 ./.logs/t2.jsonl | python3 -c 'import sys,json;print(json.loads(sys.stdin.read()).get("cache_hit"))'
# → False
```

**What’s happening**
- The **orchestrator cache** wraps T2 and keys entries with `(version_etag, input)`. After Apply (version bump), previously cached `(vN, input)` can’t satisfy `(vN+1, input)` ⇒ MISS.
- **Stage caches** (T1/T2) are independent TTL‑based LRUs. They remain deterministic but can make TTL expectations feel “sticky.” Orchestrator invalidations do **not** forcibly clear stage caches; they prevent reuse of stale *orchestrator‑level* results.

**Variations to try**
- **Kill‑switch:** set `t4.enabled=false` → no Apply; the version doesn’t change; consecutive runs remain HIT at the orchestrator layer.
- **No cache bust:** set `t4.cache_bust_mode: none` → orchestrator keys still change (MISS after version bump), but stage caches may still show their own hits.
- **TTL sensitivity:** set `t2.cache.ttl_s=0` → stage cache effectively disabled; only orchestrator cache HIT/MISS is visible.

**Troubleshooting**
- Seeing HIT after Apply? Check `t4.cache_bust_mode` (must be `on-apply`) and confirm `apply.jsonl.cache_invalidations > 0`.
- Seeing perpetual MISS? Ensure `t2.cache.enabled=true` and that inputs are identical across runs.
- Mixed signals? Remember there are **two** cache layers:
  - `.logs/t2.jsonl.cache_hit` reflects the stage’s internal cache.
  - The orchestrator wrapper uses the version‑aware key; MISS after version bump is expected.

**Invariants**
- Identical inputs + identical `version_etag` ⇒ identical outputs and HIT on second run.
- After Apply with `on-apply` busting, the *next* identical input run yields MISS due to `version_etag` change.

## Stage semantics

### T1 (propagation)
- **Decay modes:** `exp_floor` (default) or `attn_quad` via config.
- **Caps:** `radius_cap`, `iter_cap_layers` (depth), `node_budget`, `queue_budget`, optional `relax_cap`.
- **Metrics:** `pops`, `iters`, `propagations`, `radius_cap_hits`, `layer_cap_hits`, `node_budget_hits`, `cache_*`.

### T2 (retrieval + residual)
- **Query:** `input_text` + labels of nodes touched by T1 (deterministically gathered).
- **Tiers:**  
  1) `exact_semantic` → recent window (`exact_recent_days`) + `sim_threshold`  
  2) `cluster_semantic` → route to `clusters_top_m` clusters by centroid similarity  
  3) `archive` → older shards (fallback)  
- **Embeddings:** offline deterministic `BGEAdapter(dim=32)` (stable hash‑based vectors).  
- **Residuals:** keyword‑match episode text ↔ current graph labels, emit bounded `upsert_node` nudges; **never undo T1**.  
- **Metrics:** `tier_sequence`, `k_returned`, `k_used`, `sim_stats{mean,max}`, `caps.residual_cap`, `cache_*`.

# Logs

JSONL files are written to `.logs/`:

- `t1.jsonl` — propagation metrics per turn
- `t2.jsonl` — retrieval/residual metrics per turn (includes `cache_hit: bool`, `cache_size: int`)
- `t3_plan.jsonl` — plan metrics and policy details
- `t3_dialogue.jsonl` — dialogue synthesis metrics
- `t4.jsonl` — meta‑filter approvals/rejections
- `apply.jsonl` — state changes summary (includes `cache_invalidations: int`)
- **gel.jsonl** — Graph Evolution Layer events (only when `graph.enabled=true`)
  - `observe_retrieval` fields: `turn`, `agent`, `k_in`, `k_used`, `pairs_updated`, `threshold`, `mode`, `alpha`, `ms`
  - `edge_decay` fields: `turn`, `agent`, `decayed_edges`, `dropped_edges`, `half_life_turns`, `floor`, `ms`
  - `merge` fields: component signature, size, avg_w, diameter
  - `split` fields: parts/signature, thresholds
  - `promotion` fields: concept_id, label, members_count, attach_weight
- `turn.jsonl` — per‑turn roll‑up (durations, key metrics)
- `health.jsonl` — guardrail flags
- `rq_traces.jsonl` — quality traces (M7).
  - **Shadow (PR36)**: written only when the **triple gate** is satisfied and shadow is on: `perf.enabled && perf.metrics.report_memory && t2.quality.shadow && !t2.quality.enabled` → `meta.reason: "shadow"`.
  - **Enabled (PR37)**: written when the **triple gate** is satisfied and the quality fuser actually runs: `perf.enabled && perf.metrics.report_memory && t2.quality.enabled` → `meta.reason: "enabled"`. Per-item fields include `rank_sem`, `rank_fused`, and `score_fused`; `meta.alpha` and `meta.lex_hits` summarize fusion settings and lexical coverage. Default path: `logs/quality/`.
# HS1 wrap-up
# HS1:

- Deterministic tests use the hash‑based embeddings; if a test filters too hard on cosine, set `t2.sim_threshold` to `-1.0` in that test to include all candidates and let other filters (recency, tiers) drive behavior.
- Cache behavior is configurable per stage; tests include **first‑call miss → second‑call hit → invalidation by etag/index change**.
- Keep stages **pure**. Instrumentation and persistence live in the orchestrator and adapters.

## M4 Status

- PR12 (T4 Meta-Filter): ✅
- PR13 (Apply/Persist + Orchestrator wiring): ✅
- PR14 (Snapshot loader & resume): ✅
- PR15 (Cache coherency & guardrails): ✅
- PR16 (Docs & config validation): ✅

**Optional follow-ups (not required for M4):**
- CLI: inspect latest snapshot (P3 tooling)
- Cache consolidation (single layer) in a future milestone

## HS1 — Hardening Sprint

## PR17 — Config validation & hygiene

This sprint hardens configuration while keeping runtime behavior deterministic and unchanged.

### Validator CLI

```bash
python3 scripts/validate_config.py            # validate configs/config.yaml
python3 scripts/validate_config.py --strict   # fail on warnings
python3 scripts/validate_config.py path/to/other.yaml
cat configs/config.yaml | python3 scripts/validate_config.py -
```

**Exit codes**
- `0` — OK
- `1` — validation errors (or warnings when `--strict`)
- `2` — load/parse errors or bad usage

**Typical success output**
```text
OK
t4.cache: ttl_sec=600 namespaces=['t2:semantic'] cache_bust_mode=on-apply
```

### Canonical keys & TTL aliases
The validator accepts common aliases and normalizes them:

- **Stage caches (T1, T2)**: canonical key is `ttl_s` (seconds). Aliases accepted: `ttl_sec` → normalized to `ttl_s`.
- **Orchestrator cache (T4)**: canonical key is `ttl_sec` (seconds). Alias accepted: `ttl_s` → normalized to `ttl_sec`.

Use either key in your YAML; the validator will normalize to the canonical form above.

### Allowed keys (practical subset)

| Section | Key | Type | Notes |
|---|---|---|---|
| `t1.cache` | `enabled` | bool | optional; defaults to `true` in shipped config |
|  | `max_entries` | int ≥ 0 |  |
|  | `ttl_s` | int ≥ 0 | `ttl_sec` alias accepted |
| `t1` | `iter_cap`, `queue_budget`, `node_budget` | int | optional knobs |
|  | `decay`, `edge_type_mult` | number | optional knobs |
|  | `radius_cap` | int | optional |
| `t2.cache` | `enabled` | bool | optional; defaults true in shipped config |
|  | `max_entries` | int ≥ 0 |  |
|  | `ttl_s` | int ≥ 0 | `ttl_sec` alias accepted |
| `t2` | `backend` | `inmemory`\|`lancedb` | `lancedb` path remains optional/guarded |
|  | `k_retrieval` | int ≥ 1 |  |
|  | `sim_threshold` | −1.0 … 1.0 | inclusive bounds |
|  | `tiers`, `exact_recent_days`, `clusters_top_m`, `owner_scope`, `residual_cap_per_turn`, `ranking` | mixed | recognized and validated where applicable |
| `t3` | `max_rag_loops`, `max_ops_per_turn` | int ≥ 0 |  |
|  | `backend` | `rulebased`\|`llm` | default rule-based; LLM path optional/guarded |
|  | `tokens`, `temp`, `allow_reflection`, `dialogue`, `policy`, `llm` | mixed | accepted keys; kept deterministic by default |
| `t4.cache` | `enabled` | bool |  |
|  | `namespaces` | list[str] | **allowed:** `"t2:semantic"` only |
|  | `max_entries` | int ≥ 0 |  |
|  | `ttl_sec` | int ≥ 0 | `ttl_s` alias accepted |
| `t4` | `enabled` | bool | kill switch (bypasses T4+Apply) |
|  | `delta_norm_cap_l2` | number > 0 |  |
|  | `novelty_cap_per_node` | number > 0 |  |
|  | `churn_cap_edges` | int ≥ 0 |  |
|  | `cooldowns` | map[str→int ≥ 0] | keys like `EditGraph`, `CreateGraph` |
|  | `weight_min`, `weight_max` | −1.0 … 1.0 and `min < max` | clamped in Apply |
|  | `snapshot_every_n_turns` | int ≥ 1 | cadence for snapshotting |
|  | `snapshot_dir` | path | directory for snapshots |
|  | `cache_bust_mode` | `on-apply`\|`none` | orchestrator cache behavior |

> The validator also accepts other benign keys present in the shipped configs and treats unrecognized keys as warnings in the verbose path.

### Warnings & `--strict`
- **Duplicate cache namespace overlap** (informational):
  - When `t2.cache.enabled=true` **and** `"t2:semantic" ∈ t4.cache.namespaces`, you’ll see:
    ```text
    W[t4.cache.namespaces]: duplicate-cache-namespace 't2:semantic' overlaps with stage cache (t2.cache.enabled=true); deterministic but TTLs may be confusing.
    ```
  - With `--strict`, warnings cause a non-zero exit to keep CI tight.

### Error examples (deterministic messages)
```text
# Unknown key with suggestion
CONFIG INVALID
t2.backnd unknown key (did you mean 'backend')

# Out-of-range and consistency
CONFIG INVALID
t4.weight_min/weight_max must satisfy weight_min < weight_max
t2.sim_threshold must be between -1.0 and 1.0 (inclusive)

t4.cache.namespaces[t2:semantics] unknown namespace (allowed: ['t2:semantic'])
```

### Tips
- Keep `configs/config.yaml` canonical (use `ttl_s` for T1/T2 caches, `ttl_sec` for T4 cache).
- Use `python3 scripts/validate_config.py --strict` in CI to fail fast on hygiene regressions.
- If you add new config fields, update the validator and this section in the same PR.

# PR18 — Snapshot inspector & schema tag

Add a compact CLI to inspect the latest snapshot and tag new snapshots with a schema version. Loader behavior remains non‑fatal for missing/legacy snapshots.

### Inspecting snapshots

```bash
python3 scripts/inspect_snapshot.py
python3 scripts/inspect_snapshot.py --dir ./.data/snapshots --format json
```

- Exit `0` when a snapshot is found; prints a pretty summary by default.
  - Shows **graph schema** and GEL meta counts when present (PR24).
- Exit `2` if no snapshot exists or cannot be read (runtime remains tolerant).
- New snapshots include `schema_version: v1`; legacy snapshots may show `unknown`.

**Fields shown** (when available): `schema_version`, `version_etag`, `nodes`, `edges`, `last_update`, and T4 caps (`delta_norm_cap_l2`, `novelty_cap_per_node`, `churn_cap_edges`, `weight_min`, `weight_max`).
**Additional fields** (when present): `graph_schema_version`, and a compact GEL summary: node/edge counts and meta counters (`merges`, `splits`, `promotions`, `concepts`).

## PR19 — Log schemas & rotation helper

Documented the JSONL shapes we emit and added a tiny size‑based rotation script. This PR does **not** change runtime behavior.

### Log schemas (minimal fields)

All logs live under `.logs/` unless configured otherwise.

- **t1.jsonl** — per‑turn propagation metrics  
  Fields: `turn`, `agent`, `pops`, `iters`, `propagations`, `radius_cap_hits`, `layer_cap_hits`, `node_budget_hits`, `cache_hit`, `cache_size`, `ms`

- **t2.jsonl** — retrieval/residual metrics  
  Fields: `turn`, `agent`, `k_retrieved`, `k_used`, `tier_sequence[]`, `sim_stats{mean,max}`, `cache_hit`, `cache_size`, `backend`, `backend_fallback?`, `backend_fallback_reason?`, `ms`

- **t3_plan.jsonl** — plan/policy metrics  
  Fields: `turn`, `agent`, `policy_backend`, `backend`, `ops_counts`, `requested_retrieve`, `rag_used`, `reflection`, `ms_deliberate`, `ms_rag`

- **t3_dialogue.jsonl** — dialogue synthesis metrics  
  Fields: `turn`, `agent`, `backend`, `tokens`, `truncated`, `style_prefix_used`, `snippet_count`, `adapter?`, `model?`, `temperature?`, `ms`

- **t4.jsonl** — meta‑filter decisions  
  Fields: `turn`, `agent`, `approved`, `rejected_ops[]`, `reasons[]`, `metrics{caps,clamps,cooldowns}`, `ms`

- **apply.jsonl** — apply/snapshot summary  
  Fields: `turn`, `agent`, `applied`, `clamps`, `version_etag`, `snapshot_path`, `cache_invalidations`, `ms`

- **turn.jsonl** — per‑turn roll‑up  
  Fields: `turn`, `agent`, `ms_total`, `ms_t1`, `ms_t2`, `ms_t3`, `ms_t4`, `ms_apply`, `health`

- **health.jsonl** — guardrail flags  
  Fields: `turn`, `agent`, `flags[]` (e.g., `GEL_EDGES`, `GEL_MERGES`, `GEL_SPLITS` when GEL is enabled)

> Notes: Field sets are minimal and stable. Additional fields may appear; consumers should ignore unknown keys.

### Rotating logs (optional ops helper)

Use the provided script to keep JSONL files capped by size with deterministic numeric suffixes:

```bash
# Dry‑run (show actions only)
python3 scripts/rotate_logs.py --dir ./.logs --pattern '*.jsonl' --max-bytes 10_000_000 --backups 5 --dry-run

# Perform rotation
python3 scripts/rotate_logs.py --dir ./.logs --pattern '*.jsonl' --max-bytes 10_000_000 --backups 5
```

- Rotates any matched file with size **≥ `--max-bytes`**.
- Keeps `--backups` generations: `.1`, `.2`, … (oldest dropped).
- No recursion; only files directly under `--dir` are considered.
- The script does **not** append—only rotates existing files. Wire‑in is optional and non‑disruptive.

## PR20 — Performance guardrails & T4 fuzz invariants (opt-in)

Add deterministic property tests for T4 plus opt-in performance checks and a tiny microbench CLI. **No runtime behavior changes.**

### What landed
- `tests/test_t4_property.py` — deterministic fuzz/property tests (always run).
- `tests/test_perf_guardrails.py` — opt-in perf checks (skipped by default).
- `scripts/bench_t4.py` — microbench CLI for manual runs.

### Running the tests
Property tests run in the normal suite:
```bash
pytest -q tests/test_t4_property.py
```

Perf tests are **opt-in** to avoid CI flakiness. Enable via env + mark:
```bash
RUN_PERF=1 pytest -q -m perf tests/test_perf_guardrails.py
```
If you see a warning about an unknown `perf` mark, register it once in your test config.
- **pytest.ini**
  ```ini
  [pytest]
  markers =
      perf: opt-in performance tests
  ```
- **or pyproject.toml**
  ```toml
  [tool.pytest.ini_options]
  markers = [
    "perf: opt-in performance tests",
  ]
  ```

### Microbench CLI
Run synthetic workloads through the T4 meta-filter and print timing stats (median/p95) and approvals.
```bash
python3 scripts/bench_t4.py                       # defaults: --num 10000 --runs 5 --seed 1337
python3 scripts/bench_t4.py --num 20000 --runs 5
python3 scripts/bench_t4.py --num 8000 --runs 7 --json  # machine-readable output
```
**Typical output:**
```text
N=10000 runs=5 seed=1337  l2=1.5 novelty=0.3 churn=64
median=42.1ms  p95=44.3ms  min=40.6ms  max=45.0ms  thr≈237800.0 ops/s
approved median/min/max = 64/64/64
reasons total: CHURN_CAP_HIT:49312, NOVELTY_SPIKE:...
```
Notes:
- Workload is deterministic per seed; change `--seed` to vary.
- Throughput figures are indicative and machine-dependent; use for relative checks only.

## PR22 — GEL (Graph Evolution Layer): Edge update + decay

Deterministic, bounded-cost graph co-activation edges with exponential decay. **Default: OFF** — enabling this does not change core T1–T4 behavior; it only records co-activation structure and decay when turned on.

### What it does
- **observe_retrieval** (after T2): builds/updates undirected edges between top‑K retrieved items above a threshold. Weight updates are deterministic and clamped.
- **tick** (before Apply): decays all edge weights by half-life; drops edges under a floor. Decay happens before snapshot so it’s captured in `state_*`.
- **Snapshot schema**: snapshots include `graph_schema_version: "v1"` and a `gel` block `{nodes: {}, edges: {}}` with canonical edge keys `src→dst` (unicode arrow). A compact `graph` summary with counts is also written for legacy readers.
- **Logs**: events are written to `gel.jsonl` (see **Logs** section for fields).

### Configuration (YAML)
Add under top-level `graph:` in your config. All keys are validated by `scripts/validate_config.py`.

```yaml
# configs/config.yaml
graph:
  enabled: false                 # default OFF; set true to enable GEL
  coactivation_threshold: 0.20   # keep retrieved items with score ≥ threshold
  observe_top_k: 64              # consider at most K items (sorted by -score, id)
  pair_cap_per_obs: 2048         # global cap on pairs per observation
  update:
    mode: additive               # or "proportional"
    alpha: 0.02                  # additive step or proportional factor
    clamp_min: -1.0
    clamp_max: 1.0
  decay:
    half_life_turns: 200         # turns for weight to halve
    floor: 0.0                   # drop edges with |weight| < floor
  # tolerated (contracts; no behavior here)
  merge: { enabled: false, min_size: 3 }
  split: { enabled: false }
```
```md
> **Note (PR24)**: The GEL config is extended with `merge`, `split`, and `promotion` blocks. See PR24 below. Enabling these in HS1 only records metadata and logs; core retrieval/apply behavior remains unchanged.


**Determinism & bounds**
- No RNG; stable ordering and canonicalized edge keys (`a→b` where `a < b`).
- Observe work is bounded by `observe_top_k` and `pair_cap_per_obs` (prefix traversal of the K list).
- Decay is linear in current edge count and performed once per turn when Apply runs.

**Wiring (already done)**
- Orchestrator calls:
  - After T2: `gel.observe_retrieval(ctx, state, t2.retrieved, turn=turn_id, agent=agent)`
  - Before Apply: `gel.tick(ctx, state, decay_dt=1, turn=turn_id, agent=agent)`

**Inspector**
- `scripts/inspect_snapshot.py` now shows `schema_version`, counts, and (when present) `gel edges`.

**Tests**
- `tests/test_gel_update_decay.py` — updates, clamping, proportional mode, key canonicalization, decay + floor.


## PR23 — Hybrid dense+graph re‑ranking (optional)

Blend dense similarity (T2) with evidence from the Graph Evolution Layer (GEL) to reorder only the **top‑K** results. Default is **OFF**; enabling this keeps determinism and bounded work.

### What it does
- Takes the T2 list (already sorted by `(-score, id)`), considers at most `k_max` items, and computes a **graph bonus** per item using GEL edges.
- **Anchors**: the first `anchor_top_m` items act as anchors.
- **1‑hop sum**: sum of anchor→item edge weights above `edge_threshold` (disabled when `walk_hops=2`).
- **2‑hop best path** (optional): best `anchor→w→item` product, scaled by `damping`.
- **Degree normalization** (optional): divide bonus by item degree (`invdeg`).
- **Clamp** the bonus to `max_bonus` and combine: `hybrid_score = sim + lambda_graph * bonus`.
- Reorder indices **1..k-1** by `(-hybrid_score, id)`; index `0` (the top dense item) is **pinned**.

### Configuration (YAML)
Add under `t2.hybrid:` (all keys validated by `scripts/validate_config.py`). Defaults shown below:

```yaml
# configs/config.yaml
t2:
  ...
  hybrid:
    enabled: false      # default OFF
    use_graph: true     # read from GEL edges (state.graph.edges)
    k_max: 128          # only re-rank within the top-K slice
    anchor_top_m: 8     # number of anchors from the top of the list
    walk_hops: 1        # 1 or 2; when 2, 1-hop is disabled
    edge_threshold: 0.10
    lambda_graph: 0.25  # blend weight for graph bonus
    damping: 0.50       # scales the 2-hop best-path term
    degree_norm: none   # or "invdeg"
    max_bonus: 0.50     # clamp on absolute graph bonus
```

**Notes & invariants**
- With `enabled=false` **or** when the GEL edge map is empty, order is unchanged; metrics record `hybrid_used=false`.
- When `walk_hops=2`, **1‑hop** is intentionally disabled; only 2‑hop contributes (matches tests and keeps behavior intuitive).
- With a single anchor (`anchor_top_m=1`) and `walk_hops=1`, 1‑hop is disabled to avoid trivial self‑reordering.
- Sort is stable and ties break by `id` ascending. Inputs are never mutated.

### Where it lives
- Pure function: `clematis/engine/stages/hybrid.py` → `rerank_with_gel(ctx, state, items)`.
- Wired in `clematis/engine/stages/t2.py` **after** dense rescoring and **before** residuals/logging.

### Observability
Adds compact fields to `t2.jsonl`:
```json
{
  "hybrid_used": true,
  "hybrid": {
    "k_considered": 32,
    "k_reordered": 5,
    "anchor_top_m": 8,
    "walk_hops": 2,
    "edge_threshold": 0.1,
    "lambda_graph": 0.25,
    "damping": 0.5,
    "degree_norm": "invdeg",
    "k_max": 128
  }
}
```

### Caches & freshness
- The **stage cache (T2)** stores the post‑hybrid list. If GEL edges evolve between identical queries **without an Apply**, you may see a **stale re‑rank** until the stage cache TTL expires or the orchestrator cache key changes.
- The **orchestrator cache (PR15)** remains version‑aware and invalidated on Apply; enabling hybrid doesn’t change those semantics.

### Tests
- Unit: `tests/test_t2_hybrid.py` — disabled path, no‑edges path, rank‑shift with edges, 2‑hop behavior, tie‑break, clamps, degree normalization.

### Enable and try it
```bash
# In configs/config.yaml
# t2.hybrid.enabled: true
python3 scripts/run_demo.py
jq '.hybrid_used, .hybrid' < ./.logs/t2.jsonl | tail -n2
```
## PR24 — GEL: Merge / Split / Promotion (contracts; metadata-only in HS1)

Adds deterministic **interfaces and stubs** to manage higher-order structure in the Graph Evolution Layer (GEL). Default **OFF**; enabling writes metadata and logs but does **not** change retrieval or application behavior in HS1.

### What lands in PR24
- **Merge candidates**: `merge_candidates(ctx, state)` → list of components with `{nodes, size, avg_w, diameter, signature}`; sorted by `(-avg_w, -size, signature)`.
- **Apply merge**: `apply_merge(ctx, state, cand)` → records an event in `state.graph.meta.merges`; **does not** mutate edges in HS1.
- **Split candidates**: `split_candidates(ctx, state)` → weak-edge cuts produce `{parts: [list[str], list[str], ...], signature}` based on `split.weak_edge_thresh` and `min_component_size`.
- **Apply split**: `apply_split(ctx, state, cand)` → records an event in `state.graph.meta.splits`; **no** edge mutation in HS1.
- **Promotions**: `promote_clusters(ctx, state, clusters)` deterministically proposes concept nodes with ids `c::<lexmin>` and labels per mode (`lexmin` or `concat_k`). `apply_promotion(ctx, state, promo)` creates a concept node (if absent) and attaches `rel="concept"` edges to members with a bounded weight; idempotent.

### Snapshot & schema
- **`graph_schema_version`** is still written at the **top level** (e.g., `"v1"`), and PR24 adds a richer `gel.meta` block under the GEL section.
- GEL meta includes: `merges[]`, `splits[]`, `promotions[]`, and `concept_nodes_count`, and is tagged internally with `meta.schema = "v1.1"`.
- Loader remains backward-compatible: legacy snapshots without a `gel` block, or with only a compact `graph` summary, load as empty GEL.

### Validation (config surface)
The validator now accepts and bounds the following keys (all optional; defaults provided):
```yaml
graph:
  merge: { enabled: false, min_size: 3, min_avg_w: 0.20, max_diameter: 2, cap_per_turn: 4 }
  split: { enabled: false, weak_edge_thresh: 0.05, min_component_size: 2, cap_per_turn: 4 }
  promotion: { enabled: false, label_mode: lexmin, topk_label_ids: 3, attach_weight: 0.5, cap_per_turn: 2 }
```

# HS1 wrap-up

**Status:** PR17–PR24 landed. HS1 hardening complete with:
- Config hygiene & validator (PR17)
- Snapshot inspector & schema tagging (PR18)
- Log schemas & rotation helper (PR19)
- T4 property tests + perf guardrails + microbench (PR20)
- GEL edge update/decay (PR22)
- Hybrid dense+graph re-ranking (PR23)
- GEL merge/split/promotion contracts & snapshot meta (PR24)

**Runtime behavior:** Deterministic; feature flags default OFF for GEL and Hybrid. When disabled, outputs remain bit-for-bit identical to pre-HS1.

**Next tracks (post-HS1, optional):**
- M5A — LLM productization with deterministic fixtures.
- M5B — Retrieval quality & eval (nDCG@k, hit@k) with richer GEL features.
- (Note: we changed those to be tucked into M7 and M10, reverse order)
```

## M5 — Scheduler (PR26 wiring, logging only; OFF by default)

PR26 wires the scheduler at **stage boundaries** and writes `./.logs/scheduler.jsonl`. It injects immutable per-slice caps into `ctx.slice_budgets` (read by T1/T2/T3), evaluates yield reasons at boundaries (**WALL_MS > BUDGET_* > QUANTUM_EXCEEDED**), and logs them. **No preemption/rotation is enforced yet** (that’s PR27). With `scheduler.enabled: false`, behavior and logs remain byte-for-byte identical to pre-M5.

**What changed**
- Orchestrator: attaches `ctx.slice_budgets` + `slice_idx` when enabled; logs scheduler events after T1/T2/T3/T4/Apply via `append_jsonl("scheduler.jsonl", …)`.
- Stages (pure, read-only caps):
  - T1 respects `t1_iters` (depth) and `t1_pops` (queue); metrics include `iters`, `pops`.
  - T2 uses only the first `t2_k` retrieved items this slice; metrics add `k_used`.
  - T3 clamps plan ops to `t3_ops` this slice.
- `turn.jsonl` gains `slice_idx`/`yielded` fields **only when enabled**.

**Enable (example)**
```yaml
# configs/config.yaml
scheduler:
  enabled: true
  policy: round_robin
  quantum_ms: 20
  budgets: { t1_iters: 50, t2_k: 64, t3_ops: 3, wall_ms: 200 }

  fairness: { max_consecutive_turns: 1, aging_ms: 200 }  # logged only in PR26
```


## M5 — Scheduler (PR27 fairness + enforced yields + ready-set; OFF by default)

PR27 builds on PR26. Summary:
- **Fairness**: enforce `max_consecutive_turns` and deterministic aging (`idle_ms // aging_ms`) in `fair_queue`. No RNG; lex tie-break.
- **Enforced yields**: at T1/T2/T3/T4/Apply boundaries, when a reason is hit, the slice is cut and control returns to the scheduler.
- **Ready-Set hook**: `agent_ready(ctx, state, agent_id) -> (bool, str)` gate. Default always true; overrideable and deterministic.
- **Rotation**: _deferred to PR28_ (driver-level). Expect bursty `max_consecutive_turns > 1` on round-robin until then.

**What changed**
- `clematis/scheduler.py`: `next_turn` now respects `max_consecutive_turns` and aging, and can return `"RESET_CONSEC"`; `on_yield` updates clocks/counters and supports `reset=True`.
- `clematis/engine/orchestrator.py`: enforces yields at stage boundaries; augments `turn.jsonl` with `slice_idx`, `yielded: true`, `yield_reason`; writes `scheduler.jsonl` entries with `"enforced": true`; exposes `agent_ready(...)`.

**Logging additions**
- `scheduler.jsonl`: adds `"enforced": true` when a yield is applied.
- `turn.jsonl`: slice entries include `slice_idx`, `yielded: true`, `yield_reason`.
- (Queue rotation and `queue_before/queue_after` are planned for PR28.)


## M5 — Scheduler (PR28 docs, demo, queue rotation; OFF by default)

PR28 adds a rotation-aware demo and documentation updates. No core behavior changes beyond what PR27 introduced.

**What’s new**
- **Queue rotation (RR only, demo/driver level):** after an enforced yield, the selected agent is rotated head→tail. `fair_queue` does **not** rotate.
- **Pick-reason passthrough:** the demo passes `pick_reason` into the orchestrator so `scheduler.jsonl` includes it.  
- **Full `queue_before/queue_after` logging** is deferred to **PR29** (driver-authored record).

**Run the demo**
```bash
# Multi-agent loop with small caps to force frequent yields & RR rotation
python3 scripts/run_demo.py \
  --agents AgentA,AgentB,AgentC \
  --policy round_robin \
  --steps 6 \
  --t1-iters 1 --t2-k 1 --t3-ops 1

# Fair-queue (no rotation), deterministic aging tiers
python3 scripts/run_demo.py \
  --agents AgentA,AgentB,AgentC \
  --policy fair_queue \
  --steps 6
```

**Inspect logs**
```bash
tail -n 3 ./.logs/scheduler.jsonl | jq .
tail -n 3 ./.logs/turn.jsonl | jq .
```
You should see `pick_reason` on scheduler records when provided (e.g., `"ROUND_ROBIN"`, `"AGING_BOOST"`, or `"RESET_CONSEC"`).  
In RR, the demo prints the queue transition like `['A','B','C'] -> ['B','C','A']` whenever a yield happens.

**Examples (configs)**

Use the example configs under `examples/scheduler/` to get predictable behavior out-of-the-box:

- `examples/scheduler/round_robin_minimal.yaml` — frequent yields & visible RR rotation
- `examples/scheduler/fair_queue_aging.yaml` — aging tiers; **no rotation**
- `examples/scheduler/quantum_vs_wall.yaml` — shows precedence (WALL > budgets > quantum)

Run them with the demo:
```bash
python3 scripts/run_demo.py --config examples/scheduler/round_robin_minimal.yaml \
  --agents AgentA,AgentB,AgentC --steps 6

python3 scripts/run_demo.py --config examples/scheduler/fair_queue_aging.yaml \
  --agents AgentA,AgentB,AgentC --steps 6 --policy fair_queue

python3 scripts/run_demo.py --config examples/scheduler/quantum_vs_wall.yaml \
  --agents AgentA,AgentB --steps 4
```

**FAQ**
- *Why no rotation in `fair_queue`?* Aging (idle time) deterministically controls selection priority; rotating would be cosmetic and can reduce determinism.
- *Why can RR still look bursty?* With `max_consecutive_turns > 1`, RR will produce bursts of length `max_consecutive_turns` by design. Rotation happens **after** a yield.
- *Where do I see `pick_reason`?* In `scheduler.jsonl` when the driver passes it (PR28 wires this via `scenario.run_one_turn(..., pick_reason=...)`).
- *Where are `queue_before/queue_after`?* Deferred to **PR29**, where the driver will author a single enriched `scheduler.jsonl` record per yield.
- *Does any of this change stage semantics?* No. Stages remain pure; only immutable per-slice budgets are injected; rotation occurs in the driver/demo layer.

**Notes & invariants**
- Rotation is performed **only** when `scheduler.enabled=true`, `policy=round_robin`, and a yield occurs at a stage boundary.
- `fair_queue` order is governed by deterministic aging; rotating would be cosmetic and is intentionally avoided.
- Determinism holds: same inputs + same config ⇒ identical picks and yields.

**Next (PR29)**
- Single-source `scheduler.jsonl` records authored by the driver, including `queue_before/queue_after` in addition to `pick_reason`.

M5 — Scheduler (PR29 driver-authored logs + CI golden identity; OFF by default)

PR29 finishes M5 by (1) making the driver the single source of truth for scheduler.jsonl and (2) adding a CI Golden Identity Guard that ensures scheduler.enabled=false stays byte-for-byte identical to pre-M5 (after normalization).

What changed
	•	Orchestrator suppression/capture: when TurnCtx has _driver_writes_scheduler_log=True and a dict on _sched_capture, the orchestrator captures the boundary event and does not write scheduler.jsonl.
	•	Driver-authored record: the driver/demo writes one enriched scheduler.jsonl entry per enforced yield:
	•	Always includes: turn, slice, agent, policy, reason, enforced:true, stage_end, quantum_ms, wall_ms, budgets, consumed, ms.
	•	Adds pick_reason (ROUND_ROBIN | AGING_BOOST | RESET_CONSEC).
	•	For round_robin only: adds queue_before and queue_after (head→tail rotation on yield). For fair_queue, these arrays are omitted/empty.
	•	Disabled path identity: with scheduler.enabled=false, no scheduler.jsonl is written. CI verifies that normalized logs match golden fixtures.

How to use (driver-authored logging)

The demo already enables capture and authors the record:
capture = {}
line = run_one_turn(agent, state, text, cfg,
                    pick_reason=pick_reason,
                    driver_logging=True,
                    capture=capture)
# on yield: rotate (RR), then write a single enriched record
event = {**capture, "pick_reason": pick_reason}
if policy == "round_robin":
    event["queue_before"] = queue_before
    event["queue_after"]  = queue_after
append_jsonl("scheduler.jsonl", event)

Enriched record (RR example)
{
  "turn": 123,
  "slice": 2,
  "agent": "Ambrose",
  "policy": "round_robin",
  "pick_reason": "ROUND_ROBIN",
  "reason": "BUDGET_T2_K",
  "enforced": true,
  "stage_end": "T2",
  "quantum_ms": 20,
  "wall_ms": 200,
  "budgets": {"t1_iters": 50, "t2_k": 64, "t3_ops": 3},
  "consumed": {"ms": 19, "t1_iters": 12, "t2_k": 64, "t3_ops": 0},
  "queue_before": ["Ambrose","Kafka","Ringer"],
  "queue_after":  ["Kafka","Ringer","Ambrose"],
  "ms": 0
}

For policy: fair_queue, queue_before/after are omitted or [].

CI Golden Identity Guard (disabled path)
	•	Script: scripts/ci_compare_golden.py runs one disabled turn, normalizes logs, and compares against fixtures in tests/golden/pre_m5_disabled/.
	•	Workflow: .github/workflows/ci.yml job golden_disabled_identity runs after tests and fails on any diff.
	•	Normalization: drops only volatile keys: ms, now, timestamp, ts, elapsed_ms, durations_ms, uuid, run_id.
	•	Update flow (local):
rm -rf ./.logs
python3 scripts/ci_compare_golden.py --update            # writes fixtures under tests/golden/pre_m5_disabled/
git add tests/golden/pre_m5_disabled
git commit -m "refresh disabled-path golden"

	•	Pass criteria: with scheduler.enabled=false, scheduler.jsonl must not exist and normalized .logs/* match golden exactly.

Invariants & rollback
	•	Determinism holds: same inputs + same config ⇒ identical picks/yields and identical logs.
	•	Stages remain pure; only immutable per-slice budgets are injected.
	•	Rollback is trivial: set scheduler.enabled=false. CI enforces that disabled-path logs match golden.

## M6 — Perf & Compaction (PR31: T1 caps + dedupe; OFF by default)

PR31 introduces **deterministic T1 compaction** controls that bound memory without changing behavior unless explicitly enabled. Defaults preserve the **disabled‑path identity** used by CI.

**What lands**
- **Frontier cap**: keeps the T1 priority queue size ≤ `caps.frontier` (in addition to `t1.queue_budget`; the effective limit is `min(queue_budget, caps.frontier)`).
- **Visited cap**: bounds the visited set using a deterministic FIFO‑on‑first‑visit set.
- **Dedupe ring**: short‑horizon push deduplication window to avoid thrash.
- **Determinism**: no RNG, no wall clock; tie‑breaks are lexicographic on `node_id`.

**Identity by default**
- Leave PR31 keys **absent** or set `perf.enabled: false` to keep byte‑for‑byte identity with PR29 goldens.
- The validator treats absent keys as disabled; when present, numeric values must be **≥ 1**.

**Config (enable example)**
```yaml
# configs/config.yaml
perf:
  enabled: true
  metrics:
    report_memory: true   # emit counters only when both gates are true
  t1:
    caps:
      frontier: 1000      # PQ cap (effective = min(queue_budget, frontier))
      visited: 5000       # visited bound (FIFO on first-visit)
    dedupe_window: 128    # recent-push window
```

**Validator**
- Strict mode enforces bounds and hygiene:
```bash
python3 scripts/validate_config.py --strict configs/config.yaml
```
- If `perf.enabled=false` but caps/dedupe are configured, the validator emits a warning (identity path remains in effect).

**Metrics (only when `perf.enabled` **and** `perf.metrics.report_memory` are true)**
- `t1_frontier_evicted` — number of PQ evictions performed to respect `caps.frontier`.
- `t1_dedup_hits` — pushes skipped because the node id was within the dedupe window.
- `t1_visited_evicted` — entries evicted from the bounded visited set.

**Tests**
- Data structures: `tests/util/test_ring.py` (dedupe ring), `tests/util/test_lru_det.py` (deterministic LRU set/map).
- Validator: `tests/config/test_validate_perf_t1_caps.py`.
- Stage smoke (placeholder): `tests/stages/test_t1_compaction.py` (skipped until a stable test graph fixture is exposed).

**Gotchas**
- Do **not** set `frontier/visited/dedupe_window` to `0` in strict configs; omit the keys instead to disable.
- When both `t1.queue_budget` and `perf.t1.caps.frontier` are present, T1 respects the stricter bound.
- Disabled path remains identical: with `perf.enabled=false` (or no perf keys), no new metrics keys appear and logs match PR29 goldens after normalization.

```

## M6 — Perf & Compaction (PR32: Size‑aware caches for T1/T2; OFF by default)

PR32 adds a deterministic **LRU‑by‑bytes** cache utility and wires it behind `perf.t1.cache` and `perf.t2.cache`. These caches **only avoid recomputation**; enabling them **does not** change outputs or ordering. Defaults keep the **disabled‑path identity** used by CI.

**What lands**
- Shared utility `LRUBytes(max_entries, max_bytes)` with deterministic LRU order (no clocks/RNG).
- T1/T2 size‑aware caches selected when `perf.enabled=true` **and** either cap > 0. Otherwise, legacy stage caches remain as shipped.
- Oversized insertions (`cost_bytes > max_bytes`) are **rejected** to prevent thrash.

**Identity by default**
- Keep `perf.enabled: false` or set caps to `0` (or omit) to stay byte‑for‑byte identical to PR29 goldens (and PR31 disabled path).
- New metrics appear **only** when both `perf.enabled=true` and `perf.metrics.report_memory=true`.

**Config (enable example)**
```yaml
# configs/config.yaml
perf:
  enabled: true
  metrics: { report_memory: true }
  t1:
    cache: { max_entries: 512, max_bytes: 2_000_000 }
  t2:
    cache: { max_entries: 512, max_bytes: 8_000_000 }
```

**Metrics (gated)**
- `t1.cache_evictions`, `t1.cache_bytes`
- `t2.cache_evictions`, `t2.cache_bytes`

**Determinism & behavior**
- Eviction is strictly LRU→MRU (unique positions), with stable tie‑break by order; no randomness or clocks.
- Keys must be stable strings (the stages provide canonical keys; no action needed for default demo).
- With caches ON vs OFF, **results are identical**; only performance changes.

**Validator**
- Accepts `perf.t1.cache.{max_entries,max_bytes}` and `perf.t2.cache.{max_entries,max_bytes}` (≥ 0).
- Warns if any caps are set while `perf.enabled=false` (identity path; caches disabled).

**Tests**
- Utility: `tests/util/test_lru_bytes.py` (sizes, multi‑eviction, MRU on get/put, oversize reject).
- Config: `tests/config/test_validate_perf_cache_pr32.py` (strict acceptance, disabled‑perf warnings, zero‑caps disabled).
- Stage: `tests/stages/test_t2_cache_smoke.py` (bytes‑mode selection, eviction math); metrics smoke is skipped pending a stable fixture.

**Gotchas**
- Don’t set negative caps (validator rejects); use `0` or omit to disable.
- If you enable both **legacy** stage caches and **perf** caches, behavior is still deterministic, but you may see overlapping effects; prefer one layer for clarity.
```

## M6 — Perf & Compaction (PR33: T2 fp16 store + fp32 math + partition-friendly reader; opt-in runtime)

PR33 introduces a compact on-disk embedding store for T2 with **fp16 storage** and **fp32 math**, plus a **partition-friendly reader**. Defaults preserve the disabled-path identity; the runtime reader is **opt-in**.

**What lands**
- `engine/util/embed_store.py`
  - `write_shard(dir, ids, embeds, *, dtype="fp16|fp32", precompute_norms: bool)`
  - `open_reader(root, partitions={enabled, layout, path}) -> EmbedReader`
  - `EmbedReader.iter_blocks(batch)` yields `(ids, embeds_fp32, norms_fp32)` deterministically; no RNG/clocks.
- `engine/stages/t2.py`
  - Reads PR33 config and **emits gated metrics** about store dtype, partition layout, and shard count when enabled.
  - **Opt-in runtime wiring**: when `perf.enabled=true` **and** `perf.t2.reader.partitions.enabled=true`, T2 retrieves **directly from the on-disk embed store** (cosine in fp32). Metrics reflect `tier_sequence: ["embed_store"]`. When disabled, the legacy in-memory path is used.
- `configs/validate.py`
  - Accepts `perf.t2.embed_store_dtype: {fp32, fp16}` and `perf.t2.precompute_norms: bool`.
  - Accepts `perf.t2.reader.partitions.{enabled, layout, path}` with `layout ∈ {owner_quarter, none}`.
  - Accepts T2 stage knobs: `t2.reader_batch` (≥ 1) and `t2.embed_root` (non-empty string path).
  - Warns when `embed_store_dtype=fp16` and `precompute_norms=false` (parity is better with fp32 norms).

**Identity by default**
- With `perf.enabled: false` (or when reader/partitions are disabled), T2 uses the existing in-memory store and logs remain identical to PR29/PR31/PR32 disabled path.

**Config (opt-in runtime reader; identity remains default)**
```yaml
perf:
  enabled: true
  metrics: { report_memory: true }
  t2:
    embed_store_dtype: fp16        # or fp32 (identity)
    precompute_norms: true         # recommended for parity
    reader:
      partitions:
        enabled: true
        layout: owner_quarter      # or "none"
        path: ./data/t2            # root dir for shards
t2:
  reader_batch: 4096               # optional; default 8192
  embed_root: ./data/t2            # optional; overrides default root when set

Shard layout
	•	meta.json {schema:"t2:1", embed_dtype:"fp16|fp32", norms:bool, dim:int, count:int}
	•	embeddings.bin float16|float32 [N,D] (row-major)
	•	ids.tsv one id per line (lex-stable)
	•	norms.bin optional float32 [N] when precompute_norms=true

Partitions (deterministic discovery)
	•	layout: owner_quarter supports either shard subdirs or meta.json directly under the quarter path:
	•	<root>/<owner>/<quarter>/<shard-*>/meta.json
	•	<root>/<owner>/<quarter>/meta.json

Parity guarantees
	•	Math is performed in fp32; fp16 storage is cast up when read.
	•	Tests enforce Top-K identical to an fp32 baseline and score drift ≤ 1e-6.
	•	Ordering is stable: sort by (-score, id).

How to Use: (offline tooling)
from clematis.engine.util.embed_store import write_shard, open_reader
# Write
write_shard("./data/t2/shard-000", ids, embeds, dtype="fp16", precompute_norms=True)
# Read
reader = open_reader("./data/t2", partitions={"enabled": True, "layout": "owner_quarter"})
for ids_b, vecs_b, norms_b in reader.iter_blocks(batch=1024):
    ...  # feed your retrieval/scoring code

Metrics (gated; appear only if perf.enabled=true and report_memory=true)
	•	t2.embed_dtype = fp32
	•	t2.embed_store_dtype = fp16|fp32 (from shard meta if discovered)
	•	t2.precompute_norms = true|false
	•	t2.reader_shards = shard count (when reader is enabled)
	•	t2.partition_layout = owner_quarter|none
	•	t2.tier_sequence = ["embed_store"] when the reader is active

Tests
	•	Parity: tests/t2/test_fp16_parity.py
	•	Partitions: tests/t2/test_partition_reader.py
	•	Back-compat: tests/t2/test_reader_backcompat.py
	•	Integration (runtime flip): tests/stages/test_t2_reader_integration.py
	•	Metrics (placeholder, skipped until stage fixture is exposed): tests/t2/test_metrics_gated.py
	•	Validator for PR33 keys: tests/config/test_validate_perf_t2_fp16.py, tests/config/test_validate_t2_reader_batch.py

Gotchas
	•	If you use fp16, prefer precompute_norms: true for best parity.
	•	Reader currently requires meta.json; shards without it will be rejected.
	•	Disabled path remains the default; enabling the reader is opt-in and gated under perf.enabled=true.

-## M6 — Gate C (PR33.5): T2 reader parity & runtime‑flip invariants

**What this adds (no behavior change by default)**
- A required CI gate that proves the PR33 reader is **drop‑in**: Top‑K IDs/order identical and score drift ≤ `1e-6` across `{embed_store_dtype∈{fp32,fp16}} × {precompute_norms∈{true,false}}`.
- Disabled‑path identity stays enforced: with `perf.enabled=false`, the reader **must not engage** and **no new metrics keys** appear.

**Signals (only when the reader actually engages and gates are ON)**
- `metrics.tier_sequence` includes `"embed_store"`.
- `metrics.reader` object appears with: `{embed_store_dtype, precompute_norms, layout, shards, partitions?}`.

**How to run locally**
```bash
# Seed a tiny deterministic store (used by tests)
python3 scripts/seed_tiny_shard.py --out ./.data/tiny

# Reader must NOT engage when perf is OFF (identity gate)
pytest -q tests/t2/test_t2_reader_gate_identity.py

# Parity + integration suite (dtype × norms matrix covered in CI)
pytest -q \
  tests/t2/test_fp16_parity.py \
  tests/t2/test_partition_reader.py \
  tests/t2/test_reader_backcompat.py \
  tests/stages/test_t2_reader_integration.py
```

**CI (required checks)**
- Workflow: `.github/workflows/t2_reader_parity.yml`
- Jobs required in branch protection:
  - **Gate C — T2 Reader Parity / Parity matrix (dtype × norms)**
  - **Gate C — T2 Reader Parity / Reader must not engage (perf=OFF)**

**Notes**
- Env overrides used by CI are safe and optional locally: `PERF_T2_DTYPE=fp32|fp16`, `PERF_T2_PRECOMPUTE_NORMS=true|false`.
- Sorting is strictly `(-score, lex(id))` in all paths; ties are stable.
- Default configs keep `perf.enabled=false`, so PR33.5 does **not** change observable behavior.


## M6 — Gate B (metrics-only guard; CI)

**Purpose**
- Prevent any performance/reader/snapshot counters from appearing in logs unless **both** `perf.enabled=true` **and** `perf.metrics.report_memory=true`.
- Keeps the **disabled path** byte-for-byte identical to PR29 goldens (after normalization).

**How to run locally**
```bash
# OFF: no perf and no reporting -> absolutely no perf metrics in logs
rm -rf ./.logs
python3 scripts/run_demo.py --config .ci/gate_b_off.yaml --steps 2 || true
python3 scripts/ci_gate_b_assert.py OFF

# ON but NO REPORT: feature gate on, report gate off -> still no perf metrics in logs
rm -rf ./.logs
python3 scripts/run_demo.py --config .ci/gate_b_on_nometrics.yaml --steps 2 || true
python3 scripts/ci_gate_b_assert.py ON_NO_REPORT

# ON + REPORT: both gates -> perf metrics may appear (bounded, deterministic)
rm -rf ./.logs
python3 scripts/run_demo.py --config .ci/gate_b_on_withmetrics.yaml --steps 2 || true
python3 scripts/ci_gate_b_assert.py ON_WITH_REPORT
```

**Signals & invariants**
- With gates **OFF** or **ON_NO_REPORT**:
  - No `tier_sequence` appears in T2/T3 logs.
  - No `reader`, `embed_store_*`, or `snap.*` fields appear.
- With **ON_WITH_REPORT**, metrics that do appear are limited to the M6 set, e.g.:
  - `t1.*`/`t2.*` cache counters (PR31/PR32), `t2.reader` fields when the opt-in reader is actually engaged (PR33), and snapshot counters if enabled in a future PR.
- CI job (optional): `.github/workflows/gate_b.yml` runs the three scenarios above and fails on leakage.

## M6 — Snapshots (PR34/34.1): zstd compression + delta mode (safe fallback; defaults OFF)

**What this adds (no behavior change by default)**
- **Read/Write** support for full and delta snapshots (writer added in **PR34.1**); optional zstd compression (`level` 1–19).
- Delta snapshots (`delta_mode`) reconstruct from a baseline identified by `etag`; if baseline is missing or mismatched, the reader logs `SNAPSHOT_BASELINE_MISSING` and deterministically falls back to the matching full snapshot (or `{}` as a last resort).
- With `perf.enabled=false`, behavior/logs remain identical to pre‑M6.

**Config (example)**
```yaml
# examples/perf/snapshots.yaml
perf:
  enabled: true
  metrics: { report_memory: false }   # counters deferred; no new metrics in PR34/34.1
  snapshots:
    compression: zstd                 # allowed: none | zstd
    level: 5                          # 1..19
    delta_mode: true                  # write delta files when a matching baseline exists
```

**File formats (read/write)**
- Full: `snapshot-{etag}.full.json[.zst]`
- Delta: `snapshot-{etag_to}.delta.json[.zst]` with a small JSON header on line 1 and a delta payload on line 2+.

**API (offline usage)**
```python
from clematis.engine.snapshot import write_snapshot_auto, read_snapshot

# Write (auto-select delta vs full based on baseline availability)
path, wrote_delta = write_snapshot_auto(
    "./.data/snapshots",
    etag_from="aaaa",   # None for the first full snapshot
    etag_to="bbbb",
    payload={"store": {"nodes": {}}},
    compression="zstd",  # or "none"
    level=5,
    delta_mode=True,
)

# Read back
header, state = read_snapshot("./.data/snapshots", etag="bbbb")
```

**Run locally**
```bash
# Unit tests for codec, writer round‑trip, and safe fallback
pytest -q tests/snapshots/test_snapshot_delta_roundtrip.py \
         tests/snapshots/test_snapshot_writer_roundtrip.py \
         tests/snapshots/test_snapshot_fallback.py
```
> zstd support is optional; to read/write `.zst` files locally, install `zstandard` (already in dev/test extras).

**CI (smoke)**
- Workflow: `.github/workflows/snapshots_smoke.yml` runs the snapshot tests above on PRs that touch snapshot code.

**Notes**
- Defaults keep snapshots as uncompressed full files; enabling compression/delta changes persistence only, not semantics.
- ETags are computed over canonical JSON, keeping names deterministic; ordering is stable.
- No new metrics are emitted in PR34/34.1; any future snapshot counters must remain gated (`perf.enabled && perf.metrics.report_memory`).
```

## M6 — Docs & CLIs (PR35): quickstart, examples, and offline tooling

**Scope (no runtime behavior change)**
- Adds documentation and example configs for M6 perf features.
- Ships two **offline** CLIs: `mem_inspect.py` and `mem_compact.py`.
- Defaults keep the disabled‑path identity; tools never modify in‑place.

**Examples**
Located under `examples/perf/` (referenced throughout the docs):
- `caps.yaml` — PR31 T1 caps + dedupe ring
- `caches.yaml` — PR32 LRU‑by‑bytes caches (T1/T2)
- `fp16_reader.yaml` — PR33/33.5 fp16 store + partition‑friendly reader (opt‑in)
- `snapshots.yaml` — PR34 snapshot compression + delta (safe fallback)

**CLIs**

`mem_inspect` — list snapshot files and basic stats (works with `.json` and optional `.json.zst`).
```bash
python3 scripts/mem_inspect.py \
  --snapshots-dir ./.data/snapshots \
  --format json   # or: table
```
Typical JSON keys:
```json
{
  "root": "/abs/path", 
  "snapshots_dir": "/abs/path/.data/snapshots",
  "files": [
    {"name":"snapshot-aaaa.full.json","kind":"full","compressed":false,"level":0,"size_bytes":1234,"etag_to":"aaaa","delta_of":null,"codec":"none"}
  ],
  "summary": {"count":1,"total_bytes":1234,"compressed":0,"delta":0}
}
```

`mem_compact` — offline rewrite of **full** snapshots to a new directory with optional zstd compression; never in‑place.
```bash
# Dry‑run first
python3 scripts/mem_compact.py --in ./.data/snapshots --out ./.artifacts/compact --dry-run
# Then write (uncompressed)
python3 scripts/mem_compact.py --in ./.data/snapshots --out ./.artifacts/compact
# Or write compressed
python3 scripts/mem_compact.py --in ./.data/snapshots --out ./.artifacts/compact_zstd \
  --compression zstd --level 5
```
Notes:
- Requires `zstandard` only when writing/reading `.zst` files (already in dev/test extras).
- `--dtype fp32|fp16` updates header metadata only; it does **not** transform model weights.
- Delta emission is intentionally disabled in PR35 CLI to keep safety simple.

**Tests & CI**
- `tests/cli/test_mem_inspect.py`, `tests/cli/test_mem_compact.py` — smoke tests (exit 0, expected keys/files).
- CI workflow: `.github/workflows/cli_smoke.yml` runs these tests on PRs touching `scripts/`.

**Rollback & invariants**
- Tools are offline: they do not change runtime semantics or logs.
- With `perf.enabled=false`, behavior/logs remain identical to PR29 goldens (guarded in CI).

-## M7 — Retrieval Quality: shadow tracing (PR36, defaults OFF)

**Scope (no-op by default)**  
Adds the `t2.quality.*` surface and a shadow tracing path that **does not change rankings or metrics**. When the **triple gate** is ON:
- `perf.enabled: true`
- `perf.metrics.report_memory: true`
- `t2.quality.shadow: true` and `t2.quality.enabled: false`

…the system writes a single JSONL stream at `logs/quality/rq_traces.jsonl`. With defaults (both gates OFF), behavior is **identical** to pre‑PR36.

**Config (example)** — see `examples/quality/shadow.yaml`:
```yaml
perf:
  enabled: true
  metrics: { report_memory: true }
t2:
  quality:
    enabled: false         # PR36 forbids true (guarded by validator)
    shadow: true           # emit traces only; no ranking changes
    trace_dir: "logs/quality"
    redact: true           # redact human-readable fields in traces
```

**Determinism & trace schema**
Each trace record includes deterministic headers:
- `trace_schema_version` (starts at 1)
- `git_sha` (from `CLEMATIS_GIT_SHA` or `"unknown"`)
- `config_digest` (semantic knobs only; excludes `trace_dir`)
- `query_id` (Unicode **NFKC**, collapsed whitespace, lowercased; SHA1)
- `clock=0`, `seed=0`

**CI guard — Gate D (“shadow is a no‑op”)**  
Required check that asserts **no diffs** between baseline and shadow runs **except** for `rq_traces.jsonl`. Comparator script: `scripts/validate_noop_shadow.py`.

**Try it locally**
```bash
export CLEMATIS_CONFIG=examples/quality/shadow.yaml
pytest -q
python3 scripts/rq_trace_dump.py --trace_dir logs/quality --limit 5
```

**Invariants**
- Defaults keep **disabled‑path identity**.
- Enabling `t2.quality.enabled=true` is rejected in PR36 (validator guard); functional paths land in later PRs.

## M7 — Retrieval Quality: Lexical BM25 + Fusion (PR37, opt-in, gated)

**Scope (opt-in; defaults OFF)**  
Adds a deterministic lexical BM25 scorer over the current candidate set and a rank-based fusion step that combines semantic and lexical signals. When **enabled** and the **triple gate** is ON, we emit both **metrics** and **traces** for the enabled path. Disabled path remains byte-for-byte identical to pre-M7.

**Config (example)** — see [`examples/quality/lexical_fusion.yaml`](examples/quality/lexical_fusion.yaml):
```yaml
perf:
  enabled: true
  metrics: { report_memory: true }
t2:
  quality:
    enabled: true          # activate fusion
    shadow: false
    trace_dir: "logs/quality"
    redact: true
    lexical:
      bm25_k1: 1.2
      bm25_b: 0.75
      stopwords: en-basic   # or "none"
    fusion:
      mode: score_interp
      alpha_semantic: 0.6   # 0..1; higher = more semantic
```

> Note: In MMR, λ is the **diversity** weight (λ=0.0 → pure relevance; λ=1.0 → pure diversity). Ties break lex(id). Metrics: t2q.mmr.selected, t2q.mmr.lambda, t2q.diversity_avg_pairwise.

## Changelog & Releases
- See the [CHANGELOG](./CHANGELOG.md) for notable changes.
- Binary / source packages and release notes are on the [Releases] page.