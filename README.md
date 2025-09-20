# Clematis v2 — Scaffold (M1 + M2)

Minimal scaffold matching the Clematis v2 steering capsule. Stages are pure; the orchestrator handles I/O and logging.  
The demo exercises the canonical turn loop and writes structured JSONL logs for first‑class observability.

## Quick start


```bash
# Run the end-to-end demo turn (writes logs under .logs/)
python3 scripts/run_demo.py

# Run unit tests
pytest -q
```

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

**Expected success output**:
```
OK
t4.cache: ttl_sec=600 namespaces=['t2:semantic'] cache_bust_mode=on-apply
```

**On error** you get explicit field paths, e.g.:
```
t4.weight_min/weight_max must satisfy weight_min < weight_max
t2.k_retrieval must be >= 1
```

**TTL keys:**
- Stage caches (T1/T2) use `ttl_s`.
- Orchestrator cache (PR15) uses `t4.cache.ttl_sec` (alias: accepts `ttl_s`; normalized to `ttl_sec`).
```

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
    provider: qwen
    model: qwen3-4b-instruct
    temperature: 0.2
    max_tokens_default: 256
    timeout_s: 30
```

**Provide an adapter at runtime** (no network in CI):
```python
from clematis.adapters.llm import QwenLLMAdapter

def qwen_chat(prompt: str, *, model: str, max_tokens: int, temperature: float, timeout_s: int) -> str:
    # Your client code to call Qwen (DashScope/Ollama/etc.) and return text
    return "..."

state["llm_adapter"] = QwenLLMAdapter(call_fn=qwen_chat, model="qwen3-4b-instruct", temperature=0.2, timeout_s=30)
```

**Deterministic tests**
- CI uses `DeterministicLLMAdapter` (offline). No network calls. Token caps enforced the same as rule-based `speak()`.

**Fallback behavior**
- If `t3.backend: llm` but no adapter is present, orchestrator falls back to rule-based and logs:
  - `t3_plan.jsonl`: `backend_fallback`, `fallback_reason: "no_adapter"`
  - `t3_dialogue.jsonl`: `backend: "rulebased"`

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

## Logs

JSONL files are written to `.logs/`:

- `t1.jsonl` — propagation metrics per turn
- `t2.jsonl` — retrieval/residual metrics per turn (includes `cache_hit: bool`, `cache_size: int`)
- `t3_plan.jsonl` — plan metrics and policy details
- `t3_dialogue.jsonl` — dialogue synthesis metrics
- `t4.jsonl` — meta‑filter approvals/rejections
- `apply.jsonl` — state changes summary (includes `cache_invalidations: int`)
- `turn.jsonl` — per‑turn roll‑up (durations, key metrics)
- `health.jsonl` — guardrail flags

## Development tips

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

# HS1:

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
|  | `delta_norm_cap_l2` | number 
> 0 |  |
|  | `novelty_cap_per_node` | number 
> 0 |  |
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
- Exit `2` if no snapshot exists or cannot be read (runtime remains tolerant).
- New snapshots include `schema_version: v1`; legacy snapshots may show `unknown`.

**Fields shown** (when available): `schema_version`, `version_etag`, `nodes`, `edges`, `last_update`, and T4 caps (`delta_norm_cap_l2`, `novelty_cap_per_node`, `churn_cap_edges`, `weight_min`, `weight_max`).