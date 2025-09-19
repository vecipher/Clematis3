# Clematis v2 — Scaffold

Minimal scaffold matching the Clematis v2 steering capsule. Stages are pure, orchestrator handles I/O. 
This repo includes a runnable demo (`scripts/run_demo.py`) that exercises the turn loop and writes JSONL logs.

## Quick start

```bash
python3 scripts/run_demo.py
```

Logs are written to `.logs/` at the repo root.

## T1:
	•	pops = PQ pops
	•	iters = layers beyond seeds visited
	•	propagations = relaxations (edge traversals)

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

## What’s implemented

- **T1 — Keyword propagation (deterministic)**  
  Wavefront PQ with decay + budgets, cache keyed by graph `version_etag`. Deterministic seeding, tie‑breaking, and delta emission.
- **T2 — Semantic retrieval + residual (deterministic, tiered, offline)**  
  Deterministic embedding stub (BGEAdapter) + in‑memory index with exact/cluster/archive tiers. Emits small, monotonic residual nudges (never undoes T1). Cached per query.
- **T3/T4 — Stubs & meta‑filter**  
  T3 is placeholder; T4 accepts/filters proposed deltas; Apply persists and logs.

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
    template: "style_prefix| summary: {labels}. next: {intent}"
    include_top_k_snippets: 2
```

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
- `t2.jsonl` — retrieval/residual metrics per turn
- `t3_plan.jsonl`, `t3_dialogue.jsonl` — placeholders for policy/dialogue
- `t4.jsonl` — meta‑filter approvals/rejections
- `apply.jsonl` — state changes summary
- `turn.jsonl` — per‑turn roll‑up (durations, key metrics)
- `health.jsonl` — guardrail flags

## Development tips

- Deterministic tests use the hash‑based embeddings; if a test filters too hard on cosine, set `t2.sim_threshold` to `-1.0` in that test to include all candidates and let other filters (recency, tiers) drive behavior.
- Cache behavior is configurable per stage; tests include **first‑call miss → second‑call hit → invalidation by etag/index change**.
- Keep stages **pure**. Instrumentation and persistence live in the orchestrator and adapters.