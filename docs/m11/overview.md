# M11 — Field‑Control GEL (HS1) Overview (v3)

**Status:** Contracts + plumbing complete. Default **OFF**. Deterministic.
**Scope (v3):** Record co‑activations during retrieval, maintain an optional concept graph (nodes/edges) with update + decay, and expose **optional** structural ops (merge/split/promotion).
**Not in v3:** The *field‑control “nudge” planner* (v4 topic).

---

## TL;DR

- When `graph.enabled=false` (the default), the engine never touches GEL:
  - No graph updates, no merges/splits/promotions, **no `gel.jsonl`** logs.
  - Disabled path remains **byte‑for‑byte identical** to pre‑GEL behavior.
- When enabled, GEL can:
  1) Observe retrieval results and **strengthen** edges between co‑activated concepts.
  2) **Decay** edge weights over turns using a half‑life.
  3) **Optionally** run deterministic graph maintenance: merge/split/promotion. *(All OFF by default in v3.)*

---

## Where things live

- **Engine:** `clematis/engine/gel.py`
  - `observe_retrieval(...)` — co‑activation logging + edge updates
  - `tick(decay_dt)` — half‑life decay
  - `merge_candidates/apply_merge`, `split_candidates/apply_split`
  - `promote_clusters/apply_promotion`
- **Config validation & defaults:** `configs/validate.py` (`graph.*` subtree)
- **Orchestrator hooks:** guarded calls to GEL:
  - `gel_observe` after T2 retrieval (shadow‑safe; disabled path = no‑op)
  - `gel_tick` after Apply (time‑step decay)
- **Logs:** `logs/gel.jsonl` (only when `graph.enabled=true`) plus GEL counters in `logs/health.jsonl`
- **Snapshots:** `clematis/engine/snapshot.py` includes `state.graph.*` with `graph_schema_version`

---

## Configuration (reference)

```yaml
graph:
  enabled: false                # Default OFF — identity path when false
  coactivation_threshold: 0.20  # Min normalized score for co-activation
  observe_top_k: 8              # Pairing cap per retrieval observation
  update:
    mode: additive              # {additive|proportional}
    alpha: 0.02                 # Update step size
    clamp_min: -1.0
    clamp_max:  1.0
  decay:
    half_life_turns: 200        # Half-life in turns
    floor: 0.0                  # Minimum absolute weight after decay
  merge:
    enabled: false              # v3 default OFF (contracts present)
    min_size: 3                 # Small clusters only
    cap_per_turn: 2             # Deterministic cap
  split:
    enabled: false              # v3 default OFF
    weak_edge_thresh: 0.05      # Remove intra-cluster edges below this
  promotion:
    enabled: false              # v3 default OFF
    label_mode: concat_k        # {lexmin|concat_k}
    attach_weight: 0.30         # Edge weight for new promoted node
```

### Identity guarantee (disabled path)
- You may include the entire `graph` subtree in config with `enabled:false` and arbitrary non‑default values — runtime remains identical and **no GEL log file is created**.

---

## Algorithms (deterministic summaries)

### 1) Co‑activation update (`observe_retrieval`)
For a retrieval step producing items \( \{(i, s_i)\} \) with scores \( s_i \in [0,1] \), define the active set
\( A = \{ i \mid s_i \ge \theta \} \) where \( \theta = \) `coactivation_threshold`.
Create up to `observe_top_k` strongest pairs \( (u,v) \in A \) (lexicographically tie‑broken).

- **Additive mode:**
  \( w_{t+1}(u,v) = \mathrm{clamp}\big( w_t(u,v) + \alpha \cdot m(u,v),\; \mathrm{clamp\_min},\mathrm{clamp\_max} \big) \)
  where \( m(u,v) = \tfrac{s_u + s_v}{2} \) or a deterministically chosen pairwise score.
- **Proportional mode:**
  \( w_{t+1}(u,v) = \mathrm{clamp}\big( w_t(u,v)\cdot(1+\alpha \cdot m(u,v)),\; \mathrm{clamp\_min},\mathrm{clamp\_max} \big) \)

All pair selection and tie‑breaks are **lexicographic** to preserve determinism.

### 2) Decay (`tick`)
Half‑life decay over integer turns \( \Delta t \):
\( w_{t+\Delta t} = \max\big( \mathrm{floor},\; w_t \cdot 2^{-\Delta t / H} \big) \)
where \( H = \) `half_life_turns`. Applied uniformly to all edges (deterministic iteration order).

### 3) Merge (optional; default OFF)
- Candidate clusters ranked by `(avg_weight DESC, size ASC, diameter ASC, id LEX)`.
- Apply at most `cap_per_turn` merges; edges/nodes updated deterministically.

### 4) Split (optional; default OFF)
- Remove intra‑cluster edges with weight `< weak_edge_thresh`; if connected components split, instantiate new clusters. Deterministic order.

### 5) Promotion (optional; default OFF)
- Promote compact clusters to a new concept node with label strategy:
  - `lexmin`: pick lexicographically minimal member label
  - `concat_k`: deterministically concatenate top‑k member labels
- Connect promoted node back to members with `attach_weight`. Order is lexicographic.

> **Note:** Merge/Split/Promotion exist to future‑proof the graph. They are intentionally **disabled** in v3 by default.

---

## Logging & Observability

When `graph.enabled=true`:
- **`logs/gel.jsonl`**: event stream with records like:
  - `{"ts": ..., "event": "observe_retrieval", "pairs": [...], "weights_applied": ...}`
  - `{"ts": ..., "event": "edge_decay", "count": N, "half_life_turns": H}`
  - `{"ts": ..., "event": "merge_attempt", "candidates": [...], "applied": M}`
  - `{"ts": ..., "event": "split_attempt", "clusters": [...], "applied": S}`
  - `{"ts": ..., "event": "promotion_attempt", "candidates": [...], "applied": P}`
- **`logs/health.jsonl`**: summary counters (GEL_EDGES, GEL_MERGES, GEL_SPLITS, …)

When `enabled=false`, **no `gel.jsonl`** is written and no GEL counters are emitted.

---

## Determinism & Identity

- **Determinism:** All candidate lists and updates use deterministic sorting and lexicographic tie‑breaks. No RNG, no wall‑clock dependence in decisions.
- **Identity path:** With `enabled=false`, there is no difference in outputs, snapshots, or logs compared to a build without GEL.

---

## Safe enablement (v3)

1. Start with everything disabled:
   ```yaml
   graph: { enabled: false }
   ```
2. Turn on **observe+decay only**:
   ```yaml
   graph:
     enabled: true
     merge: { enabled: false }
     split: { enabled: false }
     promotion: { enabled: false }
   ```
3. Inspect `logs/gel.jsonl` and snapshots to verify edges evolve as expected.
4. (Optional) Experiment with merge/split/promotion **offline**; keep them `false` in production.

---

## Minimal examples

### Disabled (identity path)
```yaml
graph:
  enabled: false
  coactivation_threshold: 0.33
  observe_top_k: 7
  update: { mode: proportional, alpha: 0.07, clamp_min: -0.9, clamp_max: 0.9 }
  decay:  { half_life_turns: 123, floor: 0.01 }
  merge:  { enabled: true,  min_size: 2, cap_per_turn: 3 }   # inert while enabled=false
  split:  { enabled: true,  weak_edge_thresh: 0.05 }        # inert while enabled=false
  promotion: { enabled: true, label_mode: concat_k }        # inert while enabled=false
```

### Enabled (observe + decay only)
```yaml
graph:
  enabled: true
  coactivation_threshold: 0.20
  observe_top_k: 8
  update: { mode: additive, alpha: 0.02, clamp_min: -1.0, clamp_max: 1.0 }
  decay:  { half_life_turns: 200, floor: 0.0 }
  merge:  { enabled: false }
  split:  { enabled: false }
  promotion: { enabled: false }
```

## Examples

The repository includes ready-to-run configs:

- **Enabled (observe + decay only; ops OFF):** `examples/gel/enabled.yaml`
- **Disabled (identity path):** `examples/gel/disabled.yaml`

Run either via the smoke harness:

```bash
python scripts/examples_smoke.py --examples examples/gel/enabled.yaml
python scripts/examples_smoke.py --examples examples/gel/disabled.yaml
```

Or run the built-in set (includes both):

```bash
python scripts/examples_smoke.py --all
```

---

## Snapshot schema

Snapshots include `state.graph.nodes`, `state.graph.edges`, `state.graph.meta`, and a `graph_schema_version`. Round‑trips are deterministic; enabling GEL changes snapshots only when GEL is actually enabled.

---

## FAQ

**Q: Does enabling GEL change retrieval ranking?**
A: Not in v3. GEL can write state and logs; T2’s hybrid reranker has a **flagged path** and remains **OFF by default**.

**Q: Can GEL hurt identity tests?**
A: Not when `enabled=false`. We maintain tests to assert disabled‑path identity, including the presence of a full `graph.*` subtree in config.

**Q: Why keep merge/split/promotion off by default?**
A: They’re conservative maintenance passes intended for controlled experiments and v4 evolution, not baseline v3 behavior.

---

## Relation to v4 (Field‑Control)

v4 introduces a *field‑control nudge planner* that consumes GEL as a potential‑field substrate to propose nudges (new edges, bridge evidence, motif promotions, etc.). v3 lays the substrate with strict determinism and default‑OFF gates.

---

## Tests & Identity (how we verify this)

**Status:** ✅ PR92, ✅ PR93, and ✅ PR94 **landed**. Disabled path stays byte‑identical and produces **no GEL artifacts**. Examples ship under `examples/gel/`.

- **PR92 — Config + runtime log identity**
  - `tests/test_identity_disabled_path.py::test_disabled_path_identity_config_roundtrip_graph_subtree`
    Proves a full `graph.*` subtree with `enabled:false` is inert post-validation (equal to config without `graph`) via `_strip_perf_and_quality_and_graph(...)`.
  - `tests/test_identity_disabled_path.py::test_disabled_path_runtime_no_gel_and_log_identity`
    Runs a tiny smoke turn twice (baseline vs. `graph.enabled:false`) using `run_smoke_turn`; asserts **no `gel.jsonl`** and **byte-identical** logs.

- **PR93 — Runtime identity + snapshots**
  - `tests/test_gel_disabled_path_runtime.py::test_gel_disabled_path_runtime_logs_and_snapshots`
    Repeats the smoke run and additionally asserts **snapshot parity** (same files; identical content hashes when present).

**Shared helpers** (all in `tests/helpers/identity.py`):

- Normalizers for deterministic diffing: `normalize_json_line(...)`, `normalize_json_lines(...)`, `normalize_logs_dir(...)`
- Config strippers: `_strip_perf_and_quality(...)`, `_strip_perf_and_quality_and_graph(...)`
- Snapshot utilities: `collect_snapshots_from_apply(...)`, `hash_snapshots(...)`

These tests run with `CI=true` and `CLEMATIS_NETWORK_BAN=1`. If `run_smoke_turn` is unavailable on a branch, runtime tests **skip** rather than fail.

---

## Inspecting GEL state (CLI)

You can peek into snapshots with the built‑in inspector:

```bash
# Inspect the most recent snapshot (macOS/Linux)
clematis cli inspect-snapshot "$(ls -t logs/snapshots/*.jsonl | head -n1)"
```

GEL lives under `state.graph.*` in the snapshot. Handy one‑liners:

```bash
# Show graph metadata
clematis cli inspect-snapshot "$(ls -t logs/snapshots/*.jsonl | head -n1)" | jq '.state.graph.meta'

# Count nodes and edges
clematis cli inspect-snapshot "$(ls -t logs/snapshots/*.jsonl | head -n1)" | jq '{nodes: (.state.graph.nodes|length), edges: (.state.graph.edges|length)}'

# Top 10 strongest edges (u, v, weight)
clematis cli inspect-snapshot "$(ls -t logs/snapshots/*.jsonl | head -n1)" \
  | jq -r '.state.graph.edges | sort_by(-.weight) | .[:10] | .[] | [.u, .v, .weight] | @tsv'

# List node labels with degree
clematis cli inspect-snapshot "$(ls -t logs/snapshots/*.jsonl | head -n1)" \
  | jq -r '
    .state.graph as $g
    | ($g.nodes | map({id, label}) | from_entries? // (reduce .[] as $n ({}; .[$n.id] = $n.label))) as $label
    | $g.edges
    | (reduce .[] as $e ({}; .[$e.u]+=1 | .[$e.v]+=1))
    | to_entries
    | sort_by(-.value)[:10]
    | .[] | "\($label[.key] // .key)\t\(.value)"'
```

> Notes
> - When `graph.enabled=false` (default), snapshots may have `state.graph` absent or empty.
> - The commands above rely on `jq`. If `jq` isn’t installed, install it or pipe to a file to view raw JSON.

**Tip:** To generate a quick snapshot with GEL substrate enabled (observe+decay only), run:

```bash
python scripts/examples_smoke.py --examples examples/gel/enabled.yaml
```

---

## Pointers

- M11 (HS1/GEL) substrate docs + examples: complete (PR91–PR94). Close out with **PR96 — CHANGELOG & milestone**.
- See also: **Reflection (M10)** → `docs/m10/reflection.md` (section “Next: HS1/GEL”).


---

_Milestone note:_ **v3 HS1 (GEL substrate) is complete.** The field‑control **nudge planner** is **deferred to v4**.
