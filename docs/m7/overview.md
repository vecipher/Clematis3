

# M7 Overview — Observability & DevEx

This milestone adds **opt‑in quality paths**, **deterministic traces/metrics**, and a small evaluation harness. Defaults remain conservative: the **disabled path is unchanged** and fully rollbackable.

---

## Invariants (do not relax)
- **Determinism:** no RNG or wall‑clock; ties break by `lex(id)`; fixed epsilons.
- **Disabled‑path identity:** with `t2.quality.enabled=false`, behavior/logs match PR29 (baseline).
- **Triple gate** for metrics & traces:

```
perf.enabled && perf.metrics.report_memory && (t2.quality.shadow || t2.quality.enabled)
```

- **Gate D (required):** “shadow is a no‑op” comparator remains in CI.
- **Instant rollback:** set `t2.quality.enabled=false`.

Known watch‑outs:
- Comparator ignore rule currently matches any `rq_traces.jsonl` (broad); narrowing to `**/logs/quality/rq_traces.jsonl` is tracked.
- Ensure CI exports `CLEMATIS_GIT_SHA` so traces don’t show `git=unknown`.

---

## PR36 — Shadow tracing (no‑op)
**What:** Added a shadow scaffold and emitter that writes `rq_traces.jsonl` **only under the triple gate**. Shadow mode is wired but does **not** affect rankings/metrics.

**How:**
- Config: `t2.quality.shadow=true` (quality remains **disabled**).
- Tools: `scripts/rq_trace_dump.py`, `scripts/validate_noop_shadow.py` (Gate D comparator).

**Expect:**
- No change in results.
- Gated traces present when `perf.enabled && report_memory`.

---

## PR37 — Lexical BM25 + rank fusion (enabled path)
**What:** Opt‑in lexical scoring and rank‑based fusion over the candidate set. Ties break by `lex(id)`.

**How:**
- Enable with `t2.quality.enabled=true`.
- Knobs (validator‑bound): `t2.quality.lexical.{bm25_k1,bm25_b,stopwords}` and `t2.quality.fusion.{mode,alpha_semantic}`.

**Expect:**
- Enabled‑path traces (`rq_traces.jsonl`) under the triple gate with `meta.reason="enabled"`.
- `t2q.*` fusion metrics emitted under the gate.

---

## PR38 — MMR diversification (opt‑in)
**What:** Deterministic **MMR** on the fused list; selection is stable and tie‑broken by `lex(id)`.

**How:**
- Config: `t2.quality.mmr.{enabled,lambda,k}` (with bounds & types in validator).
- Metrics (gated): `t2q.mmr.selected`, plus diversity summaries (e.g., `t2q.diversity_*`).

**Expect:**
- With `mmr.enabled=false` or `t2.quality.enabled=false`, ordering equals PR37.

---

## PR39 — Normalizer & aliasing (quality paths only)
**What:** Deterministic text normalization and token‑level alias expansion used by BM25/MMR.

**How:**
- Normalizer (defaults for quality modes): NFKC → lower → collapse whitespace.
- Aliases: `t2.quality.aliasing.map_path` (YAML `{ alias: canonical }`), exact‑token rewrites; phrase canonicals expand to tokens.

**Expect:**
- Affects **only** quality paths; disabled path unchanged.
- Deterministic ordering; idempotent aliasing.

---

## PR40 — Lance partition reader parity (optional)
**What:** Optional partition‑aware reader **without** changing results; default remains `flat`.

**How:**
- Config: `t2.reader.mode ∈ { flat | partition | auto }` (default `flat`).
- Observability: gated metric `t2.reader_mode` reports the selected mode; unavailability falls back to `flat` deterministically.

**Expect:**
- Identical Top‑K IDs & scores vs flat when partition mode is available; otherwise flat behavior.

---

## PR41 — Retrieval Quality (RQ) evaluation harness
**What:** Deterministic, offline A/B evaluator (JSON/CSV) computing **recall@k / MRR / nDCG**.

**How:**
- CLI: `scripts/rq_eval.py --queries ... --truth ... --configA ... --configB ... --k 10 --out out.json [--csv out.csv] [--emit-traces]`.
- Uses your runtime entrypoint; `--emit-traces` respects the triple gate; trace reason may appear as `"eval"` or `"enabled"` depending on runtime support.

**Expect:**
- Stable outputs with input/config digests; no RNG/wall‑clock.

---

## Enablement path (typical)
1. **Shadow** tracing (PR36): verify no‑op.
2. **Fusion** (PR37): enable quality; observe gated metrics/traces.
3. **MMR** (PR38): tune `lambda`/`k`; check `t2q.mmr.selected`.
4. **Normalizer/Aliases** (PR39): add `aliasing.map_path` if needed.
5. **Reader parity** (PR40): optionally set `t2.reader.mode=auto`.
6. **Evaluation** (PR41): run `rq_eval.py` to compare A vs B.

Rollback at any time via `t2.quality.enabled=false`.
