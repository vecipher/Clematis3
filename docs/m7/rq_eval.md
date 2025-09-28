

# PR41 — Retrieval Quality (RQ) evaluation harness

Deterministic, offline A/B evaluator for retrieval quality. Given a fixed corpus (optional), a set of queries and ground truth (qrels), and two configs (A/B), the CLI computes **recall@k**, **MRR@k**, and **nDCG@k**, and writes stable **JSON** (and optional **CSV**). No network I/O, no randomness, no wall‑clock.

> Script: `scripts/rq_eval.py`

---

## Quickstart
```bash
# (from repo root; ensure your package is importable)
python -m pip install -e .

python scripts/rq_eval.py \
  --corpus  tests/fixtures/rq/corpus.jsonl \
  --queries tests/fixtures/rq/queries.tsv \
  --truth   tests/fixtures/rq/qrels.tsv \
  --configA examples/quality/lexical_fusion.yaml \
  --configB tests/examples/quality/normalizer_aliases.yaml \
  --k 10 \
  --out out/rq_eval.json \
  --csv out/rq_eval.csv \
  --emit-traces
```
- **Traces** are written only if the **triple gate** is satisfied:
  ```
  perf.enabled && perf.metrics.report_memory && (t2.quality.shadow || t2.quality.enabled)
  ```
  Trace reason may appear as `"enabled"` (runtime default) even when invoked by the evaluator.

---

## Inputs
### Queries (TSV)
Two columns, **no header**: `qid\tquery_text`
```tsv
q1	apple tart
q2	banana bread
```

### Qrels / Truth (TSV)
Three columns, **no header**: `qid\tdoc_id\trel`. Any `rel > 0` is considered relevant. `rel` is used as graded gain for nDCG.
```tsv
q1	d1	2
q2	d2	1
```

### Corpus (JSONL, optional)
Only used to compute a digest for reproducibility. One JSON object per line with at least an `id` and `text` field.
```json
{"id":"d1","text":"apple tart recipe with cinnamon"}
{"id":"d2","text":"banana bread with walnuts"}
```

### Configs (YAML)
Two configs are compared:
- **A**: baseline (often quality disabled or earlier PR features)
- **B**: candidate (e.g., fusion + MMR + aliases)

The evaluator **does not mutate** configs. If `--emit-traces` is provided, the runtime’s existing gating controls whether traces are written.

---

## Metrics (definitions)
Let `L_q` be the ranked list of doc IDs for query *q* (top‑k), and `R_q` the set of relevant doc IDs. Let `gain(d) = rel(d)` from qrels.

- **Recall@k**: `|L_q ∩ R_q| / |R_q|` (0 if `|R_q|=0`).
- **MRR@k**: `1 / rank(first relevant in L_q)` else `0`. (1‑indexed rank.)
- **nDCG@k** (graded):
  - `DCG = Σ_{i=1..k} (2^{gain(d_i)} − 1) / log2(i+1)`
  - `IDCG` is DCG on the ideal ranking by decreasing `gain` (tie‑break by `lex(id)`)
  - `nDCG = DCG / IDCG` (0 if `IDCG=0`)

Reported values are **macro averages** over queries (simple mean), plus per‑query rows.

All ties are broken by **`lex(id)`** for determinism (consistent with PR37–PR39).

---

## Output files
### JSON (stable keys)
```json
{
  "schema_version": 1,
  "k": 10,
  "corpus_digest": "…",
  "queries_digest": "…",
  "qrels_digest": "…",
  "systems": {
    "A": {"config_path": "…", "config_digest": "…", "metrics": {"macro": {"recall": 0.73, "mrr": 0.51, "ndcg": 0.60}}},
    "B": {"config_path": "…", "config_digest": "…", "metrics": {"macro": {"recall": 0.76, "mrr": 0.54, "ndcg": 0.63}}}
  },
  "delta": {"macro": {"recall": 0.03, "mrr": 0.03, "ndcg": 0.03}},
  "per_query": [
    {"qid": "q1", "A": {"recall": 1.0, "mrr": 0.5, "ndcg": 0.58, "hits": ["d7","d9"]},
                 "B": {"recall": 1.0, "mrr": 0.5, "ndcg": 0.60, "hits": ["d7","d9"]},
                 "delta": {"recall": 0.0, "mrr": 0.0, "ndcg": 0.02}}
  ]
}
```
- **Digests** are SHA‑256 of input files (and a behavioral digest of each config) for reproducibility.

### CSV
Header: `qid,system,recall,mrr,ndcg,hits`
Each query appears up to twice (system `A` and `B`).

---

## Determinism & reproducibility
- No randomness or wall‑clock; repeated runs on the same inputs yield identical outputs.
- Ties resolved by `lex(id)`.
- Inputs and configs are hashed and recorded in the JSON.

---

## Exit codes
- `0` on success; non‑zero on invalid inputs or runtime errors.

---

## Troubleshooting (CLI‑specific)
- **ImportError**: `run_t2` must be importable from `clematis.engine.stages.t2`. Install the package: `pip install -e .`.
- **PyYAML missing**: install `pyyaml`.
- **Zero metrics**: if your configs retrieve nothing for the toy fixtures, metrics may be 0.0 — not an error.
- **No traces with `--emit-traces`**: turn on the triple gate (see above). The evaluator never forces traces.

For broader M7 guidance, see `docs/m7/overview.md` and `docs/m7/troubleshooting.md`.
