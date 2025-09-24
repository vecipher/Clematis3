# M7 Troubleshooting — Observability & DevEx

This page is pragmatic: **symptom → likely cause → concrete fix**. M7 keeps defaults conservative; the **disabled path stays baseline‑identical** and you can roll back instantly with `t2.quality.enabled=false`.

---

## Quick gate checklist (traces/metrics)
**Triple gate** must be true for quality traces/metrics:

```
perf.enabled && perf.metrics.report_memory && (t2.quality.shadow || t2.quality.enabled)
```

**Verify fast**
- Set in config:
  ```yaml
  perf:
    enabled: true
    metrics:
      report_memory: true
  t2:
    quality:
      shadow: true   # or enabled: true
  ```
- After a run, check metrics for the reader mode key (PR40) as a smoke test:
  - Expect `t2.reader_mode` when the gate is on.

---

## “No traces were written”
**Cause**: Triple gate not satisfied, or quality path not active.

**Fix**
1. Ensure:
   - `perf.enabled=true`
   - `perf.metrics.report_memory=true`
   - `t2.quality.shadow=true` **or** `t2.quality.enabled=true`
2. Confirm the trace directory (`t2.quality.trace_dir`, default under `logs/quality`) is writable.
3. For CI/local provenance, set `CLEMATIS_GIT_SHA` to avoid `git=unknown` in traces:
   ```bash
   export CLEMATIS_GIT_SHA=$(git rev-parse --short HEAD 2>/dev/null || echo local)
   ```
4. Use `scripts/rq_trace_dump.py` to inspect `rq_traces.jsonl` (PR36+).

---

## “Metrics didn’t show up”
**Cause**: Gate off, or you’re on the disabled path.

**Fix**
- Turn on the triple gate (above).
- For enabled‑path metrics (PR37+), make sure:
  ```yaml
  t2:
    quality:
      enabled: true
  ```
- For MMR metrics (PR38), also set:
  ```yaml
  t2:
    quality:
      mmr:
        enabled: true
  ```
- Expect `t2q.mmr.selected` when MMR actually ran.

---

## “Shadow changed results”
**Cause**: It shouldn’t. Shadow is designed to be a **no‑op**.

**Fix**
- Run the Gate‑D comparator (PR36) locally:
  ```bash
  python scripts/validate_noop_shadow.py  # compares baseline vs shadow
  ```
- If you added files matching `rq_traces.jsonl` outside quality logs, note the ignore pattern is **broad**. Narrowing to `**/logs/quality/rq_traces.jsonl` is tracked; avoid writing similarly named files elsewhere.

---

## “MMR didn’t change ordering”
**Cause**: MMR off, `k` too small, `lambda` near 0, or token sets are identical.

**Fix**
```yaml
t2:
  quality:
    enabled: true
    mmr:
      enabled: true
      k: 10        # head considered by MMR; omit => full fused list
      lambda: 0.5  # 0=relevance‑only; 1=diversity‑heavy
```
- With `lambda=0` or `k=1`, MMR degenerates to relevance.
- If items tokenize to the same set (e.g., alias collapse), diversity may be zero → less movement; that’s expected.

---

## “Aliases didn’t apply” (PR39)
**Causes**: Bad path, wrong YAML shape, expectations beyond exact tokens.

**Fix**
1. Point to a real file:
   ```yaml
   t2:
     quality:
       aliasing:
         map_path: examples/quality/aliases.yaml
   ```
2. Use **`alias: canonical`** pairs, one per line. Canonicals support phrases; underscores are preserved in canonicals:
   ```yaml
   cuda: nvidia_cuda
   llm: large language model
   ```
3. Aliasing matches **exact tokens** (post‑normalization). Phrase‑level aliases on the **left** are out of scope.
4. Aliasing is **single‑pass and idempotent**; re‑applying has no effect.

---

## “Normalizer changed my baseline” (PR39)
**Cause**: The normalizer applies **only** in quality paths (BM25/MMR). Disabled path stays baseline‑identical.

**Fix**
- If you see diffs with `t2.quality.enabled=false`, that’s a bug. File an issue with a minimal repro.
- Normalizer stack (quality): NFKC → lower → collapse whitespace.

---

## Reader parity (PR40)
**Question**: “Did switching readers change my results?” — It shouldn’t.

**Facts**
- Config knob: `t2.reader.mode: flat | auto | partition` (default `flat`).
- Availability checks are conservative; if a partition fixture isn’t present, the code **falls back to `flat`** and emits `t2.reader_mode="flat"` under the gate.

**Fix**
- To observe selection, enable the gate and inspect metrics for `t2.reader_mode`.
- If you expected partition mode but see `flat`, ensure the dataset root exists (see `t2.embed_root`) and any required fixture metadata is mounted.

---

## Validator warnings you can safely act on
- **FP16 store without norms** (perf): enable precomputed norms for score parity.
- **MMR head vs retrieval k**: `t2.quality.mmr.k` greater than the candidate pool will effectively truncate; adjust or accept.
- **Alias map missing**: path unreadable → aliasing is a no‑op; fix the path.
- **Reader mode while perf.disabled**: `partition/auto` set but `perf.enabled=false` → no effect (falls back to flat). 

Use `validate_config_verbose(cfg)` in your code/tests to surface these early.

---

## RQ eval harness (PR41) gotchas
- **Import error**: `run_t2` must be importable from `clematis.engine.stages.t2`.
  - Install your package in editable mode: `pip install -e .`
- **PyYAML missing**: install `pyyaml` to load configs.
- **Zero metrics**: check that your configs can retrieve anything for the toy fixtures; otherwise expect zeros (not an error).
- **Traces absent with `--emit-traces`**: triple gate still applies; ensure it’s on.

Run example:
```bash
python scripts/rq_eval.py \
  --corpus tests/fixtures/rq/corpus.jsonl \
  --queries tests/fixtures/rq/queries.tsv \
  --truth   tests/fixtures/rq/qrels.tsv \
  --configA examples/quality/lexical_fusion.yaml \
  --configB tests/examples/quality/normalizer_aliases.yaml \
  --k 5 --out out/rq_eval.json --csv out/rq_eval.csv --emit-traces
```

---

## Determinism checks (when in doubt)
- **Tie‑breaks**: equal scores order by `lex(id)`.
- **Run twice**: IDs and metrics should match bit‑for‑bit given fixed inputs.
- **No wall‑clock/RNG** anywhere in M7 paths.

---

## Rollback
At any time:
```yaml
t2:
  quality:
    enabled: false
```
That restores **baseline behavior and logging** (PR29 parity).
