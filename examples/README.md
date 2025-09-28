# Scheduler examples

## Round-robin (visible rotation)
```bash
python3 scripts/run_demo.py \
  --config examples/scheduler/round_robin_minimal.yaml \
  --agents AgentA,AgentB,AgentC --steps 6
# Examples

This folder contains runnable examples for the demo runner and the M6 performance features. All examples are **safe by default** and designed to keep behavior unchanged unless you explicitly enable perf flags.

> Tip: If you want to prove disabled‑path identity locally, use:
> ```bash
> python3 scripts/run_demo.py --config .ci/disabled_path_config.yaml --steps 6 || true
> ```

---

## Scheduler examples

### Round‑robin (visible rotation)
Run the minimal round‑robin configuration and watch the agent rotation:
```bash
python3 scripts/run_demo.py \
  --config examples/scheduler/round_robin_minimal.yaml \
  --agents AgentA,AgentB,AgentC --steps 6
```

---

## Performance (M6) examples
These configs demonstrate the M6 perf controls. **All features are OFF by default** in the project; the examples turn them ON intentionally and gate metrics via `perf.metrics.report_memory`.

> **Invariant:** With `perf.enabled=false` and `t2.quality.enabled=false`, behavior must match PR29 goldens after normalization (Gate A).

### T1 caps & dedupe (PR31)
Deterministic queue caps and a dedupe ring for T1.
```bash
python3 scripts/run_demo.py --config examples/perf/caps.yaml --steps 6
```

### Size‑aware caches (PR32)
LRU‑by‑bytes caches for T1/T2 (identity preserved when disabled).
```bash
python3 scripts/run_demo.py --config examples/perf/caches.yaml --steps 6
```

### T2 fp16 store + fp32 math + partition‑friendly reader (PR33/PR33.5)
Opt‑in fp16 storage with fp32 accumulators; reader only engages when gated. Parity enforced by Gate C.
```bash
python3 scripts/run_demo.py --config examples/perf/fp16_reader.yaml --steps 6
```

### Snapshots: zstd + delta (PR34/34.1)
Compressed and delta snapshots with a **safe fallback** to full when the baseline is missing/mismatched.
```bash
python3 scripts/run_demo.py --config examples/perf/snapshots.yaml --steps 6
```
> Note: Delta mode may fallback to full if `etag_from`/baseline is unavailable; this is expected and deterministic.

---

## PR35 tooling (docs, examples, CLIs)
Small utilities shipped with M6 to inspect and prepare stores **offline** (no runtime behavior change):

- Show CLI help:
```bash
python3 scripts/mem_inspect.py --help
python3 scripts/mem_compact.py --help
```

- Examples are under `examples/perf/*.yaml`. They are referenced in the README and validated by CI.

---

## Troubleshooting
- **No metrics when gates are off**: By design, `metrics.*` (including `tier_sequence`) only appear when `perf.enabled=true` **and** `perf.metrics.report_memory=true` (Gate B).
- **Reader identity**: The T2 reader is hard‑gated; in disabled mode, `embed_store` is not engaged and the legacy tiers apply.
- **Snapshots**: If delta mode is enabled but the baseline is missing, the reader falls back to a full snapshot and logs a deterministic warning when metrics gating allows it.
