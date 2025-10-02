

# Migration — PR76 (M7/PR37–38): T2 Refactor

This migration documents the **T2 extraction** that makes `t2/core.py` lean and relocates ancillary modules under `clematis/engine/stages/t2/`.

> **Summary:** No functional changes on the identity path. Import paths changed. **No shim re‑exports** were added — update callers accordingly.

---

## Scope
- Extract quality orchestration (hybrid → fusion → MMR + shadow trace) into `t2/quality.py`.
- Move index/label helpers into `t2/state.py`.
- Centralize gated metrics assembly in `t2/metrics.py` (`assemble_metrics`, `finalize`).
- Relocate auxiliary modules under `t2/` to simplify structure.

## Breaking import changes (update required)
Old → New import paths:

```python
# OLD                                          # NEW
from clematis.engine.stages.t2_quality_mmr import ...
from clematis.engine.stages.t2_quality_norm import ...
from clematis.engine.stages.t2_quality_trace import ...
from clematis.engine.stages.t2_quality import ...
from clematis.engine.stages.t2_shard import ...
from clematis.engine.stages.t2_lance_reader import ...

# becomes

from clematis.engine.stages.t2.quality_mmr import ...
from clematis.engine.stages.t2.quality_norm import ...
from clematis.engine.stages.t2.quality_trace import ...
from clematis.engine.stages.t2.quality_ops import ...
from clematis.engine.stages.t2.shard import ...
from clematis.engine.stages.t2.lance_reader import ...
```

**Core integration points** (unchanged call semantics):

```python
# Quality orchestration
from clematis.engine.stages.t2.quality import apply_quality

# Metrics assembly
from clematis.engine.stages.t2.metrics import finalize as finalize_metrics
```

## File moves (for reference)

```text
engine/stages/t2_lance_reader.py  → engine/stages/t2/lance_reader.py
engine/stages/t2_quality_mmr.py   → engine/stages/t2/quality_mmr.py
engine/stages/t2_quality_norm.py  → engine/stages/t2/quality_norm.py
engine/stages/t2_quality_trace.py → engine/stages/t2/quality_trace.py
engine/stages/t2_quality.py       → engine/stages/t2/quality_ops.py
engine/stages/t2_shard.py         → engine/stages/t2/shard.py
```

## Config & behavior (no change on identity path)
- Identity path remains byte‑for‑byte identical with defaults.
- Shadow trace now lives in `t2/quality.py` and fires **only** if all are true:
  - `perf.enabled`
  - `perf.metrics.report_memory`
  - `t2.quality.shadow`
  - `t2.quality.enabled == false`
- Shadow trace **reason** resolution: prefers `ctx["trace_reason"]`, falls back to `cfg["perf"]["metrics"]["trace_reason"]`.

## How to migrate your code
1. Update imports per the mapping above.
2. If you referenced quality primitives directly from `t2_quality.py`, import them from `t2/quality_ops.py`.
3. If you used MMR helpers in custom code, switch to:
   ```python
   from clematis.engine.stages.t2.quality_mmr import MMRItem, avg_pairwise_distance
   ```
4. Re‑run tests.

### Quick grep to spot old paths
```bash
git grep -nE "t2_(quality|shard|lance|quality_(mmr|norm|trace))\.py|stages\.t2_"
```

## Test checklist
Run the focused suites first, then the full matrix:

```bash
pytest -q tests/integration/test_traces_write_path.py::test_traces_written_to_custom_dir_and_first_line_parses
pytest -q tests/t2/test_t2_parallel_merge.py tests/test_t2_stage.py
pytest -q tests/identity tests/cli
pytest -q
```

## Rollback
- Revert PR76. There are no data migrations; only import paths and file locations changed.

---

## Module map (post‑refactor)
```text
clematis/engine/stages/t2/
  core.py          # lean pipeline; calls quality + metrics.finalize
  quality.py       # orchestration + shadow trace (no mutation in shadow)
  state.py         # index/labels helpers
  metrics.py       # assemble_metrics + finalize
  lance_reader.py  # optional LanceDB backend
  quality_ops.py   # fusion/MMR primitives (ex‑t2_quality.py)
  quality_mmr.py   # MMRItem, avg_pairwise_distance
  quality_norm.py  # lexical/BM25 normalization
  quality_trace.py # rq_traces.jsonl writer
  shard.py         # shard/partition helpers
```
