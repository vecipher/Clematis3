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
