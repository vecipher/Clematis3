# Clematis v3 — deterministic, turn‑based agent engine

Clematis is a deterministic, turn‑based scaffold for agential AI. It models agents with concept graphs and tiered reasoning (T1→T4), uses small LLMs where needed, and keeps runtime behavior reproducible (no hidden network calls in tests/CI).

**Status:** M1–M7 are stable and hardened. M8 (Packaging & CLI polish) is in progress. PR47 shipped umbrella CLI hardening with **zero behavior change** (help determinism, first‑subcommand anchoring, single‑sentinel strip, stderr‑only breadcrumbs).

---

## Goals
- Universalisable scaffold for simulating characters/agents.
- Turn‑based core with deterministic logs and budgets.
- Modular stages: T1 (propagation) → T2 (retrieval) → T3 (planning) → T4 (meta‑filter/apply).
- Identity persistence via vector memories (BGE) + LanceDB; optional LLM planning path.
- Separation of planner/utterance phases to support future MoE.
- Practical latency targets on modest hardware; scalable to larger models later.

## Architecture (high level)
- **Memories:** LanceDB vector store (BGE small); tiered retrieval; deterministic scoring.
- **Concept graph:** nodes/edges with decay and relations; surface views for I/O.
- **Stages:**  
  T1 keyword/seeded propagation → T2 semantic retrieval (+ residual) → T3 bounded policy (rule‑based now; LLM adapter gated) → T4 meta‑filter & apply/persist.  
  Reflection and scheduler features are gated for determinism.
- **Determinism:** golden logs, identity path when gates are OFF; shadow/quality traces never affect results.

## Quick start
```bash
# install (editable)
python -m pip install -e .

# check umbrella CLI is wired
python -m clematis --version

# try a wrapper (both orders work; single leading -- is stripped by the wrapper)
python -m clematis rotate-logs -- --dir ./.logs --dry-run
# or
python -m clematis --dir ./.logs rotate-logs -- --dry-run

# For bench-t4 and seed-lance-demo, install extras: pip install clematis[cli-extras].
```

CLI details, delegation rules, and recipes live in **[docs/m8/cli.md](docs/m8/cli.md)**.

## Repository layout (brief)
- `clematis/engine/` — core stages (T1–T4), scheduler stubs, persistence, logs.
- `clematis/cli/` — umbrella + wrapper subcommands (delegates to `clematis.scripts.*`).
- `scripts/` — direct script shims (`*_hint.py`, tolerant import, single stderr hint).
- `docs/` — milestone docs and updates.
- `tests/` — deterministic tests, golden comparisons, CLI checks.

## Environment flags
- `CLEMATIS_NETWORK_BAN=1` — enforce no network (recommended in CI).
- `CLEMATIS_DEBUG=1` — enable a single stderr breadcrumb for wrapper delegation.  
  Exit codes and stdout remain identical.

## Milestones snapshot
- **M1–M4:** core stages + apply/persist + logs.
- **M5:** scheduler config and groundwork (feature‑gated; identity path when disabled).
- **M6:** memory/perf scaffolding; caches and snapshot hygiene (default‑off quality toggles).
- **M7:** observability/dev‑ex; shadow quality traces; golden tests; gate hardening.
- **M8 (current):** packaging & CLI docs/CI polish.  
  – README trimmed; canonical CLI doc split to `docs/m8/cli.md`.  
  – Add fast CLI smokes to CI (help phrase, arg‑order, sentinel strip, shim hint).

Pre‑M8 hardening notes: **`Changelog/PreM8Hardening.txt`**.  
LLM adapter + fixtures: **`docs/m3/llm_adapter.md`**.

## Contributing
- Keep changes deterministic. If a gate is OFF, results must be byte‑for‑byte identical.  
- Tests should run offline; prefer fixtures and golden logs.  
- Include small, focused PRs with a clear scope and a short DoD checklist.

---
_Read the milestone docs under `docs/` for deeper details. This README stays lean and stable._