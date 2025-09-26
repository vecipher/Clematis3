# Clematis v3 - Goals:
> - Universalisable scaffold for agential AI to simulate a character.
> - Turnbased.
> - Somewhat optimised.
> - Modularisation of functions to allow for desynced/parallelized functions for lower latency and lower apparent latency.
> - ≥15 seconds apparent latency on my M1 8GB Mac without knowing how to deal with Metal, using Qwen3:4b-instruct on int4 quant through Ollama.
> - ≥3 seconds apparent latency on my PC, (once i get it) probably mixing and matching models, though Mistral:7b looks attractive.
> - Framework to subdivide T3/LLM calls into planner/utter phases, so I can slot in my own MoE.
> - serve as a jumping-off point to make Clematis v4 
> - both simple, complex, abstract, and long-term planning through concept-graph overlays from vector-embedded memories and a little scratchpad once I actually have the processing power to afford that.

# Clematis v3 - Plans and shit:
> - Concept graph fuckery to either figure out Large Concept Models and the encoder thing its working on to make a compatibility shim and figure out a converter to turn my 384dim vector embeddings to 1024 or whatever LCM runs on.
> - Works. Turn-based system that has a dynamic world with dynamic speaking order, not enough hardware rn to do proper parallel processing so I'll just be happy if it works.

# Clematis v3 - Description:
> - An LLM system that utilises RAG, semantic search through cosine similarity, (will figure out something better) bge encoding, LanceDB, through a tiered system to maintain identity persistence.
> - Process divided into stages and systems - T1 through T4 - RAG, LanceDB, Qwen calls, BGE encoding, Dynamic Concept Graphs, (Similar to GraphRAG but at least I came up with it originally-ish I think) Reflection Cycles.


# Clematis v3 - Progress: [M1-M7], partial [M8]
> - Detailed notes live in `docs/` (see index below) to keep this README lean.
> - Previous README moved to oldREADME.md because I couldn't be fucked to edit.
> - Progressive notes per PR are supposed to live in `docs/updates/` (append-only).
> - LLM adapter + fixtures: see `docs/m3/llm_adapter.md`.
>
> For pre‑M8 hardening notes, see `Changelog/PreM8Hardening.txt`.

## Docs index

- `docs/m7/` — validator shapes, quality tracing, MMR λ semantics
- `docs/updates/` (theoretical, as of rn) — progressive PR notes; template at `docs/updates/_template.md`
- `docs/m3/llm_adapter.md` — LLM adapter, fixtures, CI guardrails
- `docs/m8/` - random shit

### Updates stream (rolling)

See docs/updates/ for progressive notes per PR (lightweight, append-only).

# More things
read the other documentation i cannot be bothered tonight