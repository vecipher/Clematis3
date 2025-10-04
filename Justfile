# Reflection utilities for developers
# Usage:
#   just regen-reflection-goldens     # refresh test goldens deterministically
#   just bench-rule                   # run rule-based bench (writes bench_rule.json)
#   just bench-llm                    # run LLM fixtures bench if fixtures exist
#   just smoke-reflection             # quick local smoke: benches + microbench tests

set shell := ["bash", "-eu", "-o", "pipefail", "-c"]

# Deterministic rule-based bench (writes bench_rule.json, also printed to stdout)
bench-rule:
	export CI=true CLEMATIS_NETWORK_BAN=1
	python -m clematis.scripts.bench_reflection -c examples/reflection/enabled.yaml | tee bench_rule.json

# Deterministic LLM fixtures bench (writes bench_llm.json); skips if fixtures missing
bench-llm:
	export CI=true CLEMATIS_NETWORK_BAN=1
	if [ -f tests/fixtures/reflection_llm.jsonl ]; then python -m clematis.scripts.bench_reflection -c examples/reflection/llm_fixture.yaml | tee bench_llm.json; else echo "LLM fixtures not present at tests/fixtures/reflection_llm.jsonl; skipping."; fi

# Regenerate reflection goldens used by tests/reflection/test_reflection_golden_*.py
regen-reflection-goldens:
	export CI=true CLEMATIS_NETWORK_BAN=1
	mkdir -p tests/reflection/goldens/enabled_rulebased
	python -m clematis.scripts.bench_reflection -c examples/reflection/enabled.yaml | tee tests/reflection/goldens/enabled_rulebased/bench_rule.json
	if [ -f tests/fixtures/reflection_llm.jsonl ]; then mkdir -p tests/reflection/goldens/enabled_llm; python -m clematis.scripts.bench_reflection -c examples/reflection/llm_fixture.yaml | tee tests/reflection/goldens/enabled_llm/bench_llm.json; else echo "LLM fixtures not present; skipping LLM golden."; fi

# Microbench tests only (no identity tests)
test-microbench:
	export CI=true CLEMATIS_NETWORK_BAN=1
	pytest -q tests/test_bench_reflection.py

# (Optional) Quick reflection smoke locally: benches + microbench tests
smoke-reflection:
	export CI=true CLEMATIS_NETWORK_BAN=1
	just bench-rule
	just bench-llm
	pytest -q tests/test_bench_reflection.py
