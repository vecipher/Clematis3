from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Protocol

# PR8: LLM adapter interface + deterministic test double + Qwen adapter thin wrapper.
# This module performs no network I/O. Real calls are injected via a user-provided callable.


def _approx_token_count(text: str) -> int:
    # Deterministic whitespace-based token approximation (keeps CI offline and stable)
    if not text:
        return 0
    return len(text.split())


@dataclass
class LLMResult:
    text: str
    tokens: int
    truncated: bool


class LLMAdapter(Protocol):
    """Minimal adapter protocol. Implementations must be deterministic for the same inputs in tests."""

    def generate(self, prompt: str, max_tokens: int, temperature: float) -> LLMResult:  # pragma: no cover - interface only
        ...


class DeterministicLLMAdapter:
    """
    Offline, deterministic adapter for tests/CI.
    Strategy: echo a compact, deterministic slice of the prompt up to max_tokens.
    No randomness; no external calls.
    """

    name = "DeterministicLLMAdapter"

    def generate(self, prompt: str, max_tokens: int, temperature: float) -> LLMResult:
        # Normalize inputs deterministically
        prompt = (prompt or "").strip()
        toks = prompt.split()
        if max_tokens <= 0:
            return LLMResult(text="", tokens=0, truncated=True)
        truncated = len(toks) > max_tokens
        kept = toks[:max_tokens]
        text = " ".join(kept)
        return LLMResult(text=text, tokens=len(kept), truncated=truncated)


class QwenLLMAdapter:
    """
    Thin wrapper for Qwen Instruct models (e.g., "qwen3-4b-instruct").
    This class itself does NO network I/O; instead, you must inject a callable that performs the request.

    Expected callable signature:
        call_fn(prompt: str, *, model: str, max_tokens: int, temperature: float, timeout_s: int) -> str

    Example wiring (outside of CI/tests):
        from some_client import qwen_chat  # your own function that hits DashScope/Ollama/etc.
        adapter = QwenLLMAdapter(call_fn=qwen_chat, model="qwen3-4b-instruct", temperature=0.2, timeout_s=30)

    In tests, do NOT use this class; prefer DeterministicLLMAdapter to keep runs offline and stable.
    """

    name = "QwenLLMAdapter"

    def __init__(self, call_fn, model: str = "qwen3-4b-instruct", temperature: float = 0.2, timeout_s: int = 30):
        if not callable(call_fn):
            raise TypeError("QwenLLMAdapter requires a callable 'call_fn'(prompt, *, model, max_tokens, temperature, timeout_s) -> str")
        self.call_fn = call_fn
        self.model = str(model)
        self.default_temperature = float(temperature)
        self.timeout_s = int(timeout_s)

    def generate(self, prompt: str, max_tokens: int, temperature: float) -> LLMResult:
        t = float(temperature if temperature is not None else self.default_temperature)
        try:
            raw = self.call_fn(prompt, model=self.model, max_tokens=int(max_tokens), temperature=t, timeout_s=self.timeout_s)
        except Exception as e:
            # Fail closed: return a minimal deterministic message instead of raising, to keep the pipeline robust.
            raw = f"[qwen:error:{type(e).__name__}]"
        # Ensure deterministic clipping to max_tokens
        toks = (raw or "").split()
        truncated = len(toks) > max_tokens if max_tokens > 0 else True
        kept = toks[: max(0, max_tokens)]
        text = " ".join(kept)
        return LLMResult(text=text, tokens=len(kept), truncated=truncated)
