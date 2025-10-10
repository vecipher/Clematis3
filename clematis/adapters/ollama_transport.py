# clematis/adapters/ollama_transport.py
from __future__ import annotations
import json
import urllib.request

def generate_with_ollama(prompt: str, *, model: str, max_tokens: int, temperature: float, timeout_s: float) -> str:
    body = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": max(0, int(max_tokens)),
        },
    }
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        "http://localhost:11434/api/generate",
        data=data,
        headers={"content-type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    # /api/generate returns {"response": "...", ...}
    return (payload.get("response") or "").strip()
