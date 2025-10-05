import os
import sys
import time
import random
import importlib
import pytest

LINUX = sys.platform.startswith("linux")

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(not (LINUX or os.getenv("PERF_SMOKE_ANY_OS") == "1"), reason="Linux-only; override locally with PERF_SMOKE_ANY_OS=1"),
    pytest.mark.skipif(os.getenv("PERF_SMOKE") != "1", reason="Set PERF_SMOKE=1 to run"),
]

def _generate_graph(num_nodes: int = 8000, target_edges: int = 50_000, seed: int = 1337):
    """
    Deterministic sparse directed graph with ~target_edges edges.
    Weights in (0.2..1.0]; rel='associates' (or 0) to avoid edge-type branching.
    """
    rng = random.Random(seed)
    edges = []
    # Prefer near-uniform out-degree; clamp self-loops; avoid duplicates cheaply
    seen = set()
    avg_out = max(1, target_edges // num_nodes)
    for u in range(num_nodes):
        k = avg_out
        for _ in range(k):
            v = rng.randrange(0, num_nodes)
            if v == u:
                v = (v + 1) % num_nodes
            key = (u, v)
            if key in seen:
                continue
            seen.add(key)
            w = 0.2 + 0.8 * rng.random()
            edges.append((u, v, w, "associates"))
            if len(edges) >= target_edges:
                break
        if len(edges) >= target_edges:
            break
    return {"num_nodes": num_nodes, "edges": edges}

def _bench_once(run_fn, graph, *, iters: int = 1) -> float:
    # Each run_fn must be a pure function of (graph) with no global warmups
    # Return seconds
    t0 = time.perf_counter()
    for _ in range(iters):
        run_fn(graph)
    return time.perf_counter() - t0


# Helper to resolve native t1 module
def _resolve_native_t1_module():
    """
    Resolve the module that exposes native T1 availability.
    Supports both shapes:
      1) clematis.native.t1.available()
      2) clematis.native.available() or clematis.native.t1 shim
    Returns a module-like object with `available()` or None if not importable.
    """
    # Preferred: dedicated submodule
    try:
        import clematis.native.t1 as nt1  # type: ignore
        return nt1
    except Exception:
        pass

    # Fallbacks: package-level shim or attribute
    try:
        import clematis.native as n  # type: ignore
        if hasattr(n, "t1") and hasattr(n.t1, "available"):
            return n.t1
        if hasattr(n, "available"):
            return n
    except Exception:
        pass

    return None

def _load_parity():
    """
    Prefer the parity harness introduced in PR98/PR99.
    Expected symbols (pick one of the two blocks below and delete the other):

    Option A (preferred):
      from clematis.native.parity import run_python_t1, run_native_t1

    Option B (toggle path):
      - force-disable native via env to get python path
      - force-enable native via env to get native path
      - both call the same public entrypoint (e.g., t1_propagate_kernel)
    """
    try:
        # ---- Option A: parity harness (replace with your actual module path if different) ----
        from clematis.native.parity import run_python_t1, run_native_t1  # type: ignore  # noqa: F401
        return ("parity", run_python_t1, run_native_t1)
    except Exception:
        try:
            import clematis.native as n  # type: ignore
            rp = getattr(n, "run_python_t1", None)
            rn = getattr(n, "run_native_t1", None)
            if callable(rp) and callable(rn):
                return ("parity", rp, rn)
        except Exception:
            pass
        # ---- Option B: single entrypoint + env toggle ----
        # Wire these two closures to your public kernel entrypoint.
        # Replace `clematis.native.t1_api` and `run_t1_kernel` with actual names.
        import clematis.native.t1 as t1mod  # provides available()
        def _run_python(graph):
            os.environ["CLEMATIS_NATIVE_T1"] = "0"
            import clematis.engine.stages.t1 as t1_py
            importlib.reload(t1_py)
            # TODO: replace with your real call — this should execute the Python kernel
            return t1_py._run_kernel_for_bench(graph)  # <-- wire to your bench hook

        def _run_native(graph):
            os.environ["CLEMATIS_NATIVE_T1"] = "1"
            import clematis.engine.stages.t1 as t1_any
            importlib.reload(t1_any)
            # TODO: replace with your real call — this should exercise the native kernel
            return t1_any._run_kernel_for_bench(graph)  # <-- wire to your bench hook

        return ("toggle", _run_python, _run_native)

@pytest.mark.timeout(60)
def test_t1_native_is_faster_by_1p8x():
    os.environ.setdefault("CI", "true")
    os.environ.setdefault("CLEMATIS_NETWORK_BAN", "1")

    # Skip if native module isn’t even importable
    nt1 = _resolve_native_t1_module()
    if nt1 is None:
        pytest.skip("native module not importable")

    if not getattr(nt1, "available", lambda: False)():
        pytest.skip("native not available")

    mode, run_py, run_rs = _load_parity()

    # Small warmups to avoid first-call overheads
    g = _generate_graph()
    _bench_once(run_py, g, iters=1)
    _bench_once(run_rs, g, iters=1)

    # Measured
    t_py = _bench_once(run_py, g, iters=2)
    t_rs = _bench_once(run_rs, g, iters=2)

    speedup = t_py / max(1e-9, t_rs)
    # Allow slight variance but keep the advisory bar meaningful
    assert speedup >= 1.8, f"Expected ≥1.8×, got {speedup:.2f}× (py={t_py:.3f}s, rs={t_rs:.3f}s, mode={mode})"
