

import numpy as np
import pytest

# We exercise two layers:
#  - engine dispatcher (records fallback counters)
#  - native wrapper (import fallback only; counters returned are just edges_processed/decay_applied)


@pytest.mark.timeout(10)
def test_engine_gel_runtime_fallback(monkeypatch):
    try:
        from clematis.engine.gel import gel_tick  # type: ignore
    except Exception:  # engine dispatcher not present in this build
        pytest.skip("engine.gel.gel_tick not available; skipping dispatcher fallback test")

    from clematis.native import gel as ng

    # Pretend native is available but make it raise at call time
    monkeypatch.setattr(ng, "available", lambda: True, raising=True)
    def boom(*a, **k):
        raise MemoryError("OOM")
    monkeypatch.setattr(ng, "tick_decay", boom, raising=True)

    w = np.asarray([0.1, 0.2, 0.3], dtype=np.float32)
    params = {"decay": {"rate": 0.6, "floor": 0.05}}
    cfg = {"perf": {"native": {"gel": {"enabled": True}}}}

    out, met = gel_tick(weights=w, params=params, cfg=cfg)

    # Python reference
    mul = np.float32(max(0.6, 0.05))
    ref = (w * mul).astype(np.float32)

    np.testing.assert_allclose(out, ref, rtol=1e-6, atol=1e-7)
    assert (met.get("native_gel", {}) or {}).get("fallback_runtime_exc", 0) >= 1


@pytest.mark.timeout(10)
def test_engine_gel_import_fallback(monkeypatch):
    try:
        from clematis.engine.gel import gel_tick  # type: ignore
    except Exception:  # engine dispatcher not present in this build
        pytest.skip("engine.gel.gel_tick not available; skipping import fallback test")

    from clematis.native import gel as ng

    # Pretend native disabled/unavailable
    monkeypatch.setattr(ng, "available", lambda: False, raising=True)

    w = np.asarray([0.05, 0.0, 1.0], dtype=np.float32)
    params = {"decay": {"rate": 0.6, "floor": 0.05}}
    cfg = {"perf": {"native": {"gel": {"enabled": True}}}}

    out, met = gel_tick(weights=w, params=params, cfg=cfg)

    # Python reference
    mul = np.float32(max(0.6, 0.05))
    ref = (w * mul).astype(np.float32)

    np.testing.assert_allclose(out, ref, rtol=1e-6, atol=1e-7)
    assert (met.get("native_gel", {}) or {}).get("fallback_import_failed", 0) >= 1
