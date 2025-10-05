

import numpy as np
import pytest

import clematis.native.gel as ng


@pytest.mark.timeout(10)
@pytest.mark.parametrize(
    "n,seed,rate,floor",
    [
        (4096, 123, 0.6, 0.05),
        (1024, 7,   0.95, 0.10),
        (513,  99,  0.5,  0.5),   # equal rate/floor → multiplier = rate
        (257,  2025, 0.2, 0.8),   # floor dominates
    ],
)
def test_gel_tick_parity_native_or_fallback(n, seed, rate, floor):
    rng = np.random.default_rng(seed)
    w = rng.random(n, dtype=np.float32)

    out, met = ng.tick_decay(w, rate=rate, floor=floor)

    # Reference (pure Python) parity
    mul = np.float32(rate if rate > floor else floor)
    ref = (w.astype(np.float32, copy=False) * mul)

    np.testing.assert_allclose(out, ref, rtol=1e-6, atol=1e-7)

    # Metrics sanity
    assert isinstance(met, dict)
    assert int(met.get("edges_processed", 0)) == int(w.size)
    expected_changed = int(np.count_nonzero(ref != w))
    assert int(met.get("decay_applied", -1)) == expected_changed


@pytest.mark.timeout(10)
def test_gel_tick_forced_fallback_parity(monkeypatch):
    # Force Python fallback regardless of platform availability
    monkeypatch.setattr(ng, "_HAVE_RS", False, raising=False)

    w = np.asarray([0.1, 0.2, 0.3, 0.0, 1.0], dtype=np.float32)
    out, met = ng.tick_decay(w, rate=0.6, floor=0.05)

    mul = np.float32(0.6)
    ref = (w * mul).astype(np.float32, copy=False)

    np.testing.assert_allclose(out, ref, rtol=1e-6, atol=1e-7)
    assert int(met.get("edges_processed", 0)) == w.size
    assert int(met.get("decay_applied", -1)) == int(np.count_nonzero(ref != w))
