import numpy as np
import pytest

# This is an FFI contract test: verify types/shapes of the Rust entrypoint.
pytestmark = pytest.mark.skipif(
    __import__("importlib").util.find_spec("clematis.native._t1_rs") is None,
    reason="native ext not built",
)


def test_contract_shapes():
    import clematis.native.t1 as nt1

    # Minimal but well-formed CSR + metadata
    indptr = np.asarray([0, 1, 2], dtype=np.int32)       # 2 rows
    indices = np.asarray([1, 0], dtype=np.int32)
    weights = np.asarray([1.0, 0.5], dtype=np.float32)
    rel_code = np.asarray([0, 0], dtype=np.int32)
    rel_mult = np.asarray([1.0, 1.0], dtype=np.float32)
    seed_nodes = np.asarray([0], dtype=np.int32)
    seed_weights = np.asarray([1.0], dtype=np.float32)
    key_rank = np.asarray([0, 1], dtype=np.int32)

    # Kernel params (FFI-shape positional args)
    rate = 0.6
    floor = 0.05
    radius_cap = 4
    iter_cap_layers = 50
    node_budget = 1.5

    d_nodes, d_vals, metrics = nt1.propagate_one_graph_rs(
        indptr,
        indices,
        weights,
        rel_code,
        rel_mult,
        seed_nodes,
        seed_weights,
        key_rank,
        rate,
        floor,
        radius_cap,
        iter_cap_layers,
        node_budget,
    )

    # Contract assertions (content/parity covered elsewhere)
    assert isinstance(metrics, dict)
    assert "iters" in metrics and isinstance(metrics["iters"], int)

    assert d_nodes.dtype == np.int32
    assert d_vals.dtype == np.float32
    assert d_nodes.ndim == 1 and d_vals.ndim == 1
    assert d_nodes.shape == d_vals.shape
