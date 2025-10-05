use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use ordered_float::NotNan;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::cmp::Ordering;
use std::collections::BinaryHeap;

const EPS: f32 = 1e-8;

/// Queue item: pop by highest |priority|, then smallest key_rank, then smallest node id.
/// We encode that by custom Ord so BinaryHeap (max-heap) yields desired order.
#[derive(Debug, Clone)]
struct QItem {
    priority: NotNan<f32>, // magnitude for ordering
    key_rank: i32,         // smaller first
    node: i32,             // smaller first
    contrib: f32,          // actual value propagated
}

impl Eq for QItem {}
impl PartialEq for QItem {
    fn eq(&self, other: &Self) -> bool {
        self.priority == other.priority
            && self.key_rank == other.key_rank
            && self.node == other.node
    }
}
impl Ord for QItem {
    fn cmp(&self, other: &Self) -> Ordering {
        // Primary: priority (higher first)
        match self.priority.cmp(&other.priority) {
            Ordering::Equal => {
                // Secondary: key_rank (smaller first)
                match other.key_rank.cmp(&self.key_rank) {
                    Ordering::Equal => other.node.cmp(&self.node), // smaller node first
                    o => o,
                }
            }
            o => o,
        }
    }
}
impl PartialOrd for QItem {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

#[allow(clippy::too_many_arguments)]
#[pyfunction]
#[pyo3(signature = (
    indptr, indices, weights, rel_code, rel_mult_edges, seed_nodes, seed_weights, key_rank,
    rate, floor, radius_cap, iter_cap_layers, node_budget
))]
fn t1_propagate_one_graph<'py>(
    py: Python<'py>,
    indptr: PyReadonlyArray1<'py, i32>,
    indices: PyReadonlyArray1<'py, i32>,
    weights: PyReadonlyArray1<'py, f32>,
    rel_code: PyReadonlyArray1<'py, i32>,        // reserved; ignored in perf-OFF
    rel_mult_edges: PyReadonlyArray1<'py, f32>,  // per-edge multipliers (parity with PR98)
    seed_nodes: PyReadonlyArray1<'py, i32>,
    seed_weights: PyReadonlyArray1<'py, f32>,
    key_rank: PyReadonlyArray1<'py, i32>,
    rate: f32,
    floor: f32,
    radius_cap: i32,
    iter_cap_layers: i32,
    node_budget: f32,
) -> PyResult<(Py<PyArray1<i32>>, Py<PyArray1<f32>>, PyObject)> {
    // Copy to owned vectors (contiguity-agnostic and simple)
    let indptr = indptr.as_array().to_owned(); // len = n_nodes+1
    let indices = indices.as_array().to_owned();
    let weights = weights.as_array().to_owned();
    let _rel_code = rel_code.as_array().to_owned();
    let rel_mult = rel_mult_edges.as_array().to_owned();
    let seed_nodes = seed_nodes.as_array().to_owned();
    let seed_weights = seed_weights.as_array().to_owned();
    let key_rank = key_rank.as_array().to_owned();

    if indptr.len() == 0 {
        return Err(PyValueError::new_err("indptr is empty"));
    }
    let n_nodes = indptr.len() - 1;

    // Sanity sizes
    if indices.len() != weights.len() || indices.len() != rel_mult.len() {
        return Err(PyValueError::new_err(
            "indices, weights, rel_mult_edges must have same length",
        ));
    }
    if seed_nodes.len() != seed_weights.len() {
        return Err(PyValueError::new_err(
            "seed_nodes and seed_weights must match in length",
        ));
    }
    if key_rank.len() != n_nodes {
        return Err(PyValueError::new_err(
            "key_rank length must equal n_nodes",
        ));
    }

    // Accumulators
    let mut acc = vec![0f32; n_nodes];
    let mut dist = vec![-1i32; n_nodes];

    // PQ
    let mut pq = BinaryHeap::<QItem>::new();

    // Seed init
    let mut local_max_delta: f32 = 0.0;
    for (s, sw) in seed_nodes.iter().zip(seed_weights.iter()) {
        let s_usize = (*s as usize);
        if s_usize >= n_nodes {
            return Err(PyValueError::new_err("seed node out of range"));
        }
        acc[s_usize] += *sw;
        dist[s_usize] = 0;
        let pr = NotNan::new(sw.abs()).unwrap_or(NotNan::new(0.0).unwrap());
        pq.push(QItem { priority: pr, key_rank: key_rank[s_usize], node: *s, contrib: *sw });
        local_max_delta = local_max_delta.max(sw.abs());
    }

    // Params
    let decay = |d: i32| -> f32 {
        let d = d.max(0) as f32;
        (rate.powf(d)).max(floor)
    };

    // Metrics
    let mut pops: i32 = 0;
    let mut propagations: i32 = 0;
    let mut layers_processed: i32 = 0;
    let mut radius_cap_hits: i32 = 0;
    let mut layer_cap_hits: i32 = 0;
    let mut node_budget_hits: i32 = 0;

    while let Some(item) = pq.pop() {
        pops += 1;
        let u = item.node as usize;
        let layer = dist[u].max(0);

        if layer > 0 && layer > layers_processed {
            layers_processed = layer;
            if layers_processed > iter_cap_layers {
                // do not expand beyond iter cap; drain queue only
                continue;
            }
        }
        if acc[u].abs() >= node_budget {
            node_budget_hits += 1;
            continue;
        }

        let row_start = indptr[u] as usize;
        let row_end = indptr[u + 1] as usize;
        if row_end <= row_start {
            continue;
        }

        for e in row_start..row_end {
            let v = indices[e] as usize;
            if v >= n_nodes {
                return Err(PyValueError::new_err("edge index out of range"));
            }
            let d = layer + 1;
            if d > radius_cap {
                radius_cap_hits += 1;
                continue;
            }
            if d > iter_cap_layers {
                layer_cap_hits += 1;
                continue;
            }
            let contrib = item.contrib * weights[e] * rel_mult[e] * decay(d);
            if contrib.abs() < EPS {
                continue;
            }
            acc[v] += contrib;
            propagations += 1;
            local_max_delta = local_max_delta.max(contrib.abs());
            if dist[v] < 0 || d < dist[v] {
                dist[v] = d;
            }
            if acc[v].abs() < node_budget {
                // push with priority = |contrib|; tie-break by key_rank and node
                let pr = NotNan::new(contrib.abs()).unwrap_or(NotNan::new(0.0).unwrap());
                pq.push(QItem { priority: pr, key_rank: key_rank[v], node: v as i32, contrib });
            } else {
                node_budget_hits += 1;
            }
        }
    }

    // Prepare outputs: non-zeros sorted by node id
    let mut nz: Vec<(i32, f32)> = acc
        .iter()
        .enumerate()
        .filter_map(|(i, &v)| if v.abs() >= EPS { Some((i as i32, v)) } else { None })
        .collect();
    nz.sort_by_key(|(i, _)| *i);

    let (nodes_vec, vals_vec): (Vec<i32>, Vec<f32>) = nz.into_iter().unzip();

    // Build metrics dict
    let metrics = PyDict::new(py);
    metrics.set_item("pops", pops)?;
    metrics.set_item("iters", layers_processed.min(iter_cap_layers))?;
    metrics.set_item("propagations", propagations)?;
    metrics.set_item("radius_cap_hits", radius_cap_hits)?;
    metrics.set_item("layer_cap_hits", layer_cap_hits)?;
    metrics.set_item("node_budget_hits", node_budget_hits)?;
    metrics.set_item("_max_delta_local", local_max_delta)?;

    Ok((
        PyArray1::from_vec(py, nodes_vec).into_py(py),
        PyArray1::from_vec(py, vals_vec).into_py(py),
        metrics.into(),
    ))
}

#[pymodule]
fn _t1_rs(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(t1_propagate_one_graph, m)?)?;
    Ok(())
}
