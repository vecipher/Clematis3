use numpy::{PyArray1, PyReadonlyArray1};
use ordered_float::NotNan;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashSet, VecDeque};

const EPS: f32 = 1e-8;

/// Fixed-size ring for recent node ids. `contains()` does not refresh order; `add()` appends
/// and evicts the oldest when capacity is exceeded.
struct DedupeRing {
    cap: usize,
    q: VecDeque<i32>,
    set: HashSet<i32>,
}
impl DedupeRing {
    fn new(cap: i32) -> Option<Self> {
        if cap <= 0 { return None; }
        Some(Self { cap: cap as usize, q: VecDeque::with_capacity(cap as usize), set: HashSet::new() })
    }
    fn contains(&self, x: i32) -> bool { self.set.contains(&x) }
    fn add(&mut self, x: i32) {
        if self.cap == 0 { return; }
        self.q.push_back(x);
        self.set.insert(x);
        if self.q.len() > self.cap {
            if let Some(y) = self.q.pop_front() { self.set.remove(&y); }
        }
    }
}

/// Deterministic LRU set with fixed capacity. `contains()` is read-only; `add()` appends and
/// evicts the oldest when capacity is exceeded. Returns `true` when an eviction occurred.
struct DeterministicLRUSet {
    cap: usize,
    q: VecDeque<i32>,
    set: HashSet<i32>,
}
impl DeterministicLRUSet {
    fn new(cap: i32) -> Option<Self> {
        if cap <= 0 { return None; }
        Some(Self { cap: cap as usize, q: VecDeque::with_capacity(cap as usize), set: HashSet::new() })
    }
    fn contains(&self, x: i32) -> bool { self.set.contains(&x) }
    /// Add returns true iff an eviction occurred.
    fn add(&mut self, x: i32) -> bool {
        if self.cap == 0 { return false; }
        if self.set.contains(&x) { return false; }
        self.q.push_back(x);
        self.set.insert(x);
        if self.q.len() > self.cap {
            if let Some(y) = self.q.pop_front() { self.set.remove(&y); }
            return true;
        }
        false
    }
}

#[allow(clippy::too_many_arguments)]
#[pyfunction]
#[pyo3(signature = (
    indptr, indices, weights, rel_code, rel_mult_edges, seed_nodes, seed_weights, key_rank,
    rate, floor, radius_cap, iter_cap_layers, node_budget, queue_cap, dedupe_window, visited_cap
))]
fn t1_propagate_one_graph<'py>(
    py: Python<'py>,
    indptr: PyReadonlyArray1<'py, i32>,
    indices: PyReadonlyArray1<'py, i32>,
    weights: PyReadonlyArray1<'py, f32>,
    rel_code: PyReadonlyArray1<'py, i32>,
    rel_mult_edges: PyReadonlyArray1<'py, f32>,
    seed_nodes: PyReadonlyArray1<'py, i32>,
    seed_weights: PyReadonlyArray1<'py, f32>,
    key_rank: PyReadonlyArray1<'py, i32>,
    rate: f32,
    floor: f32,
    radius_cap: i32,
    iter_cap_layers: i32,
    node_budget: f32,
    queue_cap: i32,
    dedupe_window: i32,
    visited_cap: i32,
) -> PyResult<(Py<PyArray1<i32>>, Py<PyArray1<f32>>, PyObject)> {
    // Own arrays (contiguous) and views
    let indptr = indptr.as_array().to_owned();
    let indices = indices.as_array().to_owned();
    let weights = weights.as_array().to_owned();

    let codes_view = rel_code.as_array();
    let rem_view = rel_mult_edges.as_array();

    if indices.len() != weights.len() {
        return Err(PyValueError::new_err("indices and weights must have same length"));
    }
    let n_edges = indices.len();

    // Per-edge relation multipliers: either provided per-edge or via (codes, table)
    let rel_mult_vec: Vec<f32> = if rem_view.len() == n_edges {
        rem_view.to_owned().to_vec()
    } else if codes_view.len() == n_edges {
        let table = rem_view;
        let mut v = Vec::with_capacity(n_edges);
        for &c in codes_view.iter() {
            if c < 0 { return Err(PyValueError::new_err("rel_code contains negative entry")); }
            let idx = c as usize;
            if idx >= table.len() { return Err(PyValueError::new_err("rel_code index out of range for rel_mult")); }
            v.push(table[idx]);
        }
        v
    } else {
        return Err(PyValueError::new_err(
            "indices, weights, rel_mult_edges must have same length (or provide rel_code matching edges)",
        ));
    };

    let seed_nodes = seed_nodes.as_array().to_owned();
    let seed_weights = seed_weights.as_array().to_owned();
    let key_rank = key_rank.as_array().to_owned();

    if indptr.len() == 0 { return Err(PyValueError::new_err("indptr is empty")); }
    let n_nodes = indptr.len() - 1;

    if seed_nodes.len() != seed_weights.len() {
        return Err(PyValueError::new_err("seed_nodes and seed_weights must match in length"));
    }
    if key_rank.len() != n_nodes {
        return Err(PyValueError::new_err("key_rank length must equal n_nodes"));
    }

    // State
    let mut acc = vec![0f32; n_nodes];
    let mut dist = vec![-1i32; n_nodes];

    // PQ item: priority by SIGNED contrib (desc), then key_rank (asc), then node id (asc)
    #[derive(Debug, Clone)]
    struct QItem { priority: NotNan<f32>, key_rank: i32, node: i32, contrib: f32, depth: i32 }
    impl Eq for QItem {}
    impl PartialEq for QItem {
        fn eq(&self, o: &Self) -> bool {
            // Equality only on the pruning identity: (priority, key_rank, node)
            self.priority == o.priority && self.key_rank == o.key_rank && self.node == o.node
        }
    }
    impl Ord for QItem {
        fn cmp(&self, o: &Self) -> Ordering {
            match self.priority.cmp(&o.priority) {
                Ordering::Equal => match o.key_rank.cmp(&self.key_rank) {
                    Ordering::Equal => o.node.cmp(&self.node),
                    ord => ord,
                },
                ord => ord,
            }
        }
    }
    impl PartialOrd for QItem { fn partial_cmp(&self, o: &Self) -> Option<Ordering> { Some(self.cmp(o)) } }

    let mut pq = BinaryHeap::<QItem>::new();

    // Caps wiring
    let caps_on = queue_cap > 0 || visited_cap > 0 || dedupe_window > 0;
    let mut in_queue = if caps_on { None } else { Some(vec![false; n_nodes]) };
    let mut ring = DedupeRing::new(dedupe_window);
    let mut visited = DeterministicLRUSet::new(visited_cap);
    let frontier_cap = if queue_cap > 0 { queue_cap as usize } else { 0 };

    // Metrics
    let mut pops: i32 = 0;
    let mut propagations: i32 = 0;
    let mut layers_processed: i32 = 0;
    let mut radius_cap_hits: i32 = 0;
    let mut layer_cap_hits: i32 = 0;
    let mut node_budget_hits: i32 = 0;
    let mut frontier_evicted: i32 = 0;
    let mut dedupe_hits: i32 = 0;
    let mut visited_evicted: i32 = 0;
    let mut local_max_delta: f32 = 0.0;

    // Decay(d) in f32: max(rate^d, floor)
    let decay_at = |d: i32| -> f32 {
        let d = d.max(0);
        let p = if rate == 0.0 { 0.0 } else { rate.powi(d) };
        if p > floor { p } else { floor }
    };

    // Helper: push (with ring + push-then-prune frontier capping)
    let mut push_with_caps = |pq: &mut BinaryHeap<QItem>, node: i32, prio: f32, kr: i32, contrib: f32, depth: i32,
                              ring: &mut Option<DedupeRing>, frontier_cap: usize,
                              dedupe_hits: &mut i32, frontier_evicted: &mut i32| {
        if let Some(r) = ring.as_ref() {
            if r.contains(node) { *dedupe_hits += 1; return; }
        }
        let entry = QItem { priority: NotNan::new(prio).unwrap_or_else(|_| NotNan::new(0.0).unwrap()), key_rank: kr, node, contrib, depth };
        pq.push(entry.clone());
        let mut accepted = true;
        if frontier_cap > 0 && pq.len() > frontier_cap {
            // Keep top-K by Ord; stable enough for our deterministic ordering
            let mut v: Vec<QItem> = pq.drain().collect();
            v.sort(); // BinaryHeap Ord: largest first → sort ascending then truncate from end
            let evict = v.len().saturating_sub(frontier_cap);
            if evict > 0 { *frontier_evicted += evict as i32; }
            // keep the last `frontier_cap` (largest)
            let start = v.len().saturating_sub(frontier_cap);
            let kept: Vec<QItem> = v.into_iter().skip(start).collect();
            accepted = kept.iter().any(|qi| qi == &entry);
            *pq = BinaryHeap::from(kept);
        }
        if let Some(r) = ring.as_mut() { if accepted { r.add(node); } }
    };

    // Seed init: accumulate and enqueue seed deltas
    for (s, sw) in seed_nodes.iter().zip(seed_weights.iter()) {
        let u = *s as usize;
        if u >= n_nodes { return Err(PyValueError::new_err("seed node out of range")); }
        acc[u] += *sw;
        dist[u] = 0;
        if sw.abs() > local_max_delta { local_max_delta = sw.abs(); }
        if let Some(ref mut iq) = in_queue {
            if !iq[u] {
                pq.push(QItem { priority: NotNan::new(*sw).unwrap(), key_rank: key_rank[u], node: *s, contrib: *sw, depth: 0 });
                iq[u] = true;
            }
        } else {
            push_with_caps(&mut pq, *s, *sw, key_rank[u], *sw, 0, &mut ring, frontier_cap, &mut dedupe_hits, &mut frontier_evicted);
        }
    }

    // Main loop
    while let Some(item) = pq.pop() {
        pops += 1;
        let u = item.node as usize;
        if let Some(ref mut iq) = in_queue { iq[u] = false; }

        // Visited LRU: skip expansion if already visited, else record and count eviction
        if let Some(ref mut v) = visited {
            if v.contains(item.node) { continue; }
            if v.add(item.node) { visited_evicted += 1; }
        }

        let layer = if dist[u] >= 0 { dist[u] } else { 0 };
        if layer > 0 && layer > layers_processed {
            layers_processed = layer;
            if iter_cap_layers > 0 && layers_processed > iter_cap_layers { continue; }
        }

        // Node budget gates expansion of this node (but we still popped it)
        if node_budget > 0.0 && acc[u].abs() >= node_budget { node_budget_hits += 1; continue; }

        let row_start = indptr[u] as usize;
        let row_end = indptr[u + 1] as usize;
        if row_end <= row_start { continue; }
        let next_depth = layer + 1;

        for e in row_start..row_end {
            let v = indices[e] as usize;
            if v >= n_nodes { return Err(PyValueError::new_err("edge index out of range")); }
            if radius_cap > 0 && next_depth > radius_cap { radius_cap_hits += 1; continue; }
            if iter_cap_layers > 0 && next_depth > iter_cap_layers { layer_cap_hits += 1; continue; }

            let contrib = item.contrib * weights[e] * rel_mult_vec[e] * decay_at(next_depth);
            if contrib.abs() < EPS { continue; }

            acc[v] += contrib;
            propagations += 1;
            if contrib.abs() > local_max_delta { local_max_delta = contrib.abs(); }
            if dist[v] < 0 || next_depth < dist[v] { dist[v] = next_depth; }

            if node_budget > 0.0 && acc[v].abs() >= node_budget { node_budget_hits += 1; continue; }

            if let Some(ref mut iq) = in_queue {
                if !iq[v] {
                    pq.push(QItem { priority: NotNan::new(contrib).unwrap(), key_rank: key_rank[v], node: v as i32, contrib, depth: next_depth });
                    iq[v] = true;
                }
            } else {
                push_with_caps(&mut pq, v as i32, contrib, key_rank[v], contrib, next_depth, &mut ring, frontier_cap, &mut dedupe_hits, &mut frontier_evicted);
            }
        }
    }

    // Outputs: all |acc| >= EPS, sorted by node id ascending
    let mut nz: Vec<(i32, f32)> = acc.iter().enumerate().filter_map(|(i, &v)| if v.abs() >= EPS { Some((i as i32, v)) } else { None }).collect();
    nz.sort_by_key(|(i, _)| *i);
    let (nodes_vec, vals_vec): (Vec<i32>, Vec<f32>) = nz.into_iter().unzip();

    // Metrics dict
    let metrics = PyDict::new(py);
    metrics.set_item("pops", pops)?;
    metrics.set_item("iters", if iter_cap_layers > 0 { layers_processed.min(iter_cap_layers) } else { layers_processed })?;
    metrics.set_item("propagations", propagations)?;
    metrics.set_item("radius_cap_hits", radius_cap_hits)?;
    metrics.set_item("layer_cap_hits", layer_cap_hits)?;
    metrics.set_item("node_budget_hits", node_budget_hits)?;
    metrics.set_item("frontier_evicted", frontier_evicted)?;
    metrics.set_item("dedupe_hits", dedupe_hits)?;
    metrics.set_item("visited_evicted", visited_evicted)?;
    metrics.set_item("_max_delta_local", local_max_delta)?;

    Ok((
        PyArray1::from_vec(py, nodes_vec).into_py(py),
        PyArray1::from_vec(py, vals_vec).into_py(py),
        metrics.into(),
    ))
}

#[allow(clippy::too_many_arguments)]
#[pyfunction]
#[pyo3(signature = (
    indptr, indices, weights, rel_code, rel_mult_edges, seed_nodes, seed_weights, key_rank,
    rate, floor, radius_cap, iter_cap_layers, node_budget, queue_cap, dedupe_window, visited_cap
))]
fn propagate_one_graph_rs<'py>(
    py: Python<'py>,
    indptr: PyReadonlyArray1<'py, i32>,
    indices: PyReadonlyArray1<'py, i32>,
    weights: PyReadonlyArray1<'py, f32>,
    rel_code: PyReadonlyArray1<'py, i32>,
    rel_mult_edges: PyReadonlyArray1<'py, f32>,
    seed_nodes: PyReadonlyArray1<'py, i32>,
    seed_weights: PyReadonlyArray1<'py, f32>,
    key_rank: PyReadonlyArray1<'py, i32>,
    rate: f32,
    floor: f32,
    radius_cap: i32,
    iter_cap_layers: i32,
    node_budget: f32,
    queue_cap: i32,
    dedupe_window: i32,
    visited_cap: i32,
) -> PyResult<(Py<PyArray1<i32>>, Py<PyArray1<f32>>, PyObject)> {
    // Canonical path: forward to t1_propagate_one_graph to avoid divergence
    t1_propagate_one_graph(
        py,
        indptr,
        indices,
        weights,
        rel_code,
        rel_mult_edges,
        seed_nodes,
        seed_weights,
        key_rank,
        rate,
        floor,
        radius_cap,
        iter_cap_layers,
        node_budget,
        queue_cap,
        dedupe_window,
        visited_cap,
    )
}


#[pymodule]
fn _t1_rs(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(t1_propagate_one_graph, m)?)?;
    m.add_function(wrap_pyfunction!(propagate_one_graph_rs, m)?)?;
    m.add_function(wrap_pyfunction!(gel_tick_decay, m)?)?;
    Ok(())
}
