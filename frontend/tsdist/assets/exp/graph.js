function pickGraph(bundle) {
    // Extremely conservative: look for a snapshot-like structure with node/edge summaries.
    // We avoid assumptions; if not found, return empty.
    const s = bundle.snapshots?.[0] ?? {};
    const nodes = Array.isArray(s.nodes)
        ? s.nodes.map((n) => ({ id: String(n.id ?? n.label ?? n.name ?? "") })).filter((n) => !!n.id)
        : Array.isArray((s.summary || {}).nodes)
            ? s.summary.nodes.map((n) => ({ id: String(n.id ?? n.label ?? n.name ?? "") })).filter((n) => !!n.id)
            : [];
    const edges = Array.isArray(s.edges)
        ? s.edges.map((e) => ({ src: String(e.src ?? e.from ?? ""), dst: String(e.dst ?? e.to ?? "") }))
            .filter((e) => !!(e.src && e.dst))
        : Array.isArray((s.summary || {}).edges)
            ? s.summary.edges.map((e) => ({ src: String(e.src ?? e.from ?? ""), dst: String(e.dst ?? e.to ?? "") }))
                .filter((e) => !!(e.src && e.dst))
            : [];
    return { nodes, edges };
}
export function renderGraph(container, bundle) {
    container.textContent = "";
    const { nodes, edges } = pickGraph(bundle);
    const N = nodes.length;
    const W = 720, H = 480, R = Math.min(W, H) * 0.4, cx = W / 2, cy = H / 2;
    // Deterministic order by id, then index.
    const sorted = nodes
        .map((n, i) => ({ n, i }))
        .sort((a, b) => (a.n.id < b.n.id ? -1 : a.n.id > b.n.id ? 1 : a.i - b.i))
        .map((x) => x.n);
    const pos = new Map();
    for (let k = 0; k < sorted.length; k++) {
        const theta = (2 * Math.PI * k) / Math.max(1, N);
        pos.set(sorted[k].id, { x: cx + R * Math.cos(theta), y: cy + R * Math.sin(theta) });
    }
    const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
    svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
    svg.setAttribute("width", String(W));
    svg.setAttribute("height", String(H));
    svg.classList.add("exp-graph");
    // Edges (drawn first)
    for (const e of edges) {
        const a = pos.get(e.src), b = pos.get(e.dst);
        if (!a || !b)
            continue;
        const line = document.createElementNS(svg.namespaceURI, "line");
        line.setAttribute("x1", a.x.toFixed(2));
        line.setAttribute("y1", a.y.toFixed(2));
        line.setAttribute("x2", b.x.toFixed(2));
        line.setAttribute("y2", b.y.toFixed(2));
        line.setAttribute("class", "exp-edge");
        svg.appendChild(line);
    }
    // Nodes
    for (const n of sorted) {
        const p = pos.get(n.id);
        const g = document.createElementNS(svg.namespaceURI, "g");
        g.setAttribute("transform", `translate(${p.x.toFixed(2)},${p.y.toFixed(2)})`);
        const c = document.createElementNS(svg.namespaceURI, "circle");
        c.setAttribute("r", "10");
        c.setAttribute("class", "exp-node");
        g.appendChild(c);
        const t = document.createElementNS(svg.namespaceURI, "text");
        t.setAttribute("text-anchor", "middle");
        t.setAttribute("dy", "-14");
        t.setAttribute("class", "exp-label");
        t.textContent = n.id;
        g.appendChild(t);
        svg.appendChild(g);
    }
    container.appendChild(svg);
}
