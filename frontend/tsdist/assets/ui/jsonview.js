function canonical(obj) {
    if (obj === null || typeof obj !== "object")
        return obj;
    if (Array.isArray(obj))
        return obj.map(canonical);
    const keys = Object.keys(obj).sort();
    const out = {};
    for (const k of keys)
        out[k] = canonical(obj[k]);
    return out;
}
export function renderJSONPre(node, data) {
    const text = JSON.stringify(canonical(data), null, 2);
    node.textContent = text + "\n";
}
