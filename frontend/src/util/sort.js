// Stable sort by key; tie-break by index to remain deterministic across engines.
export function stableBy(xs, keyFn) {
  return xs
    .map((v, i) => ({v, i, k: keyFn(v, i)}))
    .sort((a, b) => (a.k < b.k ? -1 : a.k > b.k ? 1 : a.i - b.i))
    .map(({v}) => v);
}
