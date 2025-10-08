function canonical(obj: any): any {
  if (obj === null || typeof obj !== "object") return obj;
  if (Array.isArray(obj)) return obj.map(canonical);
  const keys = Object.keys(obj).sort();
  const out: any = {};
  for (const k of keys) out[k] = canonical(obj[k]);
  return out;
}
export function renderJSONPre(node: HTMLElement, data: unknown) {
  const text = JSON.stringify(canonical(data), null, 2);
  node.textContent = text + "\n";
}
