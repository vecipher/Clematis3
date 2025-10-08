export const $ = (sel, root = document) => root.querySelector(sel);
export const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));

export function el(tag, attrs = {}, children = []) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") node.className = v;
    else if (k === "text") node.textContent = v;
    else node.setAttribute(k, v);
  }
  for (const c of children) node.append(c);
  return node;
}

export function clear(node) { while (node.firstChild) node.removeChild(node.firstChild); }
export function copyToClipboard(text) {
  // no network; rely on Clipboard API if available
  if (navigator.clipboard && navigator.clipboard.writeText) {
    return navigator.clipboard.writeText(text);
  }
  // fallback
  const ta = el("textarea", {}, [text]);
  document.body.append(ta);
  ta.select(); document.execCommand("copy");
  ta.remove();
}
