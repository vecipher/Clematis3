export const qs = <T extends Element = Element>(sel: string, root: ParentNode = document) =>
  root.querySelector(sel) as T | null;

export const qsa = <T extends Element = Element>(sel: string, root: ParentNode = document) =>
  Array.from(root.querySelectorAll(sel)) as T[];

export const el = (tag: string, attrs: Record<string, string | number | boolean> = {}, children: (Node | string)[] = []) => {
  const n = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v === false || v === null || v === undefined) continue;
    if (v === true) n.setAttribute(k, "");
    else n.setAttribute(k, String(v));
  }
  for (const c of children) n.append(c instanceof Node ? c : document.createTextNode(c));
  return n;
};

export const clear = (node: Element) => { while (node.firstChild) node.removeChild(node.firstChild); };

export const setVisible = (node: HTMLElement, vis: boolean) => { node.style.display = vis ? "" : "none"; };
