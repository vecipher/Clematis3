export const qs = (sel, root = document) => root.querySelector(sel);
export const qsa = (sel, root = document) => Array.from(root.querySelectorAll(sel));
export const el = (tag, attrs = {}, children = []) => {
    const n = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs)) {
        if (v === false || v === null || v === undefined)
            continue;
        if (v === true)
            n.setAttribute(k, "");
        else
            n.setAttribute(k, String(v));
    }
    for (const c of children)
        n.append(c instanceof Node ? c : document.createTextNode(c));
    return n;
};
export const clear = (node) => { while (node.firstChild)
    node.removeChild(node.firstChild); };
export const setVisible = (node, vis) => { node.style.display = vis ? "" : "none"; };
