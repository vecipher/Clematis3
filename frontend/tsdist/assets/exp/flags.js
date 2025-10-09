export function readHashFlag(name) {
    try {
        const h = window.location.hash.replace(/^#/, "");
        const params = new URLSearchParams(h);
        return params.get(name) === "1";
    }
    catch {
        return false;
    }
}
/** Experimental master switch: true if #experimental=1 or #exp=1 or checkbox on */
export function isExperimental() {
    const fromHash = readHashFlag("experimental") || readHashFlag("exp");
    const cb = document.querySelector("#exp-toggle");
    return !!(fromHash || (cb && cb.checked));
}
export function wireExperimentalRepaint(repaint) {
    const cb = document.querySelector("#exp-toggle");
    if (cb)
        cb.addEventListener("change", repaint);
    window.addEventListener("hashchange", repaint);
}
