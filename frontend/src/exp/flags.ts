export function readHashFlag(name: string): boolean {
  try {
    const h = window.location.hash.replace(/^#/, "");
    const params = new URLSearchParams(h);
    return params.get(name) === "1";
  } catch {
    return false;
  }
}

/** Experimental master switch: true if #experimental=1 or #exp=1 or checkbox on */
export function isExperimental(): boolean {
  const fromHash = readHashFlag("experimental") || readHashFlag("exp");
  const cb = document.querySelector<HTMLInputElement>("#exp-toggle");
  return !!(fromHash || (cb && cb.checked));
}

export function wireExperimentalRepaint(repaint: () => void) {
  const cb = document.querySelector<HTMLInputElement>("#exp-toggle");
  if (cb) cb.addEventListener("change", repaint);
  window.addEventListener("hashchange", repaint);
}
