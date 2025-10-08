export const state = {
  runs: [],          // [{id, name, bundle}]
  selectedRunId: null,
};

let seq = 0;
export function addRun(bundle, name = null) {
  const id = `run${++seq}`;
  state.runs.push({id, name: name || id, bundle});
  if (!state.selectedRunId) state.selectedRunId = id;
  return id;
}

export function selectRun(id) { state.selectedRunId = id; }
export function currentRun() { return state.runs.find(r => r.id === state.selectedRunId) || null; }
