export type Stage = "t1" | "t2" | "t4" | "apply" | "turn";

export interface RunBundle {
  meta?: Record<string, unknown>;
  runs?: unknown[];
  snapshots?: unknown[];
  logs?: Record<string, unknown[]>;
  perf?: Record<string, unknown[]>;
  stage_order?: Stage[];
}

export interface AppState {
  bundle: RunBundle | null;
  showPerf: boolean;
  selectedPerfKey: string | null;
}
