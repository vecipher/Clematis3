export type Stage = "t1" | "t2" | "t3" | "t3_reflection" | "t4" | "apply" | "turn";

export interface RunBundle {
  meta?: Record<string, unknown>;
  runs?: unknown[];
  snapshots?: unknown[];
  logs?: {
    t1?: unknown[];
    t2?: unknown[];
    t3?: unknown[];
    t3_plan?: unknown[];
    t3_dialogue?: unknown[];
    t3_reflection?: unknown[];
    t4?: unknown[];
    apply?: unknown[];
    turn?: unknown[];
    [k: string]: unknown;
  };
  perf?: Record<string, unknown[]>;
  stage_order?: Stage[];
}

export interface AppState {
  bundle: RunBundle | null;
  showPerf: boolean;
  selectedPerfKey: string | null;
}
