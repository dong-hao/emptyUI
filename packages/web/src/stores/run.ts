import { create } from "zustand";

export interface ArtifactInfo {
  id: string;
  url: string;
  nodeId: string;
  kind: string;
  handle?: string | null;
}

export interface NodeRunState {
  status: "queued" | "running" | "done" | "error";
  error?: string;
  progress?: number;
  msg?: string;
  /** result reused from the cross-run cache (executor emitted cached: true) */
  cached?: boolean;
}

interface RunState {
  runId: string | null;
  runStatus: "idle" | "queued" | "running" | "completed" | "failed" | "cancelled";
  nodeStates: Record<string, NodeRunState>;
  artifacts: ArtifactInfo[];
  setRunId: (id: string | null) => void;
  handleEvent: (event: Record<string, unknown>) => void;
  reset: () => void;
}

/** Ephemeral run state streamed over WebSocket (DESIGN.md §9). */
export const useRun = create<RunState>((set) => ({
  runId: null,
  runStatus: "idle",
  nodeStates: {},
  artifacts: [],
  setRunId: (id) =>
    set({ runId: id, runStatus: id ? "queued" : "idle", nodeStates: {}, artifacts: [] }),
  handleEvent: (event) => {
    if (event.run_id !== useRun.getState().runId) return;
    switch (event.event) {
      case "run_started":
        set({ runStatus: "running" });
        break;
      case "node_status":
        set((s) => ({
          runStatus: s.runStatus === "idle" ? "running" : s.runStatus,
          nodeStates: {
            ...s.nodeStates,
            [event.node as string]: {
              status: event.status as NodeRunState["status"],
              error: event.error as string | undefined,
              cached: event.cached === true ? true : undefined,
            },
          },
        }));
        break;
      case "node_progress":
        set((s) => ({
          nodeStates: {
            ...s.nodeStates,
            [event.node as string]: {
              ...s.nodeStates[event.node as string],
              status: "running",
              progress: event.frac as number,
              msg: event.msg as string,
            },
          },
        }));
        break;
      case "artifact":
        set((s) => {
          const id = event.id as string;
          if (s.artifacts.some((a) => a.id === id)) return s; // dedupe (polling replay)
          return {
            artifacts: [
              ...s.artifacts,
              {
                id,
                url: event.url as string,
                nodeId: event.node as string,
                kind: event.kind as string,
                handle: (event.handle as string | undefined) ?? null,
              },
            ],
          };
        });
        break;
      case "run_finished":
        set({ runStatus: event.status as RunState["runStatus"] });
        break;
    }
  },
  reset: () => set({ runId: null, runStatus: "idle", nodeStates: {}, artifacts: [] }),
}));
