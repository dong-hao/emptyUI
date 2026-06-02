import { create } from "zustand";
import type { PayloadSummary } from "../api/client";

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
  /** degradations the node announced via ctx.warn (issues.md H7) —
   *  e.g. a missing calibration. Rendered on the card (⚠ marker) and in
   *  the properties panel; deduped, order preserved. */
  warnings?: string[];
  /** result reused from the cross-run cache (executor emitted cached: true) */
  cached?: boolean;
}

/** payload summaries per node per output port (DESIGN.md §8.3) — e.g. the
 *  time extent a `ts.run` payload covers, driving the time-selector bar. */
export type PayloadMap = Record<string, Record<string, PayloadSummary>>;

interface RunState {
  runId: string | null;
  runStatus:
    | "idle"
    | "queued"
    | "running"
    | "completed"
    | "failed"
    | "cancelled"
    | "timeout";
  /** Reason of a failed run, straight from the terminal run_finished event
   *  (H3): the UI shows this in the toolbar — the status badge alone leaves
   *  a red "failed" with no explanation. */
  runError: string | null;
  nodeStates: Record<string, NodeRunState>;
  artifacts: ArtifactInfo[];
  payloads: PayloadMap;
  /** Events that arrived while runId was still null (between run creation
   *  and setRunId) — flushed by setRunId, dropped by reset. Capped. */
  pending: Record<string, unknown>[];
  /** Refcount of POST /runs requests in flight (M-F2): runStatus is still
   *  "idle" until setRunId, so Run/Run-to-here/New gate on this instead —
   *  two concurrent posts would otherwise race setRunId (last-resolved
   *  wins, the other run orphaned). */
  startingCount: number;
  beginStart: () => void;
  endStart: () => void;
  setRunId: (id: string | null) => void;
  handleEvent: (event: Record<string, unknown>) => void;
  reset: () => void;
}

/** Ephemeral run state streamed over WebSocket (DESIGN.md §7.2). */
export const useRun = create<RunState>((set) => ({
  runId: null,
  runStatus: "idle",
  runError: null,
  nodeStates: {},
  artifacts: [],
  payloads: {},
  pending: [],
  startingCount: 0,
  beginStart: () => set((s) => ({ startingCount: s.startingCount + 1 })),
  endStart: () =>
    set((s) => ({ startingCount: Math.max(0, s.startingCount - 1) })),
  setRunId: (id) => {
    set({
      runId: id,
      runStatus: id ? "queued" : "idle",
      runError: null,
      nodeStates: {},
      artifacts: [],
      payloads: {},
    });
    // flush events that arrived before the id was known (they were buffered
    // precisely for this moment); the run_id filter drops foreign ones
    const buffered = useRun.getState().pending;
    if (buffered.length) {
      set({ pending: [] });
      for (const e of buffered) useRun.getState().handleEvent(e);
    }
  },
  handleEvent: (event) => {
    const st = useRun.getState();
    if (st.runId === null) {
      // run id not yet known (the POST /runs response is in flight) — buffer
      // instead of dropping; an older run's events are discarded at flush
      set((s) => ({ pending: [...s.pending, event].slice(-200) }));
      return;
    }
    if (event.run_id !== st.runId) return;
    switch (event.event) {
      case "run_started":
        set({ runStatus: "running" });
        break;
      case "node_status":
        set((s) => ({
          runStatus: s.runStatus === "idle" ? "running" : s.runStatus,
          nodeStates: {
            ...s.nodeStates,
            // L-F8: MERGE with the prior state — a node_status replacing the
            // object dropped the progress/msg a node_progress event had
            // already streamed (the progress bar snapped away mid-run)
            [event.node as string]: {
              ...s.nodeStates[event.node as string],
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
                // M-F5: the server's `url` is root-absolute and ignores a
                // sub-path mount (root_path) — derive the URL client-side
                // via api.artifactUrl(artifact.id) where it is consumed;
                // the raw value is kept only for debugging parity
                url: event.url as string,
                nodeId: event.node as string,
                kind: event.kind as string,
                handle: (event.handle as string | undefined) ?? null,
              },
            ],
          };
        });
        break;
      case "payload": {
        // time-series extent summary (§23.4) — last run wins (poll replays
        // can repeat it; same summary, so overwrite is harmless)
        const node = event.node as string;
        const port = event.port as string;
        set((s) => ({
          payloads: {
            ...s.payloads,
            [node]: { ...(s.payloads[node] ?? {}), [port]: event.summary as PayloadSummary },
          },
        }));
        break;
      }
      case "node_warning":
        // H7: append (dedupe by content — the replay can repeat events);
        // warnings persist for the run's lifetime like artifacts
        set((s) => ({
          nodeStates: {
            ...s.nodeStates,
            [event.node as string]: {
              ...s.nodeStates[event.node as string],
              warnings: (() => {
                const prev = s.nodeStates[event.node as string]?.warnings ?? [];
                const msg = event.msg as string;
                return prev.includes(msg) ? prev : [...prev, msg];
              })(),
            },
          },
        }));
        break;
      case "run_finished":
        set({
          runStatus: event.status as RunState["runStatus"],
          // H3: keep the reason of a failed/timed-out run — the executor
          // carries it on the terminal event; a terminal status without
          // one clears it. "timeout" (M2) is a failed run whose reason
          // names the node and the remedy, so it renders identically.
          runError:
            event.status === "failed" || event.status === "timeout"
              ? ((event.error as string | undefined) || null)
              : null,
        });
        break;
    }
  },
  // startingCount survives: it tracks in-flight POSTs, not run identity —
  // zeroing it here would un-gate the buttons while a POST is still pending
  reset: () =>
    set({
      runId: null,
      runStatus: "idle",
      runError: null,
      nodeStates: {},
      artifacts: [],
      payloads: {},
      pending: [],
    }),
}));
