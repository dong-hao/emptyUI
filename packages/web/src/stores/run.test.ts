import { beforeEach, expect, test } from "vitest";
import { useRun } from "./run";

beforeEach(() => {
  useRun.getState().reset();
  useRun.getState().setRunId("r1");
});

test("a timeout terminal status carries its reason (M2)", () => {
  useRun.getState().handleEvent({
    run_id: "r1",
    event: "run_finished",
    status: "timeout",
    node: "n1",
    error: "node 'n1' exceeded its 600 s per-node deadline",
  });
  const s = useRun.getState();
  expect(s.runStatus).toBe("timeout");
  expect(s.runError).toContain("n1");
});

test("a timeout terminal status without a reason clears runError", () => {
  useRun.setState({ runError: "stale" });
  useRun.getState().handleEvent({
    run_id: "r1",
    event: "run_finished",
    status: "timeout",
  });
  expect(useRun.getState().runError).toBeNull();
});

test("node warnings are collected per node, deduped (H7)", () => {
  const emit = (msg: string) =>
    useRun.getState().handleEvent({
      run_id: "r1",
      event: "node_warning",
      node: "n1",
      msg,
    });
  emit("no box calibration (*_CLB.asc) — shape correction missing");
  emit("no box calibration (*_CLB.asc) — shape correction missing");
  emit("no coil calibration for hx");
  const state = useRun.getState().nodeStates["n1"];
  expect(state?.warnings).toEqual([
    "no box calibration (*_CLB.asc) — shape correction missing",
    "no coil calibration for hx",
  ]);
});

test("node warnings are scoped to their node and reset per run", () => {
  useRun.getState().handleEvent({
    run_id: "r1",
    event: "node_warning",
    node: "n1",
    msg: "w1",
  });
  useRun.getState().handleEvent({
    run_id: "r1",
    event: "node_warning",
    node: "n2",
    msg: "w2",
  });
  expect(useRun.getState().nodeStates["n1"]?.warnings).toEqual(["w1"]);
  expect(useRun.getState().nodeStates["n2"]?.warnings).toEqual(["w2"]);
});

test("a node's warnings clear when THAT node re-executes (M12: states survive a re-run)", () => {
  useRun.getState().handleEvent({
    run_id: "r1",
    event: "node_warning",
    node: "n1",
    msg: "w1",
  });
  // a re-run no longer wipes the store (M12) — the stale warning lingers
  // only until the node re-executes
  useRun.getState().setRunId("r2");
  expect(useRun.getState().nodeStates["n1"]?.warnings).toEqual(["w1"]);
  useRun.getState().handleEvent({
    run_id: "r2",
    event: "node_status",
    node: "n1",
    status: "running",
  });
  expect(useRun.getState().nodeStates["n1"]?.warnings).toBeUndefined();
  // an un-targeted node keeps its chip and warnings
  useRun.getState().handleEvent({
    run_id: "r2",
    event: "node_warning",
    node: "n2",
    msg: "w2",
  });
  expect(useRun.getState().nodeStates["n2"]?.warnings).toEqual(["w2"]);
});

test("a re-run keeps artifacts and payload extents visible (M12)", () => {
  useRun.getState().handleEvent({
    run_id: "r1",
    event: "artifact",
    id: "a1",
    url: "http://x/a1",
    node: "n1",
    kind: "plot",
  });
  useRun.getState().handleEvent({
    run_id: "r1",
    event: "payload",
    node: "n1",
    port: "data",
    summary: { kind: "ts.run", start: "2024-01-01", end: "2024-01-02" },
  });
  // re-run: the store must NOT blank the viewers mid-run
  useRun.getState().setRunId("r2");
  expect(useRun.getState().artifacts).toHaveLength(1);
  expect(useRun.getState().payloads["n1"]?.data).toBeDefined();
  // the fresh run's events still arrive and coexist
  useRun.getState().handleEvent({
    run_id: "r2",
    event: "artifact",
    id: "a2",
    url: "http://x/a2",
    node: "n1",
    kind: "plot",
  });
  expect(useRun.getState().artifacts.map((a) => a.id)).toEqual(["a1", "a2"]);
  // a document load / New still clears everything
  useRun.getState().reset();
  expect(useRun.getState().artifacts).toEqual([]);
  expect(useRun.getState().payloads).toEqual({});
  expect(useRun.getState().nodeStates).toEqual({});
});

test("opening a history entry hydrates the run view (L32)", () => {
  useRun.getState().loadHistory({
    run_id: "r7",
    label: "1690618B.TS5",
    created_at: 1,
    finished_at: 2,
    status: "completed",
    duration_s: 1,
    node_count: 1,
    artifact_count: 1,
    workflow: {},
    targets: [],
    summary: {},
    results: {
      nodes: { n1: { status: "done", warnings: ["w"] } },
      artifacts: [
        {
          id: "a1",
          node: "n1",
          kind: "plot",
          handle: "data",
          url: "/api/artifacts/a1/preview",
        },
      ],
      payloads: {
        n1: {
          data: {
            kind: "ts",
            start: "2024-01-01",
            end: "2024-01-02",
            absolute: false,
            sample_rate: 1,
            channels: [],
          },
        },
      },
      status: "completed",
      error: null,
    },
  });
  const s = useRun.getState();
  expect(s.runId).toBe("r7");
  expect(s.runStatus).toBe("completed");
  expect(s.runError).toBeNull();
  expect(s.nodeStates["n1"]).toEqual({ status: "done", warnings: ["w"] });
  expect(s.artifacts).toEqual([
    {
      id: "a1",
      url: "/api/artifacts/a1/preview",
      nodeId: "n1",
      kind: "plot",
      handle: "data",
    },
  ]);
  expect(s.payloads["n1"]?.data).toBeDefined();
});

test("a failed history entry keeps its error (L32)", () => {
  useRun.getState().loadHistory({
    run_id: "r8",
    label: "data.csv",
    created_at: 1,
    finished_at: 2,
    status: "failed",
    duration_s: 1,
    node_count: 1,
    artifact_count: 0,
    workflow: {},
    targets: [],
    summary: { error: "boom" },
    results: {
      nodes: { n1: { status: "error", error: "boom" } },
      artifacts: [],
      payloads: {},
      status: "failed",
      error: "boom",
    },
  });
  expect(useRun.getState().runStatus).toBe("failed");
  expect(useRun.getState().runError).toBe("boom");
  expect(useRun.getState().nodeStates["n1"]?.status).toBe("error");
});
