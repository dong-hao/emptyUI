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
  useRun.getState().setRunId("r2");
  expect(useRun.getState().nodeStates["n1"]?.warnings).toBeUndefined();
});
