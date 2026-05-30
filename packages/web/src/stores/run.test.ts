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
