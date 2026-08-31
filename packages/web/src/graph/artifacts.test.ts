import { expect, test } from "vitest";
import type { Edge } from "@xyflow/react";
import type { ArtifactInfo } from "../stores/run";
import { resolveInputArtifact } from "./artifacts";

const E = (over: Partial<Edge>): Edge =>
  ({ id: "e", source: "src", target: "dst", ...over }) as Edge;
const A = (over: Partial<ArtifactInfo>): ArtifactInfo =>
  ({
    id: "a",
    url: "http://x",
    nodeId: "src",
    kind: "plot",
    handle: null,
    ...over,
  }) as ArtifactInfo;

test("matches the artifact by the edge's source handle (docs/issues.md L3)", () => {
  const edges = [E({ sourceHandle: "tf", targetHandle: "plot" })];
  const artifacts = [
    A({ id: "table1", handle: "table" }),
    A({ id: "tf1", handle: "tf" }),
  ];
  expect(resolveInputArtifact(edges, artifacts, "dst", "plot")?.id).toBe("tf1");
});

test("no silent last-artifact fallback: an unattributed artifact is not shown", () => {
  const edges = [E({ sourceHandle: "tf", targetHandle: "plot" })];
  const artifacts = [A({ id: "table1", handle: "table" })];
  expect(resolveInputArtifact(edges, artifacts, "dst", "plot")).toBeNull();
});

test("multi-output sources pick the edge's own output, latest wins", () => {
  const edges = [
    E({ id: "e1", sourceHandle: "data1", targetHandle: "data" }),
    E({ id: "e2", sourceHandle: "data2", targetHandle: "data" }),
  ];
  const artifacts = [
    A({ id: "d1-old", handle: "data1" }),
    A({ id: "d1-new", handle: "data1" }),
    A({ id: "d2", handle: "data2" }),
  ];
  // the e1 edge resolves to data1's freshest artifact, never data2's
  expect(resolveInputArtifact(edges, artifacts, "dst", "data")?.id).toBe(
    "d1-new",
  );
});

test("an edge into a different input port does not feed the body", () => {
  const edges = [E({ sourceHandle: "plot", targetHandle: "aux" })];
  const artifacts = [A({ id: "p", handle: "plot" })];
  expect(resolveInputArtifact(edges, artifacts, "dst", "plot")).toBeNull();
});

test("absent handles on both sides still pair (normalized '' contract)", () => {
  const edges = [E({ sourceHandle: undefined, targetHandle: undefined })];
  const artifacts = [A({ id: "x", handle: null })];
  expect(resolveInputArtifact(edges, artifacts, "dst", undefined)?.id).toBe("x");
});
