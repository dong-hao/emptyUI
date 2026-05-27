import { describe, expect, it } from "vitest";
import type { Edge, Node } from "@xyflow/react";
import type { NodeManifest, PortSpec } from "../api/client";
import type { MTNodeData } from "../stores/workflow";
import { resolveInherited, type InheritContext } from "./inherit";

const port = (name: string, type: string): PortSpec => ({ name, type });

function manifest(
  type: string,
  inputs: PortSpec[],
  outputs: PortSpec[],
  inherit?: NodeManifest["inherit"],
): NodeManifest {
  return {
    type, version: "1", title: type, tooltip: "", description: "",
    category: "Processing", deps: [], params: [], inputs, outputs, inherit,
  };
}

function node(id: string, nodeType: string, params: Record<string, unknown> = {}): Node<MTNodeData> {
  return { id, position: { x: 0, y: 0 }, data: { label: id, nodeType, params } };
}

const link = (id: string, source: string, target: string): Edge =>
  ({ id, source, target, sourceHandle: "data", targetHandle: "data" });

function context(nodes: Node<MTNodeData>[], edges: Edge[], manifests: NodeManifest[]): InheritContext {
  return { nodes, edges, manifests: new Map(manifests.map((m) => [m.type, m])) };
}

// root (sets survey) -> mid (pass-through, no inherit declaration) -> tgt
// (declares that its data input supplies `survey`).
const rootMidTgt = [
  manifest("root.node", [], [port("data", "ts.run")]),
  manifest("mid.node", [port("data", "ts.run")], [port("data", "ts.run")]),
  manifest("tgt.node", [port("data", "ts.run")], [], { data: { survey: "survey" } }),
];

describe("resolveInherited", () => {
  it("walks transitively through a pass-through node to the upstream value", () => {
    const ctx = context(
      [
        node("root", "root.node", { survey: "S1" }),
        node("mid", "mid.node"),
        node("tgt", "tgt.node"),
      ],
      [link("e1", "root", "mid"), link("e2", "mid", "tgt")],
      rootMidTgt,
    );
    expect(resolveInherited(ctx, "tgt", "survey")).toEqual(["S1"]);
  });

  it("does not offer the target's own value as an inherited candidate", () => {
    const ctx = context(
      [
        node("root", "root.node", { survey: "S1" }),
        node("mid", "mid.node"),
        node("tgt", "tgt.node", { survey: "OWN" }),
      ],
      [link("e1", "root", "mid"), link("e2", "mid", "tgt")],
      rootMidTgt,
    );
    expect(resolveInherited(ctx, "tgt", "survey")).toEqual(["S1"]);
  });

  it("returns nothing and terminates on a cycle", () => {
    const loop = [manifest("loop.node", [port("data", "ts.run")], [port("data", "ts.run")])];
    const ctx = context(
      [node("a", "loop.node"), node("b", "loop.node")],
      [link("e1", "a", "b"), link("e2", "b", "a")],
      loop,
    );
    expect(resolveInherited(ctx, "a", "survey")).toEqual([]);
  });
});
