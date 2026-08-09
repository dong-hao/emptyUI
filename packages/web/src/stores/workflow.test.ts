import { describe, expect, it, vi, afterEach } from "vitest";
import type { Connection, Edge, Node } from "@xyflow/react";
import type { NodeManifest, PortSpec } from "../api/client";
import type { PayloadMap } from "./run";
import { useCatalog } from "./catalog";
import {
  connectionCreatesCycle,
  deserializeWorkflow,
  enforceWiringRules,
  portTypesCompatible,
  relativizeToAnchor,
  serializeWorkflow,
  validateConnection,
  type MTNodeData,
} from "./workflow";

const port = (name: string, type: string, extra: Partial<PortSpec> = {}): PortSpec =>
  ({ name, type, ...extra });

function manifest(type: string, inputs: PortSpec[], outputs: PortSpec[]): NodeManifest {
  return {
    type, version: "1", title: type, tooltip: "", description: "",
    category: "Processing", deps: [], params: [], inputs, outputs,
  };
}

function node(id: string, nodeType: string): Node<MTNodeData> {
  return { id, position: { x: 0, y: 0 }, data: { label: id, nodeType, params: {} } };
}

const edge = (id: string, source: string, target: string): Edge =>
  ({ id, source, target, sourceHandle: "out", targetHandle: "in" });

const connection = (source: string, target: string, targetHandle = "in"): Connection =>
  ({ source, target, sourceHandle: "out", targetHandle });

afterEach(() => {
  vi.restoreAllMocks();
});

describe("portTypesCompatible", () => {
  it("accepts equal tags, any on either side, and declared accepts", () => {
    expect(portTypesCompatible("ts.run", "ts.run")).toBe(true);
    expect(portTypesCompatible("any", "fc")).toBe(true);
    expect(portTypesCompatible("fc", "any")).toBe(true);
    expect(portTypesCompatible("ts.runs", "ts.run", ["ts.runs"])).toBe(true);
  });

  it("rejects different tags when nothing accepts them", () => {
    expect(portTypesCompatible("ts.run", "fc")).toBe(false);
    expect(portTypesCompatible("ts.runs", "ts.run")).toBe(false);
  });
});

describe("connectionCreatesCycle", () => {
  const chain = [edge("e1", "a", "b"), edge("e2", "b", "c")];

  it("rejects an edge that closes a directed cycle", () => {
    expect(connectionCreatesCycle(chain, "c", "a")).toBe(true);
  });

  it("allows an edge that keeps the graph acyclic", () => {
    expect(connectionCreatesCycle(chain, "a", "c")).toBe(false);
  });

  it("treats a self-loop as a cycle", () => {
    expect(connectionCreatesCycle([], "a", "a")).toBe(true);
  });
});

describe("validateConnection", () => {
  const manifests = [
    manifest("src.node", [], [port("out", "ts.run")]),
    manifest("tgt.node", [port("in", "ts.run")], []),
    manifest("loop.node", [port("in", "ts.run")], [port("out", "ts.run")]),
    manifest("table.node", [port("in", "tabular")], []),
    manifest("coll.node", [port("in", "ts.run", { collection_ok: true })], []),
    manifest("plain.node", [port("in", "ts.run")], []),
  ];
  const nodes = [
    node("n1", "src.node"),
    node("n2", "tgt.node"),
    node("l1", "loop.node"),
    node("l2", "loop.node"),
    node("l3", "loop.node"),
    node("tb", "table.node"),
    node("c1", "coll.node"),
    node("p1", "plain.node"),
  ];

  it("accepts a compatible connection between known ports", () => {
    expect(validateConnection(connection("n1", "n2"), nodes, [], manifests)).toBe(true);
  });

  it("rejects a self-loop even when both ports exist", () => {
    expect(validateConnection(connection("l1", "l1"), nodes, [], manifests)).toBe(false);
  });

  it("rejects an unknown target port", () => {
    expect(validateConnection(connection("n1", "n2", "nope"), nodes, [], manifests)).toBe(false);
  });

  it("rejects incompatible port types", () => {
    expect(validateConnection(connection("n1", "tb", "in"), nodes, [], manifests)).toBe(false);
  });

  it("rejects a connection that would close a cycle", () => {
    const edges = [edge("e1", "l1", "l2"), edge("e2", "l2", "l3")];
    expect(validateConnection(connection("l3", "l1"), nodes, edges, manifests)).toBe(false);
    expect(validateConnection(connection("l1", "l3"), nodes, edges, manifests)).toBe(true);
  });

  it("refuses a collection into a non-collection input, once the payload is known", () => {
    const payloads: PayloadMap = {
      n1: { out: { kind: "ts.runs", start: "", end: "", absolute: false, sample_rate: 1, channels: [] } },
    };
    expect(validateConnection(connection("n1", "p1"), nodes, [], manifests, payloads)).toBe(false);
    expect(validateConnection(connection("n1", "c1"), nodes, [], manifests, payloads)).toBe(true);
  });
});

describe("enforceWiringRules", () => {
  const manifests = new Map<string, NodeManifest>([
    ["src.node", manifest("src.node", [], [port("out", "ts.run")])],
    ["tgt.node", manifest("tgt.node", [port("in", "ts.run")], [])],
    ["spec.node", manifest("spec.node", [port("in", "spectra")], [])],
    ["loop.node", manifest("loop.node", [port("in", "ts.run")], [port("out", "ts.run")])],
  ]);
  const manifestOf = (type: string) => manifests.get(type);
  const wire = (id: string, source: string, target: string, targetPort = "in") => ({
    id, source, source_port: "out", target, target_port: targetPort,
  });

  it("keeps the last edge into a duplicated input and warns about the rest", () => {
    const warn = vi.spyOn(console, "warn").mockImplementation(() => {});
    const nodes = [node("a", "src.node"), node("b", "src.node"), node("t", "tgt.node")];
    const kept = enforceWiringRules([wire("e1", "a", "t"), wire("e2", "b", "t")], nodes, manifestOf);
    expect(kept.map((e) => e.id)).toEqual(["e2"]);
    expect(warn.mock.calls.some(([m]) => String(m).includes("duplicate edge"))).toBe(true);
  });

  it("keeps an incompatible edge but warns, leaving the verdict to the server", () => {
    const warn = vi.spyOn(console, "warn").mockImplementation(() => {});
    const nodes = [node("a", "src.node"), node("s", "spec.node")];
    const kept = enforceWiringRules([wire("e1", "a", "s")], nodes, manifestOf);
    expect(kept.map((e) => e.id)).toEqual(["e1"]);
    expect(warn.mock.calls.some(([m]) => String(m).includes("incompatible port"))).toBe(true);
  });

  it("keeps an edge that closes a cycle but warns about it", () => {
    const warn = vi.spyOn(console, "warn").mockImplementation(() => {});
    const nodes = [node("a", "loop.node"), node("b", "loop.node")];
    const kept = enforceWiringRules([wire("ea", "a", "b"), wire("eb", "b", "a")], nodes, manifestOf);
    expect(kept.map((e) => e.id)).toEqual(["ea", "eb"]);
    expect(warn.mock.calls.some(([m]) => String(m).includes("directed cycle"))).toBe(true);
  });
});

describe("document envelope (L12)", () => {
  it("deserialize names the missing arrays instead of a generic TypeError", () => {
    expect(() =>
      deserializeWorkflow({ format: "emptyui/workflow", version: 1 } as never),
    ).toThrow(/nodes.*edges|edges.*nodes/i);
    expect(() =>
      deserializeWorkflow({
        format: "emptyui/workflow",
        version: 1,
        nodes: [{ id: "a", type: "x" }],
      } as never),
    ).toThrow(/edges/i);
  });

  it("serialize refuses to invent a 'data' handle for a handle-less edge", () => {
    const bad = edge("e1", "a", "b") as unknown as Edge;
    (bad as unknown as { sourceHandle: null }).sourceHandle = null;
    expect(() =>
      serializeWorkflow(
        [node("a", "x"), node("b", "y")] as unknown as Node<MTNodeData>[],
        [bad],
      ),
    ).toThrow(/handle/);
  });
});

describe("M23: plugin-version record + save-time path relativization", () => {
  const withCatalog = (manifests: NodeManifest[]) => {
    const prev = useCatalog.getState().catalog;
    useCatalog.setState({ catalog: { nodes: manifests } });
    return () => useCatalog.setState({ catalog: prev });
  };

  afterEach(() => useCatalog.setState({ catalog: undefined }));

  it("stamps each node's plugin_version from the catalog manifest", () => {
    const restore = withCatalog([
      { ...manifest("plug.node", [], []), plugin_version: "0.6.8" },
    ]);
    const doc = serializeWorkflow([node("a", "plug.node")], []);
    restore();
    expect(doc.nodes[0].plugin_version).toBe("0.6.8");
  });

  it("omits the record for nodes the catalog does not know", () => {
    const restore = withCatalog([]);
    const doc = serializeWorkflow([node("a", "ghost.type")], []);
    restore();
    expect(doc.nodes[0].plugin_version).toBeUndefined();
  });

  it("relativizes in-kind path params under the anchor; nothing else", () => {
    const m = manifest("plug.node", [], []);
    m.params = [
      { name: "path", widget: "file", path_kind: "in" },
      { name: "dest", widget: "file", path_kind: "out" },
      { name: "rate", widget: "number" },
    ];
    const restore = withCatalog([m]);
    const n = {
      ...node("a", "plug.node"),
      data: {
        label: "a",
        nodeType: "plug.node",
        params: {
          path: "/srv/proj/data/x.csv",
          dest: "/tmp/workspace/out.png",
          rate: 15,
        },
      },
    } as unknown as Node<MTNodeData>;
    const doc = serializeWorkflow([n], [], "t", { projectDir: "/srv/proj" });
    restore();
    expect(doc.nodes[0].params.path).toBe("data/x.csv");
    expect(doc.nodes[0].params.dest).toBe("/tmp/workspace/out.png");
    expect(doc.nodes[0].params.rate).toBe(15);
  });

  it("relativizeToAnchor: longest anchor wins; relative values untouched", () => {
    const anchors = { projectDir: "/srv/proj", dataRoot: "/srv/proj/data" };
    expect(relativizeToAnchor("/srv/proj/data/x.csv", anchors)).toBe("x.csv");
    expect(relativizeToAnchor("/srv/proj/other/x.csv", anchors)).toBe(
      "other/x.csv",
    );
    expect(relativizeToAnchor("/elsewhere/x.csv", anchors)).toBe(
      "/elsewhere/x.csv",
    );
    expect(relativizeToAnchor("data/x.csv", anchors)).toBe("data/x.csv");
    expect(relativizeToAnchor("C:\\srv\\proj\\x.csv", anchors)).toBe(
      "C:\\srv\\proj\\x.csv",
    );
    expect(relativizeToAnchor("/srv/proj/x.csv", {})).toBe("/srv/proj/x.csv");
  });

  it("serializes unchanged without anchors (paths saved as-is)", () => {
    const restore = withCatalog([
      { ...manifest("plug.node", [], []), plugin_version: "0.6.8" },
    ]);
    const n = {
      ...node("a", "plug.node"),
      data: {
        label: "a",
        nodeType: "plug.node",
        params: { path: "/srv/proj/data/x.csv" },
      },
    } as unknown as Node<MTNodeData>;
    const doc = serializeWorkflow([n], []);
    restore();
    expect(doc.nodes[0].params.path).toBe("/srv/proj/data/x.csv");
    expect(doc.nodes[0].plugin_version).toBe("0.6.8"); // record still stamps
  });
});
