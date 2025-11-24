import { create } from "zustand";
import {
  addEdge,
  applyEdgeChanges,
  applyNodeChanges,
  type Connection,
  type Edge,
  type Node,
  type NodeChange,
  type EdgeChange,
} from "@xyflow/react";
import { useCatalog, portColor } from "./catalog";

export interface MTNodeData extends Record<string, unknown> {
  label: string;
  nodeType: string;
  params: Record<string, unknown>;
}

interface WorkflowState {
  nodes: Node<MTNodeData>[];
  edges: Edge[];
  selectedNodeId: string | null;
  onNodesChange: (changes: NodeChange<Node<MTNodeData>>[]) => void;
  onEdgesChange: (changes: EdgeChange[]) => void;
  onConnect: (connection: Connection) => void;
  addNode: (node: Node<MTNodeData>) => void;
  selectNode: (id: string | null) => void;
  updateNodeData: (
    id: string,
    patch: Partial<MTNodeData> | ((d: MTNodeData) => Partial<MTNodeData>),
  ) => void;
  deleteSelected: () => void;
  loadDocument: (nodes: Node<MTNodeData>[], edges: Edge[]) => void;
}

/** The workflow document is the single source of truth (DESIGN.md §9). */
export const useWorkflow = create<WorkflowState>((set, get) => ({
  nodes: [],
  edges: [],
  selectedNodeId: null,
  onNodesChange: (changes) =>
    set((s) => ({ nodes: applyNodeChanges(changes, s.nodes) })),
  onEdgesChange: (changes) =>
    set((s) => ({ edges: applyEdgeChanges(changes, s.edges) })),
  onConnect: (connection) => {
    const { catalog } = useCatalog.getState();
    const source = get().nodes.find((n) => n.id === connection.source);
    const target = get().nodes.find((n) => n.id === connection.target);
    const srcManifest = catalog?.nodes.find(
      (m) => m.type === (source?.data as MTNodeData | undefined)?.nodeType,
    );
    const tgtManifest = catalog?.nodes.find(
      (m) => m.type === (target?.data as MTNodeData | undefined)?.nodeType,
    );
    const outPort = srcManifest?.outputs.find((o) => o.name === connection.sourceHandle);
    const inPort = tgtManifest?.inputs.find((i) => i.name === connection.targetHandle);
    if (!outPort || !inPort) return; // unknown ports — silent reject

    // Silent reject mismatched types (equal, declared accepts, or any)
    const srcType = outPort.type;
    const tgtType = inPort.type;
    const compatible =
      srcType === tgtType ||
      srcType === "any" ||
      tgtType === "any" ||
      (inPort.accepts ?? []).includes(srcType);
    if (!compatible) return; // silent reject (DESIGN.md §3)

    const color = portColor(srcType, catalog);
    set((s) => {
      // one-to-one inputs: a new edge to an occupied input replaces the old one
      const without = s.edges.filter(
        (e) =>
          !(
            e.target === connection.target &&
            e.targetHandle === connection.targetHandle
          ),
      );
      return {
        edges: addEdge(
          {
            ...connection,
            id: `e${crypto.randomUUID()}`,
            style: { stroke: color, strokeWidth: 2.5 },
            data: { portType: srcType },
          },
          without,
        ),
      };
    });
  },
  addNode: (node) => set((s) => ({ nodes: [...s.nodes, node] })),
  selectNode: (id) =>
    set((s) => ({
      selectedNodeId: id,
      nodes: s.nodes.map((n) => ({ ...n, selected: n.id === id })),
    })),
  updateNodeData: (id, patch) =>
    set((s) => ({
      nodes: s.nodes.map((n) =>
        n.id === id
          ? {
              ...n,
              data: { ...n.data, ...(typeof patch === "function" ? patch(n.data) : patch) },
            }
          : n,
      ),
    })),
  deleteSelected: () =>
    set((s) => ({
      nodes: s.nodes.filter((n) => n.id !== s.selectedNodeId),
      edges: s.edges.filter(
        (e) => e.source !== s.selectedNodeId && e.target !== s.selectedNodeId,
      ),
      selectedNodeId: null,
    })),
  loadDocument: (nodes, edges) =>
    set({ nodes, edges, selectedNodeId: null }),
}));

/** Serialize the workflow document (DESIGN.md §6 format). */
export function serializeWorkflow(nodes: Node<MTNodeData>[], edges: Edge[]) {
  return {
    format: "emptyui/workflow",
    version: 1,
    meta: { name: "untitled", created: new Date().toISOString() },
    nodes: nodes.map((n) => ({
      id: n.id,
      type: n.data.nodeType,
      params: n.data.params,
      position: n.position,
    })),
    edges: edges.map((e) => ({
      id: e.id,
      source: e.source,
      source_port: e.sourceHandle ?? "data",
      target: e.target,
      target_port: e.targetHandle ?? "data",
    })),
  };
}

/** Rebuild editor state from a workflow document (inverse of serialize). */
export function deserializeWorkflow(
  doc: {
    nodes: { id: string; type: string; params?: Record<string, unknown>; position?: { x: number; y: number } }[];
    edges: { id: string; source: string; source_port: string; target: string; target_port: string }[];
  },
): { nodes: Node<MTNodeData>[]; edges: Edge[] } {
  const { catalog } = useCatalog.getState();
  const nodes: Node<MTNodeData>[] = doc.nodes.map((n) => {
    const manifest = catalog?.nodes.find((m) => m.type === n.type);
    return {
      id: n.id,
      type: "mt",
      position: n.position ?? { x: 80, y: 80 },
      data: {
        label: manifest?.title ?? n.type,
        nodeType: n.type,
        params: n.params ?? {},
      },
      style:
        n.type === "core.display.preview_plot" ||
        n.type === "core.display.view_table"
          ? { width: 340, height: 260 }
          : { width: 250, height: "auto" },
    };
  });
  const edges: Edge[] = doc.edges.map((e) => {
    // re-derive edge color from the source port type
    const src = doc.nodes.find((n) => n.id === e.source);
    const manifest = catalog?.nodes.find((m) => m.type === src?.type);
    const portType =
      manifest?.outputs.find((o) => o.name === e.source_port)?.type ?? "any";
    return {
      id: e.id,
      source: e.source,
      sourceHandle: e.source_port,
      target: e.target,
      targetHandle: e.target_port,
      style: { stroke: portColor(portType, catalog), strokeWidth: 2.5 },
      data: { portType },
    };
  });
  return { nodes, edges };
}
