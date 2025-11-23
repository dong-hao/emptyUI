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
    // Edge color/thickness follow the source port's type (DESIGN.md §3)
    const { catalog } = useCatalog.getState();
    const source = get().nodes.find((n) => n.id === connection.source);
    const manifest = catalog?.nodes.find(
      (m) => m.type === (source?.data as MTNodeData | undefined)?.nodeType,
    );
    const port = manifest?.outputs.find((o) => o.name === connection.sourceHandle);
    const color = portColor(port?.type ?? "any", catalog);
    set((s) => ({
      edges: addEdge(
        {
          ...connection,
          id: `e${crypto.randomUUID()}`,
          style: { stroke: color, strokeWidth: 2.5 },
          data: { portType: port?.type ?? "any" },
        },
        s.edges,
      ),
    }));
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
}));
