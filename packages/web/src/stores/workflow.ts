import { create } from "zustand";
import {
  addEdge,
  applyEdgeChanges,
  applyNodeChanges,
  reconnectEdge,
  type Connection,
  type Edge,
  type Node,
  type NodeChange,
  type EdgeChange,
} from "@xyflow/react";
import { useCatalog, portColor } from "./catalog";
import type { NodeManifest } from "../api/client";

export interface MTNodeData extends Record<string, unknown> {
  label: string;
  nodeType: string;
  params: Record<string, unknown>;
}

/** Internal clipboard for copy/paste (DESIGN.md §9): a subgraph snapshot of
 *  selected nodes + the edges among them (external edges are not copied —
 *  the pasted copy is independent, ComfyUI parity). */
interface NodeClipboard {
  nodes: Node<MTNodeData>[];
  edges: Edge[];
  /** repeated pastes fan out by 24 px per paste */
  pasteCount: number;
}

const PASTE_STEP = 24;

/** Port-type connection rule (DESIGN.md §3): equal tags, declared accepts, any. */
export function portTypesCompatible(
  srcType: string,
  tgtType: string,
  accepts?: string[],
): boolean {
  return (
    srcType === tgtType ||
    srcType === "any" ||
    tgtType === "any" ||
    (accepts ?? []).includes(srcType)
  );
}

/** Would adding a source→target edge close a directed cycle (target ⇝ source)? */
export function connectionCreatesCycle(
  edges: Edge[],
  source: string,
  target: string,
): boolean {
  const adj = new Map<string, string[]>();
  for (const e of edges) {
    if (!adj.has(e.source)) adj.set(e.source, []);
    adj.get(e.source)!.push(e.target);
  }
  const stack = [target];
  const seen = new Set<string>();
  while (stack.length) {
    const cur = stack.pop()!;
    if (cur === source) return true;
    if (seen.has(cur)) continue;
    seen.add(cur);
    for (const nxt of adj.get(cur) ?? []) stack.push(nxt);
  }
  return false;
}

/**
 * Full client-side connection rule (DESIGN.md §3 + §9): known ports, type
 * compatibility, no self-loop, no cycle. Used by the live
 * `isValidConnection` (drag feedback) and as defense-in-depth in
 * `onConnect`; the server re-validates before every run.
 */
export function validateConnection(
  connection: Connection,
  nodes: Node<MTNodeData>[],
  edges: Edge[],
  manifests: NodeManifest[],
): boolean {
  if (!connection.source || !connection.target) return false;
  if (connection.source === connection.target) return false; // self-loop
  const source = nodes.find((n) => n.id === connection.source);
  const target = nodes.find((n) => n.id === connection.target);
  if (!source || !target) return false;
  const srcManifest = manifests.find(
    (m) => m.type === (source.data as MTNodeData).nodeType,
  );
  const tgtManifest = manifests.find(
    (m) => m.type === (target.data as MTNodeData).nodeType,
  );
  const outPort = srcManifest?.outputs.find((o) => o.name === connection.sourceHandle);
  const inPort = tgtManifest?.inputs.find((i) => i.name === connection.targetHandle);
  if (!outPort || !inPort) return false; // unknown ports — reject
  if (!portTypesCompatible(outPort.type, inPort.type, inPort.accepts)) return false;
  if (connectionCreatesCycle(edges, connection.source, connection.target)) return false;
  return true;
}

interface WorkflowState {
  nodes: Node<MTNodeData>[];
  edges: Edge[];
  selectedNodeId: string | null;
  clipboard: NodeClipboard | null;
  /** document name carried through save/load (meta.name, DESIGN.md §6) */
  docName: string;
  onNodesChange: (changes: NodeChange<Node<MTNodeData>>[]) => void;
  onEdgesChange: (changes: EdgeChange[]) => void;
  onConnect: (connection: Connection) => void;
  /** Disconnect gestures (DESIGN.md §9): remove edges by id (drag-handle-
   *  away unlink, edge-endpoint drop on the pane) and re-route an existing
   *  edge to a new connection (validated with the full connection rule). */
  detachEdges: (edgeIds: string[]) => void;
  reconnect: (oldEdge: Edge, connection: Connection) => void;
  addNode: (node: Node<MTNodeData>) => void;
  selectNode: (id: string | null) => void;
  updateNodeData: (
    id: string,
    patch: Partial<MTNodeData> | ((d: MTNodeData) => Partial<MTNodeData>),
  ) => void;
  deleteSelected: () => void;
  /** Copy/paste of the selected subgraph (Ctrl+C/Ctrl+V/Ctrl+D, DESIGN §9). */
  copySelection: () => void;
  pasteSelection: () => void;
  duplicateSelection: () => void;
  loadDocument: (nodes: Node<MTNodeData>[], edges: Edge[], name?: string) => void;
}

/** The workflow document is the single source of truth (DESIGN.md §9). */
export const useWorkflow = create<WorkflowState>((set, get) => ({
  nodes: [],
  edges: [],
  selectedNodeId: null,
  clipboard: null,
  docName: "untitled",
  onNodesChange: (changes) =>
    set((s) => ({ nodes: applyNodeChanges(changes, s.nodes) })),
  onEdgesChange: (changes) =>
    set((s) => ({ edges: applyEdgeChanges(changes, s.edges) })),
  onConnect: (connection) => {
    const { nodes, edges } = get();
    const { catalog } = useCatalog.getState();
    // defense-in-depth: the full connection rule (the live isValidConnection
    // already rejects invalid drags — this catches programmatic adds)
    if (!validateConnection(connection, nodes, edges, catalog?.nodes ?? [])) return;

    const source = nodes.find((n) => n.id === connection.source);
    const srcManifest = catalog?.nodes.find(
      (m) => m.type === (source?.data as MTNodeData | undefined)?.nodeType,
    );
    const outPort = srcManifest?.outputs.find((o) => o.name === connection.sourceHandle);
    const color = portColor(outPort?.type ?? "any");
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
            data: { portType: outPort?.type ?? "any" },
          },
          without,
        ),
      };
    });
  },
  detachEdges: (edgeIds) =>
    set((s) => ({
      edges: s.edges.filter((e) => !edgeIds.includes(e.id)),
    })),
  reconnect: (oldEdge, connection) => {
    // re-routing an edge endpoint must satisfy the full connection rule
    const { nodes, edges } = get();
    const { catalog } = useCatalog.getState();
    if (!validateConnection(connection, nodes, edges, catalog?.nodes ?? [])) return;
    set((s) => ({
      // one-to-one inputs: rerouting onto an occupied input replaces the
      // old edge (the same rule onConnect enforces — reconnectEdge only
      // rewrites the dragged edge and knows nothing about occupancy). The
      // dragged edge itself must SURVIVE the filter: reconnectEdge looks
      // it up by id (missing -> error 007 -> the edge is lost).
      edges: reconnectEdge(
        oldEdge,
        connection,
        s.edges.filter(
          (e) =>
            e.id === oldEdge.id ||
            !(
              e.target === connection.target &&
              e.targetHandle === connection.targetHandle
            ),
        ),
      ),
    }));
  },
  addNode: (node) => set((s) => ({ nodes: [...s.nodes, node] })),
  selectNode: (id) =>
    // only the properties-panel selection; the RF-native node/edge `selected`
    // flags are maintained through onNodesChange/onEdgesChange (so
    // shift-click multi-select and box selection keep working)
    set({ selectedNodeId: id }),
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
    set((s) => {
      // full RF selection: multi-selected nodes AND selected edges (the
      // single-node fallback covers workflows restored before RF selection)
      const nodeIds = new Set(
        s.nodes.filter((n) => n.selected || n.id === s.selectedNodeId).map((n) => n.id),
      );
      return {
        nodes: s.nodes.filter((n) => !nodeIds.has(n.id)),
        edges: s.edges.filter(
          (e) => !nodeIds.has(e.source) && !nodeIds.has(e.target) && !e.selected,
        ),
        selectedNodeId: null,
      };
    }),
  copySelection: () => {
    const selected = get().nodes.filter((n) => n.selected);
    if (!selected.length) return;
    const ids = new Set(selected.map((n) => n.id));
    // only edges WITHIN the selection are copied (external wiring is not)
    const internal = get().edges.filter((e) => ids.has(e.source) && ids.has(e.target));
    set({
      clipboard: {
        nodes: selected.map((n) => ({
          ...n,
          selected: false,
          position: { ...n.position },
          data: { ...n.data, params: { ...n.data.params } },
        })),
        edges: internal.map((e) => ({ ...e, selected: false })),
        pasteCount: 0,
      },
    });
  },
  pasteSelection: () => {
    const clip = get().clipboard;
    if (!clip || !clip.nodes.length) return;
    const pasteCount = clip.pasteCount + 1;
    const dx = PASTE_STEP * pasteCount;
    const dy = PASTE_STEP * pasteCount;
    const idMap = new Map<string, string>();
    const newNodes = clip.nodes.map((n) => {
      const id = `n${crypto.randomUUID().slice(0, 6)}`;
      idMap.set(n.id, id);
      return {
        ...n,
        id,
        position: { x: n.position.x + dx, y: n.position.y + dy },
        data: { ...n.data, params: { ...n.data.params } },
      };
    });
    const newEdges = clip.edges.map((e) => ({
      ...e,
      id: `e${crypto.randomUUID()}`,
      source: idMap.get(e.source)!,
      target: idMap.get(e.target)!,
    }));
    const newNodeIds = new Set(newNodes.map((n) => n.id));
    const newEdgeIds = new Set(newEdges.map((e) => e.id));
    set((s) => ({
      // the pasted subgraph becomes the selection (ComfyUI parity)
      nodes: [...s.nodes, ...newNodes].map((n) => ({ ...n, selected: newNodeIds.has(n.id) })),
      edges: [...s.edges, ...newEdges].map((e) => ({ ...e, selected: newEdgeIds.has(e.id) })),
      selectedNodeId: newNodes[0].id,
      clipboard: { ...clip, pasteCount },
    }));
  },
  duplicateSelection: () => {
    get().copySelection();
    get().pasteSelection();
  },
  loadDocument: (nodes, edges, name) =>
    set({
      nodes,
      edges,
      selectedNodeId: null,
      ...(name ? { docName: name } : {}),
    }),
}));

/** Serialize the workflow document (DESIGN.md §6 format). */
export function serializeWorkflow(
  nodes: Node<MTNodeData>[],
  edges: Edge[],
  name?: string,
) {
  return {
    format: "emptyui/workflow",
    version: 1,
    meta: {
      name: name ?? useWorkflow.getState().docName ?? "untitled",
      created: new Date().toISOString(),
    },
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

/** Rebuild editor state from a workflow document (inverse of serialize).
 *
 *  Validates the document envelope (format + version, DESIGN.md §6) and
 *  prunes stale edges — edges referencing missing nodes or ports load
 *  silently broken otherwise. Throws on foreign/unsupported documents.
 */
export function deserializeWorkflow(
  doc: {
    format?: string;
    version?: number;
    meta?: { name?: string };
    nodes: { id: string; type: string; params?: Record<string, unknown>; position?: { x: number; y: number } }[];
    edges: { id: string; source: string; source_port: string; target: string; target_port: string }[];
  },
): { nodes: Node<MTNodeData>[]; edges: Edge[]; name: string } {
  if (doc?.format !== "emptyui/workflow") {
    throw new Error(
      "not an emptyUI workflow document (missing format: 'emptyui/workflow')",
    );
  }
  if (doc.version !== 1) {
    throw new Error(`unsupported workflow version: ${String(doc.version)}`);
  }
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
          : { width: 280, height: "auto" },
    };
  });
  const nodeIds = new Set(nodes.map((n) => n.id));
  const manifestOfType = (type: string) =>
    catalog?.nodes.find((m) => m.type === type);
  const edges: Edge[] = doc.edges
    .filter((e) => {
      // stale edge: endpoint node missing, or a port the manifest doesn't
      // declare (renamed/removed in a newer plugin version)
      const src = nodes.find((n) => n.id === e.source);
      const tgt = nodes.find((n) => n.id === e.target);
      if (!src || !tgt) return false;
      const srcManifest = manifestOfType(src.data.nodeType);
      const tgtManifest = manifestOfType(tgt.data.nodeType);
      if (srcManifest && !srcManifest.outputs.some((o) => o.name === e.source_port))
        return false;
      if (tgtManifest && !tgtManifest.inputs.some((i) => i.name === e.target_port))
        return false;
      return true;
    })
    .map((e) => {
      // re-derive edge color from the source port type
      const src = nodes.find((n) => n.id === e.source)!;
      const manifest = manifestOfType(src.data.nodeType);
      const portType =
        manifest?.outputs.find((o) => o.name === e.source_port)?.type ?? "any";
      return {
        id: e.id,
        source: e.source,
        sourceHandle: e.source_port,
        target: e.target,
        targetHandle: e.target_port,
        style: { stroke: portColor(portType), strokeWidth: 2.5 },
        data: { portType },
      };
    });
  return { nodes, edges, name: doc.meta?.name || "untitled" };
}
