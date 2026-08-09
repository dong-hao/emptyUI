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
import { useRun, type PayloadMap } from "./run";
import type { NodeManifest } from "../api/client";

/** M13: the one legacy-edge migration that keys on a node type — declared
 *  once instead of a string literal drifting from the plugin's id. */
export const EXTRACT_RUN_NODE_TYPE = "mth5.io.extract_run";

/** M13: the card-size pair, declared once (deserializeWorkflow sizes new
 *  nodes; App re-centering subtracts half the default width). */
export const DEFAULT_NODE_SIZE = { width: 280, height: 0 } as const;
export const VIEWER_NODE_SIZE = { width: 340, height: 260 } as const;

/** RFC-4122 v4 UUID: crypto.randomUUID is secure-context-only (HTTPS or
 *  localhost) — the remote single-origin serving mode is plain HTTP on the
 *  ZeroTier LAN, so fall back to getRandomValues (available everywhere). */
function uuid(): string {
  if (typeof crypto !== "undefined" &&
      typeof crypto.randomUUID === "function") {
    return crypto.randomUUID();
  }
  const b = crypto.getRandomValues(new Uint8Array(16));
  b[6] = (b[6] & 0x0f) | 0x40; // version 4
  b[8] = (b[8] & 0x3f) | 0x80; // variant 10
  const h = Array.from(b, (x) => x.toString(16).padStart(2, "0")).join("");
  return `${h.slice(0, 8)}-${h.slice(8, 12)}-${h.slice(12, 16)}-${h.slice(16, 20)}-${h.slice(20)}`;
}

/** Collision-safe node id: full UUID space, not a 6-hex slice (16.7M —
 *  long sessions with many pastes could collide, surfacing only as a
 *  server-side 422 at run time). The `n` prefix keeps CSS selectors sane. */
export function uniqueNodeId(): string {
  return `n${uuid()}`;
}

/** Stale edges: endpoint node missing, or a port the manifest doesn't
 *  declare (renamed/removed in a newer plugin version). With an EMPTY
 *  catalog every manifest check passes vacuously — callers re-run this
 *  through revalidateEdges() when the catalog arrives. */
function pruneStaleEdges<E extends { id: string; source: string; source_port: string; target: string; target_port: string }>(
  candidates: E[],
  nodes: Node<MTNodeData>[],
  manifestOfType: (type: string) => NodeManifest | undefined,
): E[] {
  return candidates.filter((e) => {
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
  });
}

export interface MTNodeData extends Record<string, unknown> {
  label: string;
  nodeType: string;
  params: Record<string, unknown>;
}

/** Internal clipboard for copy/paste (DESIGN.md §8): a subgraph snapshot of
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
 * Full client-side connection rule (DESIGN.md §3 + §8): known ports, type
 * compatibility, no self-loop, no cycle. Used by the live
 * `isValidConnection` (drag feedback) and as defense-in-depth in
 * `onConnect`; the server re-validates before every run.
 */
export function validateConnection(
  connection: Connection,
  nodes: Node<MTNodeData>[],
  edges: Edge[],
  manifests: NodeManifest[],
  payloads?: PayloadMap,
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
  // One-port runtime-shape check (§4): when the source port's payload
  // summary is known (the upstream ran once), a collection may only be
  // wired into a collection_ok input — recovers wiring-time validation
  // without dual ports. No summary → the static tags rule.
  if (payloads && outPort.type === "ts.run" && inPort.type === "ts.run"
      && !inPort.collection_ok) {
    const s = payloads[connection.source]?.[connection.sourceHandle ?? ""];
    if (s?.kind === "ts.runs") return false;
  }
  if (connectionCreatesCycle(edges, connection.source, connection.target)) return false;
  return true;
}

interface WorkflowState {
  nodes: Node<MTNodeData>[];
  edges: Edge[];
  /** M18: bumped on every change that alters the graph TOPOLOGY or node
   *  params (the inputs of the inheritance walk) — position/dimension
   *  updates (the per-frame drag path) do NOT bump, so consumers keyed on
   *  this revision don't re-render during a drag. */
  graphRev: number;
  /** L12: bumped by loadDocument so the canvas re-fits on EVERY document
   *  load (the empty->non-empty transition only covered the first one). */
  docRev: number;
  selectedNodeId: string | null;
  clipboard: NodeClipboard | null;
  /** document name carried through save/load (meta.name, DESIGN.md §6) */
  docName: string;
  onNodesChange: (changes: NodeChange<Node<MTNodeData>>[]) => void;
  onEdgesChange: (changes: EdgeChange[]) => void;
  onConnect: (connection: Connection) => void;
  /** Disconnect gestures (DESIGN.md §8): remove edges by id (drag-handle-
   *  away unlink, edge-endpoint drop on the pane) and re-route an existing
   *  edge to a new connection (validated with the full connection rule). */
  detachEdges: (edgeIds: string[]) => void;
  reconnect: (oldEdge: Edge, connection: Connection) => void;
  addNode: (node: Node<MTNodeData>) => void;
  /** Shift one node to an absolute flow position (palette placement: the
   *  card is re-centered after it renders, since its height is unknown
   *  before the manifest renders the params). */
  moveNode: (id: string, position: { x: number; y: number }) => void;
  selectNode: (id: string | null) => void;
  updateNodeData: (
    id: string,
    patch: Partial<MTNodeData> | ((d: MTNodeData) => Partial<MTNodeData>),
  ) => void;
  deleteSelected: () => void;
  /** Remove one node and its incident edges (L-F12, §8.1): the
   *  properties-panel "Delete node" acts on its own target, not the
   *  whole multi-selection. */
  deleteNode: (id: string) => void;
  /** Re-run stale-edge pruning when the catalog arrives after a load
   *  (with an empty catalog every manifest check passes vacuously —
   *  edges referencing renamed/removed ports would survive). */
  revalidateEdges: () => void;
  /** Copy/paste of the selected subgraph (Ctrl+C/Ctrl+V/Ctrl+D, DESIGN §8). */
  copySelection: () => void;
  pasteSelection: () => void;
  duplicateSelection: () => void;
  loadDocument: (
    nodes: Node<MTNodeData>[], edges: Edge[], name?: string, created?: string,
  ) => void;
  /** The document's original creation stamp (L-F12, DESIGN §6): carried
   *  through load so a save writes the original date, not today. */
  docCreated: string | undefined;
  /** Start a fresh document (toolbar "New", §8.1): empty canvas, default
   *  name — the run state is reset by the caller. */
  newWorkflow: () => void;
}

/** The workflow document is the single source of truth (DESIGN.md §8). */
export const useWorkflow = create<WorkflowState>((set, get) => ({
  nodes: [],
  edges: [],
  graphRev: 0,
  docRev: 0,
  selectedNodeId: null,
  clipboard: null,
  docName: "untitled",
  /** the document's original creation stamp (L-F12: preserved across saves,
   *  not re-stamped per save) */
  docCreated: undefined as string | undefined,
  onNodesChange: (changes) =>
    set((s) => ({
      nodes: applyNodeChanges(changes, s.nodes),
      // M18: position/dimension/select changes are the per-frame drag path
      // and don't affect inheritance — only structural ones bump the rev
      graphRev: changes.some(
        (c) => !["position", "dimensions", "select"].includes(c.type),
      )
        ? s.graphRev + 1
        : s.graphRev,
    })),
  onEdgesChange: (changes) =>
    set((s) => ({ edges: applyEdgeChanges(changes, s.edges) })),
  onConnect: (connection) => {
    const { nodes, edges } = get();
    const { catalog } = useCatalog.getState();
    // defense-in-depth: the full connection rule (the live isValidConnection
    // already rejects invalid drags — this catches programmatic adds)
    if (!validateConnection(connection, nodes, edges, catalog?.nodes ?? [], useRun.getState().payloads)) return;

    const source = nodes.find((n) => n.id === connection.source);
    const srcManifest = catalog?.nodes.find(
      (m) => m.type === (source?.data as MTNodeData | undefined)?.nodeType,
    );
    const outPort = srcManifest?.outputs.find((o) => o.name === connection.sourceHandle);
    const color = portColor(outPort?.type ?? "any", catalog);
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
            id: `e${uuid()}`,
            style: { stroke: color, strokeWidth: 2.5 },
            data: { portType: outPort?.type ?? "any" },
          },
          without,
        ),
        graphRev: s.graphRev + 1,
      };
    });
  },
  detachEdges: (edgeIds) =>
    set((s) => ({
      edges: s.edges.filter((e) => !edgeIds.includes(e.id)),
      graphRev: s.graphRev + 1,
    })),
  reconnect: (oldEdge, connection) => {
    // re-routing an edge endpoint must satisfy the full connection rule
    const { nodes, edges } = get();
    const { catalog } = useCatalog.getState();
    if (!validateConnection(connection, nodes, edges, catalog?.nodes ?? [], useRun.getState().payloads)) return;
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
      graphRev: s.graphRev + 1,
    }));
  },
  addNode: (node) =>
    set((s) => ({ nodes: [...s.nodes, node], graphRev: s.graphRev + 1 })),
  moveNode: (id, position) =>
    set((s) => ({
      nodes: s.nodes.map((n) => (n.id === id ? { ...n, position } : n)),
    })),
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
      graphRev: s.graphRev + 1,
    })),
  deleteSelected: () =>
    set((s) => {
      // H4: delete EXACTLY the visible RF selection — the multi-selected
      // nodes AND the selected edges. The panel's selectedNodeId is display
      // state: unioning it in also removed whatever node the panel happened
      // to be showing (a node the user had already deselected — box-select
      // B after clicking A used to delete A *and* B, with no undo anywhere).
      const nodeIds = new Set(
        s.nodes.filter((n) => n.selected).map((n) => n.id),
      );
      return {
        nodes: s.nodes.filter((n) => !nodeIds.has(n.id)),
        edges: s.edges.filter(
          (e) => !nodeIds.has(e.source) && !nodeIds.has(e.target) && !e.selected,
        ),
        selectedNodeId: null,
        graphRev: s.graphRev + 1,
      };
    }),
  deleteNode: (id) =>
    set((s) => ({
      nodes: s.nodes.filter((n) => n.id !== id),
      edges: s.edges.filter((e) => e.source !== id && e.target !== id),
      selectedNodeId: s.selectedNodeId === id ? null : s.selectedNodeId,
      graphRev: s.graphRev + 1,
    })),
  revalidateEdges: () => {
    const { nodes, edges } = get();
    const { catalog } = useCatalog.getState();
    if (!catalog?.nodes?.length) return;
    const manifestOfType = (type: string) =>
      catalog.nodes.find((m) => m.type === type);
    const surviving = new Set(
      pruneStaleEdges(
        edges.map((e) => ({
          id: e.id,
          source: e.source,
          source_port: e.sourceHandle ?? "",
          target: e.target,
          target_port: e.targetHandle ?? "",
        })),
        nodes,
        manifestOfType,
      ).map((e) => e.id),
    );
    set({ edges: edges.filter((e) => surviving.has(e.id)) });
  },
  copySelection: () => {
    const selected = get().nodes.filter((n) => n.selected);
    if (!selected.length) return;
    const ids = new Set(selected.map((n) => n.id));
    // only edges WITHIN the selection are copied (external wiring is not)
    const internal = get().edges.filter((e) => ids.has(e.source) && ids.has(e.target));
    set({
      clipboard: {
        // deep-copy params (DESIGN.md §8 "deep-copied params"): a shallow
        // spread shares nested arrays/objects (output_channels lists, save
        // paths) — editing a copy would mutate the original and every
        // earlier paste
        nodes: selected.map((n) => ({
          ...n,
          selected: false,
          position: { ...n.position },
          data: { ...n.data, params: structuredClone(n.data.params) },
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
      const id = uniqueNodeId();
      idMap.set(n.id, id);
      return {
        ...n,
        id,
        position: { x: n.position.x + dx, y: n.position.y + dy },
        // clone again per paste: the clipboard snapshot is reused, so the
        // pasted nodes must not share param objects with it (or with each
        // other across repeated pastes)
        data: { ...n.data, params: structuredClone(n.data.params) },
      };
    });
    const newEdges = clip.edges.map((e) => ({
      ...e,
      id: `e${uuid()}`,
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
      graphRev: s.graphRev + 1,
    }));
  },
  duplicateSelection: () => {
    get().copySelection();
    get().pasteSelection();
  },
  loadDocument: (nodes, edges, name, created) =>
    set({
      graphRev: get().graphRev + 1,
      docRev: get().docRev + 1,
      nodes,
      // the runs→data remap happens exactly once, inside
      // deserializeWorkflow — every document crosses that boundary; the
      // second copy that re-ran it here on sourceHandle was provably a
      // no-op and is gone (D2)
      edges,
      selectedNodeId: null,
      ...(name ? { docName: name } : {}),
      docCreated: created,
    }),
  newWorkflow: () =>
    set({
      graphRev: get().graphRev + 1,
      nodes: [],
      edges: [],
      selectedNodeId: null,
      clipboard: null,
      docName: "untitled",
      // a fresh document must not inherit the previous document's
      // creation timestamp (Load → New → Save stamped it)
      docCreated: undefined,
    }),
}));

/** Extract Run node ids — the one-port migration's source set (§4). */
function extractRunIds(nodes: Node<MTNodeData>[]): Set<string> {
  return new Set(
    nodes
      .filter((n) => (n.data as MTNodeData | undefined)?.nodeType === EXTRACT_RUN_NODE_TYPE)
      .map((n) => n.id));
}

/** Wiring-rule enforcement for LOADED documents (L-F1, defense-in-depth):
 *  deserializeWorkflow previously bypassed every connection rule. One edge
 *  per input is ENFORCED (the last edge wins, matching onConnect's
 *  replace-on-connect behavior) with a warning per dropped duplicate;
 *  cycle/type violations only WARN (the server re-validates before every
 *  run — silently pruning a legacy doc's edge would lose wiring with no
 *  recourse). */
export function enforceWiringRules<E extends { id: string; source: string; source_port: string; target: string; target_port: string }>(
  candidates: E[],
  nodes: Node<MTNodeData>[],
  manifestOfType: (type: string) => NodeManifest | undefined,
): E[] {
  const seen = new Set<string>();
  const cleaned: E[] = [];
  for (const e of [...candidates].reverse()) {
    const key = `${e.target}::${e.target_port}`;
    if (seen.has(key)) {
      console.warn(
        `wiring: duplicate edge into input '${e.target_port}' of node `
        + `'${e.target}' dropped (one edge per input, §3; the last edge wins)`);
      continue;
    }
    seen.add(key);
    cleaned.push(e);
  }
  cleaned.reverse();
  const nodeOf = (id: string) => nodes.find((n) => n.id === id);
  for (const e of cleaned) {
    const src = nodeOf(e.source);
    const tgt = nodeOf(e.target);
    if (!src || !tgt) continue;
    const srcManifest = manifestOfType(src.data.nodeType);
    const tgtManifest = manifestOfType(tgt.data.nodeType);
    const srcType =
      srcManifest?.outputs.find((o) => o.name === e.source_port)?.type;
    const tgtSpec = tgtManifest?.inputs.find((i) => i.name === e.target_port);
    if (srcType !== undefined && tgtSpec !== undefined &&
        !portTypesCompatible(srcType, tgtSpec.type, tgtSpec.accepts)) {
      console.warn(
        `wiring: edge '${e.id}' connects ${srcType} -> ${tgtSpec.type} `
        + `(input '${e.target_port}' of '${e.target}') — incompatible port `
        + "types; the server will reject it at run time");
    }
    if (connectionCreatesCycle(
        cleaned.filter((c) => c !== e) as unknown as Edge[],
        e.source, e.target)) {
      console.warn(
        `wiring: edge '${e.id}' (${e.source} -> ${e.target}) closes a `
        + "directed cycle — the executor will fail at run time");
    }
  }
  return cleaned;
}


/** Anchor directories for save-time path portability (issues.md M19/M23):
 *  an absolute input path living under one of these is STORED anchor-relative
 *  — the workflow then loads on any machine whose server resolves the
 *  relative path against the same anchor (project_dir, default server CWD).
 *  The settings view supplies the effective values. */
export interface SaveAnchors {
  projectDir?: string | null;
  dataRoot?: string | null;
}

/** Rewrite `value` anchor-relative when it is an absolute path under one
 *  of the anchors (longest match wins). POSIX separators; untouched
 *  otherwise. Case-sensitive: the values come from the same server that
 *  reported the anchors (the file browser), so they agree verbatim. */
export function relativizeToAnchor(
  value: string,
  anchors: SaveAnchors,
): string {
  const norm = value.replace(/\\/g, "/");
  if (!norm.startsWith("/")) return value; // relative / bare — nothing to do
  const candidates = [anchors.dataRoot, anchors.projectDir]
    .filter((a): a is string => !!a)
    .map((a) => a.replace(/\\/g, "/").replace(/\/+$/, ""));
  // longest anchor wins: a path under both the data root and a nested
  // project dir must relativize against the deeper one
  for (const a of candidates.sort((x, y) => y.length - x.length)) {
    if (norm.startsWith(`${a}/`)) return norm.slice(a.length + 1);
  }
  return value;
}

/** Serialize the workflow document (DESIGN.md §6 format).
 *
 *  M23: each node records its plugin's distribution version (from the
 *  catalog manifest) — the executor compares it at run time and warns
 *  about drift. Input paths (path_kind "in", file/directory widgets)
 *  under an anchor directory are stored anchor-relative (M19 decision:
 *  relativize on save).
 */
export function serializeWorkflow(
  nodes: Node<MTNodeData>[],
  edges: Edge[],
  name?: string,
  anchors?: SaveAnchors,
) {
  const { catalog } = useCatalog.getState();
  const manifestOf = (type: string) =>
    catalog?.nodes.find((m) => m.type === type);
  return {
    format: "emptyui/workflow",
    version: 1,
    meta: {
      name: name ?? useWorkflow.getState().docName ?? "untitled",
      created: useWorkflow.getState().docCreated ?? new Date().toISOString(),
    },
    nodes: nodes.map((n) => {
      const manifest = manifestOf(n.data.nodeType);
      const params: Record<string, unknown> = { ...n.data.params };
      if (anchors && manifest) {
        for (const spec of manifest.params) {
          if (
            spec.path_kind === "in" &&
            (spec.widget === "file" || spec.widget === "directory") &&
            typeof params[spec.name] === "string"
          ) {
            params[spec.name] = relativizeToAnchor(
              params[spec.name] as string,
              anchors,
            );
          }
        }
      }
      return {
        id: n.id,
        type: n.data.nodeType,
        params,
        position: n.position,
        plugin_version: manifest?.plugin_version,
      };
    }),
    edges: edges.map((e) => {
      // L12: defaulting a missing handle to "data" silently INVENTED wiring
      // on save (the loaded document then connects ports that were never
      // connected) — fail loudly instead
      if (e.sourceHandle == null || e.targetHandle == null) {
        throw new Error(
          `edge '${e.id}' (${e.source} -> ${e.target}) has no port handle — `
          + "refusing to save invented 'data' wiring",
        );
      }
      return {
        id: e.id,
        source: e.source,
        source_port: e.sourceHandle,
        target: e.target,
        target_port: e.targetHandle,
      };
    }),
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
    meta?: { name?: string; created?: string };
    nodes: { id: string; type: string; params?: Record<string, unknown>; position?: { x: number; y: number } }[];
    edges: { id: string; source: string; source_port: string; target: string; target_port: string }[];
  },
): { nodes: Node<MTNodeData>[]; edges: Edge[]; name: string; created?: string } {
  if (doc?.format !== "emptyui/workflow") {
    throw new Error(
      "not an emptyUI workflow document (missing format: 'emptyui/workflow')",
    );
  }
  if (doc.version !== 1) {
    throw new Error(`unsupported workflow version: ${String(doc.version)}`);
  }
  // L12: a document missing either array died as "Cannot read properties
  // of undefined (reading 'map')" behind the generic Load failed message —
  // name the actual defect
  if (!Array.isArray(doc.nodes) || !Array.isArray(doc.edges)) {
    throw new Error(
      "malformed workflow document: 'nodes' and 'edges' arrays are required",
    );
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
      // viewer sizing is manifest-driven (L-F12) — a load that races the
      // catalog falls back to the default box (reload after it arrives)
      style: manifest?.viewer
        ? { width: VIEWER_NODE_SIZE.width, height: VIEWER_NODE_SIZE.height }
        : { width: DEFAULT_NODE_SIZE.width, height: "auto" as const },
    };
  });
  const manifestOfType = (type: string) =>
    catalog?.nodes.find((m) => m.type === type);
  // migrate BEFORE pruning (§4): the legacy `runs` port is no longer
  // declared by the manifest, so an unmigrated edge is pruned as stale
  // before the migration can remap it (pruneStaleEdges runs first
  // otherwise)
  // — and the ONLY copy of the rule (D2): loadDocument used to re-run it
  // on sourceHandle after conversion, always as a no-op
  const extract = extractRunIds(nodes);
  const migrated = extract.size
    ? doc.edges.map((e) =>
        extract.has(e.source) && e.source_port === "runs"
          ? { ...e, source_port: "data" }
          : e,
      )
    : doc.edges;
  const edges: Edge[] = enforceWiringRules(
    pruneStaleEdges(migrated, nodes, manifestOfType), nodes, manifestOfType,
  ).map((e) => {
      // re-derive edge color from the source port type (L-F2: the catalog
      // carries plugin-declared port colors — the fallback table alone
      // ignored them)
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
        style: { stroke: portColor(portType, catalog), strokeWidth: 2.5 },
        data: { portType },
      };
    });
  return {
    nodes,
    edges,
    name: doc.meta?.name || "untitled",
    created: doc.meta?.created,
  } satisfies {
    nodes: Node<MTNodeData>[];
    edges: Edge[];
    name: string;
    created?: string;
  };
}
