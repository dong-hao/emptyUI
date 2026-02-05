import { useCallback, useEffect, useRef, useState } from "react";
import {
  Background,
  Controls,
  MiniMap,
  ReactFlow,
  type Node,
  type ReactFlowInstance,
} from "@xyflow/react";
import type { Connection, Edge } from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import { useQuery } from "@tanstack/react-query";
import { api } from "./api/client";
import { ensureSocket } from "./api/ws";
import { useCatalog } from "./stores/catalog";
import { nodeTypes } from "./graph/MTNode";
import { NodePalette } from "./panels/NodePalette";
import { PropertiesPanel } from "./panels/PropertiesPanel";
import { SettingsDialog } from "./panels/SettingsDialog";
import {
  deserializeWorkflow,
  serializeWorkflow,
  uniqueNodeId,
  useWorkflow,
  validateConnection,
  type MTNodeData,
} from "./stores/workflow";
import type { NodeManifest } from "./api/client";
import { useRun } from "./stores/run";

export default function App() {
  const { nodes, edges, onNodesChange, onEdgesChange, onConnect, addNode, selectedNodeId } =
    useWorkflow();
  const setCatalog = useCatalog((s) => s.setCatalog);
  const catalog = useCatalog((s) => s.catalog);
  const runStatus = useRun((s) => s.runStatus);
  const docName = useWorkflow((s) => s.docName);
  const [message, setMessage] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const [showSettings, setShowSettings] = useState(false);
  const fileInput = useRef<HTMLInputElement>(null);
  const pollRef = useRef<number | null>(null);
  const pollGenerationRef = useRef(0);
  const rfInstance = useRef<
    ReactFlowInstance<Node<MTNodeData>, Edge> | null
  >(null);

  // §8 initial view: fit the graph once it first becomes non-empty (the
  // declarative `fitView` prop is consumed on the empty initial node set in
  // React Flow v12 — an explicit call after the first node arrives covers
  // both palette-add and document load). React Flow v12 padding is a
  // FRACTION of the viewport (0.1 = 10%), not px.
  const hadNodes = useRef(false);
  useEffect(() => {
    if (nodes.length > 0 && !hadNodes.current) {
      hadNodes.current = true;
      rfInstance.current?.fitView({ maxZoom: 1, padding: 0.1 });
    } else if (nodes.length === 0) {
      hadNodes.current = false;
    }
  }, [nodes.length]);

  const stopPolling = useCallback(() => {
    pollGenerationRef.current += 1;
    if (pollRef.current !== null) {
      window.clearInterval(pollRef.current);
      pollRef.current = null;
    }
  }, []);

  useEffect(() => stopPolling, [stopPolling]);

  useQuery({
    queryKey: ["nodes"],
    queryFn: async () => {
      const c = await api.nodes();
      setCatalog(c);
      return c;
    },
  });

  useEffect(() => {
    ensureSocket();
  }, []);

  // A document loaded before the catalog arrived skips stale-edge pruning
  // vacuously — re-run it once the catalog is available.
  const loadedBeforeCatalog = useRef(false);
  useEffect(() => {
    if (catalog && loadedBeforeCatalog.current) {
      loadedBeforeCatalog.current = false;
      useWorkflow.getState().revalidateEdges();
    }
  }, [catalog]);

  // Copy/paste of the selected subgraph (DESIGN.md §8, ComfyUI parity):
  // Ctrl/Cmd+C copies the selected nodes + their internal edges, Ctrl/Cmd+V
  // pastes with fresh ids and a fan-out offset, Ctrl/Cmd+D duplicates in
  // place. Keystrokes originating in form controls (param editing!) are
  // ignored so text copy/paste works normally inside widgets.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (!(e.ctrlKey || e.metaKey)) return;
      const t = e.target as HTMLElement | null;
      if (
        t &&
        (t.tagName === "INPUT" ||
          t.tagName === "TEXTAREA" ||
          t.tagName === "SELECT" ||
          t.isContentEditable)
      ) {
        return;
      }
      const wf = useWorkflow.getState();
      const key = e.key.toLowerCase();
      if (key === "c") {
        wf.copySelection();
        e.preventDefault();
      } else if (key === "v") {
        wf.pasteSelection();
        e.preventDefault();
      } else if (key === "d") {
        wf.duplicateSelection();
        e.preventDefault();
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  // Live connection rule (DESIGN.md §3 + §8): self-loops, type mismatches
  // and cycles are rejected during the drag — React Flow greys out invalid
  // drop targets. The same rule re-validates in onConnect and server-side.
  const isValidConnection = useCallback(
    (conn: Connection | Edge) => {
      const state = useWorkflow.getState();
      return validateConnection(
        conn as Connection,
        state.nodes,
        state.edges,
        catalog?.nodes ?? [],
      );
    },
    [catalog],
  );

  // --- Disconnect gestures (DESIGN.md §8, ComfyUI parity) -----------------
  // 1. Drag a connected handle to the empty pane -> unlink. onConnectStart
  //    records the edge(s) on the starting handle; onConnectEnd detaches
  //    them when no valid new connection was made. Input handles hold one
  //    edge (one-to-one rule); an output handle with several edges is left
  //    alone (ambiguous — use click+Delete or edge-endpoint drag instead).
  const connectStartRef = useRef<{ edgeIds: string[] } | null>(null);
  const handleConnectStart = useCallback(
    (_event: unknown, { nodeId, handleId, handleType }: {
      nodeId: string | null;
      handleId: string | null;
      handleType: "source" | "target" | null;
    }) => {
      if (!nodeId || !handleId || !handleType) {
        connectStartRef.current = null;
        return;
      }
      const { edges } = useWorkflow.getState();
      const onHandle = edges.filter((e) =>
        handleType === "target"
          ? e.target === nodeId && e.targetHandle === handleId
          : e.source === nodeId && e.sourceHandle === handleId,
      );
      if (onHandle.length === 1) {
        connectStartRef.current = { edgeIds: onHandle.map((e) => e.id) };
      } else {
        connectStartRef.current = null;
      }
    },
    [],
  );
  const handleConnectEnd = useCallback(
    (_event: unknown, state: { isValid: boolean | null }) => {
      const started = connectStartRef.current;
      connectStartRef.current = null;
      if (!started || state?.isValid === true) return;
      // drag ended without a valid connection -> unlink (dropping the link
      // on an invalid target also detaches, matching ComfyUI)
      useWorkflow.getState().detachEdges(started.edgeIds);
    },
    [],
  );

  // 2. Drag either endpoint of an existing edge: drop on another compatible
  //    handle to re-route, drop on the pane to delete (RF "delete edge on
  //    drop" + reconnectEdge pattern).
  const reconnectSuccessful = useRef(false);
  const handleReconnect = useCallback((oldEdge: Edge, newConnection: Connection) => {
    reconnectSuccessful.current = true;
    useWorkflow.getState().reconnect(oldEdge, newConnection);
  }, []);
  const handleReconnectStart = useCallback(() => {
    reconnectSuccessful.current = false;
  }, []);
  const handleReconnectEnd = useCallback((_event: unknown, edge: Edge) => {
    if (!reconnectSuccessful.current) useWorkflow.getState().detachEdges([edge.id]);
  }, []);

  const handleAdd = (manifest: NodeManifest) => {
    addNode({
      id: uniqueNodeId(),
      type: "mt",
      position: { x: 80 + Math.random() * 200, y: 80 + Math.random() * 160 },
      data: {
        label: manifest.title,
        nodeType: manifest.type,
        params: Object.fromEntries(
          manifest.params
            .filter((p) => p.default !== null && p.default !== undefined)
            .map((p) => [p.name, p.default]),
        ),
      } satisfies MTNodeData,
      style:
        manifest.type === "core.display.preview_plot" ||
        manifest.type === "core.display.view_table"
          ? { width: 340, height: 260 }
          : { width: 280, height: "auto" },
    });
  };

  const startRun = async (targets?: string[]) => {
    try {
      stopPolling();
      setMessage(null);
      // Reminder: without a Display/output node the run produces nothing visible
      const hasOutputNode = nodes.some((n) => {
        const m = catalog?.nodes.find(
          (m) => m.type === (n.data as MTNodeData).nodeType,
        );
        return !!m && (m.outputs.length === 0 || m.category === "Display");
      });
      if (!hasOutputNode) {
        setNotice(
          "No Display or output node (Plot / Preview / Save) — add one to see results.",
        );
      } else {
        setNotice(null);
      }
      useRun.getState().reset();
      const doc = serializeWorkflow(nodes, edges);
      const { run_id } = await api.runWorkflow(doc, targets);
      useRun.getState().setRunId(run_id);

      // Polling fallback: works even if the WebSocket is unavailable.
      // WS events and polls are both deduped by the run store.
      const seen = new Set<string>();
      const generation = pollGenerationRef.current;
      const eventKey = (event: Record<string, unknown>, index: number) => {
        if (event.event === "artifact") return `artifact:${event.id ?? index}`;
        if (event.event === "payload") {
          // port + summary identify the event (§23.4): multi-output nodes
          // (e.g. Align Runs data1/data2) each carry their own extent
          return `payload:${event.node ?? ""}:${event.port ?? ""}:${
            JSON.stringify(event.summary ?? "")}`;
        }
        if (event.event === "run_started" || event.event === "run_finished") {
          return `${event.event}:${event.status ?? ""}`;
        }
        return JSON.stringify([
          event.event,
          event.node ?? "",
          event.status ?? "",
          event.frac ?? "",
          event.msg ?? "",
          event.error ?? "",
          // `cached` differentiates a cache-restored done from a fresh one —
          // without it the chip is missed in polling-only replay
          event.cached ?? "",
        ]);
      };
      pollRef.current = window.setInterval(async () => {
        if (generation !== pollGenerationRef.current) return;
        try {
          const run = (await api.getRun(run_id)) as {
            status: string;
            events: Record<string, unknown>[];
          };
          if (generation !== pollGenerationRef.current) return;
          for (const [index, e] of run.events.entries()) {
            const key = eventKey(e, index);
            if (!seen.has(key)) {
              seen.add(key);
              useRun.getState().handleEvent(e);
            }
          }
          if (["completed", "failed", "cancelled"].includes(run.status)) {
            stopPolling();
          }
        } catch {
          // transient poll errors are non-fatal
        }
      }, 700);
    } catch (e) {
      setMessage(e instanceof Error ? e.message : String(e));
    }
  };

  const handleRun = () => startRun();

  const handleCancel = async () => {
    const runId = useRun.getState().runId;
    if (!runId) return;
    try {
      setMessage(null);
      await api.cancelRun(runId);
    } catch (e) {
      setMessage(e instanceof Error ? e.message : String(e));
    }
  };

  const handleSave = () => {
    const doc = serializeWorkflow(nodes, edges, docName);
    const blob = new Blob([JSON.stringify(doc, null, 2)], { type: "application/json" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = `${doc.meta.name || "workflow"}.emptyui.json`;
    a.click();
    URL.revokeObjectURL(a.href);
  };

  const handleLoad = async (file: File) => {
    try {
      const doc = JSON.parse(await file.text());
      const { nodes: loaded, edges: loadedEdges, name } = deserializeWorkflow(doc);
      useWorkflow.getState().loadDocument(loaded, loadedEdges, name);
      if (!catalog?.nodes?.length) loadedBeforeCatalog.current = true;
      setMessage(null);
    } catch (e) {
      setMessage(`Load failed: ${e instanceof Error ? e.message : String(e)}`);
    }
  };

  return (
    <div className="app">
      <header className="toolbar">
        <strong>emptyUI</strong>
        <button onClick={handleRun} disabled={runStatus === "running" || nodes.length === 0}>
          ▶ Run
        </button>
        <button
          className="danger"
          onClick={handleCancel}
          disabled={runStatus !== "running" && runStatus !== "queued"}
        >
          ✕ Cancel
        </button>
        <button onClick={handleSave} disabled={nodes.length === 0}>
          Save
        </button>
        <button onClick={() => fileInput.current?.click()}>Load</button>
        <input
          ref={fileInput}
          type="file"
          accept=".json,application/json"
          hidden
          onChange={(e) => e.target.files?.[0] && handleLoad(e.target.files[0])}
        />
        <button onClick={() => setShowSettings(true)} title="Enable/disable plugins">
          ⚙ Settings
        </button>
        <span className={`run-status ${runStatus}`}>{runStatus}</span>
        {notice && <span className="notice-msg">{notice}</span>}
        {message && <span className="error-msg">{message}</span>}
        <span className="spacer" />
        <button
          className="danger"
          disabled={
            !nodes.some((n) => n.selected) &&
            !edges.some((e) => e.selected) &&
            !selectedNodeId
          }
          onClick={() => useWorkflow.getState().deleteSelected()}
        >
          Delete selected
        </button>
      </header>
      <div className="main">
        <NodePalette manifests={catalog?.nodes ?? []} onAdd={handleAdd} />
        <div className="canvas">
          <ReactFlow
            nodes={nodes}
            edges={edges}
            nodeTypes={nodeTypes}
            onNodesChange={onNodesChange}
            onEdgesChange={onEdgesChange}
            onConnect={onConnect}
            isValidConnection={isValidConnection}
            onConnectStart={handleConnectStart}
            onConnectEnd={handleConnectEnd}
            edgesReconnectable
            onReconnect={handleReconnect}
            onReconnectStart={handleReconnectStart}
            onReconnectEnd={handleReconnectEnd}
            onNodeClick={(_, node) => useWorkflow.getState().selectNode(node.id)}
            onPaneClick={() => useWorkflow.getState().selectNode(null)}
            deleteKeyCode={["Backspace", "Delete"]}
            onInit={(instance) => {
              rfInstance.current = instance;
            }}
          >
            <Background />
            <Controls />
            <MiniMap pannable zoomable />
          </ReactFlow>
        </div>
        <PropertiesPanel onRunToHere={(nodeId) => startRun([nodeId])} />
      </div>
      {showSettings && <SettingsDialog onClose={() => setShowSettings(false)} />}
    </div>
  );
}
