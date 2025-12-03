import { useCallback, useEffect, useRef, useState } from "react";
import { Background, Controls, MiniMap, ReactFlow } from "@xyflow/react";
import type { Connection, Edge } from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import { useQuery } from "@tanstack/react-query";
import { api } from "./api/client";
import { ensureSocket } from "./api/ws";
import { useCatalog } from "./stores/catalog";
import { nodeTypes } from "./graph/MTNode";
import { NodePalette } from "./panels/NodePalette";
import { PropertiesPanel } from "./panels/PropertiesPanel";
import {
  deserializeWorkflow,
  serializeWorkflow,
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
  const [message, setMessage] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const fileInput = useRef<HTMLInputElement>(null);

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

  // Live connection rule (DESIGN.md §3 + §9): self-loops, type mismatches
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

  const handleAdd = (manifest: NodeManifest) => {
    addNode({
      id: `n${crypto.randomUUID().slice(0, 6)}`,
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
          : { width: 250, height: "auto" },
    });
  };

  const handleRun = async () => {
    try {
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
      const { run_id } = await api.runWorkflow(doc);
      useRun.getState().setRunId(run_id);

      // Polling fallback: works even if the WebSocket is unavailable.
      // WS events and polls are both deduped by the run store.
      const seen = new Set<string>();
      const poll = window.setInterval(async () => {
        try {
          const run = (await api.getRun(run_id)) as {
            status: string;
            events: Record<string, unknown>[];
          };
          for (const e of run.events) {
            const key = `${e.event}:${e.node ?? e.id ?? ""}:${e.status ?? e.frac ?? ""}`;
            if (!seen.has(key)) {
              seen.add(key);
              useRun.getState().handleEvent(e);
            }
          }
          if (["completed", "failed", "cancelled"].includes(run.status)) {
            window.clearInterval(poll);
          }
        } catch {
          // transient poll errors are non-fatal
        }
      }, 700);
    } catch (e) {
      setMessage(e instanceof Error ? e.message : String(e));
    }
  };

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
    const doc = serializeWorkflow(nodes, edges);
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
      const { nodes: loaded, edges: loadedEdges } = deserializeWorkflow(doc);
      useWorkflow.getState().loadDocument(loaded, loadedEdges);
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
        <span className={`run-status ${runStatus}`}>{runStatus}</span>
        {notice && <span className="notice-msg">{notice}</span>}
        {message && <span className="error-msg">{message}</span>}
        <span className="spacer" />
        <button
          className="danger"
          disabled={!selectedNodeId}
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
            onNodeClick={(_, node) => useWorkflow.getState().selectNode(node.id)}
            onPaneClick={() => useWorkflow.getState().selectNode(null)}
            deleteKeyCode={["Backspace", "Delete"]}
            fitView
          >
            <Background />
            <Controls />
            <MiniMap pannable zoomable />
          </ReactFlow>
        </div>
        <PropertiesPanel />
      </div>
    </div>
  );
}
