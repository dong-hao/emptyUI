import { useCallback, useEffect, useRef, useState } from "react";
import {
  Background,
  Controls,
  MiniMap,
  ReactFlow,
  SelectionMode,
  type Node,
  type ReactFlowInstance,
} from "@xyflow/react";
import type { Connection, Edge } from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import { useQuery } from "@tanstack/react-query";
import { api } from "./api/client";
import { ensureSocket, useSocketState } from "./api/ws";
import { useCatalog } from "./stores/catalog";
import { nodeTypes } from "./graph/MTNode";
import { NodePalette } from "./panels/NodePalette";
import { PropertiesPanel } from "./panels/PropertiesPanel";
import { SettingsDialog } from "./panels/SettingsDialog";
import { HistoryDialog } from "./panels/HistoryDialog";
import {
  DEFAULT_NODE_SIZE,
  deserializeWorkflow,
  serializeWorkflow,
  uniqueNodeId,
  useWorkflow,
  validateConnection,
  type MTNodeData,
  type SaveAnchors,
} from "./stores/workflow";
import type { HistoryDetail, NodeManifest } from "./api/client";
import { useRun } from "./stores/run";

export default function App() {
  const { nodes, edges, onNodesChange, onEdgesChange, onConnect, addNode } =
    useWorkflow();
  const setCatalog = useCatalog((s) => s.setCatalog);
  const catalog = useCatalog((s) => s.catalog);
  const runStatus = useRun((s) => s.runStatus);
  // H3: the reason of a failed run — the status badge alone leaves a red
  // "failed" with no explanation in the toolbar
  const runError = useRun((s) => s.runError);
  const startingCount = useRun((s) => s.startingCount);
  // L-F3 (§8.1): surface WS outages loudly — the badge appears when the
  // socket drops (reconnect attempts up to the cap, then a hard stop)
  const socketStatus = useSocketState((s) => s.status);
  const docName = useWorkflow((s) => s.docName);
  const [message, setMessage] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const [showSettings, setShowSettings] = useState(false);
  const [showHistory, setShowHistory] = useState(false);
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
  // L12: a SECOND document load (nodes already on screen) skipped fitView
  // entirely — the graph landed wherever the old viewport was looking
  const docRev = useWorkflow((s) => s.docRev);
  useEffect(() => {
    if (docRev > 0) {
      rfInstance.current?.fitView({ maxZoom: 1, padding: 0.1 });
    }
  }, [docRev]);

  const stopPolling = useCallback(() => {
    pollGenerationRef.current += 1;
    if (pollRef.current !== null) {
      window.clearInterval(pollRef.current);
      pollRef.current = null;
    }
  }, []);

  useEffect(() => stopPolling, [stopPolling]);

  // H6: the palette's ONLY data source is this query — its result used to be
  // discarded, so a boot-time backend failure produced an empty palette, a
  // disabled Run and zero explanation. The status is routed to the toolbar
  // and to the palette itself, which carries the retry.
  const catalogQuery = useQuery({
    queryKey: ["nodes"],
    queryFn: async () => {
      const c = await api.nodes();
      setCatalog(c);
      return c;
    },
  });
  const catalogPending = catalogQuery.isPending;
  const catalogError = catalogQuery.isError
    ? `catalog: ${
        catalogQuery.error instanceof Error
          ? catalogQuery.error.message
          : String(catalogQuery.error)
      }`
    : null;

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

  // Copy/paste of the selected subgraph (docs/DESIGN.md §8, ComfyUI parity):
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

  // Live connection rule (docs/DESIGN.md §3 + §8): self-loops, type mismatches
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
        useRun.getState().payloads,
      );
    },
    [catalog],
  );

  // --- Disconnect gestures (docs/DESIGN.md §8, ComfyUI parity) -----------------
  // 1. Drag a connected handle to the empty pane -> unlink. onConnectStart
  //    records the edge(s) on the starting handle; onConnectEnd detaches
  //    them when no valid new connection was made. Input handles hold one
  //    edge (one-to-one rule); an output handle with several edges is left
  //    alone (ambiguous — use click+Delete or edge-endpoint drag instead).
  const connectStartRef = useRef<{ edgeIds: string[]; x: number; y: number } | null>(null);
  // onConnect* passes DOM events; onReconnect* passes React synthetic
  // MouseEvents — both carry clientX/clientY, touch events carry touches
  const pointerPos = (
    event: MouseEvent | TouchEvent | React.MouseEvent<Element>,
  ) =>
    "touches" in event
      ? { x: event.touches[0]?.clientX ?? 0, y: event.touches[0]?.clientY ?? 0 }
      : { x: event.clientX, y: event.clientY };
  const handleConnectStart = useCallback(
    (event: MouseEvent | TouchEvent, { nodeId, handleId, handleType }: {
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
        connectStartRef.current = {
          edgeIds: onHandle.map((e) => e.id),
          ...pointerPos(event),
        };
      } else {
        connectStartRef.current = null;
      }
    },
    [],
  );
  const handleConnectEnd = useCallback(
    (event: MouseEvent | TouchEvent, state: { isValid: boolean | null }) => {
      const started = connectStartRef.current;
      connectStartRef.current = null;
      if (!started || state?.isValid === true) return;
      // hair-trigger guard (L-F12): a plain CLICK on a connected handle
      // (down+up, no movement) must not unlink — only a real drag that
      // ends without a valid new connection detaches (ComfyUI parity:
      // the link visibly follows the cursor before it detaches)
      const p = pointerPos(event);
      if (Math.hypot(p.x - started.x, p.y - started.y) < 6) return;
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
  const reconnectStartRef = useRef<{ x: number; y: number } | null>(null);
  const handleReconnect = useCallback((oldEdge: Edge, newConnection: Connection) => {
    reconnectSuccessful.current = true;
    useWorkflow.getState().reconnect(oldEdge, newConnection);
  }, []);
  const handleReconnectStart = useCallback((event: React.MouseEvent<Element>) => {
    reconnectSuccessful.current = false;
    reconnectStartRef.current = pointerPos(event);
  }, []);
  const handleReconnectEnd = useCallback(
    (event: MouseEvent | TouchEvent, edge: Edge) => {
      if (reconnectSuccessful.current) return;
      // same hair-trigger guard as the pane-disconnect gesture (L-F12):
      // a plain click on the edge endpoint must not delete the edge
      const started = reconnectStartRef.current;
      reconnectStartRef.current = null;
      if (started) {
        const p = pointerPos(event);
        if (Math.hypot(p.x - started.x, p.y - started.y) < 6) return;
      }
      useWorkflow.getState().detachEdges([edge.id]);
    },
    [],
  );

  const handleAdd = (manifest: NodeManifest) => {
    // §8.1: place the new node at the center of the user's current
    // viewport — a fixed origin disappears once the user pans/zooms away
    // from it (screenToFlowPosition accounts for pan and zoom).
    const inst = rfInstance.current;
    const id = uniqueNodeId();
    let position = { x: 80 + Math.random() * 200, y: 80 + Math.random() * 160 };
    if (inst) {
      // L-F12: query the RF container directly (`.react-flow` is unique)
      // instead of coupling to the `.canvas` wrapper structure
      const el = document.querySelector<HTMLElement>(".react-flow");
      const rect = el?.getBoundingClientRect();
      const cx = rect ? rect.left + rect.width / 2 : window.innerWidth / 2;
      const cy = rect ? rect.top + rect.height / 2 : window.innerHeight / 2;
      const p = inst.screenToFlowPosition({ x: cx, y: cy });
      // M13: half the declared default card size (one source of truth)
      position = {
        x: p.x - DEFAULT_NODE_SIZE.width / 2,
        y: p.y - 40,
      };
    }
    addNode({
      id,
      type: "mt",
      position,
      data: {
        label: manifest.title,
        nodeType: manifest.type,
        params: Object.fromEntries(
          manifest.params
            .filter((p) => p.default !== null && p.default !== undefined)
            .map((p) => [p.name, p.default]),
        ),
      } satisfies MTNodeData,
      // viewer nodes get the fixed figure box — from the manifest flag,
      // never a node-type literal (L-F12)
      style: manifest.viewer
        ? { width: 340, height: 260 }
        : { width: 280, height: "auto" },
    });
    // second pass: the card's height is unknown before the manifest renders
    // the params — measure it after it mounts and shift so its CENTER sits
    // at the viewport center.
    if (inst) {
      requestAnimationFrame(() =>
        requestAnimationFrame(() => {
          const el = document.querySelector<HTMLElement>(
            `.react-flow__node[data-id="${id}"]`);
          const cr = document.querySelector<HTMLElement>(
            ".react-flow")?.getBoundingClientRect();
          if (!el || !cr) return;
          const nr = el.getBoundingClientRect();
          const zoom = inst.getViewport().zoom || 1;
          const fcen = inst.screenToFlowPosition({
            x: cr.left + cr.width / 2,
            y: cr.top + cr.height / 2,
          });
          useWorkflow.getState().moveNode(id, {
            x: fcen.x - nr.width / zoom / 2,
            y: fcen.y - nr.height / zoom / 2,
          });
        }));
    }
  };

  /** M23: the document builder shared by Run and Save — stamps the
   *  plugin-version record and stores input paths anchor-relative (M19
   *  decision: relativize on save). Settings unavailable → paths are
   *  saved as-is (the record still stamps; degradation is silent only in
   *  the sense that nothing was relativized). */
  const buildDoc = async (name?: string) => {
    let anchors: SaveAnchors | undefined;
    try {
      const view = await api.settings();
      const dep = view.deployment as Record<string, unknown>;
      anchors = {
        projectDir: typeof dep.project_dir === "string" ? dep.project_dir : null,
        dataRoot: typeof dep.data_root === "string" ? dep.data_root : null,
      };
    } catch {
      anchors = undefined;
    }
    return serializeWorkflow(nodes, edges, name, anchors);
  };

  const startRun = async (targets?: string[]) => {
    // M15: make the documented contract true — a run attempt re-arms the
    // events socket if it is down (ensureSocket is idempotent; the slow
    // background retry from the M15 reconnect fix continues regardless)
    ensureSocket();
    const run = useRun.getState();
    // M-F2: gate re-entry across the POST in-flight window — runStatus is
    // still "idle" until setRunId resolves, so a second click would start a
    // concurrent run (last-resolved wins, the other orphaned)
    if (run.startingCount > 0) return;
    run.beginStart();
    try {
      stopPolling();
      setMessage(null);
      // Reminder: without a Display/output node the run produces nothing visible
      // L12: while the catalog is still loading every lookup misses — that
      // must not read as "no terminal node" (a false warning on startup)
      const hasOutputNode = !catalog
        ? true
        : nodes.some((n) => {
            const m = catalog.nodes.find(
              (m) => m.type === (n.data as MTNodeData).nodeType,
            );
            // M13: manifest-declared, never the category string as semantics
            return !!m && (m.outputs.length === 0 || m.terminal === true);
          });
      if (!hasOutputNode) {
        // L-F12 (§8.1): don't enumerate concrete node names (they vary by
        // catalog) — describe the property the check looks for
        setNotice(
          "Run results won't be visible — add a Display node or another terminal node (one with no outputs).",
        );
      } else {
        setNotice(null);
      }
      // M12: no reset() here — a re-run keeps the previous artifacts,
      // extents and node states visible until the new run's events replace
      // them (stale per-node state clears when that node re-executes);
      // wiping happens on document load / New only (handleLoad/handleNew)
      const doc = await buildDoc();
      const { run_id } = await api.runWorkflow(doc, targets);
      useRun.getState().setRunId(run_id);

      // Polling fallback: works even if the WebSocket is unavailable.
      // WS events and polls are both deduped by the run store.
      const seen = new Set<string>();
      const generation = pollGenerationRef.current;
      const eventKey = (event: Record<string, unknown>, index: number) => {
        if (event.event === "artifact") return `artifact:${event.id ?? index}`;
        if (event.event === "payload") {
          // port + summary identify the event (history §23.4): multi-output nodes
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
      let pollFails = 0; // M15: consecutive poll failures
      pollRef.current = window.setInterval(async () => {
        if (generation !== pollGenerationRef.current) return;
        try {
          const run = (await api.getRun(run_id)) as {
            status: string;
            events: Record<string, unknown>[];
          };
          if (generation !== pollGenerationRef.current) return;
          pollFails = 0;
          for (const [index, e] of run.events.entries()) {
            const key = eventKey(e, index);
            if (!seen.has(key)) {
              seen.add(key);
              useRun.getState().handleEvent(e);
            }
          }
          if (
            ["completed", "failed", "cancelled", "timeout"].includes(
              run.status,
            )
          ) {
            stopPolling();
          }
        } catch {
          // M15: a few transient blips stay silent, but a persistently dead
          // backend must not be polled forever with no feedback — stop and
          // say so (the M15 socket retry re-streams events when it returns)
          pollFails += 1;
          if (pollFails >= 5) {
            stopPolling();
            setMessage(
              "Lost contact with the backend — run status may be stale. Check the server and re-run.",
            );
          }
        }
      }, 700);
    } catch (e) {
      setMessage(e instanceof Error ? e.message : String(e));
    } finally {
      useRun.getState().endStart();
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

  const handleSave = async () => {
    const doc = await buildDoc(docName);
    const blob = new Blob([JSON.stringify(doc, null, 2)], { type: "application/json" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = `${doc.meta.name || "workflow"}.emptyui.json`;
    a.click();
    // L12: revoking synchronously could cancel the download mid-flight in
    // some browsers — the browser has read the blob by the time this fires
    window.setTimeout(() => URL.revokeObjectURL(a.href), 10_000);
  };

  const handleLoad = async (file: File) => {
    try {
      const doc = JSON.parse(await file.text());
      const { nodes: loaded, edges: loadedEdges, name, created } =
        deserializeWorkflow(doc);
      // mirror handleNew (§8): a document load is a new working context —
      // the previous run's chips/artifacts/extents must not bleed into it
      // (and its poll timer must not keep streaming foreign events)
      stopPolling();
      useRun.getState().reset();
      useWorkflow.getState().loadDocument(loaded, loadedEdges, name, created);
      if (!catalog?.nodes?.length) loadedBeforeCatalog.current = true;
      setMessage(null);
    } catch (e) {
      setMessage(`Load failed: ${e instanceof Error ? e.message : String(e)}`);
    }
  };

  const handleNew = () => {
    if (nodes.length > 0 &&
        !window.confirm("Clear the canvas? The current workflow is not saved.")) {
      return;
    }
    // mirror handleLoad (§8): a new working context must not keep streaming
    // the previous run's events (M15 — handleLoad did this, handleNew didn't)
    stopPolling();
    useRun.getState().reset();
    useWorkflow.getState().newWorkflow();
    setMessage(null);
  };

  /** L32: open a history entry — replace the canvas with its snapshot and
   *  hydrate the run view from the stored results. App owns the transition
   *  so the poll timer and the run store cannot keep streaming the old run. */
  const handleHistoryOpen = (detail: HistoryDetail) => {
    if (
      nodes.length > 0 &&
      !window.confirm("Open this run? The current workflow will be replaced.")
    ) {
      return;
    }
    try {
      // API-seeded runs may omit the document envelope (POST /api/runs
      // accepts a bare workflow); fill the defaults so the same snapshot
      // opens through the editor's strict deserializer.
      const raw = (detail.workflow ?? {}) as Record<string, unknown>;
      const doc = { format: "emptyui/workflow", version: 1, ...raw };
      const { nodes: loaded, edges: loadedEdges, name, created } =
        deserializeWorkflow(
          doc as Parameters<typeof deserializeWorkflow>[0],
        );
      stopPolling();
      useRun.getState().loadHistory(detail);
      useWorkflow.getState().loadDocument(loaded, loadedEdges, name, created);
      if (!catalog?.nodes?.length) loadedBeforeCatalog.current = true;
      setMessage(null);
      setShowHistory(false);
    } catch (e) {
      setMessage(
        `History open failed: ${e instanceof Error ? e.message : String(e)}`,
      );
    }
  };

  return (
    <div className="app">
      <header className="toolbar">
        <strong>emptyUI</strong>
        {socketStatus !== "online" && (
          <span
            className={`socket-badge ${socketStatus}`}
            role="status"
            title={
              socketStatus === "reconnecting"
                ? "Reconnecting to the server…"
                : "Server unreachable — reconnect gave up. Reload the page to retry."
            }
          >
            {socketStatus === "reconnecting"
              ? "reconnecting…"
              : "offline"}
          </span>
        )}
        <button
          onClick={handleRun}
          disabled={
            runStatus === "running" ||
            runStatus === "queued" ||
            startingCount > 0 ||
            nodes.length === 0
          }
        >
          ▶ Run
        </button>
        <button
          className="danger"
          onClick={handleCancel}
          disabled={runStatus !== "running" && runStatus !== "queued"}
        >
          ✕ Cancel
        </button>
        <button
          onClick={handleNew}
          disabled={
            runStatus === "running" ||
            runStatus === "queued" ||
            startingCount > 0
          }
          title="Start a new (empty) workflow"
        >
          New
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
          onChange={(e) => {
            // L-F6: reset so re-picking the SAME file (after edits) fires
            // again — the browser keeps the selection otherwise
            if (e.target.files?.[0]) handleLoad(e.target.files[0]);
            e.target.value = "";
          }}
        />
        <button
          onClick={() => setShowHistory(true)}
          title="Previous runs (session-only)"
        >
          🕘 History
        </button>
        <button onClick={() => setShowSettings(true)} title="Enable/disable plugins">
          ⚙ Settings
        </button>
        <span className={`run-status ${runStatus}`} role="status">{runStatus}</span>
        {notice && (
          <span className="notice-msg" role="status">
            {notice}
          </span>
        )}
        {(message ?? runError ?? catalogError) && (
          <span
            className="error-msg"
            role="alert"
            title={message ?? runError ?? catalogError ?? undefined}
          >
            {message ?? runError ?? catalogError}
          </span>
        )}
        <span className="spacer" />
        <button
          className="danger"
          disabled={
            // H4: enabled by the VISIBLE React Flow selection only — the
            // button deletes exactly what is highlighted, so a panel-only
            // focus can never arm a delete the user is not looking at
            !nodes.some((n) => n.selected) && !edges.some((e) => e.selected)
          }
          onClick={() => useWorkflow.getState().deleteSelected()}
        >
          Delete selected
        </button>
      </header>
      <div className="main">
        <NodePalette
          manifests={catalog?.nodes ?? []}
          onAdd={handleAdd}
          pending={catalogPending}
          error={catalogError}
          onRetry={() => {
            void catalogQuery.refetch();
          }}
        />
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
            selectionMode={SelectionMode.Full}
            selectionKeyCode="Shift"
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
      {showHistory && (
        <HistoryDialog
          onClose={() => setShowHistory(false)}
          onOpen={handleHistoryOpen}
        />
      )}
    </div>
  );
}
