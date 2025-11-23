import { Background, Controls, MiniMap, ReactFlow } from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import { useQuery } from "@tanstack/react-query";
import { api } from "./api/client";
import { useCatalog } from "./stores/catalog";
import { nodeTypes } from "./graph/MTNode";
import { NodePalette } from "./panels/NodePalette";
import { PropertiesPanel } from "./panels/PropertiesPanel";
import { useWorkflow, type MTNodeData } from "./stores/workflow";

export default function App() {
  const { nodes, edges, onNodesChange, onEdgesChange, onConnect, addNode, selectedNodeId } =
    useWorkflow();
  const setCatalog = useCatalog((s) => s.setCatalog);
  const catalog = useCatalog((s) => s.catalog);

  useQuery({
    queryKey: ["nodes"],
    queryFn: async () => {
      const c = await api.nodes();
      setCatalog(c);
      return c;
    },
  });

  const handleAdd = (manifest: { title: string; type: string }) => {
    addNode({
      id: `n${crypto.randomUUID().slice(0, 6)}`,
      type: "mt",
      position: { x: 80 + Math.random() * 200, y: 80 + Math.random() * 160 },
      data: {
        label: manifest.title,
        nodeType: manifest.type,
        params: {},
      } satisfies MTNodeData,
      style: { width: 200, height: 72 },
    });
  };

  return (
    <div className="app">
      <header className="toolbar">
        <strong>emptyUI</strong>
        <span className="muted">MT workflow builder — M0 shell</span>
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
