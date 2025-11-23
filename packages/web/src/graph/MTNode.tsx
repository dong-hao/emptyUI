import { Handle, NodeResizer, Position, type NodeProps } from "@xyflow/react";
import { useCatalog, portColor } from "../stores/catalog";
import type { MTNodeData } from "../stores/workflow";

/**
 * Custom node renderer (DESIGN.md §9 + Port rendering conventions):
 * typed ports sit just inside the node edges, vertically aligned with
 * their name text (ComfyUI-style). Nodes are resizable via corner handles.
 */
export function MTNode({ data, selected }: NodeProps) {
  const catalog = useCatalog((s) => s.catalog);
  const manifest = catalog?.nodes.find((m) => m.type === data.nodeType);
  const d = data as MTNodeData;

  const inputs = manifest?.inputs ?? [];
  const outputs = manifest?.outputs ?? [];
  const rows = Math.max(inputs.length, outputs.length);

  return (
    <div className={`mt-node${selected ? " selected" : ""}`}>
      <NodeResizer
        isVisible={selected}
        minWidth={180}
        minHeight={64}
        lineClassName="mt-resize-line"
        handleClassName="mt-resize-handle"
      />
      <div className="mt-node-header">{manifest?.title ?? d.label}</div>
      <div className="mt-node-ports">
        {rows > 0 ? (
          Array.from({ length: rows }, (_, i) => {
            const input = inputs[i];
            const output = outputs[i];
            return (
              <div className="mt-row" key={`row-${i}`}>
                <span className="mt-port-label in">
                  {input && (
                    <>
                      <Handle
                        id={input.name}
                        type="target"
                        position={Position.Left}
                        className="mt-port"
                        style={{ background: portColor(input.type, catalog) }}
                        title={`${input.name}: ${input.type}`}
                      />
                      <span title={`${input.name}: ${input.type}`}>{input.name}</span>
                    </>
                  )}
                </span>
                <span className="mt-port-label out">
                  {output && (
                    <>
                      <span
                        className="mt-out-name"
                        title={`${output.name}: ${output.type}`}
                      >
                        {output.name}
                      </span>
                      <Handle
                        id={output.name}
                        type="source"
                        position={Position.Right}
                        className="mt-port"
                        style={{ background: portColor(output.type, catalog) }}
                        title={`${output.name}: ${output.type}`}
                      />
                    </>
                  )}
                </span>
              </div>
            );
          })
        ) : (
          <div className="mt-row muted-small">no ports</div>
        )}
      </div>
    </div>
  );
}

export const nodeTypes = { mt: MTNode };
