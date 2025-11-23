import { useCatalog, portColor } from "../stores/catalog";
import { useWorkflow } from "../stores/workflow";

/**
 * Properties panel auto-generated from the selected node's manifest
 * params (DESIGN.md §9). Edits write straight into the workflow store.
 */
export function PropertiesPanel() {
  const catalog = useCatalog((s) => s.catalog);
  const nodes = useWorkflow((s) => s.nodes);
  const selectedNodeId = useWorkflow((s) => s.selectedNodeId);

  const node = nodes.find((n) => n.id === selectedNodeId) ?? null;
  const manifest = node
    ? catalog?.nodes.find((m) => m.type === node.data.nodeType)
    : undefined;

  if (!node || !manifest)
    return <aside className="properties">Select a node to edit its parameters.</aside>;

  const update = (name: string, value: unknown) =>
    useWorkflow
      .getState()
      .updateNodeData(node.id, (d) => ({ params: { ...d.params, [name]: value } }));

  return (
    <aside className="properties">
      <h2>{manifest.title}</h2>
      <p className="muted">
        {manifest.type} @ v{manifest.version}
      </p>

      <label>
        <span>Label</span>
        <input
          value={String(node.data.label)}
          onChange={(e) =>
            useWorkflow.getState().updateNodeData(node.id, { label: e.target.value })
          }
        />
      </label>

      {manifest.params.length === 0 && <p className="muted">No parameters.</p>}
      {manifest.params.map((p) => {
        const value = node.data.params[p.name] ?? p.default ?? "";
        return (
          <label key={p.name}>
            <span>
              {p.label}
              {p.optional ? " (optional)" : ""}
            </span>
            {p.widget === "boolean" ? (
              <input
                type="checkbox"
                checked={Boolean(value)}
                onChange={(e) => update(p.name, e.target.checked)}
              />
            ) : p.widget === "select" ? (
              <select
                value={String(value)}
                onChange={(e) => update(p.name, e.target.value)}
              >
                {(p.options ?? []).map((o) => (
                  <option key={String(o)}>{String(o)}</option>
                ))}
              </select>
            ) : p.widget === "integer" || p.widget === "number" ? (
              <input
                type="number"
                value={value === null ? "" : String(value)}
                min={p.min}
                max={p.max}
                step={p.step}
                onChange={(e) =>
                  update(p.name, e.target.value === "" ? null : Number(e.target.value))
                }
              />
            ) : (
              <input
                value={String(value)}
                onChange={(e) => update(p.name, e.target.value)}
              />
            )}
          </label>
        );
      })}

      <h3>Ports</h3>
      <ul className="ports">
        {manifest.inputs.map((i) => (
          <li key={`in-${i.name}`}>
            <span className="dot" style={{ background: portColor(i.type, catalog) }} /> in{" "}
            {i.name}: <code>{i.type}</code>
          </li>
        ))}
        {manifest.outputs.map((o) => (
          <li key={`out-${o.name}`}>
            <span className="dot" style={{ background: portColor(o.type, catalog) }} /> out{" "}
            {o.name}: <code>{o.type}</code>
          </li>
        ))}
      </ul>

      <button className="danger" onClick={() => useWorkflow.getState().deleteSelected()}>
        Delete node
      </button>
    </aside>
  );
}
