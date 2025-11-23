import { useCatalog, portColor } from "../stores/catalog";
import { useWorkflow } from "../stores/workflow";

/**
 * Metadata panel (DESIGN.md §9, ComfyUI paradigm): node identity, ports and
 * description live here — parameters are edited inline on the node itself.
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
    return (
      <aside className="properties">
        <p className="muted">Select a node to see its details.</p>
        <p className="muted">
          Parameters are edited directly on each node; connect typed ports to
          build the pipeline, then press Run.
        </p>
      </aside>
    );

  return (
    <aside className="properties">
      <h2>{manifest.title}</h2>
      <table className="meta-table">
        <tbody>
          <tr>
            <td>node id</td>
            <td>
              <code>{node.id}</code>
            </td>
          </tr>
          <tr>
            <td>type</td>
            <td>
              <code>{manifest.type}</code>
            </td>
          </tr>
          <tr>
            <td>version</td>
            <td>{manifest.version}</td>
          </tr>
        </tbody>
      </table>
      <p className="muted">{manifest.description}</p>
      {manifest.deps.length > 0 && (
        <p className="muted">deps: {manifest.deps.join(", ")}</p>
      )}

      <h3>Ports</h3>
      <ul className="ports">
        {manifest.inputs.map((i) => (
          <li key={`in-${i.name}`}>
            <span className="dot" style={{ background: portColor(i.type, catalog) }} /> in{" "}
            {i.label ?? i.name}: <code>{i.type}</code>
          </li>
        ))}
        {manifest.outputs.map((o) => (
          <li key={`out-${o.name}`}>
            <span className="dot" style={{ background: portColor(o.type, catalog) }} /> out{" "}
            {o.label ?? o.name}: <code>{o.type}</code>
          </li>
        ))}
      </ul>

      <button className="danger" onClick={() => useWorkflow.getState().deleteSelected()}>
        Delete node
      </button>
    </aside>
  );
}
