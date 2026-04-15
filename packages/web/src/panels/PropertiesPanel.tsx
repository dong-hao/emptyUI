import { useEffect, useRef, useState } from "react";
import { useCatalog, portColor } from "../stores/catalog";
import { useRun } from "../stores/run";
import { useWorkflow } from "../stores/workflow";

/**
 * Metadata panel (DESIGN.md §8, ComfyUI paradigm): node identity, ports and
 * description live here — parameters are edited inline on the node itself.
 * Partial execution: "Run to here" runs the selected node plus its upstream
 * ancestry only (DESIGN.md §7); unchanged nodes are reused from the cache.
 */
export function PropertiesPanel({
  onRunToHere,
}: {
  onRunToHere?: (nodeId: string) => void;
}) {
  const catalog = useCatalog((s) => s.catalog);
  const nodes = useWorkflow((s) => s.nodes);
  const selectedNodeId = useWorkflow((s) => s.selectedNodeId);
  // starting a second run while one is active orphans the first run's UI;
  // startingCount also gates the POST in-flight window (runStatus is still
  // "idle" until setRunId resolves — M-F2). NB: both subscriptions must be
  // called unconditionally (hooks) — an `||` short-circuit crashed React.
  const runStatus = useRun((s) => s.runStatus);
  const startingCount = useRun((s) => s.startingCount);
  // the selected node's run state (§8.1 good-to-have): errors surface in
  // the panel as copyable text — the node card only shows an ellipsized
  // badge the user cannot read out
  const nodeState = useRun((s) =>
    selectedNodeId ? s.nodeStates[selectedNodeId] : undefined,
  );
  const runActive = ["queued", "running"].includes(runStatus) || startingCount > 0;

  const node = nodes.find((n) => n.id === selectedNodeId) ?? null;
  const manifest = node
    ? catalog?.nodes.find((m) => m.type === node.data.nodeType)
    : undefined;

  const [copied, setCopied] = useState(false);
  const copyTimer = useRef<number | undefined>(undefined);
  useEffect(() => () => window.clearTimeout(copyTimer.current), []);

  const copyError = () => {
    const text = nodeState?.error ?? "";
    if (!text) return;
    const done = () => {
      setCopied(true);
      window.clearTimeout(copyTimer.current);
      copyTimer.current = window.setTimeout(() => setCopied(false), 2000);
    };
    // navigator.clipboard exists only on secure contexts — the ZeroTier
    // http origin (e.g. 10.0.10.15:7860) is NOT one; fall back to
    // execCommand with a temporary textarea (and again when the
    // clipboard promise rejects, e.g. headless permission denials)
    const fallbackCopy = () => {
      try {
        const ta = document.createElement("textarea");
        ta.value = text;
        ta.style.position = "fixed";
        ta.style.opacity = "0";
        document.body.appendChild(ta);
        ta.select();
        document.execCommand("copy");
        ta.remove();
        done();
      } catch {
        // final fallback: the text itself stays user-selectable
      }
    };
    if (navigator.clipboard && window.isSecureContext) {
      navigator.clipboard.writeText(text).then(done).catch(fallbackCopy);
    } else {
      fallbackCopy();
    }
  };

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
      {nodeState?.status === "error" && nodeState.error && (
        <div className="node-error">
          <div className="node-error-head">
            <strong>error</strong>
            <button onClick={copyError} title="Copy the error message">
              {copied ? "copied ✓" : "copy"}
            </button>
          </div>
          <pre className="node-error-text">{nodeState.error}</pre>
        </div>
      )}
      <p className="muted">{manifest.description}</p>
      {manifest.deps.length > 0 && (
        <p className="muted">deps: {manifest.deps.join(", ")}</p>
      )}
      {(manifest.models?.length ?? 0) > 0 && (
        <div className="muted">
          <h3>Models</h3>
          <ul>
            {manifest.models!.map((m) => (
              <li key={m.id}>
                <code>{m.filename}</code> ({m.license}) — sha256{" "}
                {m.sha256.slice(0, 12)}…
              </li>
            ))}
          </ul>
        </div>
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

      {onRunToHere && (
        <button
          onClick={() => onRunToHere(node.id)}
          disabled={runActive}
          title="Run this node and everything upstream; unchanged nodes are reused from cache"
        >
          ▶ Run to here
        </button>
      )}

      {/* L-F12 (§8.1): delete the node this panel shows, not the whole
          RF multi-selection */}
      <button
        className="danger"
        onClick={() => useWorkflow.getState().deleteNode(node.id)}
      >
        Delete node
      </button>
    </aside>
  );
}
