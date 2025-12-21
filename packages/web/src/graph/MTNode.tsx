import { Handle, NodeResizer, Position, type NodeProps } from "@xyflow/react";
import { useEffect, useMemo, useState } from "react";
import { api } from "../api/client";
import { useCatalog, portColor } from "../stores/catalog";
import { useRun } from "../stores/run";
import { useWorkflow, type MTNodeData } from "../stores/workflow";
import { ParamField } from "../panels/ParamField";
import { inheritableParams, makeInheritContext, resolveInherited } from "./inherit";

/** Resolve the artifact flowing into a Preview node's `plot` input:
 *  input edge → source node + source handle → matching artifact. */
function useInputPlotArtifact(nodeId: string) {
  const edges = useWorkflow((s) => s.edges);
  const artifacts = useRun((s) => s.artifacts);
  const edge = edges.find((e) => e.target === nodeId);
  if (!edge) return null;
  const candidates = artifacts.filter((a) => a.nodeId === edge.source);
  if (candidates.length === 0) return null;
  return (
    candidates.find((a) => a.handle && a.handle === edge.sourceHandle) ??
    candidates[candidates.length - 1]
  );
}

/**
 * Custom node renderer (DESIGN.md §9 + Port rendering conventions):
 * typed ports inside the edges aligned with labels, and — following the
 * ComfyUI paradigm — the node's parameters rendered inline on the node
 * body. The right-hand panel keeps metadata (id, version, ports,
 * description, deps) and a Delete action (see PropertiesPanel).
 */
export function MTNode({ id, data, selected }: NodeProps) {
  const catalog = useCatalog((s) => s.catalog);
  const manifest = catalog?.nodes.find((m) => m.type === data.nodeType);
  const d = data as MTNodeData;
  const runState = useRun((s) => s.nodeStates[id]);
  const nodes = useWorkflow((s) => s.nodes);
  const edges = useWorkflow((s) => s.edges);

  const inputs = manifest?.inputs ?? [];
  const outputs = manifest?.outputs ?? [];
  const rows = Math.max(inputs.length, outputs.length);
  const params = manifest?.params ?? [];

  // Chain-walking inheritance context (DESIGN.md §4): survey/station/run (and
  // any other declared params) auto-fill from the upstream chain, flowing
  // through intermediate nodes.
  const inheritCtx = useMemo(
    () => makeInheritContext(catalog?.nodes, nodes, edges),
    [catalog, nodes, edges],
  );
  const inheritedOptions = (paramName: string): string[] =>
    resolveInherited(inheritCtx, id, paramName);

  // Only params the node declares as inheritable may auto-fill / become a
  // dropdown; everything else (root defaults, radio/select config) stays manual.
  const inheritable = useMemo(() => inheritableParams(manifest), [manifest]);

  const setParam = (name: string, value: unknown) => {
    const state = useWorkflow.getState();
    const updates: Record<string, unknown> = { [name]: value };
    // apply manifest cascade rules (e.g. magnetic h<->b swaps channel names;
    // remote reference forces the robust estimator)
    const spec = manifest?.params.find((p) => p.name === name);
    const actions = spec?.cascade?.[String(value)];
    if (actions) {
      for (const [target, tv] of actions) updates[target] = tv;
    }
    state.updateNodeData(id, (prev) => ({ params: { ...prev.params, ...updates } }));
  };

  // Auto-fill inheritable params that are still unset once a source is
  // connected. "Unset" includes a value equal to the manifest default (e.g.
  // channel_nomenclature defaults to "auto" — an upstream-provided keyword may
  // override it, but an explicit user choice never is).
  useEffect(() => {
    const state = useWorkflow.getState();
    const ctx = makeInheritContext(catalog?.nodes, state.nodes, state.edges);
    const cur =
      (state.nodes.find((n) => n.id === id)?.data as MTNodeData | undefined)?.params ?? {};
    const isUnset = (name: string, value: unknown) => {
      const spec = params.find((p) => p.name === name);
      const dflt = spec?.default;
      return (
        value === undefined ||
        value === null ||
        String(value).trim() === "" ||
        (dflt !== undefined && dflt !== null && String(value) === String(dflt))
      );
    };
    const updates: Record<string, unknown> = {};
    for (const p of params) {
      if (!inheritable.has(p.name)) continue;
      const opts = resolveInherited(ctx, id, p.name);
      if (opts.length === 0) continue;
      if (isUnset(p.name, cur[p.name])) updates[p.name] = opts[0];
    }
    // Inherited values bypass setParam, so apply the manifest cascade manually
    // (e.g. an inherited channel_nomenclature also fills input/output channels
    // — including the vertical magnetic channel for the tipper).
    const merged = { ...cur, ...updates };
    for (const [name, value] of Object.entries(updates)) {
      const spec = params.find((p) => p.name === name);
      const actions = spec?.cascade?.[String(value)];
      if (!actions) continue;
      for (const [target, tv] of actions) {
        if (isUnset(target, merged[target])) {
          updates[target] = tv;
          merged[target] = tv;
        }
      }
    }
    if (Object.keys(updates).length) {
      // Idempotence guard: skip the write when nothing actually differs from
      // the current params. Without it, an inherited value equal to the
      // manifest default would produce updates on every pass and — since
      // updateNodeData always creates new node objects and this effect
      // depends on `nodes` — re-trigger itself forever.
      const effective = Object.fromEntries(
        Object.entries(updates).filter(([k, v]) => String(cur[k]) !== String(v)),
      );
      if (Object.keys(effective).length) {
        state.updateNodeData(id, (prev) => ({
          params: { ...prev.params, ...effective },
        }));
      }
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [id, catalog, manifest, edges, nodes]);

  // The remote-reference checkbox only applies to the robust estimator;
  // disable it when a least-squares estimator is selected.
  const engineVal = String(d.params["estimator_engine"] ?? "Robust").toLowerCase();
  const isLeastSquares = engineVal.includes("least") || engineVal === "ols" || engineVal === "ls";

  return (
    <div className={`mt-node${selected ? " selected" : ""} status-${runState?.status ?? "idle"}`}>
      <NodeResizer
        isVisible={selected}
        minWidth={180}
        minHeight={64}
        lineClassName="mt-resize-line"
        handleClassName="mt-resize-handle"
      />
      <div className="mt-node-header" title={manifest?.tooltip || undefined}>
        {manifest?.title ?? d.label}
      </div>
      {runState?.status === "running" && (
        <div className="mt-progress" style={{ width: `${(runState.progress ?? 0) * 100}%` }} />
      )}
      {runState?.cached && (
        <div className="mt-cached" title="Result reused from cache — inputs unchanged">
          cached
        </div>
      )}
      {runState?.status === "error" && (
        <div className="mt-error" title={runState.error}>
          error: {runState.error}
        </div>
      )}

      {params.length > 0 && (
        <div className="mt-node-params">
          {params.map((p) => {
            const opts = inheritable.has(p.name) ? inheritedOptions(p.name) : [];
            const sel = opts.length > 0 ? opts : undefined;
            return (
              <ParamField
                key={p.name}
                param={p}
                value={d.params[p.name] ?? p.default ?? ""}
                onChange={(v) => setParam(p.name, v)}
                compact
                instanceId={id}
                disabled={p.name === "remote_reference" && isLeastSquares}
                selectOptions={sel}
              />
            );
          })}
        </div>
      )}

      {d.nodeType === "core.display.preview_plot" ? (
        <PreviewBody nodeId={id} />
      ) : d.nodeType === "core.display.view_table" ? (
        <TableBody nodeId={id} />
      ) : (
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
                          title={`${input.label ?? input.name}: ${input.type}`}
                        />
                      <span title={`${input.label ?? input.name}: ${input.type}`}>
                        {input.label ?? input.name}
                      </span>
                    </>
                  )}
                </span>
                <span className="mt-port-label out">
                  {output && (
                    <>
                      <span
                        className="mt-out-name"
                        title={`${output.label ?? output.name}: ${output.type}`}
                      >
                        {output.label ?? output.name}
                      </span>
                      <Handle
                        id={output.name}
                        type="source"
                        position={Position.Right}
                        className="mt-port"
                        style={{ background: portColor(output.type, catalog) }}
                        title={`${output.label ?? output.name}: ${output.type}`}
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
      )}
    </div>
  );
}

/** Inline figure viewer: fills the node body; node drag-resize scales it. */
function PreviewBody({ nodeId }: { nodeId: string }) {
  const artifact = useInputPlotArtifact(nodeId);
  const ports = (
    <div className="mt-node-ports mt-preview-ports">
      <div className="mt-row">
        <span className="mt-port-label in">
          <HandleInPlot />
          <span>plot</span>
        </span>
        <span className="mt-port-label out">
          <span>plot</span>
          <HandleOutPlot />
        </span>
      </div>
    </div>
  );
  return (
    <>
      <div className="mt-preview">
        {artifact ? (
          <>
            <img src={artifact.url} alt={`preview ${artifact.id}`} />
            <a
              className="mt-download"
              href={api.artifactUrl(artifact.id)}
              download
              aria-label="Download artifact"
              title="Download artifact"
            >
              ⤓
            </a>
          </>
        ) : (
          <span className="muted-small">connect a plot · run to render</span>
        )}
      </div>
      {ports}
    </>
  );
}

function HandleInPlot() {
  const catalog = useCatalog((s) => s.catalog);
  return (
    <Handle
      id="plot"
      type="target"
      position={Position.Left}
      className="mt-port"
      style={{ background: portColor("plot", catalog) }}
      title="plot: plot"
    />
  );
}

function HandleOutPlot() {
  const catalog = useCatalog((s) => s.catalog);
  return (
    <Handle
      id="plot"
      type="source"
      position={Position.Right}
      className="mt-port"
      style={{ background: portColor("plot", catalog) }}
      title="plot: plot"
    />
  );
}

/** Inline table viewer: fills the node body; fetches the JSON artifact. */
function TableBody({ nodeId }: { nodeId: string }) {
  const artifact = useInputPlotArtifact(nodeId);
  const [rows, setRows] = useState<Record<string, unknown>[] | null>(null);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    let active = true;
    setRows(null);
    setErr(null);
    if (!artifact) return;
    fetch(artifact.url)
      .then((r) => {
        if (!r.ok) throw new Error(`fetch ${artifact.url}: ${r.status}`);
        return r.json();
      })
      .then((data) => {
        if (active) setRows(Array.isArray(data) ? data : [data]);
      })
      .catch((e) => {
        if (active) setErr(String(e));
      });
    return () => {
      active = false;
    };
  }, [artifact]);

  const cols = rows?.length
    ? Array.from(new Set(rows.flatMap((r) => Object.keys(r))))
    : [];
  const ports = (
    <div className="mt-node-ports mt-preview-ports">
      <div className="mt-row">
        <span className="mt-port-label in">
          <HandleInTable />
          <span>table</span>
        </span>
        <span className="mt-port-label out">
          <span>table</span>
          <HandleOutTable />
        </span>
      </div>
    </div>
  );

  return (
    <>
      <div className="mt-table">
        {artifact && (
          <a
            className="mt-download"
            href={api.artifactUrl(artifact.id)}
            download
            aria-label="Download artifact"
            title="Download artifact"
          >
            ⤓
          </a>
        )}
        {err ? (
          <span className="muted-small">{err}</span>
        ) : !artifact ? (
          <span className="muted-small">connect a table · run to render</span>
        ) : !rows ? (
          <span className="muted-small">loading…</span>
        ) : (
          <table className="data-table">
            <thead>
              <tr>
                {cols.map((c) => (
                  <th key={c}>{c}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {rows.map((r, i) => (
                <tr key={i}>
                  {cols.map((c) => (
                    <td key={c}>{String(r[c] ?? "")}</td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
      {ports}
    </>
  );
}

function HandleInTable() {
  const catalog = useCatalog((s) => s.catalog);
  return (
    <Handle
      id="table"
      type="target"
      position={Position.Left}
      className="mt-port"
      style={{ background: portColor("table", catalog) }}
      title="table: table"
    />
  );
}

function HandleOutTable() {
  const catalog = useCatalog((s) => s.catalog);
  return (
    <Handle
      id="table"
      type="source"
      position={Position.Right}
      className="mt-port"
      style={{ background: portColor("table", catalog) }}
      title="table: table"
    />
  );
}

export const nodeTypes = { mt: MTNode };
