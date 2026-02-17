import { Handle, NodeResizer, Position, type NodeProps } from "@xyflow/react";
import { useEffect, useMemo, useState } from "react";
import { api } from "../api/client";
import { useCatalog, portColor } from "../stores/catalog";
import { useRun } from "../stores/run";
import { useWorkflow, type MTNodeData } from "../stores/workflow";
import { ParamField } from "../panels/ParamField";
import { parseIsoGmt, TimeBar, type TimeExtent } from "../panels/TimeBar";
import type { ParamSpec, PayloadSummary, PortSpec } from "../api/client";
import { inheritableParams, makeInheritContext, resolveInherited } from "./inherit";

/** payload summary → time-bar extent (epoch/relative seconds); null when
 *  absent or degenerate (DESIGN.md §23.4). An `mth5` archive summary
 *  resolves the run matching the node's survey/station/run params
 *  (mirroring the server's first-run fallbacks); multiple ports bind as an
 *  intersection (e.g. Align Runs shows the overlap of both inputs). */
function toExtent(
  s: PayloadSummary,
  params: Record<string, unknown>,
): TimeExtent | null {
  if (s.kind === "mth5") {
    const runs = s.runs ?? [];
    if (runs.length === 0) return null;
    const survey = String(params.survey ?? "").trim();
    const station = String(params.station ?? "").trim();
    const run = String(params.run ?? "").trim();
    const pick = runs.filter(
      (r) =>
        (!survey || r.survey === survey) &&
        (!station || r.station === station) &&
        (!run || r.run === run),
    );
    // 'time range' selection: the bar shows the union extent over all
    // matching runs (the whole coverage the range mode extracts), not the
    // first run's slice
    if (String(params.selection ?? "").trim() === "time range") {
      const starts = pick.map((r) => parseIsoGmt(String(r.start)));
      const ends = pick.map((r) => parseIsoGmt(String(r.end)));
      if (starts.some((s) => s === null) || ends.some((s) => s === null)) {
        return null;
      }
      const startSec = Math.min(...(starts as number[]));
      const endSec = Math.max(...(ends as number[]));
      return endSec > startSec
        ? { startSec, endSec, absolute: true }
        : null;
    }
    const picked = pick[0] ?? runs[0];
    if (!picked) return null;
    return tsExtent(picked.start, picked.end, true);
  }
  return tsExtent(s.start, s.end, s.absolute);
}

function intersect(a: TimeExtent, b: TimeExtent): TimeExtent | null {
  if (a.absolute !== b.absolute) return null;
  const startSec = Math.max(a.startSec, b.startSec);
  const endSec = Math.min(a.endSec, b.endSec);
  return endSec > startSec ? { startSec, endSec, absolute: a.absolute } : null;
}

function tsExtent(
  start: string | number,
  end: string | number,
  absolute: boolean,
): TimeExtent | null {
  if (absolute) {
    const a = parseIsoGmt(String(start));
    const b = parseIsoGmt(String(end));
    if (a === null || b === null || b <= a) return null;
    return { startSec: a, endSec: b, absolute: true };
  }
  const a = Number(start);
  const b = Number(end);
  if (!Number.isFinite(a) || !Number.isFinite(b) || b <= a) return null;
  return { startSec: a, endSec: b, absolute: false };
}

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
 * Custom node renderer (DESIGN.md §8 + Port rendering conventions):
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
  const inheritedOptions = (paramName: string): string[] => {
    const spec = params.find((p) => p.name === paramName);
    let opts = resolveInherited(inheritCtx, id, paramName);
    if (spec?.split_inherited) {
      // comma-separated upstream lists (e.g. Compute FC output_channels
      // "e1,e2,bz") become individual combobox choices
      opts = opts.flatMap((o) => o.split(",").map((s) => s.trim()).filter(Boolean));
    }
    return opts;
  };

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

  // Conditional-UX rules are manifest-driven (DESIGN.md §4 `enabled_when`):
  // a param is editable only when its named sibling param holds one of the
  // listed values (e.g. remote_reference only for the robust estimator).
  const isParamEnabled = (p: ParamSpec): boolean => {
    const ew = p.enabled_when;
    if (!ew) return true;
    const sibling = manifest?.params.find((q) => q.name === ew.param);
    const current = d.params[ew.param] ?? sibling?.default ?? "";
    return ew.values.includes(String(current));
  };

  // Port-analog of enabled_when (DESIGN.md §4 `populated_when`): the output
  // port is populated only when the named sibling param holds one of the
  // listed values — inactive ports render dimmed.
  const isOutputPopulated = (o: PortSpec): boolean => {
    const pw = o.populated_when;
    if (!pw) return true;
    const sibling = manifest?.params.find((q) => q.name === pw.param);
    const current = d.params[pw.param] ?? sibling?.default ?? "";
    return pw.values.includes(String(current));
  };

  // Time-selector bars (§23.4): params bound via `time_bar` are grouped by
  // their input port (start/end pair per port) and rendered as a draggable
  // acquisition bar once the upstream run streamed the payload extent.
  const payloads = useRun((s) => s.payloads);
  const barGroups = useMemo(() => {
    const groups: { port: string; startP: ParamSpec; endP?: ParamSpec }[] = [];
    for (const p of params) {
      if (!p.time_bar) continue;
      let g = groups.find((x) => x.port === p.time_bar);
      if (!g) {
        g = { port: p.time_bar, startP: p };
        groups.push(g);
      } else if (!g.endP) {
        g.endP = p;
      }
    }
    return groups;
  }, [params]);
  const barExtent = (spec: string): TimeExtent | null => {
    // `time_bar` may bind several ports (comma-sep): intersect the extents
    // of the connected ones (Align Runs → overlap; identical extents → same)
    let ext: TimeExtent | null = null;
    for (const port of spec.split(",").map((s) => s.trim()).filter(Boolean)) {
      const edge = edges.find((e) => e.target === id && e.targetHandle === port);
      if (!edge) continue;
      const s = payloads[edge.source]?.[edge.sourceHandle ?? ""];
      if (!s) continue;
      const e = toExtent(s, d.params);
      if (!e) continue;
      ext = ext ? intersect(ext, e) : e;
    }
    return ext;
  };

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
                disabled={!isParamEnabled(p)}
                selectOptions={sel}
              />
            );
          })}
          {/* time-selector bars (§23.4): one per bound input port, rendered
              under the paired Start/End boxes once the upstream has run and
              streamed its extent */}
          {barGroups.map((g) => {
            const extent = barExtent(g.port);
            if (!extent || !g.endP) return null;
            return (
              <TimeBar
                key={g.port}
                extent={extent}
                startValue={String(d.params[g.startP.name] ?? "")}
                endValue={String(d.params[g.endP.name] ?? "")}
                onChange={(which, v) =>
                  setParam(which === "start" ? g.startP.name : g.endP!.name, v)
                }
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
                        style={{
                          background: portColor(output.type, catalog),
                          opacity: isOutputPopulated(output) ? undefined : 0.25,
                        }}
                        title={
                          isOutputPopulated(output)
                            ? `${output.label ?? output.name}: ${output.type}`
                            : `${output.label ?? output.name}: ${output.type} — not populated in the current ${output.populated_when?.param} mode`
                        }
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
