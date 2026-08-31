import { Handle, NodeResizer, Position, type NodeProps } from "@xyflow/react";
import { Fragment, memo, useEffect, useMemo, useState } from "react";
import { api, type NodeManifest } from "../api/client";
import { useCatalog, portColor } from "../stores/catalog";
import { useRun } from "../stores/run";
import { useWorkflow, type MTNodeData } from "../stores/workflow";
import { ParamField } from "../panels/ParamField";
import { parseIsoGmt, TimeBar, type TimeExtent } from "../panels/TimeBar";
import { resolveInputArtifact } from "./artifacts";
import type { ParamSpec, PayloadSummary, PortSpec } from "../api/client";
import { inheritableParams, makeInheritContext, resolveInherited } from "./inherit";

/** payload summary → time-bar extent (epoch/relative seconds); null when
 *  absent or degenerate (docs/DESIGN.md §8.3). An `mth5` archive summary
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
      // M-F7 degrade: skip unparsable extents rather than vanishing the
      // bar — the union covers whatever parses; null only when nothing does
      const starts = pick
        .map((r) => parseIsoGmt(String(r.start)))
        .filter((n): n is number => n !== null);
      const ends = pick
        .map((r) => parseIsoGmt(String(r.end)))
        .filter((n): n is number => n !== null);
      if (starts.length === 0 || ends.length === 0) {
        return null;
      }
      const startSec = Math.min(...starts);
      const endSec = Math.max(...ends);
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

/** Resolve the artifact flowing into a viewer node's body: the input edge
 *  (matched on the node's declared input port — M13) names a source node +
 *  handle; only the artifact whose handle names that same output port
 *  feeds it (docs/issues.md L3 — the old "last artifact of the source"
 *  fallback showed the wrong figure for multi-output sources). */
function useInputPlotArtifact(nodeId: string, inputPort?: string) {
  const edges = useWorkflow((s) => s.edges);
  const artifacts = useRun((s) => s.artifacts);
  return useMemo(
    () => resolveInputArtifact(edges, artifacts, nodeId, inputPort),
    [edges, artifacts, nodeId, inputPort],
  );
}

/**
 * Custom node renderer (docs/DESIGN.md §8 + Port rendering conventions):
 * typed ports inside the edges aligned with labels, and — following the
 * ComfyUI paradigm — the node's parameters rendered inline on the node
 * body. The right-hand panel keeps metadata (id, version, ports,
 * description, deps) and a Delete action (see PropertiesPanel).
 */
function MTNodeBase({ id, data, selected }: NodeProps) {
  const catalog = useCatalog((s) => s.catalog);
  const manifest = catalog?.nodes.find((m) => m.type === data.nodeType);
  const d = data as MTNodeData;
  const runState = useRun((s) => s.nodeStates[id]);
  // M18: the `nodes` array mutates on every drag frame (position-only
  // updates), so subscribing to it re-rendered EVERY node per pointermove.
  // The inheritance walk now keys on the structure revision — bumped only
  // when topology or params change — and reads the arrays via getState()
  // inside the memo. `edges` keeps its subscription (it doesn't mutate
  // during node drags — not the hot path).
  const graphRev = useWorkflow((s) => s.graphRev);
  const edges = useWorkflow((s) => s.edges);

  const inputs = manifest?.inputs ?? [];
  const outputs = manifest?.outputs ?? [];
  const rows = Math.max(inputs.length, outputs.length);
  const params = useMemo(() => manifest?.params ?? [], [manifest]);

  // Chain-walking inheritance context (docs/DESIGN.md §4): survey/station/run (and
  // any other declared params) auto-fill from the upstream chain, flowing
  // through intermediate nodes.
  const inheritCtx = useMemo(
    () => {
      void graphRev; // cache key only — topology/params changes bump it (M18)
      return makeInheritContext(
        catalog?.nodes,
        useWorkflow.getState().nodes,
        edges,
      );
    },
    [catalog, graphRev, edges],
  );
  // L-F10: ONE walk per graph change (not per render per param) — the
  // inherited candidates for every declared param, memoized together.
  const inheritable = useMemo(() => inheritableParams(manifest), [manifest]);
  const inheritedByParam = useMemo(() => {
    const m: Record<string, string[]> = {};
    for (const p of params) {
      if (!inheritable.has(p.name)) continue;
      let opts = resolveInherited(inheritCtx, id, p.name);
      if (p.split_inherited) {
        // comma-separated upstream lists (e.g. Compute FC output_channels
        // "e1,e2,bz") become individual combobox choices
        opts = opts.flatMap((o) => o.split(",").map((s) => s.trim()).filter(Boolean));
      }
      m[p.name] = opts;
    }
    return m;
  }, [inheritCtx, inheritable, params, id]);

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
    // M18: graphRev stands in for the nodes array (position drags skip it)
    void graphRev;
  }, [id, catalog, manifest, edges, graphRev, inheritable, params]);

  // Conditional-UX rules are manifest-driven (docs/DESIGN.md §4 `enabled_when`):
  // a param is editable only when its named sibling param holds one of the
  // listed values (e.g. remote_reference only for the robust estimator).
  // NB: both sides are stringified case-insensitively — the manifest
  // serializes rule values via Python str() ("True"/"False" capitalized)
  // while JS String(true/false) is lowercase, so a case-sensitive compare
  // never matches boolean rules (saturated_max stayed disabled forever).
  const isParamEnabled = (p: ParamSpec): boolean => {
    const ew = p.enabled_when;
    if (!ew) return true;
    const sibling = manifest?.params.find((q) => q.name === ew.param);
    const current = d.params[ew.param] ?? sibling?.default ?? "";
    const cur = String(current).toLowerCase();
    return ew.values.map(String).some((v) => v.toLowerCase() === cur);
  };

  // Port-analog of enabled_when (docs/DESIGN.md §4 `populated_when`): the output
  // port is populated only when the named sibling param holds one of the
  // listed values — inactive ports render dimmed.
  const isOutputPopulated = (o: PortSpec): boolean => {
    const pw = o.populated_when;
    if (!pw) return true;
    const sibling = manifest?.params.find((q) => q.name === pw.param);
    const current = d.params[pw.param] ?? sibling?.default ?? "";
    return pw.values.map(String).some((v) => v.toLowerCase() === String(current).toLowerCase());
  };

  // One-port value hint (§4): the port's runtime payload kind (streamed via
  // `payload` run events) colors the handle — a ts.run-tagged source whose
  // summary says "ts.runs" renders in the collection color with a suffix.
  const outputCollection = (o: PortSpec): boolean => {
    const s = payloads[id]?.[o.name];
    return s?.kind === "ts.runs";
  };
  const outputTitle = (o: PortSpec): string => {
    const kind = payloads[id]?.[o.name]?.kind;
    const base = `${o.label ?? o.name}: ${outputCollection(o) ? "ts.runs" : o.type}`;
    if (!isOutputPopulated(o)) {
      return `${base} — not populated in the current ${o.populated_when?.param} mode`;
    }
    return kind === "ts.runs" && o.type !== "ts.runs"
      ? `${base} (runtime: collection)`
      : base;
  };

  // Time-selector bars (history §23.4): params bound via `time_bar` are grouped by
  // their input port (start/end pair per port) and rendered as a draggable
  // acquisition bar once the upstream run streamed the payload extent.
  // (The full map is needed: barExtent/suggest_from read CONNECTED
  // SOURCES' payloads — payload events are per-run, not per-frame.)
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
      {/* §7.2 (M15): the node_progress text is part of the run contract —
          render it, not just the fraction */}
      {runState?.status === "running" && runState.msg && (
        <div className="mt-progress-msg" title={runState.msg}>
          {runState.msg}
        </div>
      )}
      {runState?.cached && (
        <div className="mt-cached" title="Result reused from cache — inputs unchanged">
          cached
        </div>
      )}
      {runState?.warnings && runState.warnings.length > 0 && (
        <div
          className="mt-warning"
          title={runState.warnings.join("\n")}
        >
          ⚠ {runState.warnings.length === 1
            ? runState.warnings[0]
            : `${runState.warnings.length} warnings`}
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
            const opts = inheritedByParam[p.name] ?? [];
            let sel = opts.length > 0 ? opts : undefined;
            // select-option suggestions (history §20.1): the suggest_from port's
            // payload summary EXTENDS the options after a run (e.g. an fc
            // database's sample-rate tiers) — union of static ∪ inherited ∪
            // suggested, "auto" first. The old replacement semantics
            // orphaned an inherited/static choice (e.g. a channel name
            // resolved from the upstream chain vanished once rates arrived).
            if (p.suggest_from) {
              const edge = edges.find(
                (e) => e.target === id && e.targetHandle === p.suggest_from);
              const sum = edge
                ? payloads[edge.source]?.[edge.sourceHandle ?? ""]
                : undefined;
              // kind "fc": the database's rate tiers; kind "ts.runs": the
              // distinct sample rates across the collection's runs
              const rates = sum?.rates ?? (sum?.runs?.length
                ? Array.from(new Set(sum.runs.map((r) => r.sample_rate)))
                : undefined);
              if (rates?.length) {
                // M13: the manifest default leads the suggestions (the old
                // literal hardcoded "auto" even for params whose default
                // is something else)
                sel = Array.from(new Set([
                  ...(p.default != null ? [String(p.default)] : []),
                  ...(p.options ?? []).map(String),
                  ...(opts ?? []),
                  ...rates.map((r) => String(r)),
                ]));
              }
            }
            return (
              <ParamField
                key={p.name}
                param={p}
                // L-F9 (§8.1): a param explicitly set (even cleared to null)
                // renders its actual value; only a never-set param falls
                // back to the manifest default
                value={
                  Object.prototype.hasOwnProperty.call(d.params, p.name)
                    ? (d.params[p.name] ?? "")
                    : (p.default ?? "")
                }
                onChange={(v) => setParam(p.name, v)}
                compact
                instanceId={id}
                disabled={!isParamEnabled(p)}
                selectOptions={sel}
              />
            );
          })}
          {/* time-selector bars (history §23.4): one per bound input port, rendered
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

      {/* body renderer declared by the manifest's viewer flag (L-F12) */}
      {manifest?.viewer === "plot" ? (
        <PreviewBody nodeId={id} manifest={manifest} />
      ) : manifest?.viewer === "table" ? (
        <TableBody nodeId={id} manifest={manifest} />
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
                        title={outputTitle(output)}
                      >
                        {output.label ?? output.name}
                        {outputCollection(output) ? " (collection)" : ""}
                      </span>
                      <Handle
                        id={output.name}
                        type="source"
                        position={Position.Right}
                        className="mt-port"
                        style={{
                          background: outputCollection(output)
                            ? portColor("ts.runs", catalog)
                            : portColor(output.type, catalog),
                          opacity: isOutputPopulated(output) ? undefined : 0.25,
                        }}
                        title={outputTitle(output)}
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
function PreviewBody({ nodeId, manifest }: { nodeId: string; manifest: NodeManifest }) {
  const artifact = useInputPlotArtifact(
    nodeId,
    manifest.inputs[0]?.name,
  );
  const ports = <ViewerPorts manifest={manifest} />;
  return (
    <>
      <div className="mt-preview">
        {artifact ? (
          <>
            {/* M-F5: derive the URL client-side — the server's event url is
                root-absolute and ignores a sub-path mount (root_path),
                404-ing the <img>/table fetch under e.g. /tools/emptyui/ */}
            <img src={api.artifactUrl(artifact.id)} alt={`preview ${artifact.id}`} />
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

/** M13: viewer port rows come from the manifest — a viewer plugin declaring
 *  other port names renders handles matching its own edges (the hardcoded
 *  plot/table literals rendered a card that rejected its own edge). */
function ViewerPorts({ manifest }: { manifest: NodeManifest }) {
  const catalog = useCatalog((s) => s.catalog);
  return (
    <div className="mt-node-ports mt-preview-ports">
      <div className="mt-row">
        <span className="mt-port-label in">
          {manifest.inputs.map((p) => (
            <Fragment key={p.name}>
              <Handle
                id={p.name}
                type="target"
                position={Position.Left}
                className="mt-port"
                style={{ background: portColor(p.type, catalog) }}
                title={`${p.label ?? p.name}: ${p.type}`}
              />
              <span title={`${p.label ?? p.name}: ${p.type}`}>
                {p.label ?? p.name}
              </span>
            </Fragment>
          ))}
        </span>
        <span className="mt-port-label out">
          {manifest.outputs.map((p) => (
            <Fragment key={p.name}>
              <span>{p.label ?? p.name}</span>
              <Handle
                id={p.name}
                type="source"
                position={Position.Right}
                className="mt-port"
                style={{ background: portColor(p.type, catalog) }}
                title={`${p.label ?? p.name}: ${p.type}`}
              />
            </Fragment>
          ))}
        </span>
      </div>
    </div>
  );
}

/** Inline table viewer: fills the node body; fetches the JSON artifact. */
function TableBody({ nodeId, manifest }: { nodeId: string; manifest: NodeManifest }) {
  const artifact = useInputPlotArtifact(
    nodeId,
    manifest.inputs[0]?.name,
  );
  const [rows, setRows] = useState<Record<string, unknown>[] | null>(null);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    let active = true;
    setRows(null);
    setErr(null);
    if (!artifact) return;
    fetch(api.artifactUrl(artifact.id))
      .then((r) => {
        if (!r.ok) throw new Error(`fetch artifact ${artifact.id}: ${r.status}`);
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
  const ports = <ViewerPorts manifest={manifest} />;

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



// M18: React Flow's NodeWrapper re-renders whenever its internal node
// object is replaced — which position-only drag changes do every frame
// ({...n, position}) — and re-invokes a plain-function node component. The
// memo keeps that from reaching us: through a drag every MTNode prop stays
// referentially stable (data is not touched by position updates), so only
// the dragged node (dragging/positionAbsolute prop flips) re-renders.
export const MTNode = memo(MTNodeBase);

export const nodeTypes = { mt: MTNode };
