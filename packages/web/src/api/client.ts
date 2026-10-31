export interface PortSpec {
  name: string;
  type: string;
  required?: boolean;
  accepts?: string[];
  description?: string;
  label?: string;
  /** rendering-only: the port is populated only when the named sibling
   *  param holds one of the listed values (Extract Run's dual outputs,
   *  docs/DESIGN.md §4) */
  populated_when?: { param: string; values: string[] } | null;
  /** one-port convention (§4): this INPUT accepts a run collection
   *  (ts.runs) arriving on the static port — the wiring check consults
   *  the source port's runtime payload summary */
  collection_ok?: boolean;
}

export interface ParamSpec {
  name: string;
  widget: string;
  label?: string;
  default?: unknown;
  optional?: boolean;
  options?: unknown[];
  min?: number;
  max?: number;
  step?: number;
  description?: string;
  /** Placeholder text for text inputs (e.g. "auto (detect from data)"). */
  placeholder?: string;
  /** Auto-change rules: value key -> [targetParam, targetValue][] */
  cascade?: Record<string, [string, unknown][]>;
  /** Conditional-UX rule (docs/DESIGN.md §4): editable only when the named
   *  sibling param holds one of the listed values. Rendering-only. */
  enabled_when?: { param: string; values: string[] } | null;
  /** Split inherited comma-separated candidates into individual options. */
  split_inherited?: boolean;
  /** Time-selector bar binding (docs/DESIGN.md §8.3): input port name whose
   *  payload summary (extent, via `payload` run events) drives the bar. */
  time_bar?: string;
  /** M19/M23: "in" reads data (relativized on save against the project
   *  dir / data root); "out" writes a product (never relativized). */
  path_kind?: string;
  /** Select-option suggestions (docs/DESIGN.md §8.3): input port name whose
   *  payload summary offers the options (e.g. an fc database's sample-rate
   *  tiers) — the static options are extended after a run. */
  suggest_from?: string | null;
}

/** Extent summary of a ts payload, streamed via `payload` run events
 *  (docs/DESIGN.md §8.3): absolute ISO bounds (UTC) or relative seconds. */
export interface RunExtent {
  survey: string;
  station: string;
  run: string;
  start: string;
  end: string;
  sample_rate: number;
}

export interface PayloadSummary {
  kind?: "ts" | "mth5" | "fc" | "ts.runs";
  start: string | number;
  end: string | number;
  absolute: boolean;
  sample_rate: number;
  channels: string[];
  /** kind "mth5": per-run extents within the archive */
  runs?: RunExtent[];
  /** kind "fc": the merged database's rate tiers (descending, Hz) —
   *  offered as select suggestions by the plot nodes */
  rates?: number[];
}

/** inherit: input port -> { targetParam: upstreamParam } */
export type InheritMap = Record<string, Record<string, string>>;

/** One runtime knob in GET /api/settings (docs/DESIGN.md §7.4). */
export interface LiveSetting {
  value: number | null;
  type: string;
  default: number | null;
  description: string;
  editable: boolean;
  source: "platform" | "ui" | "default";
}

export interface SettingsView {
  /** live-editable knobs (whitelist, validated server-side) */
  live: Record<string, LiveSetting>;
  /** deployment-level rows — display only (env/restart) */
  deployment: Record<string, unknown>;
  /** reserved for the postponed §12.3 disk cache */
  planned: Record<string, unknown>;
}

export interface NodeManifest {
  type: string;
  version: string;
  title: string;
  /** one-sentence hover tip (hover protocol, docs/DESIGN.md §4) */
  tooltip: string;
  /** full details for the properties panel (shown on click) */
  description: string;
  category: string;
  /** utility/test nodes: excluded from the palette by default */
  hidden?: boolean;
  /** inline canvas-body renderer the node card mounts (manifest-driven —
   *  the frontend never switches on node types, L-F12): "plot" | "table",
   *  absent = the standard port rows */
  viewer?: "plot" | "table" | null;
  /** M13: "this run produces something the user can see" — declared on the
   *  node class, never inferred from the category string */
  terminal?: boolean;
  /** M23: the plugin distribution version — recorded per node when the
   *  workflow is saved; the executor compares it at run time and warns
   *  on the node about version drift. */
  plugin_version?: string;
  inputs: PortSpec[];
  outputs: PortSpec[];
  params: ParamSpec[];
  deps: string[];
  /** input port -> {targetParam, upstreamParam} for chain-walking auto-fill */
  inherit?: InheritMap;
  /** declared model weights (§8 plugin guide: declare, don't bundle) */
  models?: {
    id: string;
    filename: string;
    sha256: string;
    url?: string;
    license?: string;
  }[];
}

export interface Catalog {
  nodes: NodeManifest[];
  /** UI metadata for port types declared by installed nodes (color/label). */
  port_types?: Record<string, { color?: string; label?: string }>;
}

export interface FileEntry {
  name: string;
  path: string;
  is_dir: boolean;
}

export interface FileListing {
  path: string;
  parent: string | null;
  entries: FileEntry[];
}

export interface PluginInfo {
  name: string;
  version: string;
  node_count: number;
  bundled: boolean;
  enabled: boolean;
  /** present when discovery/registration failed — the plugin is skipped
   *  and the row shows why (docs/issues.md M20) */
  error?: string;
}

/** One session-history row (docs/issues.md L32): the server's in-memory run
 *  table, cleared on restart. The label is the input-data identity derived
 *  server-side (never node titles). */
export interface HistoryEntry {
  run_id: string;
  label: string;
  created_at: number | null;
  finished_at: number | null;
  status: string;
  duration_s: number | null;
  node_count: number;
  artifact_count: number;
}

export interface HistoryArtifact {
  id: string;
  node: string;
  kind: string;
  handle?: string | null;
  url: string;
}

export interface HistoryNodeResult {
  status?: string;
  error?: string;
  warnings?: string[];
}

/** Compact result snapshot stored with a history entry (not the progress
 *  event stream): final node states, artifacts and payload summaries. */
export interface HistoryResults {
  nodes: Record<string, HistoryNodeResult>;
  artifacts: HistoryArtifact[];
  payloads: Record<string, Record<string, PayloadSummary>>;
  status?: string;
  error?: string | null;
}

export interface HistoryDetail extends HistoryEntry {
  /** the submitted workflow document; the editor fills a missing envelope
   *  on Open/Export (API-seeded runs may post a bare workflow) */
  workflow: unknown;
  targets: string[];
  summary: Record<string, unknown>;
  results: HistoryResults;
}

/** API base, derived at runtime (docs/DESIGN.md §11.3 single-origin convention):
 *  the API always lives one segment below the document root — so this works
 *  at "/", under a sub-path mount (e.g. /tools/emptyui/), and in dev (Vite
 *  proxies /api). No build-time env or hardcoded mount is needed. */
function apiBase(): string {
  const base = new URL("api", document.baseURI);
  return base.pathname.replace(/\/$/, "");
}

const BASE = apiBase();

async function getJson<T>(path: string): Promise<T> {
  const res = await fetch(`${BASE}${path}`);
  if (!res.ok) throw new Error(`GET ${path}: ${res.status}`);
  return res.json();
}

export const api = {
  nodes: () => getJson<Catalog>("/nodes"),
  files: (path = "") =>
    getJson<FileListing>(`/files?path=${encodeURIComponent(path)}`),
  // runtime settings (docs/DESIGN.md §7.4): live knobs editable, deployment rows
  // read-only, planned (§12.3 cache) knobs reserved
  settings: () => getJson<SettingsView>("/settings"),
  putSettings: async (values: Record<string, number | null>): Promise<SettingsView> => {
    const res = await fetch(`${BASE}/settings`, {
      method: "PUT",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ values }),
    });
    if (!res.ok) {
      const body = await res.json().catch(() => null);
      throw new Error(body?.detail ?? `PUT /settings: ${res.status}`);
    }
    return res.json();
  },
  // targets: partial execution — run only these nodes plus their ancestors
  // ("execute to here", docs/DESIGN.md §7); omitted = run everything
  runWorkflow: async (workflow: unknown, targets?: string[]): Promise<{ run_id: string }> => {
    const res = await fetch(`${BASE}/runs`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(targets?.length ? { workflow, targets } : { workflow }),
    });
    if (!res.ok) {
      const body = await res.json().catch(() => null);
      throw new Error(
        Array.isArray(body?.detail)
          ? body.detail.map((e: { msg?: string }) => e.msg ?? JSON.stringify(e)).join("; ")
          : (body?.detail ?? `POST /runs: ${res.status}`),
      );
    }
    return res.json();
  },
  getRun: (runId: string) => getJson<Record<string, unknown>>(`/runs/${runId}`),
  cancelRun: async (runId: string): Promise<{ run_id: string; status: string }> => {
    const res = await fetch(`${BASE}/runs/${runId}/cancel`, { method: "POST" });
    if (!res.ok) throw new Error(`POST /runs/${runId}/cancel: ${res.status}`);
    return res.json();
  },
  artifactUrl: (artifactId: string) => `${BASE}/artifacts/${artifactId}`,
  // plugin enable/disable (docs/DESIGN.md §5.1): returns the updated plugin list
  plugins: () => getJson<PluginInfo[]>("/plugins"),
  setPlugin: async (name: string, enabled: boolean): Promise<PluginInfo[]> => {
    const action = enabled ? "enable" : "disable";
    const res = await fetch(`${BASE}/plugins/${encodeURIComponent(name)}/${action}`, {
      method: "POST",
    });
    if (!res.ok) {
      const body = await res.json().catch(() => null);
      throw new Error(
        body?.detail ?? `POST /plugins/${name}/${action}: ${res.status}`,
      );
    }
    return (await res.json()).plugins as PluginInfo[];
  },
  // session-only run history (docs/issues.md L32): finished runs, newest
  // first; the server clears the table on restart
  history: () => getJson<HistoryEntry[]>("/history"),
  historyEntry: (runId: string) =>
    getJson<HistoryDetail>(`/history/${encodeURIComponent(runId)}`),
  deleteHistory: async (runId: string): Promise<{ deleted: boolean }> => {
    const res = await fetch(`${BASE}/history/${encodeURIComponent(runId)}`, {
      method: "DELETE",
    });
    if (!res.ok) {
      const body = await res.json().catch(() => null);
      throw new Error(body?.detail ?? `DELETE /history/${runId}: ${res.status}`);
    }
    return res.json();
  },
  clearHistory: async (): Promise<{ deleted: number }> => {
    const res = await fetch(`${BASE}/history`, { method: "DELETE" });
    if (!res.ok) {
      const body = await res.json().catch(() => null);
      throw new Error(body?.detail ?? `DELETE /history: ${res.status}`);
    }
    return res.json();
  },
};

export function wsUrl(): string {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  return `${proto}://${location.host}${BASE}/ws`;
}
