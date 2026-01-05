export interface PortSpec {
  name: string;
  type: string;
  required?: boolean;
  accepts?: string[];
  description?: string;
  label?: string;
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
  /** Conditional-UX rule (DESIGN.md §4): editable only when the named
   *  sibling param holds one of the listed values. Rendering-only. */
  enabled_when?: { param: string; values: string[] } | null;
  /** Split inherited comma-separated candidates into individual options. */
  split_inherited?: boolean;
  /** Time-selector bar binding (DESIGN.md §23.4): input port name whose
   *  payload summary (extent, via `payload` run events) drives the bar. */
  time_bar?: string;
}

/** Extent summary of a ts payload, streamed via `payload` run events
 *  (DESIGN.md §23.4): absolute ISO bounds (UTC) or relative seconds. */
export interface PayloadSummary {
  start: string | number;
  end: string | number;
  absolute: boolean;
  sample_rate: number;
  channels: string[];
}

/** inherit: input port -> { targetParam: upstreamParam } */
export type InheritMap = Record<string, Record<string, string>>;

/** One runtime knob in GET /api/settings (DESIGN.md §22). */
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
  /** reserved for the postponed §21 disk cache */
  planned: Record<string, unknown>;
}

export interface NodeManifest {
  type: string;
  version: string;
  title: string;
  /** one-sentence hover tip (hover protocol, DESIGN.md §4) */
  tooltip: string;
  /** full details for the properties panel (shown on click) */
  description: string;
  category: string;
  /** utility/test nodes: excluded from the palette by default */
  hidden?: boolean;
  inputs: PortSpec[];
  outputs: PortSpec[];
  params: ParamSpec[];
  deps: string[];
  /** input port -> {targetParam, upstreamParam} for chain-walking auto-fill */
  inherit?: InheritMap;
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
}

/** API base, derived at runtime (DESIGN.md §17.3 single-origin convention):
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
  // runtime settings (DESIGN.md §22): live knobs editable, deployment rows
  // read-only, planned (§21 cache) knobs reserved
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
  // ("execute to here", DESIGN.md §7); omitted = run everything
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
  // plugin enable/disable (DESIGN.md §5.1): returns the updated plugin list
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
};

export function wsUrl(): string {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  return `${proto}://${location.host}${BASE}/ws`;
}
