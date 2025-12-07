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
}

/** inherit: input port -> { targetParam: upstreamParam } */
export type InheritMap = Record<string, Record<string, string>>;

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

const BASE = "/api";

async function getJson<T>(path: string): Promise<T> {
  const res = await fetch(`${BASE}${path}`);
  if (!res.ok) throw new Error(`GET ${path}: ${res.status}`);
  return res.json();
}

export const api = {
  nodes: () => getJson<Catalog>("/nodes"),
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
};

export function wsUrl(): string {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  return `${proto}://${location.host}/api/ws`;
}
