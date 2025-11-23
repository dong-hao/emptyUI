export interface PortSpec {
  name: string;
  type: string;
  required?: boolean;
  accepts?: string[];
  description?: string;
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
}

export interface NodeManifest {
  type: string;
  version: string;
  title: string;
  description: string;
  category: string;
  inputs: PortSpec[];
  outputs: PortSpec[];
  params: ParamSpec[];
  deps: string[];
}

export interface Catalog {
  nodes: NodeManifest[];
  types: Record<string, { color?: string; label?: string }>;
}

const BASE = "/api";

async function getJson<T>(path: string): Promise<T> {
  const res = await fetch(`${BASE}${path}`);
  if (!res.ok) throw new Error(`GET ${path}: ${res.status}`);
  return res.json();
}

export const api = {
  health: () => getJson<{ status: string }>("/health"),
  nodes: () => getJson<Catalog>("/nodes"),
};
