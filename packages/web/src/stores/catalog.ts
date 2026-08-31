import { create } from "zustand";
import type { Catalog } from "../api/client";

interface CatalogState {
  catalog: Catalog | null;
  setCatalog: (c: Catalog) => void;
}

export const useCatalog = create<CatalogState>((set) => ({
  catalog: null,
  setCatalog: (catalog) => set({ catalog }),
}));

/** Fallback port-type colors (ComfyUI-style socket coloring). The canonical
 *  colors come from the backend catalog (`GET /api/nodes` → port_types,
 *  docs/DESIGN.md §3 — the backend is the single source of truth); this table
 *  only covers older/offline backends and must mirror the backend's
 *  CORE_PORT_TYPES. */
export const PORT_COLORS: Record<string, string> = {
  any: "#95a5a6",
  table: "#f39c12",
  plot: "#d81b8c",
  mth5: "#9b59b6",
  calibration: "#af7ac5",
  decimation: "#a569bd",
  config: "#6c3483",
  bands: "#d2b4de",
  "ts.run": "#2980b9",
  "ts.channel": "#7fb3d5",
  "ts.runs": "#5dade2",
  fc: "#186a3b",
  "tf.level": "#e74c3c",
  tf: "#c0392b",
};

export function portColor(type: string, catalog?: Catalog | null): string {
  // plugin-declared types win (plugin guide §"Port types"): the catalog
  // carries UI metadata for every port type declared by an installed node
  const declared = catalog?.port_types?.[type]?.color;
  if (declared) return declared;
  return PORT_COLORS[type] ?? "#95a5a6";
}
