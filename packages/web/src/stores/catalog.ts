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

/** Port-type colors (ComfyUI-style socket coloring), following the
 *  DESIGN.md §3 guide: blue=timeseries, dark green=FC, red=transfer
 *  functions, purple=metadata/config, gray/yellow=generic. */
export const PORT_COLORS: Record<string, string> = {
  any: "#95a5a6",
  table: "#f39c12",
  plot: "#d81b8c",
  mth5: "#9b59b6",
  calibration: "#af7ac5",
  selection: "#8e44ad",
  decimation: "#a569bd",
  config: "#6c3483",
  bands: "#d2b4de",
  "ts.run": "#2980b9",
  "ts.channel": "#7fb3d5",
  fc: "#186a3b",
  "tf.level": "#e74c3c",
  tf: "#c0392b",
};

export function portColor(type: string): string {
  return PORT_COLORS[type] ?? "#95a5a6";
}
