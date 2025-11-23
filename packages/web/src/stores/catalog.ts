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
 *  DESIGN.md §3 guide: blue=timeseries, dark green=FC/spectra,
 *  red=transfer functions, purple=metadata/config, gray/yellow=generic.
 *  Server-provided metadata overrides these via catalog.types. */
export const PORT_COLORS: Record<string, string> = {
  any: "#95a5a6",
  file: "#7f8c8d",
  table: "#f39c12",
  image: "#f1c40f",
  "mt.archive": "#9b59b6",
  "mt.response": "#af7ac5",
  "mt.selection": "#8e44ad",
  "mt.decimation": "#a569bd",
  "mt.config": "#6c3483",
  "mt.bands": "#d2b4de",
  "ts.run": "#2980b9",
  "ts.channel": "#7fb3d5",
  spectra: "#186a3b",
  "tf.level": "#e74c3c",
  "mt.tf": "#c0392b",
  plot: "#d81b8c",
};

export function portColor(type: string, catalog: Catalog | null): string {
  return catalog?.types?.[type]?.color ?? PORT_COLORS[type] ?? "#95a5a6";
}
