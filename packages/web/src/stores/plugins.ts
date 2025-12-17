import { create } from "zustand";
import { api, type PluginInfo } from "../api/client";
import { useCatalog } from "./catalog";

interface PluginsState {
  plugins: PluginInfo[];
  loading: boolean;
  error: string | null;
  load: () => Promise<void>;
  toggle: (name: string, enabled: boolean) => Promise<void>;
}

/** Plugin enable/disable state (DESIGN.md §5.1).
 *
 * Toggling goes through the backend (in-place registry hot reload) and then
 * refetches the node catalog — the palette is catalog-driven, so hiding/
 * showing a plugin's nodes on the left panel is automatic. Runtime toggles
 * are in-memory only: no settings file is created by default. */
export const usePlugins = create<PluginsState>((set) => ({
  plugins: [],
  loading: false,
  error: null,
  load: async () => {
    set({ loading: true, error: null });
    try {
      set({ plugins: await api.plugins(), loading: false });
    } catch (e) {
      set({ error: e instanceof Error ? e.message : String(e), loading: false });
    }
  },
  toggle: async (name, enabled) => {
    set({ error: null });
    try {
      const plugins = await api.setPlugin(name, enabled);
      set({ plugins });
      // catalog-driven palette: refetch so the toggle takes effect live
      useCatalog.getState().setCatalog(await api.nodes());
    } catch (e) {
      set({ error: e instanceof Error ? e.message : String(e) });
      throw e;
    }
  },
}));
