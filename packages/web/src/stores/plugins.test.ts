import { beforeEach, expect, test, vi } from "vitest";

/** The plugins store is the only place that swaps the catalog after a
 *  runtime toggle (docs/DESIGN.md §5.1). The e2e suite observes the backend
 *  catalog, not the frontend refetch, so pin the refetch here (L30). */
vi.mock("../api/client", () => ({
  api: {
    setPlugin: vi.fn(),
    nodes: vi.fn(),
  },
}));
vi.mock("../api/queryClient", () => ({
  queryClient: { setQueryData: vi.fn() },
}));

import { api } from "../api/client";
import { queryClient } from "../api/queryClient";
import { useCatalog } from "./catalog";
import { usePlugins } from "./plugins";

beforeEach(() => {
  vi.clearAllMocks();
  useCatalog.setState({ catalog: null });
});

test("a plugin toggle refetches and publishes the catalog (L30)", async () => {
  const plugins = [
    {
      name: "example",
      version: "0.1.0",
      node_count: 2,
      bundled: false,
      enabled: false,
    },
  ];
  const catalog = { nodes: [], port_types: {} };
  vi.mocked(api.setPlugin).mockResolvedValue(plugins);
  vi.mocked(api.nodes).mockResolvedValue(catalog);

  await usePlugins.getState().toggle("example", false);

  expect(api.setPlugin).toHaveBeenCalledWith("example", false);
  expect(api.nodes).toHaveBeenCalledOnce();
  expect(useCatalog.getState().catalog).toEqual(catalog);
  // the query cache is reconciled so a stale in-flight refetch cannot
  // resurrect the just-disabled plugin's nodes
  expect(queryClient.setQueryData).toHaveBeenCalledWith(["nodes"], catalog);
});
