import { QueryClient } from "@tanstack/react-query";

/** Shared QueryClient (docs/DESIGN.md §5.1): stores outside the React tree
 *  (e.g. the plugins store) must reconcile the query cache after a
 *  catalog swap — otherwise a focus-refetch carrying a PRE-toggle /nodes
 *  snapshot can overwrite the fresh catalog and resurrect a just-disabled
 *  plugin's nodes. */
export const queryClient = new QueryClient();
