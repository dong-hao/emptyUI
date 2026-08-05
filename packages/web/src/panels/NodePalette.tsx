import { useMemo, useState } from "react";
import type { NodeManifest } from "../api/client";

/** Fixed category display order; unknown categories follow, alphabetically. */
const CATEGORY_ORDER = ["Data I/O", "Processing", "Display"];

interface Props {
  manifests: NodeManifest[];
  onAdd: (manifest: NodeManifest) => void;
  /** the catalog request is in flight (or retrying) — say so, instead of
   *  showing an empty list with no clue that anything was expected */
  pending?: boolean;
  /** terminal catalog failure reason, e.g. "catalog: GET /nodes: 500" */
  error?: string | null;
  /** user-triggered refetch of GET /api/nodes */
  onRetry?: () => void;
}

/**
 * Node palette grouped by category, populated from GET /api/nodes.
 *
 * Ordering protocol (DESIGN.md §4): categories in the fixed order above,
 * nodes alphabetically by title within each category. Nodes with
 * `hidden: true` (utility/test nodes, e.g. the echo smoke test) are
 * excluded from the palette — they remain executable when present in a
 * saved workflow.
 */
export function NodePalette({ manifests, onAdd, pending, error, onRetry }: Props) {
  const [query, setQuery] = useState("");

  const grouped = useMemo(() => {
    const q = query.toLowerCase();
    const filtered = manifests.filter(
      (m) =>
        !m.hidden && (m.title + m.type).toLowerCase().includes(q),
    );
    const map = new Map<string, NodeManifest[]>();
    for (const m of filtered) {
      if (!map.has(m.category)) map.set(m.category, []);
      map.get(m.category)!.push(m);
    }
    // fixed category order first, then any custom categories alphabetically
    const categories = [...map.keys()].sort((a, b) => {
      const ia = CATEGORY_ORDER.indexOf(a);
      const ib = CATEGORY_ORDER.indexOf(b);
      if (ia !== -1 && ib !== -1) return ia - ib;
      if (ia !== -1) return -1;
      if (ib !== -1) return 1;
      return a.localeCompare(b);
    });
    return categories.map((category) => ({
      category,
      items: map.get(category)!.sort((a, b) => a.title.localeCompare(b.title)),
    }));
  }, [manifests, query]);

  return (
    <aside className="palette">
      <input
        placeholder="Search nodes…"
        value={query}
        onChange={(e) => setQuery(e.target.value)}
      />
      {/* H6: a catalog that never arrives used to render as a silently empty
          list — the two states below are the difference between "empty" and
          "we could not find out" */}
      {pending && !error && (
        <p className="palette-status muted">Loading node catalog…</p>
      )}
      {error && (
        <div className="palette-error" role="alert">
          <strong>Node catalog unavailable</strong>
          <p className="muted">{error}</p>
          {onRetry && (
            <button className="primary" onClick={onRetry}>
              Retry catalog
            </button>
          )}
        </div>
      )}
      {grouped.map(({ category, items }) => (
        <div key={category} className="palette-category">
          <h3>{category}</h3>
          {items.map((m) => (
            <button
              key={m.type}
              // hover protocol: one sentence + the type tag; the full
              // description is in the properties panel (click the node).
              // L12: no || description fallback here — the server already
              // derives tooltip from the description's first sentence, so
              // a duplicated untruncated fallback just diverges in shape.
              title={`${m.tooltip}\n${m.type}`}
              onClick={() => onAdd(m)}
            >
              {m.title}
            </button>
          ))}
        </div>
      ))}
    </aside>
  );
}
