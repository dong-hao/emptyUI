import { useMemo, useState } from "react";
import type { NodeManifest } from "../api/client";

interface Props {
  manifests: NodeManifest[];
  onAdd: (manifest: NodeManifest) => void;
}

/** Node palette grouped by category, populated from GET /api/nodes. */
export function NodePalette({ manifests, onAdd }: Props) {
  const [query, setQuery] = useState("");

  const grouped = useMemo(() => {
    const filtered = manifests.filter((m) =>
      (m.title + m.type).toLowerCase().includes(query.toLowerCase()),
    );
    const map = new Map<string, NodeManifest[]>();
    for (const m of filtered) {
      if (!map.has(m.category)) map.set(m.category, []);
      map.get(m.category)!.push(m);
    }
    return map;
  }, [manifests, query]);

  return (
    <aside className="palette">
      <input
        placeholder="Search nodes…"
        value={query}
        onChange={(e) => setQuery(e.target.value)}
      />
      {[...grouped.entries()].map(([category, items]) => (
        <div key={category} className="palette-category">
          <h3>{category}</h3>
          {items.map((m) => (
            <button
              key={m.type}
              title={`${m.type}\n${m.description}`}
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
