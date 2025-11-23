import { useEffect, useState } from "react";

interface Entry {
  name: string;
  path: string;
  is_dir: boolean;
}

interface Props {
  kind: "file" | "directory";
  initialPath?: string;
  onSelect: (path: string) => void;
  onClose: () => void;
}

/** Server-side file browser (backend GET /api/files) for path widgets. */
export function FileBrowser({ kind, initialPath, onSelect, onClose }: Props) {
  const [path, setPath] = useState(initialPath ?? "");
  const [parent, setParent] = useState<string | null>(null);
  const [entries, setEntries] = useState<Entry[]>([]);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const res = await fetch(`/api/files?path=${encodeURIComponent(path)}`);
        if (!res.ok) throw new Error((await res.json()).detail ?? res.status);
        const body = await res.json();
        if (!cancelled) {
          setPath(body.path);
          setParent(body.parent);
          setEntries(body.entries);
          setError(null);
        }
      } catch (e) {
        if (!cancelled) setError(e instanceof Error ? e.message : String(e));
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [path]);

  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div className="modal" onClick={(e) => e.stopPropagation()}>
        <div className="modal-head">
          <strong>Select {kind}</strong>
          <span className="muted modal-path">{path || "…"}</span>
          <button onClick={onClose}>✕</button>
        </div>
        {error && <div className="error-msg modal-error">{error}</div>}
        <div className="modal-list">
          {parent && path && (
            <button className="modal-row" onClick={() => setPath(parent)}>
              ⬆ ..
            </button>
          )}
          {entries.map((e) => (
            <button
              key={e.path}
              className="modal-row"
              onClick={() => (e.is_dir ? setPath(e.path) : onSelect(e.path))}
              disabled={kind === "file" ? false : !e.is_dir && kind === "directory"}
              title={e.path}
            >
              {e.is_dir ? "📁" : "📄"} {e.name}
            </button>
          ))}
          {entries.length === 0 && !error && (
            <div className="muted modal-empty">empty directory</div>
          )}
        </div>
        {kind === "directory" && (
          <div className="modal-foot">
            <button className="primary" onClick={() => onSelect(path)}>
              Use this folder
            </button>
          </div>
        )}
      </div>
    </div>
  );
}
