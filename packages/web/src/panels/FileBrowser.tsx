import { useEffect, useRef, useState } from "react";
import { api, type FileEntry } from "../api/client";
import { useDialogFocus } from "./dialogA11y";

interface Props {
  kind: "file" | "directory";
  initialPath?: string;
  onSelect: (path: string) => void;
  onClose: () => void;
}

/** Server-side file browser (backend GET /api/files) for path widgets. */
export function FileBrowser({ kind, initialPath, onSelect, onClose }: Props) {
  const dialogRef = useRef<HTMLDivElement>(null);
  useDialogFocus(dialogRef, onClose);
  const [path, setPath] = useState(initialPath ?? "");
  const [parent, setParent] = useState<string | null>(null);
  const [entries, setEntries] = useState<FileEntry[]>([]);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const body = await api.files(path);
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

  // L-F11 (§8.1): Esc closes the dialog
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div
        className="modal"
        role="dialog"
        aria-modal="true"
        aria-label={`Select ${kind}`}
        ref={dialogRef}
        onClick={(e) => e.stopPropagation()}
      >
        <div className="modal-head">
          <strong>Select {kind}</strong>
          <span className="muted modal-path">{path || "…"}</span>
          <button onClick={onClose} aria-label="Close">
            ✕
          </button>
        </div>
        {error && <div className="error-msg modal-error" title={error}>{error}</div>}
        <div className="modal-list">
          {parent && path && (
            <button className="modal-row" onClick={() => setPath(parent)}>
              ⬆ ..
            </button>
          )}
          {entries.map((e: FileEntry) => (
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
