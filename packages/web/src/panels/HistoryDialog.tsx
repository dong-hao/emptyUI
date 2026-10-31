import { useCallback, useEffect, useRef, useState } from "react";
import { api, type HistoryDetail, type HistoryEntry } from "../api/client";
import { useDialogFocus } from "./dialogA11y";

interface Props {
  onClose: () => void;
  /** App owns the canvas/run transition (stop polling, load document,
   *  hydrate the historical results, close this dialog). */
  onOpen: (detail: HistoryDetail) => void;
}

function fmtTime(epoch: number | null): string {
  if (epoch === null || !Number.isFinite(epoch)) return "—";
  return new Date(epoch * 1000).toLocaleString();
}

function fmtDuration(seconds: number | null): string {
  if (seconds === null || !Number.isFinite(seconds)) return "—";
  const total = Math.max(0, Math.round(seconds));
  if (total < 60) return `${total} s`;
  return `${Math.floor(total / 60)} min ${total % 60} s`;
}

function safeFilename(label: string): string {
  const base = label.replace(/[\\/:*?"<>|]+/g, "_").trim() || "workflow";
  return `${base}.emptyui.json`;
}

/**
 * Session-only run history (docs/issues.md L32).
 *
 * The list is the server's in-memory run table: finished runs only, newest
 * first. The label is the input-data identity derived server-side (never
 * node titles — the source nodes are the common prefix of nearly every
 * workflow). Open restores the snapshot and its compact results; Export
 * downloads the snapshot; Delete/Clear remove the entry and its workspace.
 * A restart clears the table (ComfyUI parity).
 */
export function HistoryDialog({ onClose, onOpen }: Props) {
  const [entries, setEntries] = useState<HistoryEntry[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [query, setQuery] = useState("");
  const [busy, setBusy] = useState<string | null>(null);
  const dialogRef = useRef<HTMLDivElement>(null);
  useDialogFocus(dialogRef, onClose);

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      setEntries(await api.history());
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void load();
  }, [load]);

  const filtered = (entries ?? []).filter((e) => {
    const q = query.trim().toLowerCase();
    if (!q) return true;
    return (
      e.label.toLowerCase().includes(q) ||
      e.status.toLowerCase().includes(q)
    );
  });

  const withDetail = async (
    entry: HistoryEntry,
  ): Promise<HistoryDetail | null> => {
    setBusy(entry.run_id);
    setError(null);
    try {
      return await api.historyEntry(entry.run_id);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
      return null;
    } finally {
      setBusy(null);
    }
  };

  const handleOpen = async (entry: HistoryEntry) => {
    const detail = await withDetail(entry);
    if (detail) onOpen(detail);
  };

  const handleExport = async (entry: HistoryEntry) => {
    const detail = await withDetail(entry);
    if (!detail) return;
    // mirror Open: API-seeded snapshots may omit the envelope; the export
    // must load back through the editor's strict deserializer
    const raw = (detail.workflow ?? {}) as Record<string, unknown>;
    const doc = { format: "emptyui/workflow", version: 1, ...raw };
    const blob = new Blob([JSON.stringify(doc, null, 2)], {
      type: "application/json",
    });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = safeFilename(entry.label);
    a.click();
    // L12: revoking synchronously can cancel the download mid-flight
    window.setTimeout(() => URL.revokeObjectURL(a.href), 10_000);
  };

  const handleDelete = async (entry: HistoryEntry) => {
    if (!window.confirm(`Delete run "${entry.label}"? This cannot be undone.`)) {
      return;
    }
    setBusy(entry.run_id);
    setError(null);
    try {
      await api.deleteHistory(entry.run_id);
      await load();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(null);
    }
  };

  const handleClear = async () => {
    if (!window.confirm("Clear all run history? This cannot be undone.")) return;
    setBusy("*");
    setError(null);
    try {
      await api.clearHistory();
      await load();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(null);
    }
  };

  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div
        className="modal history-dialog"
        role="dialog"
        aria-modal="true"
        aria-label="Run history"
        ref={dialogRef}
        onClick={(e) => e.stopPropagation()}
      >
        <div className="modal-head">
          <strong>Run history</strong>
          <span className="muted">
            {entries
              ? `${entries.length} run${entries.length === 1 ? "" : "s"}`
              : ""}
          </span>
          <span className="spacer" />
          <button onClick={() => void load()} disabled={loading || busy !== null}>
            Refresh
          </button>
          <button
            className="danger"
            onClick={() => void handleClear()}
            disabled={!entries?.length || busy !== null}
          >
            Clear all
          </button>
          <button onClick={onClose} aria-label="Close">
            ✕
          </button>
        </div>
        {error && (
          <div className="error-msg modal-error" role="alert" title={error}>
            {error}
          </div>
        )}
        <div className="modal-list history-list">
          <input
            className="history-filter"
            placeholder="Filter by data or status…"
            aria-label="Filter runs"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
          />
          {loading && !entries && (
            <div className="modal-empty">Loading history…</div>
          )}
          {!loading && entries && entries.length === 0 && (
            <div className="modal-empty">
              No runs yet — press ▶ Run to create one.
            </div>
          )}
          {entries && entries.length > 0 && filtered.length === 0 && (
            <div className="modal-empty">No runs match “{query}”.</div>
          )}
          {filtered.map((entry) => (
            <div className="history-row" key={entry.run_id}>
              <div className="history-row-main">
                <span className={`history-status ${entry.status}`}>
                  {entry.status}
                </span>
                <span className="history-label" title={entry.label}>
                  {entry.label}
                </span>
              </div>
              <div className="history-row-meta">
                {fmtTime(entry.created_at)} · {fmtDuration(entry.duration_s)} ·{" "}
                {entry.node_count} node{entry.node_count === 1 ? "" : "s"} ·{" "}
                {entry.artifact_count} artifact
                {entry.artifact_count === 1 ? "" : "s"}
              </div>
              <div className="history-row-actions">
                <button
                  className="primary"
                  onClick={() => void handleOpen(entry)}
                  disabled={busy !== null}
                >
                  Open
                </button>
                <button
                  onClick={() => void handleExport(entry)}
                  disabled={busy !== null}
                >
                  Export
                </button>
                <button
                  className="danger"
                  onClick={() => void handleDelete(entry)}
                  disabled={busy !== null}
                >
                  Delete
                </button>
              </div>
            </div>
          ))}
        </div>
        <div className="modal-foot history-foot">
          <span className="muted">
            Session-only — cleared when the server restarts.
          </span>
          <span className="spacer" />
          <button onClick={onClose}>Close</button>
        </div>
      </div>
    </div>
  );
}
