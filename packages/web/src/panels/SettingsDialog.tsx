import { useEffect } from "react";
import { usePlugins } from "../stores/plugins";

interface Props {
  onClose: () => void;
}

/** Toolbar ⚙ Settings dialog — plugin enable/disable (DESIGN.md §5.1).
 *
 * Bundled plugins (core/mth5/aurora) are always on and shown as read-only
 * info rows; third-party plugins carry an enabled tick box. The palette is
 * catalog-driven, so toggling a plugin hides/shows its nodes immediately.
 * With no third-party plugin installed the dialog shows the read-only
 * bundled rows plus an empty-state note (requirement 4: silent when there
 * is nothing to switch). */
export function SettingsDialog({ onClose }: Props) {
  const { plugins, loading, error, load, toggle } = usePlugins();

  useEffect(() => {
    void load();
  }, [load]);

  const thirdParty = plugins.filter((p) => !p.bundled);

  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div className="modal settings-dialog" onClick={(e) => e.stopPropagation()}>
        <div className="modal-head">
          <strong>Settings — Plugins</strong>
          <button onClick={onClose}>✕</button>
        </div>
        {error && <div className="error-msg modal-error">{error}</div>}
        <div className="modal-list">
          {loading && <div className="modal-empty">Loading plugins…</div>}
          {!loading && thirdParty.length === 0 && (
            <div className="modal-empty">
              No third-party plugins installed. Bundled plugins below are
              always on.
            </div>
          )}
          {!loading && (
            <table className="plugin-table">
              <tbody>
                {plugins.map((p) => (
                  <tr key={p.name} className={p.bundled ? "bundled" : ""}>
                    <td>
                      <label className="plugin-row" title={p.name}>
                        <input
                          type="checkbox"
                          disabled={p.bundled}
                          checked={p.enabled}
                          onChange={(e) =>
                            toggle(p.name, e.target.checked).catch(() => {
                              /* error surfaced via the store; checkbox
                                 keeps the previous (server) state */
                            })
                          }
                        />
                        <span className="plugin-name">{p.name}</span>
                      </label>
                    </td>
                    <td className="plugin-meta">{p.version && `v${p.version}`}</td>
                    <td className="plugin-meta">
                      {p.node_count} node{p.node_count === 1 ? "" : "s"}
                    </td>
                    <td className="plugin-note">
                      {p.bundled
                        ? "bundled"
                        : p.enabled
                          ? ""
                          : "disabled"}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>
        <div className="modal-foot">
          <button className="primary" onClick={onClose}>
            Close
          </button>
        </div>
      </div>
    </div>
  );
}
