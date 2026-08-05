import { useEffect, useRef, useState } from "react";
import { api, type SettingsView } from "../api/client";
import { useDialogFocus } from "./dialogA11y";
import { usePlugins } from "../stores/plugins";

interface Props {
  onClose: () => void;
}

/** One live-editable number row: empty input = null ("off"/unlimited). */
function LiveKnobRow({
  name,
  setting,
  draft,
  onChange,
}: {
  name: string;
  setting: SettingsView["live"][string];
  draft: string;
  onChange: (v: string) => void;
}) {
  const label = name.replaceAll("_", " ");
  return (
    <tr>
      <td>
        <label className="plugin-row" title={setting.description}>
          <span className="plugin-name">{label}</span>
        </label>
      </td>
      <td className="plugin-meta">
        <input
          type="number"
          min={0}
          step="any"
          disabled={!setting.editable}
          value={draft}
          placeholder={setting.default === null ? "off" : String(setting.default)}
          onChange={(e) => onChange(e.target.value)}
          style={{ width: 90 }}
        />
      </td>
      <td className="plugin-meta">{setting.source}</td>
      <td className="plugin-note">
        {!setting.editable && "set via env/platform"}
      </td>
    </tr>
  );
}

/** Toolbar ⚙ Settings dialog (DESIGN.md §7.4).
 *
 * Sections: Plugins (§5.1 enable/disable, bundled read-only), Server &
 * retention (live-editable knobs, explicit Save), Deployment (read-only
 * env/restart-level rows), Planned (reserved §12.3 cache knobs, inert).
 */
export function SettingsDialog({ onClose }: Props) {
  const { plugins, loading, error, load, toggle } = usePlugins();
  const [view, setView] = useState<SettingsView | null>(null);
  const [loadErr, setLoadErr] = useState<string | null>(null);
  // L-F11 (§8.1): Esc closes the dialog
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);
  const [saveErr, setSaveErr] = useState<string | null>(null);
  const [drafts, setDrafts] = useState<Record<string, string>>({});
  const [saving, setSaving] = useState(false);

  useEffect(() => {
    void load();
    api.settings()
      .then((v) => {
        setView(v);
        const d: Record<string, string> = {};
        for (const [k, s] of Object.entries(v.live)) d[k] = s.value === null ? "" : String(s.value);
        setDrafts(d);
      })
      .catch((e) => setLoadErr(String(e)));
  }, [load]);

  const apply = async () => {
    setSaving(true);
    setSaveErr(null);
    try {
      const values: Record<string, number | null> = {};
      for (const [k, s] of Object.entries(view?.live ?? {})) {
        if (!s.editable) continue;
        const raw = (drafts[k] ?? "").trim();
        // L12: Number("") is NaN and JSON.stringify(NaN) is null — a typo
        // silently cleared the knob instead of being named
        const n = raw === "" ? null : Number(raw);
        if (n !== null && Number.isNaN(n)) {
          throw new Error(
            `'${s.description || k}' is not a number — fix it before saving`,
          );
        }
        values[k] = n;
      }
      const updated = await api.putSettings(values);
      setView(updated);
      const d: Record<string, string> = {};
      for (const [k, s] of Object.entries(updated.live)) d[k] = s.value === null ? "" : String(s.value);
      setDrafts(d);
    } catch (e) {
      setSaveErr(e instanceof Error ? e.message : String(e));
    } finally {
      setSaving(false);
    }
  };

  const thirdParty = plugins.filter((p) => !p.bundled);
  const dirty = view
    ? Object.entries(view.live).some(([k, s]) => {
        if (!s.editable) return false;
        const raw = (drafts[k] ?? "").trim();
        const num = raw === "" ? null : Number(raw);
        return num !== s.value && !(raw === "" && s.value === null);
      })
    : false;

  const dialogRef = useRef<HTMLDivElement>(null);
  useDialogFocus(dialogRef, onClose);

  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div
        className="modal settings-dialog"
        role="dialog"
        aria-modal="true"
        aria-label="Settings"
        ref={dialogRef}
        onClick={(e) => e.stopPropagation()}
      >
        <div className="modal-head">
          <strong>Settings</strong>
          <button onClick={onClose} aria-label="Close">
            ✕
          </button>
        </div>
        {(error || loadErr) && (
          <div className="error-msg modal-error" title={error ?? loadErr ?? undefined}>
            {error ?? loadErr}
          </div>
        )}
        <div className="modal-list">
          <h3 className="settings-section">Server &amp; retention</h3>
          {view && (
            <table className="plugin-table">
              <tbody>
                {Object.entries(view.live).map(([name, s]) => (
                  <LiveKnobRow
                    key={name}
                    name={name}
                    setting={s}
                    draft={drafts[name] ?? ""}
                    onChange={(v) => setDrafts((d) => ({ ...d, [name]: v }))}
                  />
                ))}
              </tbody>
            </table>
          )}
          {!view && !loadErr && <div className="modal-empty">Loading settings…</div>}
          {view && (
            <p className="modal-empty">
              Empty = off/unlimited. Saved settings persist only when
              EMPTYUI_SETTINGS_PATH is configured.
            </p>
          )}

          <h3 className="settings-section">Deployment (read-only)</h3>
          {view && (
            <table className="plugin-table">
              <tbody>
                {Object.entries(view.deployment).map(([k, v]) => (
                  <tr key={k}>
                    <td><span className="plugin-name">{k.replaceAll("_", " ")}</span></td>
                    <td className="plugin-meta">
                      <code>{v === null || v === undefined ? "unset" : String(v)}</code>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}

          <h3 className="settings-section">Planned — disk cache (§12.3)</h3>
          {view && (
            <table className="plugin-table">
              <tbody>
                {Object.entries(view.planned).map(([k, v]) => {
                  const s = v as { default?: unknown; description?: string };
                  return (
                    <tr key={k}>
                      <td>
                        <span className="plugin-name" title={s.description}>
                          {k.replaceAll("_", " ")}
                        </span>
                      </td>
                      <td className="plugin-meta">
                        <input
                          type="number"
                          step="any"
                          disabled
                          value={s.default === null ? "" : String(s.default)}
                          style={{ width: 90 }}
                        />
                      </td>
                      <td className="plugin-note">planned</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          )}

          <h3 className="settings-section">Plugins</h3>
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
                    <td
                      className="plugin-note"
                      title={p.error || undefined}
                    >
                      {p.bundled
                        ? "bundled"
                        : p.error
                          ? p.error
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
          {saveErr && <span className="error-msg" title={saveErr}>{saveErr}</span>}
          <button
            className="primary"
            disabled={!dirty || saving || !!view && Object.values(view.live).every((s) => !s.editable)}
            onClick={() => void apply()}
          >
            {saving ? "Saving…" : "Save"}
          </button>
          <button onClick={onClose}>Close</button>
        </div>
      </div>
    </div>
  );
}
