import { useState } from "react";
import type { ParamSpec } from "../api/client";
import { FileBrowser } from "./FileBrowser";

interface Props {
  param: ParamSpec;
  value: unknown;
  onChange: (v: unknown) => void;
  compact?: boolean;
  disabled?: boolean;
  /** Unique per node-instance id, used to namespace DOM ids (radio group
   *  names, datalist ids) so sibling instances of the same node type don't
   *  share form state. */
  instanceId?: string;
  /** Render as a select with these options (e.g. values inherited from a
   *  connected upstream node). Falls back to the default widget when empty. */
  selectOptions?: unknown[];
}

/** One schema-driven widget (widget vocabulary from the manifest).
 *  Used both inline on nodes (ComfyUI-style) and in the properties panel. */
export function ParamField({ param, value, onChange, compact, disabled, instanceId, selectOptions }: Props) {
  const [browsing, setBrowsing] = useState(false);
  const set = (v: unknown) => onChange(v);
  const str = value === null || value === undefined ? "" : String(value);
  const domScope = instanceId ?? "global";

  let field: JSX.Element;
  if (selectOptions && selectOptions.length > 0) {
    // Editable combobox (datalist): shows the upstream-provided options as
    // suggestions while remaining fully typeable, so an inherited value is a
    // convenience rather than a lock-in. The free-text entry must be coerced
    // back to the param's declared type — a raw string here used to break
    // number params downstream ('10' <= 0 TypeError).
    const listId = `dl-${domScope}-${param.name}`;
    const opts = Array.from(
      new Set([
        ...selectOptions.map(String),
        ...(value === null || value === undefined ? [] : [String(value)]),
      ]),
    );
    const coerce = (raw: string) => {
      if (param.widget === "number" || param.widget === "integer") {
        return raw.trim() === "" ? null : Number(raw);
      }
      return raw;
    };
    field = (
      <>
        <input
          list={listId}
          value={str}
          disabled={disabled}
          placeholder={param.placeholder}
          onChange={(e) => set(coerce(e.target.value))}
        />
        <datalist id={listId}>
          {opts.map((o) => (
            <option key={o} value={o} />
          ))}
        </datalist>
      </>
    );
  } else {
    switch (param.widget) {
      case "boolean":
        field = (
          <input
            type="checkbox"
            checked={Boolean(value)}
            disabled={disabled}
            onChange={(e) => set(e.target.checked)}
          />
        );
        break;
      case "select":
        field = (
          <select value={str} disabled={disabled} onChange={(e) => set(e.target.value)}>
            {(param.options ?? []).map((o) => (
              <option key={String(o)}>{String(o)}</option>
            ))}
          </select>
        );
        break;
      case "radio":
        field = (
          <span className="radio-group">
            {(param.options ?? []).map((o) => (
              <label key={String(o)} className="radio-option">
                <input
                  type="radio"
                  name={`radio-${domScope}-${param.name}`}
                  value={String(o)}
                  checked={String(o) === str}
                  onChange={(e) => set(e.target.value)}
                />
                {String(o)}
              </label>
            ))}
          </span>
        );
        break;
      case "integer":
      case "number":
        field = (
          <input
            type="number"
            value={str}
            min={param.min}
            max={param.max}
            step={param.step}
            onChange={(e) => {
              const raw = e.target.value;
              if (raw === "") {
                set(null);
                return;
              }
              // intermediate typing states ("e", "-", "1e") parse to NaN —
              // keep the raw text in the box instead of storing NaN (which
              // JSON-serializes to null and defeats the inherit unset check)
              const n = Number(raw);
              set(Number.isFinite(n) ? n : raw);
            }}
          />
        );
        break;
      case "file":
      case "directory":
        field = (
          <span className="path-field">
            <input
              value={str}
              placeholder={`path to ${param.widget}…`}
              onChange={(e) => set(e.target.value)}
            />
            <button
              className="browse-btn"
              title={`Browse server filesystem for a ${param.widget}`}
              onClick={() => setBrowsing(true)}
            >
              …
            </button>
            {browsing && (
              <FileBrowser
                kind={param.widget as "file" | "directory"}
                initialPath={str}
                onSelect={(p) => {
                  set(p);
                  setBrowsing(false);
                }}
                onClose={() => setBrowsing(false)}
              />
            )}
          </span>
        );
        break;
      default:
        field = (
          <input
            value={str}
            placeholder={param.placeholder}
            onChange={(e) => set(e.target.value)}
          />
        );
    }
  }

  return (
    <div
      className={`param-field${compact ? " compact" : ""}${disabled ? " disabled" : ""} nodrag nopan`}
      title={param.description || param.label}
    >
      <span className="param-label">{param.label}</span>
      {field}
    </div>
  );
}
