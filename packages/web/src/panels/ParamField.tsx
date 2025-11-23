import { useState } from "react";
import type { ParamSpec } from "../api/client";
import { FileBrowser } from "./FileBrowser";

interface Props {
  param: ParamSpec;
  value: unknown;
  onChange: (v: unknown) => void;
  compact?: boolean;
}

/** One schema-driven widget (widget vocabulary from the manifest).
 *  Used both inline on nodes (ComfyUI-style) and in the properties panel. */
export function ParamField({ param, value, onChange, compact }: Props) {
  const [browsing, setBrowsing] = useState(false);
  const set = (v: unknown) => onChange(v);
  const str = value === null || value === undefined ? "" : String(value);

  let field: JSX.Element;
  switch (param.widget) {
    case "boolean":
      field = (
        <input
          type="checkbox"
          checked={Boolean(value)}
          onChange={(e) => set(e.target.checked)}
        />
      );
      break;
    case "select":
      field = (
        <select value={str} onChange={(e) => set(e.target.value)}>
          {(param.options ?? []).map((o) => (
            <option key={String(o)}>{String(o)}</option>
          ))}
        </select>
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
          onChange={(e) => set(e.target.value === "" ? null : Number(e.target.value))}
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
      field = <input value={str} onChange={(e) => set(e.target.value)} />;
  }

  return (
    <div className={`param-field${compact ? " compact" : ""}`} title={param.description || param.label}>
      <span className="param-label">{param.label}</span>
      {field}
    </div>
  );
}
