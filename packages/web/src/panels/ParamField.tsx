import { useRef, useState } from "react";
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

/** ComfyUI-parity text editing (DESIGN.md §8.1): the input holds a local
 *  draft while focused — typing never touches the workflow store, so the
 *  caret stays wherever the user put it for as long as they like (writing
 *  per keystroke raced the store round-trip: React re-rendered with the
 *  stale snapshot, re-asserted `.value`, and the caret jumped to the end).
 *  The draft commits on Enter or blur (coerced as before); Escape reverts
 *  to the store value. While not editing, the box displays the store
 *  value, so external updates (inherit auto-fill, cascade, workflow load,
 *  TimeBar writes) appear immediately. */
function CommitInput({
  value,
  onCommit,
  widget,
  disabled,
  placeholder,
  list,
  type,
  min,
  max,
  step,
  ariaLabel,
  id,
}: {
  value: unknown;
  onCommit: (v: unknown) => void;
  widget: string;
  disabled?: boolean;
  placeholder?: string;
  list?: string;
  type?: string;
  min?: number | string;
  max?: number | string;
  step?: number | string;
  /** L-F11 (§8.1): compact-node inputs have no <label> element */
  ariaLabel?: string;
  /** L1: the visible param label points here via htmlFor */
  id?: string;
}) {
  const [draft, setDraftState] = useState<string | null>(null);
  // mirror ref: the blur/keydown handlers must see the CURRENT draft even
  // when cancel() flips the state and blur() fires in the same tick
  // (otherwise Escape's cancel is re-committed by the stale onBlur closure)
  const draftRef = useRef<string | null>(null);
  const setDraft = (v: string | null) => {
    draftRef.current = v;
    setDraftState(v);
  };
  const current = value === null || value === undefined ? "" : String(value);
  const editing = draft !== null;

  const coerce = (raw: string): unknown => {
    if (widget === "number" || widget === "integer") {
      const t = raw.trim();
      if (t === "") return null;
      const n = Number(t);
      if (!Number.isFinite(n)) return NaN; // invalid intermediate → revert
      return widget === "integer" ? Math.round(n) : n;
    }
    if (widget === "list") {
      // M-F8: a list param's value is an ARRAY — the old string fallback
      // committed a raw string into the workflow (type corruption).
      // JSON-aware with comma fallback: valid JSON arrays round-trip with
      // typing (numbers stay numbers); hand-typed "a, b" still works.
      const t = raw.trim();
      if (t === "") return [];
      try {
        const v: unknown = JSON.parse(t);
        if (Array.isArray(v)) return v;
      } catch {
        // fall through to the comma heuristic
      }
      return t.split(",").map((s) => s.trim()).filter(Boolean);
    }
    return raw;
  };

  const commit = () => {
    const d = draftRef.current;
    if (d === null) return;
    const v = coerce(d);
    setDraft(null);
    if (typeof v === "number" && Number.isNaN(v)) return; // revert, keep store
    if (String(v) !== current) onCommit(v);
  };
  const cancel = () => setDraft(null);

  return (
    <input
      id={id}
      list={list}
      type={type}
      value={editing ? draft : current}
      disabled={disabled}
      placeholder={placeholder}
      aria-label={ariaLabel}
      min={min}
      max={max}
      step={step}
      onChange={(e) => setDraft(e.target.value)}
      onKeyDown={(e) => {
        // L-F11 (§8.1): don't let commit/revert keys bubble into dialog
        // Esc-to-close handlers (FileBrowser/Settings)
        if (e.key === "Enter") {
          e.stopPropagation();
          commit();
          (e.target as HTMLInputElement).blur();
        } else if (e.key === "Escape") {
          e.stopPropagation();
          cancel();
          (e.target as HTMLInputElement).blur();
        }
      }}
      onBlur={commit}
    />
  );
}

/** Multi-line code editor (M-F8): same draft/commit semantics as
 *  CommitInput, but Enter inserts a newline (commit on blur or Ctrl/Cmd+
 *  Enter; Escape reverts). The old fallback rendered code as a bare
 *  single-line input — newlines in a snippet were destroyed. */
function CodeInput({
  value,
  onCommit,
  disabled,
  placeholder,
  id,
}: {
  value: unknown;
  onCommit: (v: unknown) => void;
  disabled?: boolean;
  placeholder?: string;
  /** L1: the visible param label points here via htmlFor */
  id?: string;
}) {
  const [draft, setDraftState] = useState<string | null>(null);
  const draftRef = useRef<string | null>(null);
  const setDraft = (v: string | null) => {
    draftRef.current = v;
    setDraftState(v);
  };
  const current = value === null || value === undefined ? "" : String(value);
  const commit = () => {
    const d = draftRef.current;
    if (d === null) return;
    setDraft(null);
    if (d !== current) onCommit(d);
  };
  return (
    <textarea
      id={id}
      className="code-input"
      value={draft !== null ? draft : current}
      disabled={disabled}
      placeholder={placeholder}
      rows={Math.min(12, Math.max(2, current.split("\n").length))}
      onChange={(e) => setDraft(e.target.value)}
      onKeyDown={(e) => {
        if (e.key === "Escape") {
          setDraft(null);
          e.currentTarget.blur();
        } else if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) {
          commit();
          e.currentTarget.blur();
        }
      }}
      onBlur={commit}
    />
  );
}

/** One schema-driven widget (widget vocabulary from the manifest).
 *  Used both inline on nodes (ComfyUI-style) and in the properties panel. */
export function ParamField({ param, value, onChange, compact, disabled, instanceId, selectOptions }: Props) {
  const [browsing, setBrowsing] = useState(false);
  const set = (v: unknown) => onChange(v);
  const str = value === null || value === undefined ? "" : String(value);
  const domScope = instanceId ?? "global";
  // L1: every param control is labeled — the visible label is a real
  // <label htmlFor> (the radio group carries the name via aria-label)
  const fieldId = `pf-${domScope}-${param.name}`;

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
    field = (
      <>
        <CommitInput
          value={value}
          onCommit={set}
          widget={param.widget}
          disabled={disabled}
          placeholder={param.placeholder}
          list={listId}
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
            id={fieldId}
            type="checkbox"
            // L-F4 (§8.1): string/number values round-trip through JSON and
            // text inputs — "false"/0 must not render as checked
            checked={
              value === true || value === "true" || value === 1 || value === "1"
            }
            disabled={disabled}
            onChange={(e) => set(e.target.checked)}
          />
        );
        break;
      case "select": {
        // L-F9: an explicitly-set value renders even when it is not among
        // the declared options (documents saved before a relabeling — the
        // pre-sentinel "" options). A blank value is NOT injected: the
        // browser then displays the first option, which post-aliasing is
        // exactly what the stored "" means.
        const opts = Array.from(
          new Set([
            ...(param.options ?? []).map(String),
            ...(str ? [str] : []),
          ]),
        );
        field = (
          <select id={fieldId} value={str} disabled={disabled} onChange={(e) => set(e.target.value)}>
            {opts.map((o) => (
              <option key={o}>{o}</option>
            ))}
          </select>
        );
        break;
      }
      case "radio":
        field = (
          <span className="radio-group" role="radiogroup" aria-label={param.label ?? param.name}>
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
          <CommitInput
            id={fieldId}
            type="number"
            value={value}
            onCommit={set}
            widget={param.widget}
            disabled={disabled}
            min={param.min ?? undefined}
            max={param.max ?? undefined}
            step={param.step ?? undefined}
          />
        );
        break;
      case "file":
      case "directory":
        field = (
          <span className="path-field">
            <CommitInput
              id={fieldId}
              value={value}
              onCommit={set}
              widget={param.widget}
              disabled={disabled}
              placeholder={`path to ${param.widget}…`}
            />
            <button
              className="browse-btn"
              aria-label={`Browse server filesystem for a ${param.widget}`}
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
      case "code":
        field = (
          <CodeInput
            id={fieldId}
            value={value}
            onCommit={set}
            disabled={disabled}
            placeholder={param.placeholder}
          />
        );
        break;
      case "list":
        // M-F8: single-line editor with JSON-aware/array coercion (the
        // generic string fallback committed a raw string into an array param)
        field = (
          <CommitInput
            id={fieldId}
            value={value}
            onCommit={set}
            widget="list"
            disabled={disabled}
            placeholder={param.placeholder ?? "item, item, …"}
          />
        );
        break;
      default:
        field = (
          <CommitInput
            id={fieldId}
            value={value}
            onCommit={set}
            widget={param.widget}
            disabled={disabled}
            placeholder={param.placeholder}
            ariaLabel={param.label ?? param.name}
          />
        );
    }
  }

  return (
    <div
      className={`param-field${compact ? " compact" : ""}${disabled ? " disabled" : ""} nodrag nopan`}
      title={param.description || param.label}
    >
      {param.widget === "radio" ? (
        <span className="param-label">{param.label}</span>
      ) : (
        <label htmlFor={fieldId} className="param-label">
          {param.label}
        </label>
      )}
      {field}
    </div>
  );
}
