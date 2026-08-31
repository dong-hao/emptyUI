/**
 * Time-selector bar (docs/DESIGN.md §8.3): a horizontal acquisition bar whose
 * length spans the upstream ts payload's extent, with two draggable handles
 * sandwiching the selected time slot. Synced bidirectionally with the
 * paired Start/End edit boxes — which are the node's own `start`/`end`
 * string params (the bar only writes through their onChange).
 *
 * Guards (mirroring the server-side ones): handles stay within the record
 * extent, cannot cross, and the slot is kept > 1 s (the server rejects
 * slots ≤ 1 s; boxes have second resolution, so the clamp keeps ≥ 2 s).
 *
 * Extent may be absolute (epoch seconds — ISO GMT display) or relative
 * (bare seconds display), mirroring the `payload` run-event summary.
 */

import { useEffect, useRef } from "react";

const MIN_GAP_S = 2.0; // > 1 s at second resolution

export interface TimeExtent {
  /** seconds — epoch seconds when absolute, else relative to series start */
  startSec: number;
  endSec: number;
  absolute: boolean;
}

/** "2023-06-01T12:00:00" (param) → epoch seconds (UTC). null = unparsable.
 *  Offset-aware (M-F7): a trailing Z or an explicit ±HH:MM[:SS] offset is
 *  kept verbatim — appending Z unconditionally turned "…+00:00" into
 *  "…+00:00Z" → NaN → the bar vanished for offset-aware extents. */
export function parseIsoGmt(value: string): number | null {
  const t = value.trim().replace(" ", "T");
  const withZone = /(?:Z|z|[+-]\d{2}:?\d{2})$/.test(t)
    ? t
    : /T/.test(t)
      ? `${t}Z`
      : `${t}T00:00:00Z`;
  const ms = Date.parse(withZone);
  return Number.isNaN(ms) ? null : ms / 1000;
}

/** epoch seconds → "YYYY-MM-DD HH:MM:SS" (UTC) */
export function fmtIsoGmt(epochSec: number): string {
  const d = new Date(epochSec * 1000);
  const p = (n: number) => String(n).padStart(2, "0");
  return (
    `${d.getUTCFullYear()}-${p(d.getUTCMonth() + 1)}-${p(d.getUTCDate())} ` +
    `${p(d.getUTCHours())}:${p(d.getUTCMinutes())}:${p(d.getUTCSeconds())}`
  );
}

/** param value → seconds within the extent; null = empty/unparsable */
function parseBound(
  value: string,
  extent: TimeExtent,
  which: "start" | "end",
): number | null {
  const text = (value ?? "").trim();
  if (!text) return which === "start" ? extent.startSec : extent.endSec;
  if (extent.absolute) {
    const t = parseIsoGmt(text);
    if (t !== null) return t;
    // M-F7 degrade: unparsable text renders the handle at the track edge
    // (clamped by frac()) instead of vanishing the whole bar — the box
    // keeps its draft so the user can fix the entry
    return which === "start" ? extent.startSec : extent.endSec;
  }
  const rel = Number(text);
  // bare numbers are RELATIVE SECONDS on both sides (the server's
  // parse_bound semantics) — in an absolute extent they measure from the
  // record start; the old fallback treated them as epoch seconds, drawing
  // the handle at the track edge and disagreeing with the server clamp
  return Number.isFinite(rel) ? extent.startSec + rel : null;
}

function fmtBound(sec: number, extent: TimeExtent): string {
  return extent.absolute ? fmtIsoGmt(sec) : String(Math.round(sec * 1000) / 1000);
}

interface TimeBarProps {
  extent: TimeExtent;
  startValue: string;
  endValue: string;
  onChange: (which: "start" | "end", value: string) => void;
}

export function TimeBar({ extent, startValue, endValue, onChange }: TimeBarProps) {
  const start = parseBound(startValue, extent, "start");
  const end = parseBound(endValue, extent, "end");
  // L-F5 (§8.1): the window-level drag listeners survive an unmount that
  // happens mid-drag (node removed / extent swapped) — track the active
  // cleanup and run it when the component goes away
  const cleanupRef = useRef<(() => void) | null>(null);
  useEffect(() => () => { cleanupRef.current?.(); }, []);
  if (start === null || end === null) return null;

  const span = extent.endSec - extent.startSec;
  // shared clamp: keep the bound inside the extent and on its own side of
  // the other handle (mirrors the server-side selection clamps)
  const clampTarget = (which: "start" | "end", t: number) => {
    const other = which === "start" ? end : start;
    if (span > MIN_GAP_S) {
      if (which === "start") t = Math.min(t, other - MIN_GAP_S);
      else t = Math.max(t, other + MIN_GAP_S);
    }
    return Math.min(extent.endSec, Math.max(extent.startSec, t));
  };
  // display clamp: a box value outside the extent draws at the track edge
  // (the server clamps the actual selection with a warning)
  const frac = (t: number) =>
    Math.min(100, Math.max(0, ((t - extent.startSec) / span) * 100));

  const drag = (which: "start" | "end") => (e: React.PointerEvent<HTMLDivElement>) => {
    e.preventDefault();
    const track = e.currentTarget.parentElement;
    if (!track) return;
    e.currentTarget.setPointerCapture(e.pointerId);
    const move = (ev: PointerEvent) => {
      const rect = track.getBoundingClientRect();
      const f = Math.min(1, Math.max(0, (ev.clientX - rect.left) / rect.width));
      const t = clampTarget(which, extent.startSec + f * span);
      onChange(which, fmtBound(t, extent));
    };
    const up = () => {
      window.removeEventListener("pointermove", move);
      window.removeEventListener("pointerup", up);
    };
    window.addEventListener("pointermove", move);
    window.addEventListener("pointerup", up);
    cleanupRef.current = up;
  };

  // L-F11 (§8.1): keyboard-operable handles (arrows nudge the bound by
  // 60 s, Shift+arrows by 1 h) + slider semantics for AT users
  const keyStep = (which: "start" | "end") =>
    (e: React.KeyboardEvent<HTMLDivElement>) => {
      const dir =
        e.key === "ArrowLeft" ? -1 : e.key === "ArrowRight" ? 1 : 0;
      if (!dir) return;
      e.preventDefault();
      const cur = which === "start" ? start : end;
      const t = clampTarget(which, cur + dir * (e.shiftKey ? 3600 : 60));
      if (t !== cur) onChange(which, fmtBound(t, extent));
    };

  const handle = (which: "start" | "end") => {
    const cur = which === "start" ? start : end;
    return {
      className: "time-bar-handle",
      style: { left: `calc(${frac(cur)}% - 3px)` },
      title: `${which} ${fmtBound(cur, extent)}`,
      onPointerDown: drag(which),
      role: "slider" as const,
      tabIndex: 0,
      "aria-label": `selection ${which}`,
      "aria-valuemin": extent.startSec,
      "aria-valuemax": extent.endSec,
      "aria-valuenow": cur,
      onKeyDown: keyStep(which),
    };
  };

  return (
    <div className="time-bar nodrag nopan">
      <div className="time-bar-track">
        {/* L-F5 (§8.1): crossing box values degrade (rect spans the sorted
            pair, each handle keeps its own position) instead of vanishing */}
        <div
          className="time-bar-sel"
          style={{
            left: `${frac(Math.min(start, end))}%`,
            width: `${frac(Math.max(start, end)) - frac(Math.min(start, end))}%`,
          }}
        />
        <div {...handle("start")} />
        <div {...handle("end")} />
      </div>
      <div className="time-bar-ticks">
        <span>{fmtBound(extent.startSec, extent)}</span>
        <span>{fmtBound(extent.endSec, extent)}</span>
      </div>
    </div>
  );
}
