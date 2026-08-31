"""Time-selector window resolution shared by TS-consuming nodes (history §23.4).

Bounds come from the frontend time-selector bar / the paired edit boxes and
are clamped into the record with a warning — out-of-range selection
degrades, never rejects (old records keep processing). An explicitly
bounded slot must be > :data:`MIN_SLOT_S`; a bare number is relative
seconds, an ISO datetime is UTC and needs an absolute record start.
"""

from __future__ import annotations

import logging
from datetime import datetime

from .types import is_unknown_start

_log = logging.getLogger("emptyui")

#: minimum explicitly bounded slot (docs/DESIGN.md §8.3): a narrower window
#: carries no useful MT signal and usually indicates a slider/box slip
MIN_SLOT_S = 1.0


def parse_iso_utc(text: str) -> datetime:
    """Parse an ISO datetime (space or T separator) to a naive UTC datetime."""
    return datetime.fromisoformat(text.strip().replace(" ", "T")[:19])


def parse_bound(value: str, anchor: datetime | None, rec_len: float,
                what: str, start_offset: float = 0.0) -> float:
    """One bound -> seconds-from-series-start, clamped into [0, rec_len].

    A bare number is relative seconds (from the series start); an ISO
    datetime is measured from the record start (``start_offset`` — the
    series' own offset — is subtracted). Warns on clamping.
    """
    text = str(value).strip()
    try:
        rel = float(text)
    except ValueError:
        pass
    else:
        clamped = min(max(rel, 0.0), rec_len)
        if clamped != rel:
            _log.warning("%s=%.4g s falls outside the record (length %.3g s)"
                         " — clamped", what, rel, rec_len)
        return clamped
    if anchor is None:
        raise ValueError(
            f"{what}='{value}' is not a number and the record has no "
            "absolute start time — use relative seconds")
    try:
        dt = parse_iso_utc(text)
    except ValueError as exc:
        raise ValueError(
            f"{what}='{value}' is not an ISO datetime or seconds number"
        ) from exc
    rel = (dt - anchor).total_seconds() - start_offset
    clamped = min(max(rel, 0.0), rec_len)
    if clamped != rel:
        _log.warning(
            "%s=%s falls outside the record (%s .. +%.0f s) — clamped",
            what, value, anchor.isoformat(sep=" ", timespec="seconds"),
            rec_len)
    return clamped


def resolve_anchor(start_datetime: str, log=_log,
                   allow_placeholder_axis: bool = False) -> datetime | None:
    """The record's absolute-start anchor, or None when unknown.

    Placeholder epochs (1980/1970 — digitized paper records) mean "no
    absolute time": degraded with a warning, ISO bounds become unusable
    (relative seconds still work) — never a rejection. When
    ``allow_placeholder_axis`` (Extract Run: the run's own time_period is
    the selection axis shown in the bar, placeholder included), the epoch
    is treated as a real axis instead.
    """
    if not start_datetime:
        return None
    if not allow_placeholder_axis and is_unknown_start(start_datetime):
        log.warning(
            "record starts at a placeholder epoch — time selector bounds "
            "are relative seconds, ISO values ignored")
        return None
    try:
        return datetime.fromisoformat(start_datetime[:19])
    except ValueError:
        return None


def resolve_window_s(start_param: str, end_param: str, *, rec_len: float,
                     start_datetime: str, start_offset: float = 0.0,
                     enforce_min_slot: bool = True,
                     allow_placeholder_axis: bool = False,
                     what: str = "time slot") -> tuple[float, float]:
    """Resolve the selector params to (s0, s1), seconds-from-series-start.

    Guards: clamped into the record (warn, never reject); an explicitly
    bounded slot must be > :data:`MIN_SLOT_S` (``enforce_min_slot`` — the
    legacy duration-only path opts out: a 1 s waveform view of a 500 Hz run
    is legitimate). Empty bounds mean the record boundary.
    """
    anchor = resolve_anchor(start_datetime,
                            allow_placeholder_axis=allow_placeholder_axis)
    bounded = bool(str(start_param).strip() or str(end_param).strip())
    s0 = (parse_bound(start_param, anchor, rec_len, "start", start_offset)
          if str(start_param).strip() else 0.0)
    s1 = (parse_bound(end_param, anchor, rec_len, "end", start_offset)
          if str(end_param).strip() else rec_len)
    if enforce_min_slot and bounded and s1 - s0 <= MIN_SLOT_S:
        raise ValueError(
            f"selected {what} is {s1 - s0:.3g} s — the minimum slot is "
            f"> {MIN_SLOT_S:g} s (start={start_param or 'record start'}, "
            f"end={end_param or 'record end'}, record length {rec_len:.3g} s)")
    return s0, s1
