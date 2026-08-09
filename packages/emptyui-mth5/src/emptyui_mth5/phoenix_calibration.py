"""Phoenix MTU-5A ASC calibration tables (DESIGN.md §9.2).

The manufacturer exports two ASC table kinds: ``*_CLB.asc`` (the box /
instrument response, one table per channel) and ``*_CLC.asc`` (one file per
induction coil). Layout (3 header lines + rows):

    C:\\test\\new\\MTU-1690_CLB : Instrument Calibrations
    MTU-1690.CLB    Normalizing  E-gain: 10  H-gain: 3
            Channel      Freq (Hz)        mV / mV        Degrees
                  1       317.6471       1.033872             -2
                  ...

The tables are normalized SHAPE corrections: the CLB amplitudes are relative
to the normalizing E/H gains (which match the recording TBL's ``EGN``/``HGN``
when the box was calibrated at the deployed gain setting) and the CLC
amplitudes are relative to the normalizing coil factor (1000 mV/nT = the TBL
``HNOM`` feedback constant expressed in mV). The absolute scale is carried
entirely by the TBL scalar chain (applied at ingest, units mV/km / nT), so
the ASC filters operate on the already-scalar-calibrated channels with
``units_in == units_out`` — removing them is a plain per-frequency complex
division (the same divide-by semantics aurora applies between STFT and
estimation).

Observed layouts (parsed 2026-09-16): CLB = 5 channels x 99 frequencies
ASCENDING 1.3113e-05 -> 317.647 Hz (channel numbering = the TBL
``channel_keys``: ex:1, ey:2, hx:3, hy:4, hz:5); CLC = a single channel per
file with frequencies DESCENDING (the parser sorts), range down to 4.196e-04
Hz (or 1.3113e-05 Hz, depending on the coil's calibration source). Below the
lowest measured frequency the coil response is dB/dt physics (amplitude
proportional to f, +90 deg phase) and the box response is its AC coupling —
so the default low-frequency extension is linear in f (see
:func:`build_phoenix_filters`).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

#: TBL channel numbers (``channel_keys``) -> standard component names.
PHOENIX_CHANNEL_NUMBERS = {1: "ex", 2: "ey", 3: "hx", 4: "hy", 5: "hz"}

#: CLC files label every row channel 1 — the standard component the coil
#: serves is decided by the loader (TBL HXSN/HYSN/HZSN serial match).
CLC_PLACEHOLDER_CHANNEL = 1

_KINDD_RE = re.compile(r"\.(CLB|CLC)\b", re.IGNORECASE)
_SERIAL_RE = re.compile(r"^([A-Za-z0-9_-]+)\.(CLB|CLC)\b", re.IGNORECASE)
_EH_GAINS_RE = re.compile(
    r"Normalizing\s+(?:gain[s]?\s+)?E-gain\s*:\s*([\d.eE+-]+)"
    r"(?:\s+(?:gain[s]?\s+)?H-gain\s*:\s*([\d.eE+-]+))?", re.IGNORECASE)
_FACTOR_RE = re.compile(
    r"Normalizing\s+(?:Factor\s*:\s*)?([\d.eE+-]+)\s*mV/nT", re.IGNORECASE)
_KIND_TITLE_RE = re.compile(r":\s*(Coil|Instrument)\s+Calibrations", re.IGNORECASE)


@dataclass
class PhoenixChannelTable:
    """One channel's normalized response table (ascending frequencies)."""

    frequencies: np.ndarray  # Hz, strictly increasing
    amplitudes: np.ndarray   # normalized (mV/mV)
    phases_deg: np.ndarray


@dataclass
class PhoenixCalibration:
    """A parsed ASC calibration file.

    ``channels`` maps TBL channel numbers to tables for a CLB (1..5) and
    holds the single table under channel 1 for a CLC (the component it
    serves is resolved at attach time from the TBL coil serials).
    """

    kind: str  # "CLB" (box) | "CLC" (coil)
    serial: str
    normalizing: dict[str, float]
    channels: dict[int, PhoenixChannelTable]
    source: str = ""
    extra: dict = field(default_factory=dict)


def parse_asc_calibration(path: str | Path) -> PhoenixCalibration:
    """Parse a Phoenix MTU-5A ``*_CLB.asc`` / ``*_CLC.asc`` table.

    Row layout: ``<channel> <freq Hz> <mV/mV> <degrees>``. Frequencies are
    returned strictly ascending regardless of the file's ordering (CLB files
    ascend, CLC files descend). Raises :class:`ValueError` on malformed
    tables (missing header, no rows, duplicate/non-positive frequencies).
    """
    path = Path(path)
    lines = [ln.strip() for ln in
             path.read_text(encoding="utf-8", errors="replace").splitlines()
             if ln.strip()]
    if len(lines) < 4:
        raise ValueError(
            f"{path.name}: too few lines for a Phoenix ASC calibration "
            "table (need title, normalizing, header and at least one row)")

    kind_m = _KINDD_RE.search(lines[1]) or _KINDD_RE.search(lines[0])
    if kind_m is None:
        raise ValueError(
            f"{path.name}: cannot tell CLB (box) from CLC (coil) — neither "
            f"header carries the .CLB/.CLC marker: {lines[1]!r}")
    kind = kind_m.group(1).upper()

    serial = ""
    serial_m = _SERIAL_RE.match(lines[1])
    if serial_m:
        serial = serial_m.group(1)
    if not serial:
        raise ValueError(
            f"{path.name}: cannot read the instrument serial from the "
            f"normalizing line {lines[1]!r}")

    normalizing: dict[str, float] = {}
    gain_m = _EH_GAINS_RE.search(lines[1])
    if gain_m:
        normalizing["e_gain"] = float(gain_m.group(1))
        if gain_m.group(2) is not None:
            normalizing["h_gain"] = float(gain_m.group(2))
    factor_m = _FACTOR_RE.search(lines[1])
    if factor_m:
        normalizing["factor_mv_per_nt"] = float(factor_m.group(1))
    if not normalizing:
        raise ValueError(
            f"{path.name}: cannot read the normalizing gains/factor from "
            f"{lines[1]!r} (expected 'E-gain: N H-gain: N' or "
            "'Normalizing Factor: N mV/nT')")

    rows: list[tuple[int, float, float, float]] = []
    for ln in lines[3:]:
        toks = ln.split()
        if len(toks) != 4:
            continue  # stray/footer lines are skipped, not fatal
        try:
            ch = int(toks[0])
            f, a, p = float(toks[1]), float(toks[2]), float(toks[3])
        except ValueError:
            continue
        rows.append((ch, f, a, p))
    if not rows:
        raise ValueError(f"{path.name}: no calibration rows found")

    by_ch: dict[int, list[tuple[float, float, float]]] = {}
    for ch, f, a, p in rows:
        by_ch.setdefault(ch, []).append((f, a, p))

    channels: dict[int, PhoenixChannelTable] = {}
    for ch, vals in by_ch.items():
        vals.sort(key=lambda r: r[0])
        f = np.array([v[0] for v in vals], dtype=float)
        a = np.array([v[1] for v in vals], dtype=float)
        p = np.array([v[2] for v in vals], dtype=float)
        if np.any(f <= 0) or not np.all(np.isfinite(f)):
            raise ValueError(
                f"{path.name} channel {ch}: frequencies must be finite and "
                "strictly positive")
        if np.any(np.diff(f) <= 0):
            raise ValueError(
                f"{path.name} channel {ch}: duplicate or non-increasing "
                "frequencies after sorting")
        if not (np.all(np.isfinite(a)) and np.all(np.isfinite(p))):
            raise ValueError(
                f"{path.name} channel {ch}: non-finite amplitudes/phases")
        channels[ch] = PhoenixChannelTable(f, a, p)

    return PhoenixCalibration(kind=kind, serial=serial, normalizing=normalizing,
                              channels=channels, source=path.name)


def find_asc_calibrations(directory: str | Path,
                          box_serial: str = "",
                          coil_serials: dict[str, str] | None = None,
                          ) -> dict[str, Path]:
    """Auto-pick Phoenix ASC calibration files by TBL serial (history §24.12).

    ``coil_serials`` maps standard magnetic components (hx/hy/hz) to the
    TBL coil serial (``HXSN``/``HYSN``/``HZSN``). A CLB file matches when
    its name contains the box serial, a CLC file when its name contains the
    coil serial (case-insensitive; the TBL stores plain numbers like
    ``1690`` while the file is ``MTU-1690_CLB.asc`` — substring match).

    Returns ``{"box": Path|None, "hx": Path|None, ...}``; unmatched kinds
    map to ``None`` (the loader warns, it never fails the ingest).
    """
    directory = Path(directory)
    out: dict[str, Path | None] = {"box": None}
    if not directory.is_dir():
        return out
    files = sorted(p for p in directory.iterdir()
                   if p.is_file() and p.suffix.lower() == ".asc")
    if box_serial:
        for p in files:
            if "clb" in p.stem.lower() and str(box_serial).lower() in p.stem.lower():
                out["box"] = p
                break
    for comp, serial in (coil_serials or {}).items():
        out[comp] = None
        if not serial:
            continue
        for p in files:
            if ("clc" in p.stem.lower()
                    and str(serial).lower() in p.stem.lower()):
                out[comp] = p
                break
    return out


def _extend_below(table: PhoenixChannelTable, mode: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Prepend a synthetic row below the measured range so interpolation
    behaves at deep decimation levels.

    ``"linear"`` (default, history §24.12): amplitude scaled proportional to
    ``f / f_min`` at ``f_min / 1000`` with the lowest phase held — the exact
    shape of both a dB/dt induction coil and an AC-coupled box below their
    measured range, and it makes the subsequent linear extrapolation of the
    interp1d grid follow the same proportional law arbitrarily deep.
    ``"flat"``: the repo-wide FAP convention (first amplitude, zero phase;
    log-extrapolation was found to be worse, DESIGN.md §9.1).
    """
    f, a, p = table.frequencies, table.amplitudes, table.phases_deg
    if mode == "flat":
        return (np.concatenate([[f[0] / 1e3], f]),
                np.concatenate([[a[0]], a]),
                np.concatenate([[0.0], p]))
    if mode == "linear":
        return (np.concatenate([[f[0] / 1e3], f]),
                np.concatenate([[a[0] / 1e3], a]),
                np.concatenate([[p[0]], p]))
    raise ValueError(f"unknown extend_below mode {mode!r} (flat|linear)")


def build_phoenix_filters(cal: PhoenixCalibration,
                          comp_map: dict[int, str] | None = None,
                          units: dict[str, str] | None = None,
                          extend_below: str = "linear",
                          ) -> dict[str, object]:
    """Build per-channel mt_metadata FAP filters from a parsed ASC table.

    ``comp_map`` maps channel numbers to standard components (default: the
    TBL numbering {1: ex, ..., 5: hz}); for a CLC pass ``{1: "hx"}`` (the
    component the coil serves). ``units`` overrides the per-component
    units — default milliVolt per kilometer (electric) / nanoTesla
    (magnetic). The filters are SHAPE corrections on the scalar-calibrated
    channels: ``units_in == units_out``, gain 1, and removing them is the
    per-frequency complex division aurora applies (divide by
    amplitude * exp(i*phase)).
    """
    from mt_metadata.timeseries.filters import FrequencyResponseTableFilter

    cmap = comp_map or PHOENIX_CHANNEL_NUMBERS
    out: dict[str, object] = {}
    for num, table in cal.channels.items():
        comp = cmap.get(num)
        if comp is None:
            continue
        f, a, p = _extend_below(table, extend_below)
        unit = (units or {}).get(comp) or (
            "milliVolt per kilometer" if comp.startswith("e") else "nanoTesla")
        # phases are passed in radians: mt_metadata's constructor applies a
        # max(|phase|) > 6*pi degrees->radians heuristic that silently
        # misreads small-phase tables (a -2 deg phase would stay -2 rad)
        fap = FrequencyResponseTableFilter(
            frequencies=f, amplitudes=a, phases=np.deg2rad(p),
            instrument_type=("induction coil" if cal.kind == "CLC"
                             else "MTU-5A box"),
            units_in=unit, units_out=unit, gain=1.0, sequence_number=1,
        )
        fap.name = f"phoenix_{cal.kind.lower()}_{cal.serial.lower()}_{comp}"
        out[comp] = fap
    return out
