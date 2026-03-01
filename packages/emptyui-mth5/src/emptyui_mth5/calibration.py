"""Instrument calibration: manufacturer FAP files -> mt_metadata filters.

DESIGN §16.1: the LEMI logger outputs physical units (nT, mV/km) assuming
unity gain and zero phase; the manufacturer calibration files provide the
frequency-dependent correction. This module parses the manufacturer FAP
files (``.chn`` LEMI-417-era, ``.def`` KMS-200-era wrapper) and the mt_io
JSON calibration format into per-channel ``FrequencyResponseTableFilter``
objects and attaches them to a run the way mt_timeseries expects
(``run.filters`` + per-channel dataset attrs), so ``RunTS.calibrate()``
applies them and the association survives an MTH5 round trip.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
from emptyui import Node, Output, Param

from .nomenclature import is_electric_name, is_magnetic_name

#: FAP channel labels (with/without '#' prefix) -> physical channel names.
FAP_CHANNELS = {
    "#HX": "bx", "#HY": "by", "#HZ": "bz", "#EX": "e1", "#EY": "e2",
    "HX": "bx", "HY": "by", "HZ": "bz", "EX": "e1", "EY": "e2",
    "BX": "bx", "BY": "by", "BZ": "bz", "E1": "e1", "E2": "e2",
}

_HEADER_RE = re.compile(r"^['\"]?#FP['\"]?\s+(\d+)\s+(\d+)")


def _default_unit(channel: str) -> str:
    """Best-effort default physical unit for a channel name (DESIGN §19).

    Magnetic channels (hx/hy/hz, bx/by/bz, h1..h3) default to nanoTesla
    (coil/fluxgate archives in A/m must be stamped from their real unit
    attrs, never guessed); electric channels (ex/ey, e1..) default to
    mV/km (LEMI logger output). Auxiliary/unknown channels have no
    meaningful default — an empty string (mt_metadata normalizes the
    filter units to "unknown") instead of a silently wrong guess.
    """
    if is_magnetic_name(channel):
        return "nanoTesla"
    if is_electric_name(channel):
        return "mV/km"
    return ""


def _validate_calibration_table(channel: str, frequencies, amplitudes,
                                phases_deg) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Validate and normalize one frequency-response table.

    Frequency-response interpolation requires a usable one-dimensional grid:
    frequencies must be finite, positive, and strictly increasing. FAP
    modulus values must be finite and non-negative, and phase values must be
    finite. Validation happens before flat extension so malformed or empty
    tables cannot fail later at ``f[0]`` or be handed to mt_metadata.
    """
    f = np.asarray(frequencies, dtype=float)
    a = np.asarray(amplitudes, dtype=float)
    p = np.asarray(phases_deg, dtype=float)
    if not (f.ndim == a.ndim == p.ndim == 1):
        raise ValueError(
            f"calibration channel '{channel}': frequency-response arrays "
            "must be one-dimensional"
        )
    if not (f.size == a.size == p.size):
        raise ValueError(
            f"calibration channel '{channel}': frequencies/amplitudes/phases "
            f"size mismatch ({f.size}/{a.size}/{p.size})"
        )
    if f.size == 0:
        raise ValueError(
            f"calibration channel '{channel}': frequency-response table is empty"
        )
    if not np.all(np.isfinite(f)) or np.any(f <= 0):
        raise ValueError(
            f"calibration channel '{channel}': frequencies must be finite "
            "and strictly positive"
        )
    if np.any(np.diff(f) <= 0):
        raise ValueError(
            f"calibration channel '{channel}': frequencies must be strictly "
            "increasing with no duplicates"
        )
    if not np.all(np.isfinite(a)) or np.any(a < 0):
        raise ValueError(
            f"calibration channel '{channel}': amplitudes must be finite "
            "and non-negative"
        )
    if not np.all(np.isfinite(p)):
        raise ValueError(
            f"calibration channel '{channel}': phases must be finite"
        )
    return f, a, p


def parse_fap(path) -> dict:
    """Parse a manufacturer FAP calibration file (``.chn`` / ``.def``).

    Returns ``{"channels": {name: {"frequencies": np.ndarray,
    "amplitudes": np.ndarray, "phases_deg": np.ndarray}}}`` — one table per
    channel (they share the frequency grid in the FAP layout, but each channel
    carries its own copy so downstream code never mixes grids). The LEMI-417-era
    layout is a header line ``'#FP' <NTF> <NCH> / NTF,NCH DR-TRANSFORMED
    (MOD,ARG)``, a channel-label line (``#HX #HY #HZ #EX #EY``), then ``NTF``
    rows of ``freq mod arg mod arg ...`` (modulus, argument in degrees, per
    channel; the first column is the frequency in Hz). The KMS-200 ``.def``
    wrapper carries the same FAP block — leading wrapper/comment lines are
    skipped. Raises :class:`ValueError` when the FAP block cannot be located.
    """
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    header = None
    for i, ln in enumerate(lines):
        m = _HEADER_RE.match(ln)
        if m:
            header = (int(m.group(1)), int(m.group(2)), i)
            break
    if header is None:
        raise ValueError(
            f"cannot locate the FAP header ('#FP' <NTF> <NCH>) in {path}"
        )
    ntf, nch, hidx = header

    labels = None
    for ln in lines[hidx + 1:hidx + 6]:
        mapped = [FAP_CHANNELS.get(t.strip("'\"").upper()) for t in ln.split()]
        known = [m for m in mapped if m]
        if len(known) == nch:
            labels = known
            break
    if labels is None:
        raise ValueError(
            f"cannot locate the {nch}-channel label line in {path}"
        )

    freqs: list[float] = []
    rows: list[list[float]] = []
    for ln in lines[hidx + 1:]:
        if ln[0] == "'" or ln.startswith("#"):
            continue
        toks = ln.split()
        if len(toks) < 1 + 2 * nch:
            continue
        try:
            vals = [float(t) for t in toks[:1 + 2 * nch]]
        except ValueError:
            continue
        freqs.append(vals[0])
        rows.append(vals)
    if len(rows) != ntf:
        raise ValueError(
            f"{path}: expected {ntf} FAP rows, found {len(rows)}"
        )

    frequencies = np.asarray(freqs, dtype=float)
    channels: dict[str, dict[str, np.ndarray]] = {}
    for j, name in enumerate(labels):
        mods = np.asarray([r[1 + 2 * j] for r in rows], dtype=float)
        args = np.asarray([r[2 + 2 * j] for r in rows], dtype=float)
        channels[name] = {
            "frequencies": frequencies.copy(),
            "amplitudes": mods,
            "phases_deg": args,
        }
    return {"channels": channels}


def build_fap_filters(rows, prefix: str = "cal", extend_flat: bool = True,
                      units: dict | None = None) -> dict:
    """Build per-channel ``FrequencyResponseTableFilter`` objects.

    ``rows`` is the :func:`parse_fap` result. ``extend_flat`` (default) pads
    the table down to 0.001·f_min with the first row's modulus and zero
    phase — below the measured range the response is treated as constant
    (log-extrapolation was found to be worse, DESIGN §16.1). ``units`` may
    override the per-channel units (default: nanoTesla / mV·km by channel
    prefix).
    """
    from mt_metadata.timeseries.filters import FrequencyResponseTableFilter

    out: dict[str, object] = {}
    for ch, table in rows["channels"].items():
        f, a, p = _validate_calibration_table(
            ch,
            table["frequencies"],
            table["amplitudes"],
            table["phases_deg"],
        )
        if extend_flat:
            f = np.concatenate([[f[0] / 1e3], f])
            a = np.concatenate([[a[0]], a])
            p = np.concatenate([[0.0], p])
        unit = (units or {}).get(ch) or _default_unit(ch)
        # phases are passed in radians: mt_metadata's constructor applies a
        # max(|phase|) > 6*pi degrees->radians heuristic that silently
        # misreads small-phase tables (a -2 deg phase would stay -2 rad);
        # the parsed FAP ARG column is degrees, so convert explicitly
        # (same guard as the Phoenix ASC path)
        fap = FrequencyResponseTableFilter(
            frequencies=f, amplitudes=a, phases=np.deg2rad(p),
            instrument_type="flux gate magnetometer",
            units_in=unit, units_out=unit,
            gain=1.0, sequence_number=1,
        )
        fap.name = f"{prefix}_{ch}"
        out[ch] = fap
    return out


def parse_json_calibration(path) -> dict:
    """Parse an mt_io per-channel JSON calibration into FAP-style tables.

    The mt_io JSON layout is ``{"Calibration": {"gain", "Freq": [...],
    "Re": [...], "Im": [...]}}`` per channel key. Each channel keeps its **own**
    frequency grid (channels in one file may be tabulated on different grids —
    sharing one grid silently corrupts the others). Returns the same structure
    as :func:`parse_fap` (phases in degrees, amplitude = |Re + i·Im|).
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    channels: dict[str, dict[str, np.ndarray]] = {}
    for ch, payload in data.items():
        cal = payload.get("Calibration", payload)
        freq = np.asarray(cal["Freq"], dtype=float)
        re_ = np.asarray(cal["Re"], dtype=float)
        im_ = np.asarray(cal["Im"], dtype=float)
        if not (freq.ndim == re_.ndim == im_.ndim == 1):
            raise ValueError(
                f"calibration channel '{ch}': Freq/Re/Im arrays must be "
                "one-dimensional"
            )
        if not (freq.size == re_.size == im_.size):
            raise ValueError(
                f"calibration channel '{ch}': Freq/Re/Im size mismatch "
                f"({freq.size}/{re_.size}/{im_.size})"
            )
        channels[ch] = {
            "frequencies": freq,
            "amplitudes": np.hypot(re_, im_),
            "phases_deg": np.degrees(np.arctan2(im_, re_)),
        }
    return {"channels": channels}


def load_calibration(path, prefix: str = "cal",
                     extend_flat: bool = True) -> dict:
    """Load a calibration file (``.chn`` / ``.def`` FAP or mt_io JSON) into
    per-channel filters. ``extend_flat`` applies only to FAP files."""
    path = Path(path)
    if path.suffix.lower() in (".chn", ".def"):
        rows = parse_fap(path)
        return build_fap_filters(rows, prefix=prefix, extend_flat=extend_flat)
    if path.suffix.lower() == ".json":
        rows = parse_json_calibration(path)
        return build_fap_filters(rows, prefix=prefix, extend_flat=False)
    raise ValueError(
        f"unsupported calibration file type: {path.name} "
        "(expected .chn, .def or .json)"
    )


def find_calibration_file(directory, station_id: str = "") -> Path | None:
    """Auto-pickup a calibration file in ``directory``.

    Prefers a file whose name contains the station id (per-pair files are
    the accurate ones), otherwise the first ``.chn``/``.def``/``.json``
    found. Returns ``None`` when the directory has none.
    """
    directory = Path(directory)
    if not directory.is_dir():
        return None
    candidates = [
        p for p in sorted(directory.iterdir())
        if p.suffix.lower() in (".chn", ".def", ".json") and p.is_file()
    ]
    if not candidates:
        return None
    if station_id:
        for p in candidates:
            if station_id.lower() in p.name.lower():
                return p
    return candidates[0]


def attach_filters(run, filters, strict: bool = True) -> object:
    """Attach per-channel filters to an mt_timeseries run.

    Registers the filter objects in ``run.filters`` and marks each channel's
    dataset attrs with the modern ``filters`` (AppliedFilter) format, so
    ``RunTS.calibrate()`` applies them and the association survives an MTH5
    round trip (``from_runts`` / ``to_runts``). Returns the run.

    With ``strict=True`` (default, used when archiving a run) a filter whose
    channel is absent from the run is an error; with ``strict=False`` (used by
    the Calibrate node at estimation time) it is skipped with a warning.

    Channels without a calibration filter (e.g. LEMI temperature channels)
    get a unity coefficient filter so ``RunTS.calibrate()`` passes them
    through without the "no filters / corrupt units" warnings.
    """
    import logging

    from mt_metadata.timeseries.filters import CoefficientFilter

    log = logging.getLogger("emptyui")
    run_filters = dict(run.filters)
    for ch, fap in filters.items():
        if ch not in run.dataset.data_vars:
            if strict:
                raise ValueError(
                    f"calibration channel '{ch}' not present in the run "
                    f"(available: {list(run.dataset.data_vars)})"
                )
            log.warning(
                "calibration channel '%s' not present in the run %s — skipped",
                ch, list(run.dataset.data_vars),
            )
            continue
        run_filters[fap.name] = fap
        run.dataset[ch].attrs["filters"] = [
            {"applied_filter": {"name": fap.name, "applied": True}}]
    for ch in run.dataset.data_vars:
        if ch in filters:
            continue
        if run.dataset[ch].attrs.get("filters"):
            # the channel already carries a real archived response chain —
            # never replace it with unity (the response would be silently
            # dropped from all downstream FC/TF math)
            continue
        unit = run.dataset[ch].attrs.get("units") or _default_unit(ch)
        unity = CoefficientFilter(gain=1.0, units_in=unit, units_out=unit)
        unity.name = f"cal_{ch}_unity"
        run_filters[unity.name] = unity
        run.dataset[ch].attrs["filters"] = [
            {"applied_filter": {"name": unity.name, "applied": True}}]
    run.filters = run_filters
    try:
        clear = getattr(run, "clear_channel_cache", None)
    except Exception:  # noqa: BLE001 — mt_timeseries raises NameError
        clear = None
    if callable(clear):
        # invalidate cached ChannelTS objects: their response chains were
        # resolved at first access — the newly attached filters must win
        clear()
    return run


class LoadCalibration(Node):
    type = "mth5.calibration.load"
    title = "Load Calibration"
    tooltip = "Parse a manufacturer calibration file into per-channel instrument-response filters."
    description = (
        "Parse a manufacturer calibration file (LEMI-417 .chn / KMS-200 .def "
        "FAP, or mt_io JSON) into per-channel filters and expose them as a "
        "'calibration' payload for the 'Calibrate' node. The LEMI logger "
        "assumes unity gain / zero phase; the FAP response (modulus, "
        "argument in degrees, first column = frequency in Hz) is the "
        "frequency-dependent correction."
    )
    category = "Data I/O"
    inputs = ()
    outputs = (Output("calibration", "calibration"),)
    params = (
        Param("path", "file", label="Calibration file", path_kind="in"),
        Param("extend_flat", "boolean", label="Flat below table",
              default=True,
              description="Treat the response as constant (first row, zero "
                          "phase) below the measured frequency range; "
                          "log-extrapolation was found to be worse."),
    )

    def execute(self, ctx, path: str,
                extend_flat: bool = True) -> dict[str, Any]:
        filters = load_calibration(path, extend_flat=extend_flat)
        ctx.progress(1.0, f"loaded calibration for {sorted(filters)}")
        return {"calibration": filters}
