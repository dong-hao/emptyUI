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

#: FAP channel labels (with/without '#' prefix) -> physical channel names.
FAP_CHANNELS = {
    "#HX": "bx", "#HY": "by", "#HZ": "bz", "#EX": "e1", "#EY": "e2",
    "HX": "bx", "HY": "by", "HZ": "bz", "EX": "e1", "EY": "e2",
    "BX": "bx", "BY": "by", "BZ": "bz", "E1": "e1", "E2": "e2",
}

_HEADER_RE = re.compile(r"^['\"]?#FP['\"]?\s+(\d+)\s+(\d+)")


def _default_unit(channel: str) -> str:
    return "nanoTesla" if channel.startswith("b") else "mV/km"


def parse_fap(path) -> dict:
    """Parse a manufacturer FAP calibration file (``.chn`` / ``.def``).

    Returns ``{"frequencies": np.ndarray, "channels": {name: (amplitudes,
    phases_deg)}}``. The LEMI-417-era layout is a header line
    ``'#FP' <NTF> <NCH> / NTF,NCH DR-TRANSFORMED (MOD,ARG)``, a channel-label
    line (``#HX #HY #HZ #EX #EY``), then ``NTF`` rows of
    ``freq mod arg mod arg ...`` (modulus, argument in degrees, per channel;
    the first column is the frequency in Hz). The KMS-200 ``.def`` wrapper
    carries the same FAP block — leading wrapper/comment lines are skipped.
    Raises :class:`ValueError` when the FAP block cannot be located.
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
    channels: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for j, name in enumerate(labels):
        mods = np.asarray([r[1 + 2 * j] for r in rows], dtype=float)
        args = np.asarray([r[2 + 2 * j] for r in rows], dtype=float)
        channels[name] = (mods, args)
    return {"frequencies": frequencies, "channels": channels}


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

    freqs = np.asarray(rows["frequencies"], dtype=float)
    out: dict[str, object] = {}
    for ch, (mods, args) in rows["channels"].items():
        f, a, p = freqs, mods, args
        if extend_flat:
            f = np.concatenate([[freqs[0] / 1e3], freqs])
            a = np.concatenate([[mods[0]], mods])
            p = np.concatenate([[0.0], args])
        unit = (units or {}).get(ch) or _default_unit(ch)
        fap = FrequencyResponseTableFilter(
            frequencies=f, amplitudes=a, phases=p,
            instrument_type="flux gate magnetometer",
            units_in=unit, units_out=unit,
            gain=1.0, sequence_number=1,
        )
        fap.name = f"{prefix}_{ch}"
        out[ch] = fap
    return out


def parse_json_calibration(path) -> dict:
    """Parse an mt_io per-channel JSON calibration into FAP-style rows.

    The mt_io JSON layout is ``{"Calibration": {"gain", "Freq": [...],
    "Re": [...], "Im": [...]}}`` per channel key. Returns the same structure
    as :func:`parse_fap` (phases in degrees, amplitude = |Re + i·Im|).
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    channels: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    freqs = None
    for ch, payload in data.items():
        cal = payload.get("Calibration", payload)
        freq = np.asarray(cal["Freq"], dtype=float)
        re_ = np.asarray(cal["Re"], dtype=float)
        im_ = np.asarray(cal["Im"], dtype=float)
        mods = np.hypot(re_, im_)
        args = np.degrees(np.arctan2(im_, re_))
        channels[ch] = (mods, args)
        freqs = freq if freqs is None else freqs
    return {"frequencies": freqs, "channels": channels}


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


def attach_filters(run, filters) -> object:
    """Attach per-channel filters to an mt_timeseries run.

    Registers the filter objects in ``run.filters`` and marks each channel's
    dataset attrs with the modern ``filters`` (AppliedFilter) format, so
    ``RunTS.calibrate()`` applies them and the association survives an MTH5
    round trip (``from_runts`` / ``to_runts``). Returns the run.

    Channels without a calibration filter (e.g. LEMI temperature channels)
    get a unity coefficient filter so ``RunTS.calibrate()`` passes them
    through without the "no filters / corrupt units" warnings.
    """
    from mt_metadata.timeseries.filters import CoefficientFilter

    run_filters = dict(run.filters)
    for ch, fap in filters.items():
        if ch not in run.dataset.data_vars:
            raise ValueError(
                f"calibration channel '{ch}' not present in the run "
                f"(available: {list(run.dataset.data_vars)})"
            )
        run_filters[fap.name] = fap
        run.dataset[ch].attrs["filters"] = [
            {"applied_filter": {"name": fap.name, "applied": True}}]
    for ch in run.dataset.data_vars:
        if ch in filters:
            continue
        unit = run.dataset[ch].attrs.get("units") or _default_unit(ch)
        unity = CoefficientFilter(gain=1.0, units_in=unit, units_out=unit)
        unity.name = f"cal_{ch}_unity"
        run_filters[unity.name] = unity
        run.dataset[ch].attrs["filters"] = [
            {"applied_filter": {"name": unity.name, "applied": True}}]
    run.filters = run_filters
    return run


def register(registry) -> None:
    registry.add(LoadCalibration)


class LoadCalibration(Node):
    type = "mth5.calibration.load"
    title = "Load Calibration"
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
        Param("data_path", "string", label="Calibration file"),
        Param("extend_flat", "boolean", label="Flat below table",
              default=True,
              description="Treat the response as constant (first row, zero "
                          "phase) below the measured frequency range; "
                          "log-extrapolation was found to be worse."),
    )

    def execute(self, ctx, data_path: str,
                extend_flat: bool = True) -> dict[str, Any]:
        filters = load_calibration(data_path, extend_flat=extend_flat)
        ctx.progress(1.0, f"loaded calibration for {sorted(filters)}")
        return {"calibration": filters}
