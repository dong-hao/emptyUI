"""Shared processing helpers for the aurora plugin.

ComputeFC/EstimateTF split the aurora pipeline at the FC boundary. This module
holds the signal-processing core: RunTS/TimeSeries normalization, windowed
STFT to Fourier coefficients (fc), and MT impedance estimation (OLS / robust
RME / remote-reference RME_RR, all aurora-backed via :func:`_solve_band`).
"""

from __future__ import annotations

import logging
from itertools import pairwise

import numpy as np
import pandas as pd
from emptyui_core_nodes.types import ensure_finite
from emptyui_mth5.nomenclature import (
    STANDARD_INPUTS,
    STANDARD_OUTPUTS,
    channel_map,
    default_channels,
    infer_keyword,
    is_electric_name,
    is_magnetic_name,
    magnetic_unit_for,
    magnetic_unit_from_channels,
)

MU0 = 4.0 * np.pi * 1e-7

_log = logging.getLogger("emptyui")


def _split_names(value) -> list[str]:
    """Split a comma-separated channel list (or an iterable of names); None -> []."""
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(c).strip() for c in value if str(c).strip()]
    return [c.strip() for c in str(value).split(",") if c.strip()]


def _resolve_magnetic_unit(magnetic_unit, keyword, input_channels) -> str:
    """Resolve the magnetic-unit mode: explicit 'h'/'b' wins; otherwise derive
    it from the nomenclature keyword (mt_metadata CHANNEL_MAPS) or, failing
    that, from the actual channel names (bx/by -> fluxgate B, hx/hy -> coil H).

    Note the mode only drives the nT->A/m conversion via
    :func:`_magnetic_to_h`: 'b' always converts (fluxgate B is nT), while
    'h' converts unless the payload units already say A/m (coil archives are
    nT too — see DESIGN §16.4 USArray verification).
    """
    mu = str(magnetic_unit or "auto").strip().lower()
    if mu in ("h", "b"):
        return mu
    if keyword:
        return magnetic_unit_for(keyword)
    return magnetic_unit_from_channels(input_channels)


def _resolve_keyword(channel_nomenclature, input_channels, output_channels,
                     available=None):
    """Resolve the nomenclature keyword: explicit value, or best-effort
    inference. With explicit channel fields, inference uses those names (so a
    user requesting the e3/e4 pair on a 4-E-channel run resolves to lemi34);
    with auto (empty) fields it falls back to the channels actually present in
    the data."""
    kw = str(channel_nomenclature or "").strip().lower()
    if kw and kw != "auto":
        return kw
    requested = _split_names(input_channels) + _split_names(output_channels)
    if requested:
        return infer_keyword(requested)
    if available:
        return infer_keyword(available)
    return "default"


def _resolve_io_channels(available, channel_nomenclature, input_channels,
                         output_channels):
    """Resolve the actual input/output channel names for a compute_fc call.

    Empty (or "auto") fields are resolved automatically from the nomenclature
    and the data: inputs = the horizontal magnetic pair, outputs = the
    electric pair **plus** the vertical magnetic channel (tipper) — each
    included only when the channel is actually present. This makes the tipper
    available by default while keeping an impedance-only FC when the data has
    no vertical channel, and a tipper-only FC when the data carries only
    magnetic channels.

    Explicit names are used as-is; an explicit output channel missing from the
    data is dropped with a warning (the vertical magnetic channel is a
    best-effort extension), while a missing input channel is an error.
    """
    available = [str(c) for c in available]
    keyword = _resolve_keyword(
        channel_nomenclature, input_channels, output_channels, available
    )
    ins = _split_names(input_channels)
    outs = _split_names(output_channels)
    auto_ins = not ins or ins == ["auto"]
    auto_outs = not outs or outs == ["auto"]
    if auto_ins or auto_outs:
        d_ins, d_outs = default_channels(keyword)
    if auto_ins:
        ins = d_ins
    if auto_outs:
        outs = d_outs

    # Normalize to the exact names present in the data (case-insensitive —
    # e.g. a CSV with 'Ex,Ey,Hx,Hy' satisfies the standard 'ex,ey,hx,hy').
    def present(name) -> str | None:
        low = str(name).strip().lower()
        for c in available:
            if str(c).lower() == low:
                return c
        return None

    resolved_ins = [present(c) for c in ins]
    missing_ins = [c for c in resolved_ins if c is None]
    ins = [c for c in resolved_ins if c is not None]
    resolved_outs = [present(c) for c in outs]
    missing = [c for c in resolved_outs if c is None]
    outs = [c for c in resolved_outs if c is not None]
    if missing:
        _log.warning(
            "dropping output channel(s) %s not present in the data %s "
            "(tipper is a best-effort extension — the vertical magnetic "
            "channel is included only when available)",
            missing, available,
        )
    if len(ins) != 2:
        raise ValueError(
            f"need two input channels (horizontal magnetic Hx/Hy or Bx/By) — "
            f"resolved {ins} (missing {missing_ins}) from data {available}; "
            f"set input_channels explicitly (e.g. 'bx,by')"
        )
    if not outs:
        if auto_outs and len(ins) == 2:
            # A magnetic-only run (no electric/vertical channels) is a remote-
            # reference station: the FC carries the magnetic channels, which is
            # exactly what EstimateTF's ref_fc consumes.
            _log.warning(
                "no electric/vertical output channels in the data %s — using "
                "the magnetic channels %s as outputs (remote-reference fc)",
                available, ins,
            )
            outs = list(ins)
        else:
            raise ValueError(
                f"no output channels available — requested {output_channels!r} "
                f"against data {available}"
            )
    return ins, outs, keyword


def to_channels(value, check_finite: bool = True):
    """Normalize a ts.run payload to (channels: dict[str, np.ndarray],
    sample_rate: float, start_s: float, start_dt: pd.Timestamp | None).

    Accepts:
      - mth5 RunTS (has .dataset xarray, .sample_rate, .start datetime)
      - core TimeSeries (has .channels dict, .sample_rate, .start float s)

    ``check_finite`` guards every channel against NaN/inf; pass False to
    defer the guard until the used channel subset is known (callers that
    resolve in/out channels check only those, so a NaN in an unused
    auxiliary channel cannot hard-fail the node).
    """
    # mth5 RunTS / mt_timeseries
    if hasattr(value, "dataset") and not isinstance(value, dict):
        ds = value.dataset
        sample_rate = float(value.sample_rate)
        start_s = 0.0
        start_dt = _start_datetime(getattr(value, "start", None))
        channels = {str(k): np.asarray(v.values, dtype=float) for k, v in ds.data_vars.items()}
        if check_finite:
            ensure_finite(channels, "Compute FC")
        return channels, sample_rate, start_s, start_dt
    # core-nodes TimeSeries
    if hasattr(value, "channels") and hasattr(value, "sample_rate"):
        sample_rate = float(value.sample_rate)
        start_s = float(getattr(value, "start", 0.0))
        start_dt = _start_datetime(
            getattr(value, "start_datetime", None) or getattr(value, "start", None)
        )
        channels = {str(k): np.asarray(v, dtype=float) for k, v in value.channels.items()}
        if check_finite:
            ensure_finite(channels, "Compute FC")
        return channels, sample_rate, start_s, start_dt
    raise TypeError(f"cannot normalize ts.run value of type {type(value).__name__}")


def _start_datetime(value):
    """Return a pandas Timestamp for an absolute start time, or None.

    Handles datetime / pandas Timestamp, an ISO string (core TimeSeries), and
    mt_metadata MTime. Returns None for relative float-second starts.
    """
    if value is None or isinstance(value, (int, float, np.number)):
        return None
    if isinstance(value, str):
        try:
            return pd.Timestamp(value)
        except (TypeError, ValueError):
            return None
    try:
        return pd.Timestamp(value)
    except (TypeError, ValueError):
        pass
    # mt_metadata MTime and similar expose iso_str / isoformat
    for attr in ("iso_str", "isoformat"):
        fn = getattr(value, attr, None)
        if callable(fn):
            try:
                return pd.Timestamp(str(fn()))
            except (TypeError, ValueError):
                pass
    return None


def compute_fc(
    value,
    input_channels,
    output_channels,
    window_length_s,
    overlap,
    window_type,
    magnetic_unit="auto",
    e_dipole_length_m=None,
    start_datetime=None,
    channel_nomenclature=None,
):
    """Windowed STFT -> fc xarray Dataset (dims frequency x time).

    The input is assumed to be already detrended (use the Detrend node to
    remove the DC/absolute-field baseline first — fluxgate magnetometers
    report a huge DC offset that would otherwise swamp the autopower and drive
    Z -> 0).

    ``magnetic_unit``: "auto" (default) derives the mode from the channel
    nomenclature / actual channel names — "b" (fluxgate gives absolute B,
    converted B->H by dividing by μ0) or "h" (coil sensors — mth5 archives
    coil data in nT too, so they are converted to A/m as well; only data
    whose units explicitly say A/m, or unit-less synthetic payloads, are
    left untouched). An explicit "h"/"b" always wins (legacy workflows keep
    working — "h" there means "already A/m").
    ``channel_nomenclature``: mt_metadata CHANNEL_MAPS keyword (e.g. "lemi12")
    used to interpret channel names; when None/auto it is inferred and recorded
    on the fc so EstimateTF can map back to standard TF coordinates.

    ``e_dipole_length_m``: LEMI-style instruments store E as the raw potential
    difference (mV) between electrodes, not an E-field. When set, output (E)
    channels are converted mV -> V/m by dividing by the dipole length
    (E = raw_mV * 1e-3 / dipole_m).

    Output channels are classified by name: **electric** outputs (ex/ey/e1/e2/…)
    get the E scaling above (mV/km -> V/m when no dipole length), while
    **magnetic** outputs (bz/hz) get the same B->H conversion as the inputs (so
    the fc carries H throughout and a tipper Bz/H is dimensionless). Output
    channels are not required to be exactly the electric pair — a vertical
    magnetic output extends the fc to support a geomagnetic transfer function.

    ``start_datetime`` optionally overrides the absolute start time used for the
    ``time`` coordinate (e.g. the run's start). When available (from the source
    RunTS or this arg) the fc ``time`` coordinate is ``datetime64`` and a
    ``start_datetime`` attr is recorded — required for mth5 FC persistence.
    """
    import xarray as xr
    from scipy.signal import stft

    channels, sample_rate, start_s, src_start = to_channels(value, check_finite=False)
    if start_datetime is None:
        start_datetime = src_start
    units = _channel_units(value)
    nperseg = max(8, int(window_length_s * sample_rate))
    noverlap = int(nperseg * overlap)
    win = _window(window_type)
    input_channels, output_channels, keyword = _resolve_io_channels(
        list(channels), channel_nomenclature, input_channels, output_channels
    )
    # NaN guard only on the channels that are actually STFT'd — a NaN in an
    # unused auxiliary channel must not hard-fail the node
    ensure_finite(
        {c: channels[c] for c in dict.fromkeys(input_channels + output_channels)},
        "Compute FC",
    )
    b_to_h = _resolve_magnetic_unit(magnetic_unit, keyword, input_channels) == "b"
    e_scale = (1e-3 / e_dipole_length_m) if e_dipole_length_m else None

    fc_vars = {}
    for comp in input_channels + output_channels:
        if comp not in channels:
            raise ValueError(
                f"channel '{comp}' not in data {list(channels)}; "
                "LEMI data uses e1,e2,bx,by,bz — set input/output channels accordingly"
            )
        x = channels[comp]
        if comp in input_channels:
            x = _magnetic_to_h(x, comp, b_to_h, units)
        elif is_magnetic_name(comp):  # vertical magnetic output (bz/hz) — tipper
            x = _magnetic_to_h(x, comp, b_to_h, units)  # same units as inputs
        elif is_electric_name(comp):  # electric output channel
            if e_scale is not None:
                # raw potential (mV) -> V/m via dipole length
                x = x * e_scale
            else:
                # LEMI E in mV/km == uV/m == 1e-6 V/m
                x = x * 1e-6
        # unknown output channel: leave units untouched
        f, t, z = stft(
            x, fs=sample_rate, nperseg=nperseg, noverlap=noverlap,
            window=win, boundary=None, padded=False,
        )
        fc_vars[comp] = (("frequency", "time"), z)

    time_coord = t + start_s
    start_iso = ""
    if start_datetime is not None:
        start_ts = pd.Timestamp(start_datetime)
        time_coord = start_ts + pd.to_timedelta(t, unit="s")
        start_iso = start_ts.isoformat()

    fc = xr.Dataset(
        data_vars=fc_vars,
        coords={"frequency": f, "time": time_coord},
        attrs={
            "sample_rate": float(sample_rate),
            "num_samples_window": int(nperseg),
            "window_length_s": float(window_length_s),
            "overlap": float(overlap),
            "window_type": window_type,
            "input_channels": ",".join(input_channels),
            "output_channels": ",".join(output_channels),
            "magnetic_unit": b_to_h and "b" or "h",
            "channel_nomenclature": keyword,
            "e_dipole_length_m": e_dipole_length_m or 0.0,
            "start_datetime": start_iso,
        },
    )
    return fc


def compute_fc_levels(
    value,
    input_channels,
    output_channels,
    num_decimations=0,
    factor=4,
    num_samples_window=256,
    overlap=0.5,
    window_type="hamming",
    magnetic_unit="auto",
    e_dipole_length_m=None,
    start_datetime=None,
    channel_nomenclature=None,
):
    """Cascading-decimation multi-level STFT -> ``{level: xr.Dataset}``.

    ``num_decimations`` is the number of cascading decimations performed, so it
    also equals the deepest decimation level: 0 = no decimation (just level 0 at
    the native sample rate); 1 = decimate once (levels 0 and 1); N = levels 0..N.
    Each deeper level decimates the previous level's (already-normalized) series
    by ``factor`` and re-STFTs it with the **same constant ``num_samples_window``**
    (aurora-style), so the window *duration* grows at deeper levels and
    low-frequency resolution improves. Level *i*'s sample rate = native / factor^i.

    Each level's fc is an xr.Dataset with ``(frequency, time)`` dims and a
    ``datetime64`` ``time`` coordinate when an absolute start is known (needed
    for mth5 FC persistence). Per-level metadata (``sample_rate``,
    ``decimation_level``, window params) is recorded in each fc's attrs.
    """
    import xarray as xr

    channels, sample_rate, start_s, start_dt = to_channels(value, check_finite=False)
    if start_datetime is None:
        start_datetime = start_dt
    units = _channel_units(value)
    input_channels, output_channels, keyword = _resolve_io_channels(
        list(channels), channel_nomenclature, input_channels, output_channels
    )
    # NaN guard only on the channels that are actually normalized/STFT'd —
    # a NaN in an unused auxiliary channel must not hard-fail the node
    ensure_finite(
        {c: channels[c] for c in dict.fromkeys(input_channels + output_channels)},
        "Compute FC",
    )
    b_to_h = _resolve_magnetic_unit(magnetic_unit, keyword, input_channels) == "b"
    e_scale = (1e-3 / e_dipole_length_m) if e_dipole_length_m else None

    num_decimations = max(0, int(num_decimations))
    levels: dict[int, xr.Dataset] = {}
    cur = channels
    cur_rate = sample_rate
    for level in range(num_decimations + 1):
        # normalize the raw data only once (level 0); deeper levels re-use the
        # already-normalized series passed down from the previous level.
        norm = (
            _normalize_channels(cur, input_channels, output_channels, b_to_h, e_scale,
                                units)
            if level == 0
            else cur
        )
        f, t, fc_vars = _stft_channels(
            norm, cur_rate, num_samples_window, overlap, window_type,
            input_channels, output_channels,
        )
        time_coord = t + start_s
        start_iso = ""
        if start_datetime is not None:
            start_ts = pd.Timestamp(start_datetime)
            time_coord = start_ts + pd.to_timedelta(t, unit="s")
            start_iso = start_ts.isoformat()

        fc = xr.Dataset(
            data_vars=fc_vars,
            coords={"frequency": f, "time": time_coord},
            attrs={
                "sample_rate": float(cur_rate),
                "decimation_level": int(level),
                "factor": int(factor),
                "num_samples_window": int(num_samples_window),
                "overlap": float(overlap),
                "window_type": window_type,
                "input_channels": ",".join(input_channels),
                "output_channels": ",".join(output_channels),
                "magnetic_unit": b_to_h and "b" or "h",
                "channel_nomenclature": keyword,
                "e_dipole_length_m": e_dipole_length_m or 0.0,
                "start_datetime": start_iso,
            },
        )
        levels[int(level)] = fc

        if level < num_decimations:
            cur = _decimate_channels(norm, factor)
            cur_rate = cur_rate / float(factor)

    return levels


def _channel_units(value):
    """Per-channel unit strings from a ts.run payload (best-effort).

    Returns ``{channel: units}`` for payloads that carry units (mth5 RunTS
    xarray attrs — real archives say ``nanoTesla`` / ``milliVolt per
    kilometer``; synthetic/core payloads usually carry none, so ``{}``).
    """
    if hasattr(value, "dataset"):
        ds = value.dataset
        out: dict[str, str] = {}
        for name in ds.data_vars:
            u = ds[name].attrs.get("units")
            if u:
                out[str(name)] = str(u)
        return out
    return {}


def _magnetic_to_h(x, comp, b_to_h, units):
    """Convert a magnetic channel from nT to A/m (B -> H, /mu0).

    The fc must carry H in A/m for the impedance to be in ohm (V/m per A/m);
    nT data must therefore be divided by mu0. ``b_to_h`` (fluxgate B, mode
    'b') always converts; mode 'h' (coil) is archived in nT by mth5 as well,
    so it converts too — unless the payload explicitly says the data are
    already in A/m (or carries no unit info at all, e.g. synthetic tests,
    which pass mode 'h' for unit-neutral A/m data).
    """
    if b_to_h:
        return x * 1e-9 / MU0
    u = str((units or {}).get(comp, "")).lower()
    if u and ("nano" in u or u in ("nt", "nanotesla")):
        return x * 1e-9 / MU0
    return x


def _normalize_channels(channels, input_channels, output_channels, b_to_h, e_scale,
                        units=None):
    """Apply instrument unit normalization (B->H, mV->V/m) to a channel dict."""
    out = {}
    for comp, x in channels.items():
        x = np.asarray(x, dtype=float)
        if comp in input_channels:
            x = _magnetic_to_h(x, comp, b_to_h, units)
        elif comp in output_channels:
            if is_magnetic_name(comp):  # vertical magnetic output (tipper)
                x = _magnetic_to_h(x, comp, b_to_h, units)  # Bz -> Hz, same as inputs
            elif is_electric_name(comp):
                if e_scale is not None:
                    x = x * e_scale  # raw potential (mV) -> V/m
                else:
                    x = x * 1e-6  # LEMI E in mV/km == uV/m
        out[comp] = x
    return out


def _decimate_channels(channels, factor):
    """Cascade-decimate every channel by an integer factor (scipy decimate)."""
    from scipy.signal import decimate

    factor = max(2, int(factor))
    return {k: np.asarray(decimate(v, factor)) for k, v in channels.items()}


def _stft_channels(channels, sample_rate, num_samples, overlap, window_type,
                   input_channels, output_channels):
    """Windowed STFT of a channel dict -> (f, t, {channel: (freq,time) complex})."""
    from scipy.signal import stft

    nperseg = max(8, int(num_samples))
    noverlap = int(nperseg * overlap)
    win = _window(window_type)
    f = t = None
    fc_vars = {}
    for comp in input_channels + output_channels:
        x = np.asarray(channels[comp], dtype=float)
        fi, ti, zi = stft(
            x, fs=sample_rate, nperseg=nperseg, noverlap=noverlap,
            window=win, boundary=None, padded=False,
        )
        f, t = fi, ti
        fc_vars[comp] = (("frequency", "time"), zi)
    return f, t, fc_vars


def _window(name: str) -> str:
    """Map our window_type vocabulary to scipy window names."""
    return {"hanning": "hann"}.get(name, name)


def _resolve_estimator(estimator_engine, remote_reference) -> str:
    """Map the user-facing estimator choice + remote_reference to an internal
    aurora engine (``OLS`` / ``RME`` / ``RME_RR``).

    Accepts both the current two-option vocabulary (``Least Squares`` /
    ``Robust``) and the legacy three-value one (``OLS`` / ``RME`` / ``RME_RR``)
    so previously saved workflows keep loading. There is no least-squares +
    remote-reference path by design; requesting one raises a clear error.
    """
    e = str(estimator_engine).strip().lower().replace("_", " ")
    if e in ("ols", "least squares", "ls"):
        if remote_reference:
            raise ValueError(
                "least-squares remote reference is not supported — aurora's "
                "remote-reference estimator is robust (RME_RR). Use estimator "
                "'Robust' with remote reference, or 'Least Squares' without."
            )
        return "OLS"
    if e in ("rme", "rme rr", "robust"):
        return "RME_RR" if remote_reference else "RME"
    raise ValueError(f"unknown estimator engine: {estimator_engine!r}")


def _normalize_fc_layout(fc):
    """Normalize an fc to a ``(frequency, time)`` data layout.

    emptyUI's :func:`compute_fc` produces ``(frequency, time)``; mth5's FC
    persistence stores ``(time, frequency)`` and ``to_xarray`` returns that
    layout. The band loop indexes ``fc[channel][f_idx, :]`` (frequency rows),
    so read-back FCs must be transposed first.
    """
    if "frequency" not in fc.dims or "time" not in fc.dims:
        raise ValueError("fc must have 'frequency' and 'time' dimensions")
    first = next(iter(fc.data_vars))
    if fc[first].dims[0] != "frequency":
        fc = fc.transpose("frequency", "time")
    return fc


def _check_remote_grid(ref_freq, local_freq, ref_time, local_time):
    """Shared remote-reference FC grid validation (coherence sort + estimate).

    RME_RR pairs local/remote samples by row, so the FCs must share identical
    frequency and window (time) grids — the runs have to be cropped to their
    common acquisition overlap (Align Runs) before computing FCs.
    """
    if not np.array_equal(ref_freq, local_freq):
        raise ValueError(
            "remote-reference fc has a different frequency grid than the "
            "local fc — compute both with the same window settings"
        )
    if not np.array_equal(ref_time, local_time):
        raise ValueError(
            "remote-reference fc has a different time grid than the local "
            "fc — the runs are not aligned; crop both to their common "
            "acquisition overlap (Align Runs) before computing FCs"
        )


def _level_ref(ref_levels, level):
    """Remote fc for one decimation level, failing loudly on a partial match.

    ``ref_levels is None`` (nothing connected) is the documented self-reference
    case and is passed through; but a multi-level remote container that lacks
    the local fc's level would silently degrade those levels to a self-reference
    (≈RME) — that is almost always a window/decimation mismatch between the
    two Compute FC nodes, so it must be an error instead.
    """
    if ref_levels is None:
        return None
    if isinstance(ref_levels, dict):
        if level not in ref_levels:
            raise ValueError(
                f"remote-reference fc has no decimation level {level} "
                f"(has {sorted(ref_levels)}) — compute both stations' FCs "
                "with the same decimation/window settings"
            )
        return ref_levels[level]
    return ref_levels


def _simple_coherence(x, y):
    """Pairwise coherence² between two complex series (band ensemble)."""
    denom = np.sum(np.abs(x) ** 2) * np.sum(np.abs(y) ** 2)
    if denom <= 0:
        return 0.0
    return float(np.clip(np.abs(np.sum(x * np.conj(y))) ** 2 / denom, 0.0, 1.0))


def _multiple_coherence(E, H):
    """Per-window multiple coherence² of each output (rows of E) on the
    magnetic inputs (rows of H), over the band's frequency-bin ensemble.

    ``E`` is (n_out, n_bins) and ``H`` is (n_in, n_bins) complex FC samples.
    Standard multiple coherence: coh²_o = (S_oH S_HH⁻¹ S_Ho) / S_oo from the
    bin-ensemble cross-spectral matrices. Bands with too few bins (n_bins <
    n_in + 1) fall back to the minimum pairwise coherence (conservative);
    degenerate (all-zero) outputs score 0.
    """
    n_out, n_bins = E.shape
    n_in = H.shape[0]
    out = np.zeros(n_out)
    if n_bins < n_in + 1:
        for o in range(n_out):
            out[o] = min(_simple_coherence(E[o], H[i]) for i in range(n_in))
        return out
    S_HH_inv = np.linalg.pinv(H @ H.conj().T)
    for o in range(n_out):
        e = E[o]
        S_oo = float(np.real(e @ e.conj()))
        if S_oo <= 0:
            out[o] = 0.0
            continue
        S_Ho = H @ e.conj()
        c = float(np.real(S_Ho.conj() @ (S_HH_inv @ S_Ho)) / S_oo)
        if not np.isfinite(c):
            c = min(_simple_coherence(e, H[i]) for i in range(n_in))
        out[o] = float(np.clip(c, 0.0, 1.0))
    return out


def _coherence_weight(coh2, lo, hi, mode, ramp=0.05):
    """Weight for a window's coherence² outside [lo, hi].

    ``mode``: ``off`` → 1 always; ``drop`` → 0 outside the range; ``weight``
    → smooth trapezoid (linear ramp of width ``ramp`` outside the bounds, so
    borderline windows are down-weighted, not hard-cut). The upper bound
    matters because perfect (≈1) coherence can be a noise/overfit signature.
    """
    if mode == "off":
        return 1.0
    if coh2 < lo:
        if mode == "drop":
            return 0.0
        return float(np.clip((coh2 - (lo - ramp)) / ramp, 0.0, 1.0))
    if coh2 > hi:
        if mode == "drop":
            return 0.0
        return float(np.clip(((hi + ramp) - coh2) / ramp, 0.0, 1.0))
    return 1.0


def _resolve_band_style(band_style):
    """Normalize the ``band_style`` value to one of the two supported
    schemes (DESIGN.md §14.9): ``"l2u"`` (default) — the fixed reference
    template with log-uniform window-starvation coarsening — or
    ``"uniform"`` — the uniform log2-grid partition with merge-based level
    combination (formerly named ``"log"``, legacy alias accepted). The
    retired ``"emtf"`` value is accepted as a legacy alias of ``"l2u"``
    (identical scheme was not — the auto-derived disjoint EMTF-style table
    was replaced 2026-09-11 by the l2u scheme); the retired ``"index"``
    mode raises a clear error.
    """
    style = str(band_style or "l2u").strip().lower()
    if style == "log":  # legacy alias, identical behavior
        style = "uniform"
    if style == "emtf":  # legacy alias, superseded by l2u (2026-09-11)
        style = "l2u"
    if style not in ("l2u", "uniform"):
        raise ValueError(
            f"unknown band_style '{band_style}' (expected 'l2u' or "
            "'uniform'; the retired 'index' mode was removed 2026-09-09 — "
            "use 'l2u' for the fixed-template log-uniform scheme)")
    return style


def coherence_sort_fc(local, ref, input_channels, output_channels,
                      coherence_min=0.25, coherence_max=0.98, mode="weight",
                      trim=False, periods_per_octave=4, band_style="l2u",
                      bands_override=None):
    """EMTF-style coherence sort of one decimation level's FC (DESIGN §16.2).

    Returns ``(local_out, ref_out, rows)`` — copies of the FCs with the
    per-window coherence weights applied (``weight``: multiplied by √w;
    ``drop``/``trim``: excluded samples set to NaN), plus diagnostic rows.
    In two-FC mode the criterion is the **conservative minimum** of the
    E–remote-H signal coherence and the local↔remote magnetic transfer
    coherence (aurora's RME_RR signal-covariance estimate degrades when the
    local and remote magnetic fields are uncorrelated), and a single shared
    weight per row is applied to **both** FCs (exact weighted-LS semantics).

    ``band_style`` selects the band-construction scheme (DESIGN.md §14.9,
    shared with :func:`estimate_tf`): ``"l2u"`` (default) uses the
    reference-template table (``bands_override`` holds this level's
    ``(f_hi, f_lo)`` pairs); ``"uniform"`` is the uniform log2-grid
    partition. It should match the ``band_style`` used on the corresponding
    ``Estimate TF`` call for consistent banding, exactly like
    ``periods_per_octave``.
    """
    local = _normalize_fc_layout(local)
    ref = _normalize_fc_layout(ref) if ref is not None else None
    if ref is not None:
        _check_remote_grid(
            ref["frequency"].values, local["frequency"].values,
            ref["time"].values, local["time"].values,
        )
    ins = [c for c in input_channels if c in local.data_vars]
    outs = [c for c in output_channels if c in local.data_vars]
    if not ins or not outs:
        raise ValueError(
            f"could not determine input/output channels from the fc — "
            f"data_vars: {list(local.data_vars)}"
        )

    local_out = local.copy(deep=True)
    ref_out = ref.copy(deep=True) if ref is not None else None

    if bands_override is not None:
        periods, band_bins_list = _bands_from_freq_pairs(local, bands_override)
    else:
        periods, band_bins_list = _resolve_period_bands(
            local, periods_per_octave, band_style)
    rows = []
    for b in range(len(periods)):
        band_bins = band_bins_list[b]
        if band_bins.size == 0:
            continue
        E = np.stack([local[oc].values[band_bins, :] for oc in outs], axis=0)
        H_local = np.stack([local[ic].values[band_bins, :] for ic in ins], axis=0)
        n_win = E.shape[2]
        coh_e = np.zeros(n_win)  # E on local-H (single-FC) or remote-H (two-FC)
        coh_x = np.full(n_win, 1.0)  # local↔remote magnetic transfer
        for w in range(n_win):
            H_local_w = H_local[:, :, w]
            H_pred_w = H_local_w
            if ref is not None:
                H_pred_w = np.stack(
                    [ref[ic].values[band_bins, w] for ic in ins], axis=0
                )
            ce = _multiple_coherence(E[:, :, w], H_pred_w)
            coh_e[w] = float(ce.min()) if ce.size else 0.0
            if ref is not None:
                R = np.stack([ref[ic].values[band_bins, w] for ic in ins], axis=0)
                cx = _multiple_coherence(H_local_w, R)
                coh_x[w] = float(cx.min()) if cx.size else 0.0
        coh = np.minimum(coh_e, coh_x) if ref is not None else coh_e
        # degenerate bands (fewer bins than inputs+1) fit the regression
        # exactly, so their coherence is trivially 1.0 — the "coh ≈ 1 is
        # suspicious" rule must not apply there
        if band_bins.size < len(ins) + 1:
            coh = np.minimum(coh, coherence_max - 1e-9)
        weights = np.array([_coherence_weight(c, coherence_min, coherence_max, mode)
                            for c in coh])
        trimmed = False
        if trim:
            keep = weights > 0
            if keep.sum() == 0:
                trimmed = True  # nothing retained in this band
            elif float(np.median(coh[keep])) < coherence_min:
                trimmed = True
        if mode != "off" and not trimmed:
            scale = np.sqrt(weights)
            # fully down-weighted samples are excluded (NaN) rather than
            # zeroed — zero rows would inflate the regression degrees of
            # freedom; estimate_tf skips non-finite rows.
            scale[scale <= 0.0] = np.nan
            for oc in outs + ins:
                vals = local_out[oc].values
                vals[band_bins, :] = vals[band_bins, :] * scale[None, :]
            if ref_out is not None:
                for ic in ins:
                    vals = ref_out[ic].values
                    vals[band_bins, :] = vals[band_bins, :] * scale[None, :]
        if trimmed:
            for oc in outs + ins:
                local_out[oc].values[band_bins, :] = np.nan
            if ref_out is not None:
                for ic in ins:
                    ref_out[ic].values[band_bins, :] = np.nan
        rows.append({
            "period": float(periods[b]),
            "n_windows": int(n_win),
            "n_retained": int(np.count_nonzero(weights > 0)) if mode != "off" else int(n_win),
            "median_coherence_e": float(np.median(coh_e)),
            "median_coherence_transfer": float(np.median(coh_x)) if ref is not None else None,
            "median_coherence": float(np.median(coh)),
            "trimmed": trimmed,
        })
    return local_out, ref_out, rows


def coherence_sort_fcs(levels, ref_levels, input_channels, output_channels,
                       coherence_min=0.25, coherence_max=0.98, mode="weight",
                       trim=False, periods_per_octave=4, band_style="l2u",
                       min_bins_per_band=40.0, t_short=None):
    """Coherence-sort every decimation level (see :func:`coherence_sort_fc`).

    ``levels`` is the multi-level fc container; ``ref_levels`` is an optional
    same-shaped container (or a single fc) for the remote station. Returns
    ``(levels_out, ref_levels_out, rows)`` where the rows carry the
    decimation level for the diagnostics table. For ``band_style="l2u"``
    the reference-template band table is derived once for the whole
    container (like :func:`estimate_tf_levels`) so the sort's banding
    matches the subsequent ``Estimate TF`` call — pass the **same**
    ``min_bins_per_band`` and ``t_short`` values as the estimator
    (2026-09-11 review: these refine the l2u table and were previously
    hard-coded/missing here, silently banding differently from Estimate TF).
    """
    bands_override = None
    style = _resolve_band_style(band_style)
    if style == "l2u":
        bands_override = _l2u_band_table(
            levels, min_bins_per_band=min_bins_per_band, t_short=t_short)
    out_levels: dict[int, object] = {}
    out_ref: dict[int, object] = {} if ref_levels is not None else None
    rows: list[dict] = []
    for level, fc in sorted(levels.items()):
        ref = _level_ref(ref_levels, level)
        l_out, r_out, level_rows = coherence_sort_fc(
            fc, ref, input_channels, output_channels,
            coherence_min=coherence_min, coherence_max=coherence_max,
            mode=mode, trim=trim, periods_per_octave=periods_per_octave,
            band_style=band_style,
            bands_override=None if bands_override is None
            else bands_override.get(level, []),
        )
        out_levels[int(level)] = l_out
        if ref_levels is not None:
            out_ref[int(level)] = r_out
        for r in level_rows:
            r["decimation_level"] = int(level)
            rows.append(r)
    return out_levels, out_ref, rows


def estimate_tf(
    fc,
    input_channels,
    output_channels,
    estimator_engine="RME_RR",
    remote_reference=False,
    periods_per_octave=4,
    ref_fc=None,
    band_style="l2u",
    bands_override=None,
):
    """MT transfer-function estimate -> (periods, Z[n_out x n_in complex], Z_err).

    Z rows = outputs (e.g. ex,ey and optionally bz/hz), cols = inputs (hx,hy).
    Z_err holds the standard error of each complex element (same orientation).
    With outputs = [ex,ey] this is the 2x2 impedance; adding a vertical
    magnetic output (bz/hz) extends it to a 3x2 whose third row is the
    geomagnetic transfer function (tipper). The fc already
    holds H for the magnetic inputs (B->H done in compute_fc when the
    instrument is fluxgate), so impedance is in ohm (V/m per A/m).

    ``estimator_engine`` selects the estimator (all aurora-backed):
      - ``Least Squares`` : ordinary least squares (aurora ``estimate_ols``).
      - ``Robust``        : robust M-estimator (aurora ``RME``); with
                            ``remote_reference=True`` uses the robust
                            remote-reference estimator (aurora ``RME_RR``).
    There is intentionally **no least-squares + remote-reference** path: aurora
    offers remote reference only through the robust ``RME_RR``, and we keep
    that design (see DESIGN.md §13). For backward compatibility the legacy
    engine values ``OLS`` / ``RME`` / ``RME_RR`` are still accepted.

    Period bands (DESIGN.md §14.9): ``band_style="l2u"`` (default) uses the
    reference-template band table with log-uniform window-starvation
    coarsening (:func:`_l2u_band_table`) — the same fixed template at every
    decimation level, starved levels rebuilt as a coarser uniform grid;
    ``band_style="uniform"`` uses the uniform log2-grid partition
    (:func:`_period_bands`, ``periods_per_octave`` bands per octave,
    merge-based across overlapping levels). The legacy ``band_style="log"``
    value is accepted as an alias of ``"uniform"`` and ``"emtf"`` as an
    alias of ``"l2u"`` (the retired auto-derived EMTF-style table was
    replaced by l2u on 2026-09-11).

    ``ref_fc`` optionally supplies a *second station's* Fourier coefficients
    used as the remote reference (magnetic channels only). When omitted (and
    remote reference is requested), the local input channels are reused as the
    reference — a self-reference (degenerate but well-defined, equals RME).
    This is also what a duplicated/identical "dummy remote station" reduces to.

    FCs read back from an mth5 archive carry a ``(time, frequency)`` layout
    (mth5 ``to_xarray``), while :func:`compute_fc` produces
    ``(frequency, time)``; both are normalized to ``(frequency, time)`` here.
    """

    engine = _resolve_estimator(estimator_engine, remote_reference)

    fc = _normalize_fc_layout(fc)
    freq = fc["frequency"].values
    ok = freq > 0  # exclude DC bin (period would be infinite)
    fc = fc.isel(frequency=ok)

    if ref_fc is not None:
        ref_fc = _normalize_fc_layout(ref_fc)
        ref_fc = ref_fc.isel(frequency=ref_fc["frequency"].values > 0)
        _check_remote_grid(
            ref_fc["frequency"].values, freq[ok],
            ref_fc["time"].values, fc["time"].values,
        )

    # Realistic, octave-aligned period bands for this decimation level
    # (DESIGN.md §14.9: band_style selects the uniform log2-grid partition
    # or the reference-template table; bands_override carries a precomputed
    # table (frequency pairs from _l2u_band_table — disjoint by design,
    # nothing to merge) when the container-level derivation succeeded
    # upstream).
    if bands_override is not None:
        periods_out, band_bins_list = _bands_from_freq_pairs(fc, bands_override)
    else:
        periods_out, band_bins_list = _resolve_period_bands(
            fc, periods_per_octave, band_style)

    nbands = len(periods_out)
    n_in = len(input_channels)
    n_out = len(output_channels)
    # Z rows = outputs (ex,ey,...,bz), cols = inputs (hx,hy): a 2x2 impedance
    # when outputs are ex/ey, extended to a 3x2 (with a bz/hz tipper row) when a
    # vertical channel is included.
    Zout = np.zeros((nbands, n_out, n_in), dtype=complex)
    Zerr = np.zeros((nbands, n_out, n_in), dtype=float)

    for b in range(nbands):
        band_bins = band_bins_list[b]
        if band_bins.size == 0:
            continue
        # gather samples across selected frequencies and all windows
        H = []  # inputs  -> [n, n_in]
        E = []  # outputs -> [n, n_out]
        R = []  # reference inputs -> [n, n_in]
        for f_idx in band_bins:
            H.append(np.stack([fc[ic].values[f_idx, :] for ic in input_channels], axis=-1))
            E.append(np.stack([fc[oc].values[f_idx, :] for oc in output_channels], axis=-1))
            if engine == "RME_RR":
                if ref_fc is not None:
                    R.append(np.stack(
                        [ref_fc[ic].values[f_idx, :] for ic in input_channels], axis=-1))
                else:  # self-reference (dummy/duplicated remote station)
                    R.append(np.stack(
                        [fc[ic].values[f_idx, :] for ic in input_channels], axis=-1))
        H = np.concatenate(H, axis=0)
        E = np.concatenate(E, axis=0)
        if engine == "RME_RR":
            R_band = np.concatenate(R, axis=0)
        else:
            R_band = None
        # skip non-finite FC samples (coherence-sort drop/trim sets NaN);
        # for remote reference the union of local/remote rows is dropped so
        # the row pairing stays aligned (aurora RME_RR rejects NaN).
        ok = np.all(np.isfinite(H), axis=1) & np.all(np.isfinite(E), axis=1)
        if R_band is not None:
            ok = ok & np.all(np.isfinite(R_band), axis=1)
        if not np.all(ok):
            H = H[ok]
            E = E[ok]
            if R_band is not None:
                R_band = R_band[ok]
        if H.shape[0] <= len(input_channels):
            continue
        Zout[b], Zerr[b] = _solve_band(H, E, R_band, input_channels, output_channels, engine)

    # All estimators now come from aurora and share aurora's impedance
    # convention, so no sign/phase rotation is applied.
    return periods_out, Zout, Zerr


def _period_bands(fc, periods_per_octave=4):
    """Uniform log2-grid period bands -> (band-center periods, edges).

    The ``"uniform"`` band scheme (formerly named ``"log"``): each level's
    highest trusted frequency is **Fs/4** (so the
    shortest period is 4/Fs, avoiding the noisy Nyquist-edge bins) and its
    longest period is **window_length/4** (N/(4·Fs)). Band centers sit on a
    global log2 (octave) grid with ``periods_per_octave`` points per octave
    anchored at 4/Fs. Because decimation factors are powers of 2, consecutive
    levels' grids land on the *same* absolute periods wherever they overlap —
    so the merged TF has consistent frequency points across levels (and can be
    combined by the merge). Returns (periods shortest→longest, log2-frequency
    edges high→low) for the :func:`estimate_tf` band loop.
    """
    fs = float(fc.attrs.get("sample_rate") or 0.0)
    if not fs:
        # fc without attrs (e.g. read back from mth5 before enrichment): the
        # Nyquist bin is fs/2, so the native rate is recoverable from the grid.
        fs = 2.0 * float(fc["frequency"].values.max())
    ppo = max(1, int(periods_per_octave))
    p_short = 4.0 / fs  # Fs/4 as the highest trusted frequency

    N = int(fc.attrs.get("num_samples_window") or 0)
    if not N:
        wl = fc.attrs.get("window_length_s")
        if wl:
            N = round(float(wl) * fs)
    p_long = N / (4.0 * fs) if N else None  # window_length / 4

    longest_present = 1.0 / float(fc["frequency"].values[fc["frequency"].values > 0].min())
    if p_long is None or p_long > longest_present:
        p_long = longest_present
    if p_long <= p_short:
        p_long = longest_present

    M = max(0, int(np.ceil(ppo * np.log2(p_long / p_short))))
    logp = np.log2(p_short) + np.arange(M + 1) / ppo
    periods_out = 2.0 ** logp  # shortest -> longest

    # binning edges in log2 frequency (high -> low), one boundary per band
    logf = -logp
    d = 1.0 / ppo
    edges = np.empty(M + 2)
    edges[0] = logf[0] + d / 2.0
    edges[1:M + 1] = (logf[:-1] + logf[1:]) / 2.0
    edges[M + 1] = logf[-1] - d / 2.0
    return periods_out, edges


def _container_geometry(levels):
    """Common per-level geometry of a multi-level fc container, or ``None``
    when it cannot be derived (missing attrs, non-uniform level ratios,
    non-constant window). Returns a list (shallow→deep) of dicts with
    ``fs`` (level sample rate), ``df`` (bin spacing), ``f_ceil`` (Fs/4) and
    ``f_floor`` (window/4) plus the cascade ``factor``."""
    items = sorted(levels.items())
    infos = []
    window = None
    for _, fc in items:
        fs = float(fc.attrs.get("sample_rate") or 0.0)
        if not fs:
            return None
        n = int(fc.attrs.get("num_samples_window") or 0)
        if not n:
            wl = fc.attrs.get("window_length_s")
            if not wl:
                return None
            n = round(float(wl) * fs)
        if window is None:
            window = n
        elif n != window:  # windows must be constant across levels (D2)
            return None
        infos.append({"fs": fs, "df": fs / n,
                      "f_ceil": fs / 4.0, "f_floor": 4.0 * fs / n})
    factor = 1
    rates = [i["fs"] for i in infos]
    if len(rates) > 1:
        ratios = [round(r / s) for r, s in pairwise(rates)]
        if len(set(ratios)) != 1 or ratios[0] < 2:
            return None
        factor = ratios[0]
    return {"levels": [lv for lv, _ in items], "infos": infos,
            "factor": factor, "window": window}


#: Reference band template in window-128 native bins (DESIGN.md §14.9):
#: the classic EMTF ladder the official USArray grid repeats verbatim at
#: levels 1-3 (L1-L3 = {[15-16],[12],[9-10],[7-8],[6],[5]}), hand-crafted
#: ~3 bands/octave. Scaled to the container's window (×window/128), the
#: SAME template is used at every decimation level — the periods scale by
#: the cascade factor per level (the official grid's structure).
_L2_TEMPLATE_128 = ((15, 16), (12, 12), (9, 10), (7, 8), (6, 6), (5, 5))


def _l2u_template(window):
    """The reference template scaled to the container's window (×window/128,
    rounded, overlaps merged, clamped to the valid bin range). At window 128
    and 256 this returns the official template family exactly (the official's
    window-256 template is the doubled form); other windows get the closest
    scaled equivalent."""
    scale = float(window) / 128.0
    out = []
    for k1, k2 in _L2_TEMPLATE_128:
        a = int(np.floor(k1 * scale + 0.5))
        b = int(np.floor(k2 * scale + 0.5))
        a = max(a, 1)
        b = min(b, int(window) // 2)
        if a <= b:
            if out and b >= out[-1][0]:  # descending k: overlap check
                out[-1] = (min(out[-1][0], a), max(out[-1][1], b))
            else:
                out.append((a, b))
    return tuple(out)


def _l2u_intervals(lv, template, geom):
    """Map the template's bin indices to frequency intervals for one level
    (same bins at every level; the level's own df scales the periods by the
    cascade factor per level — the official grid's structure)."""
    infos = geom["infos"]
    idx = geom["levels"].index(lv)
    df = infos[idx]["df"]
    return [((k2 + 0.5) * df, (k1 - 0.5) * df) for k1, k2 in template]


def _l2u_band_table(levels, min_bins_per_band=40.0, t_short=None):
    """Reference-template band table with log-uniform window-starvation
    coarsening (DESIGN.md §14.9, the default frequency-grid method).

    The reference template (:func:`_l2u_template`) is applied to EVERY
    decimation level (same bin indices; periods scale by the cascade factor
    per level). Per level, if every band's total samples (n_windows × band
    width) exceed ``min_bins_per_band``, the template is kept unchanged;
    otherwise the level's grid is REBUILT as a coarser uniform table:
    ``n - floor(n/3)`` bands, each widened to the minimal width whose total
    samples exceed the threshold, re-spaced uniformly in log period across
    the template's span (endpoints included), bands sharing bins dropped
    until no overlap or ≤2 bands remain. This reproduces the official
    USArray grid's strategy: the ppo-dense head untouched, the
    window-starved deepest levels coarsened to the official counts. The
    head clip ``t_short`` drops bands whose center period is shorter
    (default None = no clip).

    Returns ``{level: [(f_hi, f_lo), ...]}`` (frequency intervals,
    high→low), or ``None`` when the container lacks the metadata needed to
    derive it (callers fall back to the per-level bin walk).
    """
    geom = _container_geometry(levels)
    if geom is None:
        return None
    template = _l2u_template(geom["window"])
    out = {}
    for lv, fc in sorted(levels.items()):
        n_windows = int(fc.sizes.get("time", 0))
        if n_windows <= 0 or len(template) <= 2:
            out[lv] = _l2u_intervals(lv, template, geom)
            continue
        cur = [list(b) for b in template]
        totals = [n_windows * (k2 - k1 + 1) for k1, k2 in cur]
        if not min_bins_per_band or min(totals) > min_bins_per_band:
            out[lv] = _l2u_intervals(lv, template, geom)
            continue
        df = geom["infos"][geom["levels"].index(lv)]["df"]
        p_lo = 1.0 / (np.sqrt(cur[0][0] * cur[0][1]) * df)   # shortest period
        p_hi = 1.0 / (np.sqrt(cur[-1][0] * cur[-1][1]) * df)  # longest period
        k_lo_min = cur[-1][0]  # template span: k descends (short → long)
        k_hi_max = cur[0][1]
        w = int(np.floor(min_bins_per_band / n_windows)) + 1
        if n_windows * w <= min_bins_per_band:
            out[lv] = _l2u_intervals(lv, template, geom)  # cannot satisfy
            continue
        n_target = max(2, len(cur) - len(cur) // 3)
        chosen = None
        while n_target >= 2:
            # log-uniform placement covering the template's period span
            # (endpoints included — the coarsened grid keeps the coverage)
            centers = np.geomspace(p_lo, p_hi, n_target)
            bands = []
            for c in centers:
                kc = 1.0 / (c * df)
                k1 = int(np.floor(kc - (w - 1) / 2.0 + 0.5))
                k2 = k1 + w - 1
                # clamp into the template's bin span by SHIFTING (keep the
                # width — truncating would drop below the threshold)
                if k1 < k_lo_min:
                    k1, k2 = k_lo_min, k_lo_min + w - 1
                if k2 > k_hi_max:
                    k2, k1 = k_hi_max, k_hi_max - w + 1
                bands.append((k1, k2))
            bands.sort()
            overlap = any(bands[i][1] >= bands[i + 1][0]
                          for i in range(len(bands) - 1))
            chosen = bands
            if not overlap:
                break
            if n_target == 2:
                # two bands cannot fit disjointly at this width (the clamp
                # shifts both inward — extreme starvation): shrink the width
                # until they tile the span (best effort — the threshold can
                # no longer be met, the documented floor behavior)
                w = min(w, (k_hi_max - k_lo_min + 1) // 2)
                chosen = [(k_hi_max - w + 1, k_hi_max),
                          (k_lo_min, k_lo_min + w - 1)]
                break
            n_target -= 1
        # shortest → longest period (descending k), like the template path
        chosen = sorted(chosen, reverse=True)
        out[lv] = [((k2 + 0.5) * df, (k1 - 0.5) * df) for k1, k2 in chosen]
    return _clip_head_bands(out, t_short)


def _clip_head_bands(owned, t_short):
    """Head clip (§14.9): drop bands whose center period is shorter than
    ``t_short`` seconds (no-op for None/0 — legacy behavior)."""
    if not t_short or t_short <= 0:
        return owned
    out = {}
    for lv, pairs in owned.items():
        kept = [(hi, lo) for hi, lo in pairs
                if 1.0 / np.sqrt(hi * lo) >= t_short]
        if kept:
            out[lv] = kept
    return out


def _bands_from_freq_pairs(fc, pairs):
    """Map disjoint ``(f_hi, f_lo)`` intervals to integer bin positions on
    ``fc``'s OWN frequency axis (layout-agnostic: works with or without the
    DC bin, ``(frequency, time)`` or ``(time, frequency)`` — call after
    layout normalization). Bin membership is by frequency value, so bins are
    disjoint across bands automatically; a band with no bin in its interval
    is kept as an empty entry (callers skip empties). Band periods are the
    geometric means of the ACTUAL selected edge bins, so each reported
    frequency is self-consistent with the data that produced it — and since
    every band is owned by exactly one level, no cross-level label alignment
    is needed and the merge is a no-op.

    Returns ``(periods_out, band_bins)`` — periods shortest→longest.
    """
    freq_vals = np.asarray(fc["frequency"].values, dtype=float)
    periods_out = []
    band_bins = []
    for hi, lo in pairs:
        k_lo = int(np.searchsorted(freq_vals, lo, side="right"))
        k_hi = int(np.searchsorted(freq_vals, hi, side="right")) - 1
        if k_hi < k_lo:
            periods_out.append(np.nan)  # no bin of this fc in the interval
            band_bins.append(np.empty(0, dtype=int))
            continue
        periods_out.append(1.0 / np.sqrt(freq_vals[k_lo] * freq_vals[k_hi]))
        band_bins.append(np.arange(k_lo, k_hi + 1))
    return np.array(periods_out), band_bins


def _period_bands_indexed(fc, periods_per_octave=4):
    """Discrete, bin-snapped period bands (DESIGN.md §14.6) — the
    per-level fallback of the ``"l2u"`` scheme: used when the
    container-level table (:func:`_l2u_band_table`) could not be derived
    (fcs without metadata); each level then walks its own bin grid.

    Note: per-level walks give slightly different band labels per level, so
    a multi-level merge of these fallback bands can interleave
    near-duplicates at level overlaps (the historical "zigzag" defect) —
    only reached in the metadata-less degenerate case.

    Classic EMTF (and aurora's own default ``bs_256_26.cfg`` band-setup
    table) defines bands as explicit, non-overlapping ``index_min``/
    ``index_max`` integer ranges into the raw FFT-bin grid — hand-tuned once
    for a specific window length / decimation factor. :func:`_period_bands`
    instead partitions a *continuous* log2(period) grid and only afterwards
    tests which raw bins fall in each nominal band; because that grid is
    agnostic to the actual (linearly spaced) FFT-bin grid, nominal bands
    near the long-period end of a decimation level can silently collect
    zero raw bins (dropped after estimation), in an outcome that depends on
    floating-point edge coincidences rather than being controlled.

    This function instead walks the *integer* FFT-bin index grid directly,
    from ``index_ceiling`` (the Fs/4 cutoff, same as :func:`_period_bands`'s
    ``p_short``) down to ``index_floor`` (the window-length reliability cap,
    same as ``p_long``, clamped to the data's available range and to never
    include the DC bin), at a target ratio of ``2**(1/periods_per_octave)``
    per step, snapped to whole bins. This guarantees, by construction: every
    raw bin in ``[index_floor, index_ceiling]`` is assigned to exactly one
    band (no gaps, no overlaps, no silently-empty bands), and band width is
    monotonically non-decreasing toward higher frequency — matching the
    shape of aurora's bundled ``bs_256_26.cfg`` table, but derived
    automatically for any sample rate / window length / decimation factor
    instead of requiring a hand-authored config file tied to one specific
    configuration.

    Each band's period label is the geometric mean of the frequencies at its
    *actual* selected ``index_min``/``index_max`` (not a pre-imagined
    continuous grid point), so the reported period is always self-consistent
    with the raw data that produced it.

    Returns ``(periods_out, index_ranges)`` — periods shortest→longest, and
    one ``(index_min, index_max)`` inclusive integer pair per period, into
    the fc's raw (including-DC) frequency axis.
    """
    freq_vals = np.asarray(fc["frequency"].values, dtype=float)
    n_freq = freq_vals.size
    positive_mask = freq_vals > 0
    if not positive_mask.any():
        raise ValueError("fc has no positive-frequency bins to band")
    first_positive_idx = int(np.argmax(positive_mask))

    fs = float(fc.attrs.get("sample_rate") or 0.0)
    if not fs:
        fs = 2.0 * float(freq_vals[positive_mask].max())
    ppo = max(1, int(periods_per_octave))

    if n_freq > 1:
        delta_f = float((freq_vals[-1] - freq_vals[0]) / (n_freq - 1))
    else:
        delta_f = fs / 2.0

    def _index_of(frequency: float) -> int:
        pos = (frequency - freq_vals[0]) / delta_f if delta_f else 0.0
        return int(np.clip(round(pos), first_positive_idx, n_freq - 1))

    p_short = 4.0 / fs  # Fs/4, same reliability cutoff as _period_bands
    N = int(fc.attrs.get("num_samples_window") or 0)
    if not N:
        wl = fc.attrs.get("window_length_s")
        if wl:
            N = round(float(wl) * fs)
    p_long = N / (4.0 * fs) if N else None
    longest_present = 1.0 / float(freq_vals[positive_mask].min())
    if p_long is None or p_long > longest_present:
        p_long = longest_present
    if p_long <= p_short:
        p_long = longest_present

    index_ceiling = _index_of(1.0 / p_short)
    index_floor = _index_of(1.0 / p_long)
    if index_ceiling <= index_floor:
        index_ceiling = min(n_freq - 1, index_floor + 1)
    if index_ceiling <= index_floor:
        # only one usable bin in this level's reliable range
        only = index_floor
        return np.array([1.0 / freq_vals[only]]), [(only, only)]

    ratio = 2.0 ** (1.0 / ppo)
    index_ranges: list[tuple[int, int]] = []
    hi = index_ceiling
    while hi >= index_floor:
        lo = round(hi / ratio)
        lo = min(lo, hi - 1)
        lo = max(lo, index_floor)
        index_ranges.append((lo, hi))
        hi = lo - 1

    periods_out = np.array([
        1.0 / np.sqrt(freq_vals[lo] * freq_vals[hi]) for lo, hi in index_ranges
    ])
    return periods_out, index_ranges


def _resolve_period_bands(fc, periods_per_octave=4, band_style="l2u"):
    """Dispatch to the two band-construction schemes (DESIGN.md §14.9):
    the uniform log2-grid bands (:func:`_period_bands`) or the discrete
    bin-snapped partition (:func:`_period_bands_indexed` — the per-level
    fallback used when :func:`_l2u_band_table` could not derive the
    container-level table, e.g. for fcs without metadata; each level then
    walks its own bin grid).

    Returns ``(periods_out, band_bins)`` — ``band_bins[b]`` is an
    ``np.ndarray`` of integer positions into ``fc``'s frequency axis for
    band ``b`` (may be empty for the uniform style near a level's
    long-period end — callers already handle empty bands by skipping them).
    """
    style = _resolve_band_style(band_style)
    if style == "l2u":
        periods_out, index_ranges = _period_bands_indexed(fc, periods_per_octave)
        band_bins = [np.arange(lo, hi + 1) for lo, hi in index_ranges]
        return periods_out, band_bins
    periods_out, edges = _period_bands(fc, periods_per_octave)
    with np.errstate(divide="ignore"):
        logf = np.log2(fc["frequency"].values)
    band_bins = []
    for b in range(len(periods_out)):
        sel = (logf <= edges[b]) & (logf > edges[b + 1])
        band_bins.append(np.where(sel)[0])
    return periods_out, band_bins


def _band_datasets(H, E, R, input_channels, output_channels):
    """Build xarray Datasets for aurora's regression from stacked band samples.

    Returns ``(X, Y, Z)`` where ``Z`` is ``None`` unless a remote reference is
    in use. Arrays are shaped [n_obs, n_in] / [n_obs, n_out]; aurora expects
    channels as data_vars and handles any number of outputs (so a bz/hz output
    is estimated as a tipper row as well as the ex/ey impedance).
    """
    X = _ds({ic: H[:, i] for i, ic in enumerate(input_channels)})
    Y = _ds({oc: E[:, i] for i, oc in enumerate(output_channels)})
    Z = None
    if R is not None:
        Z = _ds({ic: R[:, i] for i, ic in enumerate(input_channels)})
    return X, Y, Z


def _ds(channels):
    import xarray as xr

    return xr.Dataset({name: ("obs", col) for name, col in channels.items()})


def _solve_band(H, E, R, input_channels, output_channels, engine):
    """Estimate one frequency band via an aurora estimator (OLS/RME/RME_RR).

    Returns ``(Z, Z_err)`` — Z[2x2] in emptyUI's (output, input) orientation,
    already in aurora's impedance convention (aurora's b is
    [n_input, n_output]), and the standard error of each complex impedance
    element (same orientation). The errors follow aurora's own
    ``TTFZ.standard_error()``: ``std = sqrt(|cov_ss_inv[i,i] * cov_nn[o,o]|)``
    where ``cov_ss_inv`` is the inverse signal covariance of the inputs and
    ``cov_nn`` the noise covariance of the outputs. RME/RME_RR populate these
    during estimation (``IterControl(return_covariance=True)``, the default);
    OLS does not, so the classical covariance ``(XᴴX)⁻¹·σ²`` is computed from
    the residuals instead.
    """
    from aurora.transfer_function.regression.base import RegressionEstimator
    from aurora.transfer_function.regression.iter_control import IterControl
    from aurora.transfer_function.regression.RME import RME
    from aurora.transfer_function.regression.RME_RR import RME_RR

    X, Y, Z = _band_datasets(H, E, R, input_channels, output_channels)
    if engine == "OLS":
        est = RegressionEstimator(X, Y)
        est.estimate_ols(mode="solve")
        Z_err = _ols_band_errors(est, len(output_channels))
    elif engine == "RME_RR":
        est = RME_RR(X=X, Y=Y, Z=Z, iter_control=IterControl())
        est.estimate()
        Z_err = _estimator_band_errors(est, input_channels, output_channels)
    else:
        est = RME(X=X, Y=Y, iter_control=IterControl())
        est.estimate()
        Z_err = _estimator_band_errors(est, input_channels, output_channels)
    return np.asarray(est.b).T, Z_err


def _cov_diag(values, channel_names, channels):
    """Diagonal of a covariance matrix re-ordered to ``channels`` order.

    aurora's covariance DataArrays use its own channel-name coordinate order;
    the values array is indexed by the *declared* name order, so this maps our
    channel list onto it defensively.
    """
    idx = [channel_names.index(c) for c in channels]
    return np.asarray(values)[np.ix_(idx, idx)]


def _cov_coords(cov):
    """Channel names from an aurora covariance DataArray coordinate."""
    for key in ("input_channel_1", "output_channel_1"):
        if key in cov.coords:
            return list(cov.coords[key].values)
    return list(cov.dims[0] if cov.dims else [])


def _std_from_cov(cov_ss_inv, cov_nn, input_channels, output_channels):
    """aurora ``TTFZ.standard_error()``: sqrt(|cov_ss_inv[i,i] * cov_nn[o,o]|)."""
    n_in, n_out = len(input_channels), len(output_channels)
    std = np.zeros((n_out, n_in), dtype=float)
    if cov_ss_inv is None or cov_nn is None:
        return std
    cs = _cov_diag(cov_ss_inv.values, _cov_coords(cov_ss_inv), input_channels)
    cn = _cov_diag(cov_nn.values, _cov_coords(cov_nn), output_channels)
    for o in range(n_out):
        for i in range(n_in):
            std[o, i] = np.sqrt(abs(cs[i, i] * cn[o, o]))
    return std


def _estimator_band_errors(est, input_channels, output_channels):
    """Standard errors from an RME/RME_RR estimator's computed covariances."""
    return _std_from_cov(est.cov_ss_inv, est.cov_nn, input_channels, output_channels)


def _ols_band_errors(est, n_out):
    """Classical OLS standard errors from residuals: (XᴴX)⁻¹ · σ².

    aurora's OLS path does not populate covariances, so the textbook
    covariance is computed here (same form aurora uses for RME):
    ``cov_ss_inv = (XᴴX)⁻¹`` and ``cov_nn = diag(residual SS) / dof``.
    """
    X = np.asarray(est.X, dtype=complex)
    Y = np.asarray(est.Y, dtype=complex)
    b = np.asarray(est.b, dtype=complex)
    n_obs, n_in = X.shape
    dof = n_obs - n_in
    if dof <= 0 or n_out == 0:
        return np.zeros((n_out, n_in), dtype=float)
    cs = np.linalg.inv(X.conj().T @ X)
    res = Y - X @ b  # [n_obs, n_out]
    ssr = (res.conj().T @ res).real  # [n_out, n_out]
    cn = ssr / dof
    std = np.zeros((n_out, n_in), dtype=float)
    for o in range(n_out):
        for i in range(n_in):
            std[o, i] = np.sqrt(abs(cs[i, i].real * cn[o, o]))
    return std


_OUT_COORDS = tuple(STANDARD_OUTPUTS)  # ("ex", "ey", "hz")
_IN_COORDS = tuple(STANDARD_INPUTS)  # ("hx", "hy")


def build_tf(periods, Z, input_channels, output_channels, channel_nomenclature=None,
             Z_err=None, survey_id=None, station_id=None, run_id=None,
             latitude=None, longitude=None, elevation=None, start_datetime=None,
             processing_notes=None):
    """Build an mt_metadata TF carrying the actual channel names plus the
    channel_nomenclature map (aurora-style).

    mt_metadata's TF uses ``channel_nomenclature`` (standard -> actual) to
    define its coordinate names — ``tf.ex``/``tf.hx`` return the *actual*
    instrument names and ``has_impedance``/``impedance``/``tipper`` are all
    nomenclature-aware. The TF is therefore built with the actual names as
    coordinates and the map attached, exactly like aurora's
    ``TF(channel_nomenclature=...)``, so aurora tooling (``TTFZ``,
    ``TransferFunctionCollection``) can consume it directly. Exporters that
    require standard names (EDI/XML) standardize on write; ``ExportTF`` maps
    back for the ZMM writer.

    The actual channel names are resolved through the mt_metadata CHANNEL_MAPS
    reverse mapping for the given nomenclature keyword: lemi12 maps e1/e2 (and
    bx/by/bz), lemi34 maps e3/e4, phoenix123 maps e1/e2/h1/h2/h3, default maps
    ex/ey/hx/hy/hz. When no keyword is given it is inferred from the channel
    names. Both the impedance (ex,ey rows) and the tipper (hz row) are
    populated when the corresponding output channel is present.

    ``Z_err`` (optional, same shape as ``Z``) is stored as
    ``tf.transfer_function_error`` (the std of each complex element) so plots
    and exports (EDI/XML/ZMM) can carry error bars / error lines. The station
    identity (survey/station/run id + lat/lon/elevation) is attached to
    ``tf.survey_metadata``/``tf.station_metadata`` — the single point of truth
    for plot titles and export headers (DESIGN.md §15).
    """
    import xarray as xr
    from mt_metadata.transfer_functions.core import TF, Station, Survey

    keyword = _resolve_keyword(
        channel_nomenclature, input_channels, output_channels
    )
    mapping = channel_map(keyword)  # standard -> actual

    out_coords = [mapping[s] for s in _OUT_COORDS]
    in_coords = [mapping[s] for s in _IN_COORDS]
    reverse = {v.lower(): k for k, v in mapping.items()}

    data = np.zeros((len(periods), len(_OUT_COORDS), len(_IN_COORDS)), dtype=complex)
    for ri, oc in enumerate(output_channels):
        std = reverse.get(str(oc).strip().lower())
        if std is None or std not in _OUT_COORDS:
            continue
        for ci, ic in enumerate(input_channels):
            ist = reverse.get(str(ic).strip().lower())
            if ist is None or ist not in _IN_COORDS:
                continue
            data[:, _OUT_COORDS.index(std), _IN_COORDS.index(ist)] = Z[:, ri, ci]

    tf = TF()
    tf.channel_nomenclature = {s: mapping[s] for s in ("ex", "ey", "hx", "hy", "hz")}
    arr = xr.DataArray(
        data,
        dims=("period", "output", "input"),
        coords={
            "period": periods,
            "output": out_coords,
            "input": in_coords,
        },
    )
    # use the property setter (tf.dataset[...] = arr writes to the xarray
    # Dataset only, not the TF's canonical impedance storage)
    tf.transfer_function = arr
    if Z_err is not None:
        err = np.zeros((len(periods), len(_OUT_COORDS), len(_IN_COORDS)), dtype=float)
        for ri, oc in enumerate(output_channels):
            std = reverse.get(str(oc).strip().lower())
            if std is None or std not in _OUT_COORDS:
                continue
            for ci, ic in enumerate(input_channels):
                ist = reverse.get(str(ic).strip().lower())
                if ist is None or ist not in _IN_COORDS:
                    continue
                err[:, _OUT_COORDS.index(std), _IN_COORDS.index(ist)] = Z_err[:, ri, ci]
        tf.transfer_function_error = xr.DataArray(
            err,
            dims=("period", "output", "input"),
            coords={
                "period": periods,
                "output": out_coords,
                "input": in_coords,
            },
        )
    if station_id or survey_id:
        if survey_id:
            surv = Survey()
            surv.id = survey_id
            tf.survey_metadata = surv
        st = Station()
        st.id = station_id or "local"
        if latitude is not None:
            st.location.latitude = float(latitude)
        if longitude is not None:
            st.location.longitude = float(longitude)
        if elevation is not None:
            st.location.elevation = float(elevation)
        tf.station_metadata = st
    if run_id or processing_notes:
        rm = tf.run_metadata
        if run_id:
            rm.id = run_id
        if processing_notes:
            rm.comments.value = processing_notes
    return tf


def estimate_tf_levels(
    levels,
    input_channels,
    output_channels,
    estimator_engine="Robust",
    remote_reference=False,
    periods_per_octave=4,
    ref_levels=None,
    channel_nomenclature=None,
    band_style="l2u",
    min_bins_per_band=40.0,
    t_short=None,
):
    """Estimate a per-level TF for every decimation level in a multi-level fc
    container.

    Returns ``{level: {"periods": P, "Z": Z[2x2], "Z_err": Z_err,
    "input_channels": ..., "output_channels": ..., "channel_nomenclature": ...,
    "survey_id"/"station_id"/"run_id"/"station_latitude"/...: ...}}``.
    ``ref_levels`` is an optional same-shaped container (or a single fc) used
    as the remote reference per level. ``band_style`` selects the band
    construction scheme (DESIGN.md §14.9):

    - ``"l2u"`` (default): the reference-template band table with
      log-uniform window-starvation coarsening (:func:`_l2u_band_table`) —
      the same fixed template at every decimation level; starved levels are
      rebuilt as a coarser uniform grid, nothing to merge.
    - ``"uniform"``: uniform log2-grid partition (:func:`_period_bands`) —
      overlapping levels estimate the same nominal bands and
      ``merge_tf_levels`` keeps the deepest.

    For ``"l2u"``, band construction happens once for the whole container;
    if the container's metadata is insufficient (missing attrs, non-uniform
    level ratios) the per-level bin walk is used as a fallback.
    ``min_bins_per_band`` (window-starvation threshold on total samples per
    band) and ``t_short`` (head clip) refine the table per DESIGN.md §14.9;
    the latter is a no-op at its default (None).
    """
    keyword = _resolve_keyword(channel_nomenclature, input_channels, output_channels)
    bands_override = None
    style = _resolve_band_style(band_style)
    if style == "l2u":
        bands_override = _l2u_band_table(
            levels, min_bins_per_band=min_bins_per_band, t_short=t_short)
    result = {}
    for level, fc in sorted(levels.items()):
        if bands_override is not None:
            bands = bands_override.get(level, [])
            if not bands:
                continue  # head clip / coarsening left this level no bands
        else:
            bands = None
        ref_fc = _level_ref(ref_levels, level)
        periods, Z, Z_err = estimate_tf(
            fc, input_channels, output_channels,
            estimator_engine=estimator_engine,
            remote_reference=remote_reference,
            periods_per_octave=periods_per_octave,
            ref_fc=ref_fc,
            band_style=band_style,
            bands_override=bands,
        )
        entry = {
            "periods": periods,
            "Z": Z,
            "Z_err": Z_err,
            "input_channels": list(input_channels),
            "output_channels": list(output_channels),
            "channel_nomenclature": keyword,
        }
        entry.update(_identity_attrs(fc))
        result[int(level)] = entry
    return result


def _identity_attrs(fc):
    """Station identity attrs carried on an fc (set by ComputeFC)."""
    return {
        k: fc.attrs[k]
        for k in ("survey_id", "station_id", "run_id", "station_latitude",
                  "station_longitude", "station_elevation", "start_datetime")
        if fc.attrs.get(k) not in (None, "")
    }


def merge_tf_levels(per_level):
    """Merge per-decimation-level TF estimates into a single broadband
    (periods, Z, Z_err).

    Levels now share a common octave grid and overlap (each covers from Fs/4 up
    to window/4), so the same period can be estimated by several levels. Because
    a deeper level has a longer window (more cycles of a long period), for each
    period we keep the estimate from the **deepest** level that covers it;
    degenerate (all-zero) bands are dropped. The resulting grid is strictly
    ascending and free of duplicates.
    """
    merged: dict[float, np.ndarray] = {}
    merged_err: dict[float, np.ndarray] = {}
    for lv in sorted(per_level, reverse=True):  # deepest level wins
        P = np.asarray(per_level[lv]["periods"])
        if P.size == 0:
            continue  # a level may legitimately contribute nothing (clipped)
        Z = np.asarray(per_level[lv]["Z"])
        err = per_level[lv].get("Z_err")
        if err is None:
            # a hand-built level payload may carry no errors — treat as zeros
            err = np.zeros(len(Z))
        else:
            err = np.asarray(err, dtype=float)
            if err.ndim == 0:
                err = np.full(len(Z), float(err))
        ok = ~np.all(np.abs(Z).reshape(len(Z), -1) == 0, axis=1)
        for p, z, e in zip(P[ok], Z[ok], err[ok]):
            key = round(float(np.log2(p)), 8)  # shared log2 grid -> exact dedup
            merged.setdefault(key, z)
            merged_err.setdefault(key, e)
    if not merged:
        raise ValueError("no non-degenerate TF bands to merge")
    keys = sorted(merged)
    P = 2.0 ** np.asarray(keys)
    Z = np.stack([merged[k] for k in keys])
    Z_err = np.stack([merged_err[k] for k in keys])
    return P, Z, Z_err
