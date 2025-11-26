"""Shared processing helpers for the aurora plugin.

ComputeFC/EstimateTF split the aurora pipeline at the FC boundary. This module
holds the signal-processing core: RunTS/TimeSeries normalization, windowed
STFT to Fourier coefficients (fc), and MT impedance estimation (OLS / robust
RME / remote-reference RME_RR, all aurora-backed via :func:`_solve_band`).
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
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
        raise ValueError(
            f"no output channels available — requested {output_channels!r} "
            f"against data {available}"
        )
    return ins, outs, keyword


def to_channels(value):
    """Normalize a ts.run payload to (channels: dict[str, np.ndarray],
    sample_rate: float, start_s: float, start_dt: pd.Timestamp | None).

    Accepts:
      - mth5 RunTS (has .dataset xarray, .sample_rate, .start datetime)
      - core TimeSeries (has .channels dict, .sample_rate, .start float s)
    """
    # mth5 RunTS / mt_timeseries
    if hasattr(value, "dataset") and not isinstance(value, dict):
        ds = value.dataset
        sample_rate = float(value.sample_rate)
        start_s = 0.0
        start_dt = _start_datetime(getattr(value, "start", None))
        channels = {str(k): np.asarray(v.values, dtype=float) for k, v in ds.data_vars.items()}
        return channels, sample_rate, start_s, start_dt
    # core-nodes TimeSeries
    if hasattr(value, "channels") and hasattr(value, "sample_rate"):
        sample_rate = float(value.sample_rate)
        start_s = float(getattr(value, "start", 0.0))
        start_dt = _start_datetime(
            getattr(value, "start_datetime", None) or getattr(value, "start", None)
        )
        channels = {str(k): np.asarray(v, dtype=float) for k, v in value.channels.items()}
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
    converted B->H by dividing by μ0) or "h" (coil sensors already give H).
    An explicit "h"/"b" always wins (legacy workflows keep working).
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

    channels, sample_rate, start_s, src_start = to_channels(value)
    if start_datetime is None:
        start_datetime = src_start
    nperseg = max(8, int(window_length_s * sample_rate))
    noverlap = int(nperseg * overlap)
    win = _window(window_type)
    input_channels, output_channels, keyword = _resolve_io_channels(
        list(channels), channel_nomenclature, input_channels, output_channels
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
            if b_to_h:
                # LEMI fluxgate B is in nT; H [A/m] = B[T]/mu0 = B[nT]*1e-9/mu0
                x = x * 1e-9 / MU0
            # else: coil sensor already gives H [A/m], unchanged
        elif is_magnetic_name(comp):  # vertical magnetic output (bz/hz) — tipper
            # same units as the magnetic inputs, so Bz/H is dimensionless
            if b_to_h:
                x = x * 1e-9 / MU0
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

    channels, sample_rate, start_s, start_dt = to_channels(value)
    if start_datetime is None:
        start_datetime = start_dt
    input_channels, output_channels, keyword = _resolve_io_channels(
        list(channels), channel_nomenclature, input_channels, output_channels
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
            _normalize_channels(cur, input_channels, output_channels, b_to_h, e_scale)
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


def _normalize_channels(channels, input_channels, output_channels, b_to_h, e_scale):
    """Apply instrument unit normalization (B->H, mV->V/m) to a channel dict."""
    out = {}
    for comp, x in channels.items():
        x = np.asarray(x, dtype=float)
        if comp in input_channels and b_to_h:
            x = x * 1e-9 / MU0  # fluxgate B [nT] -> H [A/m]
        elif comp in output_channels:
            if is_magnetic_name(comp):  # vertical magnetic output (tipper)
                if b_to_h:
                    x = x * 1e-9 / MU0  # Bz -> Hz, same units as inputs
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


def estimate_tf(
    fc,
    input_channels,
    output_channels,
    estimator_engine="RME_RR",
    remote_reference=False,
    periods_per_octave=4,
    ref_fc=None,
):
    """MT transfer-function estimate -> (periods, Z[n_out x n_in complex]).

    Z rows = outputs (e.g. ex,ey and optionally bz/hz), cols = inputs (hx,hy).
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

    Period bands are **octave-aligned** (:func:`_period_bands`): ``periods_per_octave``
    bands per power-of-2 octave, spanning from the level's Nyquist period up to
    the shorter of the next level's Nyquist period (an exact octave boundary) or
    the window-length reliability cap — so the extremes (longest period = window
    length, mixing with DC) are avoided and adjacent decimation levels share one
    period for a clean merge.

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
    freq = freq[ok]
    fc = fc.isel(frequency=ok)
    logf = np.log2(freq)

    if ref_fc is not None:
        ref_fc = _normalize_fc_layout(ref_fc)
        ref_fc = ref_fc.isel(frequency=ref_fc["frequency"].values > 0)
        if not np.array_equal(ref_fc["frequency"].values, freq):
            raise ValueError(
                "remote-reference fc has a different frequency grid than the "
                "local fc — compute both with the same window settings"
            )

    # Realistic, octave-aligned period bands for this decimation level.
    periods_out, edges = _period_bands(fc, periods_per_octave)

    nbands = len(periods_out)
    n_in = len(input_channels)
    n_out = len(output_channels)
    # Z rows = outputs (ex,ey,...,bz), cols = inputs (hx,hy): a 2x2 impedance
    # when outputs are ex/ey, extended to a 3x2 (with a bz/hz tipper row) when a
    # vertical channel is included.
    Zout = np.zeros((nbands, n_out, n_in), dtype=complex)

    for b in range(nbands):
        sel = (logf <= edges[b]) & (logf > edges[b + 1])
        if sel.sum() == 0:
            continue
        # gather samples across selected frequencies and all windows
        H = []  # inputs  -> [n, n_in]
        E = []  # outputs -> [n, n_out]
        R = []  # reference inputs -> [n, n_in]
        for f_idx in np.where(sel)[0]:
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
        if H.shape[0] < 2:  # too few observations for a stable estimate
            continue
        R_band = np.concatenate(R, axis=0) if engine == "RME_RR" else None
        Zout[b] = _solve_band(H, E, R_band, input_channels, output_channels, engine)

    # All estimators now come from aurora and share aurora's impedance
    # convention, so no sign/phase rotation is applied.
    return periods_out, Zout


def _period_bands(fc, periods_per_octave=4):
    """Realistic, octave-aligned period bands -> (band-center periods, edges).

    Per the feedback: each level's highest trusted frequency is **Fs/4** (so the
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

    Returns Z[2x2] in emptyUI's (output, input) orientation, already in
    aurora's impedance convention (aurora's b is [n_input, n_output]).
    """
    from aurora.transfer_function.regression.base import RegressionEstimator
    from aurora.transfer_function.regression.iter_control import IterControl
    from aurora.transfer_function.regression.RME import RME
    from aurora.transfer_function.regression.RME_RR import RME_RR

    X, Y, Z = _band_datasets(H, E, R, input_channels, output_channels)
    if engine == "OLS":
        est = RegressionEstimator(X, Y)
        est.estimate_ols(mode="solve")
    elif engine == "RME_RR":
        est = RME_RR(X=X, Y=Y, Z=Z, iter_control=IterControl())
        est.estimate()
    else:
        est = RME(X=X, Y=Y, iter_control=IterControl())
        est.estimate()
    return np.asarray(est.b).T


_OUT_COORDS = tuple(STANDARD_OUTPUTS)  # ("ex", "ey", "hz")
_IN_COORDS = tuple(STANDARD_INPUTS)  # ("hx", "hy")


def build_tf(periods, Z, input_channels, output_channels, channel_nomenclature=None):
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
    """
    import xarray as xr
    from mt_metadata.transfer_functions.core import TF

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
):
    """Estimate a per-level TF for every decimation level in a multi-level fc
    container.

    Returns ``{level: {"periods": P, "Z": Z[2x2], "input_channels": ...,
    "output_channels": ..., "channel_nomenclature": ...}}``. ``ref_levels`` is
    an optional same-shaped container (or a single fc) used as the remote
    reference per level.
    """
    keyword = _resolve_keyword(channel_nomenclature, input_channels, output_channels)
    result = {}
    for level, fc in sorted(levels.items()):
        ref_fc = ref_levels.get(level) if isinstance(ref_levels, dict) else ref_levels
        periods, Z = estimate_tf(
            fc, input_channels, output_channels,
            estimator_engine=estimator_engine,
            remote_reference=remote_reference,
            periods_per_octave=periods_per_octave,
            ref_fc=ref_fc,
        )
        result[int(level)] = {
            "periods": periods,
            "Z": Z,
            "input_channels": list(input_channels),
            "output_channels": list(output_channels),
            "channel_nomenclature": keyword,
        }
    return result


def merge_tf_levels(per_level):
    """Merge per-decimation-level TF estimates into a single broadband
    (periods, Z).

    Levels now share a common octave grid and overlap (each covers from Fs/4 up
    to window/4), so the same period can be estimated by several levels. Because
    a deeper level has a longer window (more cycles of a long period), for each
    period we keep the estimate from the **deepest** level that covers it;
    degenerate (all-zero) bands are dropped. The resulting grid is strictly
    ascending and free of duplicates.
    """
    merged: dict[float, np.ndarray] = {}
    for lv in sorted(per_level, reverse=True):  # deepest level wins
        P = np.asarray(per_level[lv]["periods"])
        Z = np.asarray(per_level[lv]["Z"])
        ok = ~np.all(np.abs(Z).reshape(len(Z), -1) == 0, axis=1)
        for p, z in zip(P[ok], Z[ok]):
            key = round(float(np.log2(p)), 8)  # shared log2 grid -> exact dedup
            merged.setdefault(key, z)
    if not merged:
        raise ValueError("no non-degenerate TF bands to merge")
    keys = sorted(merged)
    P = 2.0 ** np.asarray(keys)
    Z = np.stack([merged[k] for k in keys])
    return P, Z
