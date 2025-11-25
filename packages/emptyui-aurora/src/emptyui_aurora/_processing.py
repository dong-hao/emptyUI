"""Shared processing helpers for the aurora plugin.

ComputeFC/EstimateTF split the aurora pipeline at the FC boundary. This module
holds the signal-processing core: RunTS/TimeSeries normalization, windowed
STFT to Fourier coefficients (fc), and MT impedance estimation (OLS / robust
RME / remote-reference RME_RR, all aurora-backed via :func:`_solve_band`).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

MU0 = 4.0 * np.pi * 1e-7


def _is_electric(name: str) -> bool:
    """True for electric (E) output component names (ex,ey,e1,e2,...)."""
    n = name.strip().lower()
    return n in ("ex", "ey", "e1", "e2", "e3") or (
        len(n) >= 2 and n[0] == "e" and n[1].isdigit()
    )


def _is_magnetic(name: str) -> bool:
    """True for magnetic (B/H) component names (bx,by,bz,hx,hy,hz)."""
    return name.strip().lower() in ("hx", "hy", "hz", "bx", "by", "bz")


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
    magnetic_unit="h",
    e_dipole_length_m=None,
    start_datetime=None,
):
    """Windowed STFT -> fc xarray Dataset (dims frequency x time).

    The input is assumed to be already detrended (use the Detrend node to
    remove the DC/absolute-field baseline first — fluxgate magnetometers
    report a huge DC offset that would otherwise swamp the autopower and drive
    Z -> 0).

    ``magnetic_unit``: "h" (coil sensors already give H) or "b" (fluxgate
    gives absolute B) — for "b", input channels are converted B->H (divide by
    μ0) so the fc always carries H.

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
    b_to_h = magnetic_unit == "b"
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
        elif _is_magnetic(comp):  # vertical magnetic output (bz/hz) — tipper
            # same units as the magnetic inputs, so Bz/H is dimensionless
            if b_to_h:
                x = x * 1e-9 / MU0
        elif _is_electric(comp):  # electric output channel
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
            "magnetic_unit": magnetic_unit,
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
    magnetic_unit="h",
    e_dipole_length_m=None,
    start_datetime=None,
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
    b_to_h = magnetic_unit == "b"
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
                "magnetic_unit": magnetic_unit,
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
            if _is_magnetic(comp):  # vertical magnetic output (tipper)
                if b_to_h:
                    x = x * 1e-9 / MU0  # Bz -> Hz, same units as inputs
            elif _is_electric(comp):
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
    """

    engine = _resolve_estimator(estimator_engine, remote_reference)

    freq = fc["frequency"].values
    ok = freq > 0  # exclude DC bin (period would be infinite)
    freq = freq[ok]
    fc = fc.isel(frequency=ok)
    logf = np.log2(freq)

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
    fs = float(fc.attrs["sample_rate"])
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


_OUT_COORDS = ["ex", "ey", "hz"]
_IN_COORDS = ["hx", "hy"]
#: Map user channel names onto mt_metadata's standard TF positions.
_OUT_NAME = {
    "e1": "ex", "ex": "ex",
    "e2": "ey", "ey": "ey", "e3": "ey",
    "bz": "hz", "hz": "hz",
}
_IN_NAME = {
    "bx": "hx", "hx": "hx",
    "by": "hy", "hy": "hy",
}


def build_tf(periods, Z, input_channels, output_channels):
    """Build an mt_metadata TF with standard ex/ey/hz x hx/hy coordinates.

    User channel names are mapped onto the standard positions: electric
    e1/e2 (or ex/ey) -> ex/ey, vertical bz/hz -> hz; magnetic bx/by (or
    hx/hy) -> hx/hy. Both the impedance (ex,ey rows) and the tipper (hz row)
    are populated when the corresponding output channel is present.
    """
    import xarray as xr
    from mt_metadata.transfer_functions.core import TF

    out_rows = [_OUT_NAME.get(c.strip().lower()) for c in output_channels]
    in_cols = [_IN_NAME.get(c.strip().lower()) for c in input_channels]

    data = np.zeros((len(periods), len(_OUT_COORDS), len(_IN_COORDS)), dtype=complex)
    for ri, oc in enumerate(out_rows):
        if oc is None or oc not in _OUT_COORDS:
            continue
        for ci, ic in enumerate(in_cols):
            if ic is None or ic not in _IN_COORDS:
                continue
            data[:, _OUT_COORDS.index(oc), _IN_COORDS.index(ic)] = Z[:, ri, ci]

    tf = TF()
    arr = xr.DataArray(
        data,
        dims=("period", "output", "input"),
        coords={
            "period": periods,
            "output": _OUT_COORDS,
            "input": _IN_COORDS,
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
):
    """Estimate a per-level TF for every decimation level in a multi-level fc
    container.

    Returns ``{level: {"periods": P, "Z": Z[2x2], "input_channels": ...,
    "output_channels": ...}}``. ``ref_levels`` is an optional same-shaped
    container (or a single fc) used as the remote reference per level.
    """
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
