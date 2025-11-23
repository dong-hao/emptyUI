"""Shared processing helpers for the aurora plugin.

ComputeFC/EstimateTF split the aurora pipeline at the FC boundary. This module
holds the signal-processing core: RunTS/TimeSeries normalization, windowed
STFT to Fourier coefficients (fc), and a first-pass MT impedance estimate from
cross-powers.

Estimator note: the current EstimateTF uses a least-squares (LSQ) MT impedance
estimate per period band. Robust (RME/RME_RR) regression is a planned upgrade;
the architecture keeps the estimator swappable inside EstimateTF.
"""

from __future__ import annotations

import numpy as np

MU0 = 4.0 * np.pi * 1e-7


def to_channels(value):
    """Normalize a ts.run payload to (channels: dict[str, np.ndarray],
    sample_rate: float, start_s: float).

    Accepts:
      - mth5 RunTS (has .dataset xarray, .sample_rate)
      - core TimeSeries (has .channels dict, .sample_rate)
    """
    # mth5 RunTS / mt_timeseries
    if hasattr(value, "dataset") and not isinstance(value, dict):
        ds = value.dataset
        sample_rate = float(value.sample_rate)
        start_s = 0.0
        channels = {str(k): np.asarray(v.values, dtype=float) for k, v in ds.data_vars.items()}
        return channels, sample_rate, start_s
    # core-nodes TimeSeries
    if hasattr(value, "channels") and hasattr(value, "sample_rate"):
        sample_rate = float(value.sample_rate)
        start_s = float(getattr(value, "start", 0.0))
        channels = {str(k): np.asarray(v, dtype=float) for k, v in value.channels.items()}
        return channels, sample_rate, start_s
    raise TypeError(f"cannot normalize ts.run value of type {type(value).__name__}")


def compute_fc(
    value,
    input_channels,
    output_channels,
    window_length_s,
    overlap,
    window_type,
    magnetic_unit="h",
    e_dipole_length_m=None,
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
    """
    import xarray as xr
    from scipy.signal import stft

    channels, sample_rate, start_s = to_channels(value)
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
        else:  # output (electric) channel
            if e_scale is not None:
                # raw potential (mV) -> V/m via dipole length
                x = x * e_scale
            else:
                # LEMI E in mV/km == uV/m == 1e-6 V/m
                x = x * 1e-6
        f, t, z = stft(
            x, fs=sample_rate, nperseg=nperseg, noverlap=noverlap,
            window=win, boundary=None, padded=False,
        )
        fc_vars[comp] = (("frequency", "time"), z)

    fc = xr.Dataset(
        data_vars=fc_vars,
        coords={"frequency": f, "time": t + start_s},
        attrs={
            "sample_rate": float(sample_rate),
            "window_length_s": float(window_length_s),
            "overlap": float(overlap),
            "window_type": window_type,
            "input_channels": ",".join(input_channels),
            "output_channels": ",".join(output_channels),
            "magnetic_unit": magnetic_unit,
            "e_dipole_length_m": e_dipole_length_m or 0.0,
        },
    )
    return fc


def _window(name: str) -> str:
    """Map our window_type vocabulary to scipy window names."""
    return {"hanning": "hann"}.get(name, name)


def estimate_tf(
    fc,
    input_channels,
    output_channels,
    estimator_engine="RME_RR",
    remote_reference=False,
    periods_per_decade=8,
):
    """First-pass LSQ MT impedance estimate -> (periods, Z[2x2 complex]).

    Z rows = outputs (-> ex,ey), cols = inputs (-> hx,hy). The fc already
    holds H for the magnetic inputs (B->H done in compute_fc when the
    instrument is fluxgate), so impedance is in ohm (V/m per A/m).
    """

    if remote_reference:
        raise ValueError("remote reference not implemented yet (single station)")

    freq = fc["frequency"].values
    ok = freq > 0  # exclude DC bin (period would be infinite)
    freq = freq[ok]
    period = 1.0 / freq
    # subset the fc data along frequency
    fc = fc.isel(frequency=ok)

    # bin periods logarithmically
    logp = np.log10(period)
    pmin, pmax = np.min(logp), np.max(logp)
    nbands = max(1, int(np.ceil((pmax - pmin) * periods_per_decade)))
    edges = np.linspace(pmax, pmin, nbands + 1)  # high->low period
    periods_out = 10.0 ** ((edges[:-1] + edges[1:]) / 2.0)

    Zout = np.zeros((nbands, 2, 2), dtype=complex)

    for b in range(nbands):
        sel = (logp <= edges[b]) & (logp > edges[b + 1])
        if sel.sum() == 0:
            continue
        # gather samples across selected frequencies and all windows
        H = []  # inputs (hx, hy) -> [n, 2]
        E = []  # outputs (ex, ey) -> [n, 2]
        for f_idx in np.where(sel)[0]:
            hx = fc[input_channels[0]].values[f_idx, :]  # already H
            hy = fc[input_channels[1]].values[f_idx, :]
            ex = fc[output_channels[0]].values[f_idx, :]
            ey = fc[output_channels[1]].values[f_idx, :]
            H.append(np.stack([hx, hy], axis=-1))
            E.append(np.stack([ex, ey], axis=-1))
        H = np.concatenate(H, axis=0)
        E = np.concatenate(E, axis=0)
        # least squares: Z = (E H^H)(H H^H)^{-1} (pseudo-inverse is robust
        # to ill-conditioned / near-singular autopower for sparse bands)
        EhH = E.conj().T @ H
        HhH = H.conj().T @ H
        Zout[b] = EhH @ np.linalg.pinv(HhH)

    # MT impedance sign/phase convention (verified against aurora on the
    # synthetic halfspace test12rr): with E = Z·H, our raw least-squares solve
    # is +90° from the mt_metadata/aurora convention (which reports Zxy phase
    # ~+45°, Zyx ~-135° for a halfspace). Rotating by +90° (x j) aligns us
    # with aurora/EDI.
    Zout = 1j * Zout

    return periods_out, Zout


def build_tf(periods, Z, input_channels, output_channels):
    """Build an mt_metadata TF with standard ex/ey/hx/hy coordinates."""
    import xarray as xr
    from mt_metadata.transfer_functions.core import TF

    # mt_metadata impedance conventions: output dims ex,ey,hz; input dims hx,hy
    out_coords = ["ex", "ey", "hz"]
    in_coords = ["hx", "hy"]
    # our Z rows map to ex,ey; cols map to hx,hy (by role)
    row_map = {0: "ex", 1: "ey"}
    col_map = {0: "hx", 1: "hy"}

    data = np.zeros((len(periods), len(out_coords), len(in_coords)), dtype=complex)
    for ri, oc in row_map.items():
        for ci, ic in col_map.items():
            data[:, out_coords.index(oc), in_coords.index(ic)] = Z[:, ri, ci]

    tf = TF()
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
