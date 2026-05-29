from __future__ import annotations

import numpy as np


def synthetic_source(seed: int, n: int = 8192, sr: float = 8.0,
                     kind: str = "sines"):
    """(sr, n, t, hx, hy) — a shared source field.

    ``kind``: ``"sines"`` (default, strong periodic signals + noise, for
    coherence/estimation tests) or ``"white"`` (pure noise, for alignment
    tests). Returns the time vector and the two source components; sensor
    noise is added by :func:`synthetic_runts`.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n) / sr
    if kind == "white":
        return sr, n, t, rng.standard_normal(n), rng.standard_normal(n)
    hx = 50.0 * np.sin(2 * np.pi * 0.5 * t) + 5 * rng.standard_normal(n)
    hy = 40.0 * np.sin(2 * np.pi * 1.3 * t) + 5 * rng.standard_normal(n)
    return sr, n, t, hx, hy


def synthetic_runts(source, *, e_noise: float = 8.0, h_noise_segment=None,
                    z=(1.0, 3.0, 2.0, 1.0),
                    magnetic=("hx", "hy"), electric=("ex", "ey"),
                    with_e: bool = True, start_datetime: str = "",
                    seed: int = 0):
    """Fake RunTS-like ts.run (xarray dataset) built from a shared source.

    The recorded H carries sensor noise (and optionally a strong noisy
    segment); E is driven by the clean source (true physics: sensor noise
    does not drive E). ``z`` = (zxx, zxy, zyx, zyy) time-domain coefficients;
    ``magnetic``/``electric`` set the actual channel names.
    """
    import xarray as xr

    sr, n, t, hx, hy = source
    rng = np.random.default_rng(seed)
    hx_r = hx + 0.1 * rng.standard_normal(n)
    hy_r = hy + 0.1 * rng.standard_normal(n)
    if h_noise_segment is not None:
        s, e = h_noise_segment
        hx_r[s:e] += 200.0 * rng.standard_normal(e - s)
        hy_r[s:e] += 200.0 * rng.standard_normal(e - s)
    ex = z[0] * hx + z[1] * hy + e_noise * rng.standard_normal(n)
    ey = z[2] * hx + z[3] * hy + e_noise * rng.standard_normal(n)

    vars_ = {magnetic[0]: hx_r, magnetic[1]: hy_r}
    if with_e:
        vars_.update({electric[0]: ex, electric[1]: ey})

    class FakeRunTS:
        sample_rate = sr

        def __init__(self):
            import pandas as pd

            self.dataset = xr.Dataset(
                {k: ("time", v) for k, v in vars_.items()},
                coords={"time": t},
            )
            self.start = (pd.Timestamp(start_datetime) if start_datetime
                          else None)
            self.start_datetime = start_datetime

    return FakeRunTS()


def compute_fc_seconds(value, input_channels, output_channels, window_length_s,
                       overlap, window_type, **kw):
    """Level-0 FC with the window given in seconds (the retired single-level
    ``compute_fc`` surface). Delegates to ``compute_fc_levels`` with the recipe
    the fixtures were written against: no decimation, prewhitening off unless
    passed explicitly."""
    from emptyui_aurora._processing import compute_fc_levels, to_channels

    _, sample_rate, _, _ = to_channels(value, check_finite=False)
    kw.setdefault("prewhitening", "")
    levels = compute_fc_levels(
        value, input_channels, output_channels, num_decimations=0,
        num_samples_window=max(8, int(window_length_s * sample_rate)),
        overlap=overlap, window_type=window_type, **kw)
    if 0 not in levels:
        raise ValueError(
            f"series shorter than one window ({window_length_s} s at "
            f"{sample_rate} Hz): compute_fc_levels returned no level 0")
    return levels[0]


def synthetic_fc(runts, ins, outs, **kw):
    """FC for synthetic runs: 256 s windows, 50% overlap, Hamming."""
    return compute_fc_seconds(runts, ins, outs, window_length_s=256, overlap=0.5,
                              window_type="hamming", **kw)
