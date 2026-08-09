"""Prewhitening + per-window detrend STFT options (history §24, E1/E2 enablers).

The first-difference prewhitening must be *exactly* inverted by the
frequency-domain recoloring (÷ (1 − e^{−j2πfΔt})) so ratio-based TF products
are unchanged on clean signals while colored-noise leakage drops.
"""

from __future__ import annotations

import numpy as np
import pytest
from _synthetic import compute_fc_seconds
from emptyui_aurora._processing import compute_fc_levels
from emptyui_core_nodes.types import TimeSeries


def _payload(tone):
    return TimeSeries(channels={"hx": tone.copy(), "hy": tone.copy()},
                      sample_rate=8.0)


def _tone(fs=8.0, n=2048, f0=0.5):
    t = np.arange(n) / fs
    return 3.0 * np.sin(2 * np.pi * f0 * t), t


def _kw(**over):
    kw = {"window_length_s": 256.0, "overlap": 0.5, "window_type": "boxcar"}
    kw.update(over)
    return kw


def test_first_difference_recoloring_recovers_fc():
    """On a leakage-free signal (integer-cycle tone, one window) the
    prewhitened+recolored FC must match the un-prewhitened FC — the exact
    inverse filter undoes the differencing."""
    tone, _ = _tone()
    base = compute_fc_seconds(_payload(tone), ["hx", "hy"], ["hx", "hy"], **_kw(prewhitening=""))
    pw = compute_fc_seconds(_payload(tone), ["hx", "hy"], ["hx", "hy"],
                    **_kw(prewhitening="first difference"))
    f = base["frequency"].values
    k = int(np.argmin(np.abs(f - 0.5)))
    zb = base["hx"].values[k, 0]
    zp = pw["hx"].values[k, 0]
    assert abs(zp) == pytest.approx(abs(zb), rel=2e-3)
    assert np.angle(zp) == pytest.approx(np.angle(zb), abs=2e-3)


def test_prewhitening_nans_dc_bin():
    """Recoloring is undefined at f=0 — the DC bin is NaN (it is excluded
    from bands and skipped by the estimator's non-finite row filter)."""
    tone, _ = _tone()
    pw = compute_fc_seconds(_payload(tone), ["hx", "hy"], ["hx", "hy"],
                    **_kw(prewhitening="first difference"))
    assert np.isnan(pw["hx"].values[0, 0])
    base = compute_fc_seconds(_payload(tone), ["hx", "hy"], ["hx", "hy"], **_kw(prewhitening=""))
    assert np.isfinite(base["hx"].values[0, 0])


def test_prewhitening_reduces_colored_noise_leakage():
    """1/f-like colored noise: differencing whitens the low-frequency
    dominance before the STFT, so out-of-band leakage into a clean tone bin
    drops (this is the EMTF/aurora motivation for prewhitening)."""
    rng = np.random.default_rng(7)
    fs, n = 8.0, 32768
    t = np.arange(n) / fs
    tone = 3.0 * np.sin(2 * np.pi * 2.0 * t)
    # cumulative white noise = red (1/f²) noise — strongly colored
    red = np.cumsum(rng.standard_normal(n))
    red /= np.abs(red).std() * 10  # 10x weaker than the tone
    x = tone + red

    kw = {"window_length_s": 256.0, "overlap": 0.5, "window_type": "boxcar"}
    base = compute_fc_seconds(TimeSeries(channels={"hx": x.copy(), "hy": x.copy()},
                                 sample_rate=fs), ["hx", "hy"], ["hx", "hy"], **kw, prewhitening="")
    pw = compute_fc_seconds(TimeSeries(channels={"hx": x.copy(), "hy": x.copy()},
                               sample_rate=fs), ["hx", "hy"], ["hx", "hy"],
                    **kw, prewhitening="first difference")
    k = int(np.argmin(np.abs(base["frequency"].values - 2.0)))
    zb, zp = abs(base["hx"].values[k, :].mean()), abs(pw["hx"].values[k, :].mean())
    # the tone bin is preserved (within leakage of the differencing edge
    # effects) while the red-noise floor around it drops hard
    assert zp == pytest.approx(zb, rel=0.05)
    off = np.ones(len(base["frequency"].values), bool)
    off[max(0, k - 8):k + 8] = False
    fb = np.nanmean(np.abs(base["hx"].values[off, :]))
    fp = np.nanmean(np.abs(pw["hx"].values[off, :]))
    assert np.isfinite(fp) and fp < 0.5 * fb


def test_per_window_detrend_removes_trend():
    """Per-window linear detrend removes a linear trend's spectral
    contribution (the residual equals the detrended tone's own leakage), while
    the un-detrended STFT carries the ramp's leakage on top."""
    tone, t = _tone()
    ramp = 0.01 * t
    kw = _kw(prewhitening="")
    base = compute_fc_seconds(_payload(tone + ramp), ["hx", "hy"], ["hx", "hy"], **kw)
    detr = compute_fc_seconds(_payload(tone + ramp), ["hx", "hy"], ["hx", "hy"],
                      **_kw(prewhitening="", per_window_detrend="linear"))
    tone_detr = compute_fc_seconds(_payload(tone), ["hx", "hy"], ["hx", "hy"],
                           **_kw(prewhitening="", per_window_detrend="linear"))
    f = base["frequency"].values
    k = int(np.argmin(np.abs(f - 0.5)))
    off = np.ones(len(f), bool)
    off[k] = False
    a_base = np.abs(base["hx"].values[off, 0]).max()
    a_detr = np.abs(detr["hx"].values[off, 0]).max()
    a_tone_detr = np.abs(tone_detr["hx"].values[off, 0]).max()
    # without detrend the ramp adds leakage on top of the detrended tone's own
    assert a_base > 1.5 * a_tone_detr
    # detrend is linear: detrend(tone + ramp) == detrend(tone), the ramp removed
    # exactly. (Comparing with the un-detrended tone was only accidentally true
    # under the hamming taper: a boxcar window leaks the tone into the detrend.)
    assert a_detr == pytest.approx(a_tone_detr, rel=0.05)


def test_options_threaded_through_levels_and_attrs():
    """compute_fc_levels passes the options to every level and records them
    in the fc attrs (provenance, history §24 protocol)."""
    tone, _ = _tone(n=8192)
    levels = compute_fc_levels(
        _payload(tone), ["hx", "hy"], ["hx", "hy"],
        num_decimations=2, factor=4, num_samples_window=256,
        overlap=0.5, window_type="boxcar",
        prewhitening="first difference", per_window_detrend="linear")
    assert sorted(levels) == [0, 1, 2]
    for fc in levels.values():
        assert fc.attrs["prewhitening"] == "first difference"
        assert fc.attrs["per_window_detrend"] == "linear"
        assert np.isnan(fc["hx"].values[0, 0])  # DC NaN at every level
    # shapes: deeper levels decimate but keep the same window in samples
    assert levels[1]["hx"].shape[1] < levels[0]["hx"].shape[1]


def test_unknown_prewhitening_rejected():
    tone, _ = _tone(n=256)
    with pytest.raises(ValueError, match="unknown prewhitening"):
        compute_fc_seconds(_payload(tone), ["hx", "hy"], ["hx", "hy"],
                   **_kw(prewhitening="arma"))
    with pytest.raises(ValueError, match="unknown prewhitening"):
        compute_fc_levels(_payload(tone), ["hx", "hy"], ["hx", "hy"],
                          prewhitening="ar3")


def test_per_level_taper_override():
    """history §24.5 P2: levels >= deep_from_level use the deep taper; the override
    equals computing that level with the deep taper directly (the cascade's
    decimation is taper-independent)."""
    tone, _ = _tone(n=8192)
    levels = compute_fc_levels(
        _payload(tone), ["hx", "hy"], ["hx", "hy"],
        num_decimations=2, factor=4, num_samples_window=256,
        overlap=0.5, window_type="hamming",
        window_type_deep="boxcar", deep_from_level=1)
    assert levels[0].attrs["window_type"] == "hamming"
    assert levels[1].attrs["window_type"] == "boxcar"
    assert levels[2].attrs["window_type"] == "boxcar"
    all_box = compute_fc_levels(
        _payload(tone), ["hx", "hy"], ["hx", "hy"],
        num_decimations=2, factor=4, num_samples_window=256, overlap=0.5,
        window_type="boxcar")
    np.testing.assert_allclose(levels[1]["hx"].values, all_box[1]["hx"].values)
    # empty override = base taper everywhere (legacy behavior)
    legacy = compute_fc_levels(
        _payload(tone), ["hx", "hy"], ["hx", "hy"],
        num_decimations=1, factor=4, num_samples_window=256,
        overlap=0.5, window_type="hamming")
    assert all(fc.attrs["window_type"] == "hamming"
               for fc in legacy.values())
