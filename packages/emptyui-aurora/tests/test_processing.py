"""Synthetic unit tests for the aurora processing core (no external data)."""

from __future__ import annotations

import numpy as np
import pytest
from emptyui_aurora._processing import build_tf, compute_fc, estimate_tf


def _synthetic_runts():
    """A fake ts.run-like object (duck-typed like an mth5 RunTS dataset)."""
    import xarray as xr

    sr = 8.0
    n = 8192
    t = np.arange(n) / sr
    rng = np.random.default_rng(0)
    # input magnetic signals + noise, outputs are linear combos + noise
    hx = 50.0 * np.sin(2 * np.pi * 0.5 * t) + 5 * rng.standard_normal(n)
    hy = 40.0 * np.sin(2 * np.pi * 1.3 * t) + 5 * rng.standard_normal(n)
    # impedance: zxx=1+2j, zxy=3-1j, zyx=2+0j, zyy=1+1j applied in time
    ex = 1.0 * hx + 3.0 * hy + rng.standard_normal(n) * 8
    ey = 2.0 * hx + 1.0 * hy + rng.standard_normal(n) * 8

    class FakeRunTS:
        sample_rate = sr

        def __init__(self):
            self.dataset = xr.Dataset(
                {"ex": ("time", ex), "ey": ("time", ey),
                 "hx": ("time", hx), "hy": ("time", hy)},
                coords={"time": t},
            )

    return FakeRunTS()


def test_compute_fc_and_estimate_tf():
    runts = _synthetic_runts()
    fc = compute_fc(runts, ["hx", "hy"], ["ex", "ey"],
                    window_length_s=256, overlap=0.5, window_type="hamming")
    assert "frequency" in fc.dims and "time" in fc.dims
    assert set(fc.data_vars) == {"ex", "ey", "hx", "hy"}

    periods, Z = estimate_tf(fc, ["hx", "hy"], ["ex", "ey"], periods_per_decade=8)
    assert periods.size > 1
    assert np.all(np.isfinite(Z))

    # crude check: recovered impedance finite and roughly non-trivial
    Zxy = np.median(np.abs(Z[:, 0, 1]))
    Zyx = np.median(np.abs(Z[:, 1, 0]))
    assert Zxy > 0
    assert Zyx > 0


def test_magnetic_unit_h_vs_b_conversion():
    from emptyui_aurora._processing import MU0

    runts = _synthetic_runts()
    fc_h = compute_fc(runts, ["hx", "hy"], ["ex", "ey"],
                      window_length_s=256, overlap=0.5, window_type="hamming",
                      magnetic_unit="h")
    fc_b = compute_fc(runts, ["hx", "hy"], ["ex", "ey"],
                      window_length_s=256, overlap=0.5, window_type="hamming",
                      magnetic_unit="b")
    # fluxgate B [nT] -> H [A/m]: fc input is H = B*1e-9/mu0, so b-case is
    # the h-case (already H) scaled by 1e-9/mu0
    ratio = np.median(np.abs(fc_b["hx"].values) / np.abs(fc_h["hx"].values))
    assert np.isclose(ratio, 1e-9 / MU0, rtol=1e-6)


def test_build_tf_returns_mt_metadata_tf():
    runts = _synthetic_runts()
    fc = compute_fc(runts, ["hx", "hy"], ["ex", "ey"],
                    window_length_s=256, overlap=0.5, window_type="hamming")
    periods, Z = estimate_tf(fc, ["hx", "hy"], ["ex", "ey"])
    tf = build_tf(periods, Z, ["hx", "hy"], ["ex", "ey"])
    assert hasattr(tf, "dataset")
    assert tf.period.size == periods.size
    # impedance populated on ex/ey x hx/hy
    arr = tf.dataset["transfer_function"]
    assert arr.sel(output="ex", input="hx").size == periods.size


def test_remote_reference_rejected():
    runts = _synthetic_runts()
    fc = compute_fc(runts, ["hx", "hy"], ["ex", "ey"],
                    window_length_s=256, overlap=0.5, window_type="hamming")
    with pytest.raises(ValueError):
        estimate_tf(fc, ["hx", "hy"], ["ex", "ey"], remote_reference=True)
