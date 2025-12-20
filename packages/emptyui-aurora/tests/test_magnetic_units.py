"""Magnetic-unit handling: mth5 coil data are archived in nT (like fluxgate
B) and must be converted to A/m so the fc impedance is in ohm (V/m per A/m).

Regression for the USArray verification (DESIGN §16.4): the coil/'h' path
used to leave nT data untouched, which scaled Z by mu0*1e9 (~1.26e3) and
rho by ~1.6e6. Payloads that carry explicit A/m units (or none — the
synthetic test fixtures) must stay untouched.
"""

from __future__ import annotations

import numpy as np
import pytest
from emptyui.context import RunContext
from emptyui.store import DataStore
from emptyui_aurora._processing import (
    compute_fc_levels,
    estimate_tf_levels,
    merge_tf_levels,
)

MU0 = 4.0 * np.pi * 1e-7

#: E[mV/km] = 1e6 * Z * H[A/m]; H[A/m] = H[nT] * 1e-9 / mu0
N_TO_AM = 1e-9 / MU0


def _ctx(tmp_path):
    return RunContext(workspace=tmp_path, store=DataStore())


def _runts(hx, hy, ex, ey, sr=8.0, units_h="nanoTesla", units_e="milliVolt per kilometer"):
    """RunTS-like payload with per-channel units attrs (mth5-style)."""
    import xarray as xr

    ds = xr.Dataset(
        {
            "hx": ("time", hx),
            "hy": ("time", hy),
            "ex": ("time", ex),
            "ey": ("time", ey),
        },
        coords={"time": np.arange(len(hx)) / sr},
    )
    ds["hx"].attrs["units"] = units_h
    ds["hy"].attrs["units"] = units_h
    ds["ex"].attrs["units"] = units_e
    ds["ey"].attrs["units"] = units_e

    class FakeRunTS:
        sample_rate = sr

    obj = FakeRunTS()
    obj.dataset = ds
    obj.start = None
    obj.start_datetime = ""
    return obj


def _field(rng, n=16384, sr=8.0, z=(1.0, 3.0, -1.0, 4.0)):
    """Shared source + E driven by H through the true impedance z (2x2).

    Returns (hx, hy, ex, ey) in the given physical units:
    - nT mode: H in nT, E in mV/km = 1e6 * (Z @ H[A/m]).
    - A/m mode: H in A/m, E in mV/km = 1e6 * (Z @ H[A/m]).
    """
    t = np.arange(n) / sr
    hx = 50.0 * np.sin(2 * np.pi * 0.5 * t) + 0.5 * rng.standard_normal(n)
    hy = 40.0 * np.sin(2 * np.pi * 1.3 * t) + 0.5 * rng.standard_normal(n)
    ha = np.vstack([hx, hy]) * N_TO_AM
    e_am = np.array([[z[0], z[1]], [z[2], z[3]]]) @ ha
    ex = e_am[0] * 1e6 + 2.0 * rng.standard_normal(n)
    ey = e_am[1] * 1e6 + 2.0 * rng.standard_normal(n)
    return hx, hy, ex, ey


def _recovered_z(runts):
    levels = compute_fc_levels(
        runts, ["hx", "hy"], ["ex", "ey"],
        num_decimations=1, factor=4, num_samples_window=256,
        overlap=0.5, window_type="hamming", channel_nomenclature="default",
    )
    per = estimate_tf_levels(
        levels, ["hx", "hy"], ["ex", "ey"], estimator_engine="Robust",
        periods_per_octave=4, channel_nomenclature="default",
    )
    _, Z, _ = merge_tf_levels(per)
    return Z


def test_coil_nt_data_is_converted_to_am(tmp_path):
    """mth5 coil archive (units nT / mV per km) -> Z in ohm (V/m per A/m)."""
    rng = np.random.default_rng(11)
    runts = _runts(*_field(rng))
    Z = _recovered_z(runts)
    ok = np.abs(Z[:, 0, 1]) > 1e-9
    assert np.median(Z[ok, 0, 0].real) == pytest.approx(1.0, abs=0.5)
    assert np.median(Z[ok, 0, 1].real) == pytest.approx(3.0, abs=0.5)
    assert np.median(Z[ok, 1, 0].real) == pytest.approx(-1.0, abs=0.5)
    assert np.median(Z[ok, 1, 1].real) == pytest.approx(4.0, abs=0.5)


def test_already_am_data_is_not_converted(tmp_path):
    """Explicit A/m units -> no nT->A/m scaling; Z still recovered."""
    rng = np.random.default_rng(12)
    hx, hy, ex, ey = _field(rng)
    # H already in A/m, E in mV/km: Z = E*1e-6 / H  (unchanged by the fix)
    ex_am, ey_am = ex, ey
    runts = _runts(hx * N_TO_AM, hy * N_TO_AM, ex_am, ey_am,
                   units_h="ampere per meter")
    Z = _recovered_z(runts)
    ok = np.abs(Z[:, 0, 1]) > 1e-9
    assert np.median(Z[ok, 0, 0].real) == pytest.approx(1.0, abs=0.5)
    assert np.median(Z[ok, 0, 1].real) == pytest.approx(3.0, abs=0.5)
    assert np.median(Z[ok, 1, 0].real) == pytest.approx(-1.0, abs=0.5)
    assert np.median(Z[ok, 1, 1].real) == pytest.approx(4.0, abs=0.5)


def test_unitless_payload_keeps_legacy_behavior(tmp_path):
    """No unit attrs (synthetic fixtures) + magnetic_unit='h' = no conversion."""
    rng = np.random.default_rng(13)
    hx, hy, ex, ey = _field(rng)
    # legacy synthetic convention: H in A/m already, E in mV/km scaled so
    # that Z is recovered only when NO magnetic conversion is applied
    runts = _runts(hx * N_TO_AM, hy * N_TO_AM, ex, ey, units_h="")
    levels = compute_fc_levels(
        runts, ["hx", "hy"], ["ex", "ey"],
        num_decimations=1, factor=4, num_samples_window=256,
        overlap=0.5, window_type="hamming", magnetic_unit="h",
        channel_nomenclature="default",
    )
    per = estimate_tf_levels(
        levels, ["hx", "hy"], ["ex", "ey"], estimator_engine="Robust",
        periods_per_octave=4, channel_nomenclature="default",
    )
    _, Z, _ = merge_tf_levels(per)
    ok = np.abs(Z[:, 0, 1]) > 1e-9
    assert np.median(Z[ok, 0, 1].real) == pytest.approx(3.0, abs=0.5)


def test_fluxgate_nt_always_converts(tmp_path):
    """Explicit 'b' (fluxgate) converts even with unknown units."""
    rng = np.random.default_rng(14)
    hx, hy, ex, ey = _field(rng)
    runts = _runts(hx, hy, ex, ey, units_h="")
    levels = compute_fc_levels(
        runts, ["hx", "hy"], ["ex", "ey"],
        num_decimations=1, factor=4, num_samples_window=256,
        overlap=0.5, window_type="hamming", magnetic_unit="b",
        channel_nomenclature="default",
    )
    per = estimate_tf_levels(
        levels, ["hx", "hy"], ["ex", "ey"], estimator_engine="Robust",
        periods_per_octave=4, channel_nomenclature="default",
    )
    _, Z, _ = merge_tf_levels(per)
    ok = np.abs(Z[:, 0, 1]) > 1e-9
    assert np.median(Z[ok, 0, 1].real) == pytest.approx(3.0, abs=0.5)
