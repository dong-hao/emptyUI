"""Synthetic unit tests for the aurora processing core (no external data)."""

from __future__ import annotations

import numpy as np
import pytest
from emptyui_aurora._processing import (
    build_tf,
    compute_fc,
    compute_fc_levels,
    estimate_tf,
)


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

    periods, Z = estimate_tf(fc, ["hx", "hy"], ["ex", "ey"], periods_per_octave=8)
    assert periods.size > 1
    assert np.all(np.isfinite(Z))

    # crude check: recovered impedance finite and roughly non-trivial
    Zxy = np.median(np.abs(Z[:, 0, 1]))
    Zyx = np.median(np.abs(Z[:, 1, 0]))
    assert Zxy > 0
    assert Zyx > 0


def test_tipper_bz_output_recovered():
    """Outputs e1,e2,bz give a 3x2 TF: 2x2 impedance plus the vertical-field
    geomagnetic transfer function (tipper) as the hz row. bz is magnetic, so it
    must NOT receive the electric (mV/km -> V/m) scaling — the tipper is
    dimensionless and recovered ~ [0.5, -0.3]."""
    import xarray as xr

    sr = 8.0
    n = 16384
    t = np.arange(n) / sr
    rng = np.random.default_rng(7)
    # broadband inputs so signal is present across all period bands
    hx = rng.standard_normal(n)
    hy = rng.standard_normal(n)
    ex = 1.0 * hx + 3.0 * hy + 0.1 * rng.standard_normal(n)  # impedance [[1,3],[-1,4]]
    ey = -1.0 * hx + 4.0 * hy + 0.1 * rng.standard_normal(n)
    bz = 0.5 * hx - 0.3 * hy + 0.1 * rng.standard_normal(n)  # tipper [0.5,-0.3]

    class FakeRunTS:
        sample_rate = sr

        def __init__(self):
            self.dataset = xr.Dataset(
                {"e1": ("time", ex), "e2": ("time", ey), "bz": ("time", bz),
                 "bx": ("time", hx), "by": ("time", hy)},
                coords={"time": t},
            )

    runts = FakeRunTS()
    fc = compute_fc(runts, ["bx", "by"], ["e1", "e2", "bz"],
                    window_length_s=128, overlap=0.5, window_type="hamming")
    periods, Z = estimate_tf(fc, ["bx", "by"], ["e1", "e2", "bz"],
                             periods_per_octave=8)
    assert Z.shape[1:] == (3, 2)  # n_out x n_in (includes the tipper row)
    ok = np.all(np.isfinite(Z), axis=(1, 2))
    # tipper row is dimensionless and recovered: Tzx ~ 0.5, Tzy ~ -0.3
    assert np.median(Z[:, 2, 0][ok].real) == pytest.approx(0.5, abs=0.15)
    assert np.median(Z[:, 2, 1][ok].real) == pytest.approx(-0.3, abs=0.15)

    tf = build_tf(periods, Z, ["bx", "by"], ["e1", "e2", "bz"])
    arr = tf.dataset["transfer_function"]
    # tipper populated on the hz row; impedance on ex/ey x hx/hy — selected
    # through the nomenclature-aware accessors (actual names as coordinates)
    assert np.any(np.abs(arr.sel(output=tf.hz).values) > 0)
    assert np.any(np.abs(arr.sel(output=tf.ex, input=tf.hx).values) > 0)
    assert tf.channel_nomenclature == {
        "ex": "e1", "ey": "e2", "hx": "bx", "hy": "by", "hz": "bz",
    }


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


def test_magnetic_unit_auto_derived_from_channel_names():
    """'auto' must derive the B/H mode from the nomenclature/channel names:
    bx/by (fluxgate, nT) -> B->H conversion; hx/hy (coil, A/m) -> unchanged."""
    import xarray as xr
    from emptyui_aurora._processing import compute_fc

    sr = 8.0
    n = 8192
    t = np.arange(n) / sr
    rng = np.random.default_rng(0)
    hx = 50 * np.sin(2 * np.pi * 0.5 * t) + 5 * rng.standard_normal(n)
    hy = 40 * np.sin(2 * np.pi * 1.3 * t) + 5 * rng.standard_normal(n)
    ex = 1.0 * hx + 3.0 * hy + rng.standard_normal(n) * 8
    ey = 2.0 * hx + 1.0 * hy + rng.standard_normal(n) * 8

    class FakeRunTS:
        sample_rate = sr

        def __init__(self):
            self.dataset = xr.Dataset(
                {"e1": ("time", ex), "e2": ("time", ey),
                 "bx": ("time", hx), "by": ("time", hy)},
                coords={"time": t},
            )

    runts = FakeRunTS()

    fc_b_auto = compute_fc(runts, ["bx", "by"], ["e1", "e2"],
                           window_length_s=256, overlap=0.5, window_type="hamming",
                           magnetic_unit="auto")
    fc_b_manual = compute_fc(runts, ["bx", "by"], ["e1", "e2"],
                             window_length_s=256, overlap=0.5, window_type="hamming",
                             magnetic_unit="b")
    np.testing.assert_allclose(fc_b_auto["bx"].values, fc_b_manual["bx"].values)
    assert fc_b_auto.attrs["magnetic_unit"] == "b"
    assert fc_b_auto.attrs["channel_nomenclature"] == "lemi12"

    # hx/hy naming (default nomenclature) means coil H: unchanged, unit 'h'
    runts_h = _synthetic_runts()
    fc_h_auto = compute_fc(runts_h, ["hx", "hy"], ["ex", "ey"],
                           window_length_s=256, overlap=0.5, window_type="hamming",
                           magnetic_unit="auto")
    fc_h_manual = compute_fc(runts_h, ["hx", "hy"], ["ex", "ey"],
                             window_length_s=256, overlap=0.5, window_type="hamming",
                             magnetic_unit="h")
    np.testing.assert_allclose(fc_h_auto["hx"].values, fc_h_manual["hx"].values)
    assert fc_h_auto.attrs["magnetic_unit"] == "h"
    assert fc_h_auto.attrs["channel_nomenclature"] == "default"


def test_build_tf_lemi34_four_channel():
    """A 4-E-channel LEMI instrument (lemi34: ex=e3, ey=e4) must map both
    electric outputs onto the standard ex/ey TF rows — the previous hardcoded
    map dropped e4 entirely."""
    import xarray as xr

    sr = 8.0
    n = 16384
    t = np.arange(n) / sr
    rng = np.random.default_rng(11)
    hx = rng.standard_normal(n)
    hy = rng.standard_normal(n)
    ex = 1.0 * hx + 3.0 * hy + 0.1 * rng.standard_normal(n)
    ey = -1.0 * hx + 4.0 * hy + 0.1 * rng.standard_normal(n)

    class FakeRunTS:
        sample_rate = sr

        def __init__(self):
            self.dataset = xr.Dataset(
                {"e3": ("time", ex), "e4": ("time", ey),
                 "bx": ("time", hx), "by": ("time", hy)},
                coords={"time": t},
            )

    runts = FakeRunTS()
    fc = compute_fc(runts, ["bx", "by"], ["e3", "e4"],
                    window_length_s=128, overlap=0.5, window_type="hamming")
    assert fc.attrs["channel_nomenclature"] == "lemi34"
    periods, Z = estimate_tf(fc, ["bx", "by"], ["e3", "e4"],
                             periods_per_octave=8)
    tf = build_tf(periods, Z, ["bx", "by"], ["e3", "e4"])
    arr = tf.dataset["transfer_function"]
    # both electric outputs are mapped: ex row (e3) and ey row (e4) populated —
    # selected via the nomenclature accessors (actual names as coordinates)
    assert tf.channel_nomenclature == {
        "ex": "e3", "ey": "e4", "hx": "bx", "hy": "by", "hz": "bz",
    }
    assert np.any(np.abs(arr.sel(output=tf.ex, input=tf.hx).values) > 0)
    assert np.any(np.abs(arr.sel(output=tf.ey, input=tf.hx).values) > 0)
    assert np.any(np.abs(arr.sel(output=tf.ey, input=tf.hy).values) > 0)


def test_build_tf_phoenix123_nomenclature():
    """phoenix123 channels (h1/h2/h3, e1/e2) map onto the standard TF grid."""
    from emptyui_aurora._processing import build_tf

    periods = np.array([2.0, 4.0, 8.0])
    Z = np.zeros((3, 2, 2), dtype=complex)
    Z[:, 0, 0] = 1 + 0j  # ex x hx
    Z[:, 1, 1] = 2 + 0j  # ey x hy
    tf = build_tf(periods, Z, ["h1", "h2"], ["e1", "e2"],
                  channel_nomenclature="phoenix123")
    arr = tf.dataset["transfer_function"]
    assert tf.channel_nomenclature == {
        "ex": "e1", "ey": "e2", "hx": "h1", "hy": "h2", "hz": "h3",
    }
    assert np.any(np.abs(arr.sel(output=tf.ex, input=tf.hx).values) > 0)
    assert np.any(np.abs(arr.sel(output=tf.ey, input=tf.hy).values) > 0)


def test_estimate_tf_accepts_mth5_read_fc_layout():
    """FCs read back from an mth5 archive have a (time, frequency) layout and
    no attrs; the band loop must transpose them and still estimate."""
    import xarray as xr
    from emptyui_aurora._processing import estimate_tf

    sr = 8.0
    n = 8192
    t = np.arange(n) / sr
    rng = np.random.default_rng(3)
    hx = rng.standard_normal(n)
    hy = rng.standard_normal(n)
    ex = 1.0 * hx + 3.0 * hy + 0.1 * rng.standard_normal(n)
    ey = -1.0 * hx + 4.0 * hy + 0.1 * rng.standard_normal(n)

    class FakeRunTS:
        sample_rate = sr

        def __init__(self):
            self.dataset = xr.Dataset(
                {"ex": ("time", ex), "ey": ("time", ey),
                 "hx": ("time", hx), "hy": ("time", hy)},
                coords={"time": t},
            )

    fc = compute_fc(FakeRunTS(), ["hx", "hy"], ["ex", "ey"],
                    window_length_s=128, overlap=0.5, window_type="hamming")
    # mth5 stores (time, frequency); emulate the read-back layout + bare attrs
    mth5_layout = fc.transpose("time", "frequency").copy()
    mth5_layout.attrs = {}
    periods, Z = estimate_tf(mth5_layout, ["hx", "hy"], ["ex", "ey"],
                             estimator_engine="Least Squares", periods_per_octave=8)
    assert periods.size > 1
    assert np.all(np.isfinite(Z))
    assert np.any(np.abs(Z) > 0)


def test_remote_reference_grid_mismatch_raises_clear_error():
    """A remote-reference fc with a different window (frequency grid) must fail
    with a clear message instead of aurora's bare Exception."""
    runts = _synthetic_runts()
    fc = compute_fc(runts, ["hx", "hy"], ["ex", "ey"],
                    window_length_s=256, overlap=0.5, window_type="hamming")
    ref = compute_fc(runts, ["hx", "hy"], ["ex", "ey"],
                     window_length_s=512, overlap=0.5, window_type="hamming")
    with pytest.raises(ValueError, match="frequency grid"):
        estimate_tf(fc, ["hx", "hy"], ["ex", "ey"],
                    estimator_engine="Robust", remote_reference=True,
                    ref_fc=ref, periods_per_octave=8)


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


@pytest.mark.parametrize("engine,rr", [
    ("Least Squares", False),
    ("Robust", False),
    ("Robust", True),
])
def test_estimator_engines_run(engine, rr):
    runts = _synthetic_runts()
    fc = compute_fc(runts, ["hx", "hy"], ["ex", "ey"],
                    window_length_s=256, overlap=0.5, window_type="hamming")
    periods, Z = estimate_tf(fc, ["hx", "hy"], ["ex", "ey"],
                             estimator_engine=engine, remote_reference=rr,
                             periods_per_octave=8)
    assert periods.size > 1
    assert np.all(np.isfinite(Z))
    assert np.any(np.abs(Z) > 0)


def test_legacy_estimator_engine_names_still_accepted():
    """Old OLS/RME/RME_RR values keep loading previously saved workflows."""
    runts = _synthetic_runts()
    fc = compute_fc(runts, ["hx", "hy"], ["ex", "ey"],
                    window_length_s=256, overlap=0.5, window_type="hamming")
    for engine in ("OLS", "RME", "RME_RR"):
        rr = engine == "RME_RR"
        _, Z = estimate_tf(fc, ["hx", "hy"], ["ex", "ey"],
                           estimator_engine=engine, remote_reference=rr,
                           periods_per_octave=8)
        assert np.all(np.isfinite(Z))


def test_ls_remote_reference_rejected():
    """There is no least-squares + remote-reference path by design."""
    runts = _synthetic_runts()
    fc = compute_fc(runts, ["hx", "hy"], ["ex", "ey"],
                    window_length_s=256, overlap=0.5, window_type="hamming")
    with pytest.raises(ValueError):
        estimate_tf(fc, ["hx", "hy"], ["ex", "ey"],
                    estimator_engine="Least Squares", remote_reference=True)


def test_remote_reference_uses_duplicated_dummy_station():
    """RME_RR with a 'dummy' remote station.

    We don't have two synchronized stations on disk, so we model the remote
    reference by duplicating the original data file — computing a second fc
    from a copy of the same time series and passing it as ref_fc. Identical
    data makes this a self-reference, but it exercises the full ref_fc path.
    """
    runts = _synthetic_runts()
    fc = compute_fc(runts, ["hx", "hy"], ["ex", "ey"],
                    window_length_s=256, overlap=0.5, window_type="hamming")
    # duplicate the raw data as a dummy remote station -> separate fc
    remote = compute_fc(runts, ["hx", "hy"], ["ex", "ey"],
                        window_length_s=256, overlap=0.5, window_type="hamming")
    periods, Z = estimate_tf(fc, ["hx", "hy"], ["ex", "ey"],
                             estimator_engine="Robust", remote_reference=True,
                             ref_fc=remote, periods_per_octave=8)
    assert periods.size > 1
    assert np.all(np.isfinite(Z))
    assert np.any(np.abs(Z) > 0)

    # the estimator_engine must be honored: robust (not silently OLS)
    _, Z_ols = estimate_tf(fc, ["hx", "hy"], ["ex", "ey"],
                           estimator_engine="Least Squares", periods_per_octave=8)
    assert not np.allclose(Z, Z_ols)


def test_compute_fc_levels_cascading_decimation():
    """Multi-level ComputeFC: N levels, sample rates halved per level, and the
    constant-window STFT gives progressively lower frequency resolution."""
    runts = _synthetic_runts()  # native sample_rate 8.0
    levels = compute_fc_levels(runts, ["hx", "hy"], ["ex", "ey"],
                               num_decimations=2, factor=2, num_samples_window=64,
                               overlap=0.5, window_type="hamming")
    assert list(levels) == [0, 1, 2]
    for level, fc in levels.items():
        assert fc.attrs["decimation_level"] == level
        assert fc.attrs["sample_rate"] == pytest.approx(8.0 / 2 ** level)
        assert set(fc.data_vars) == {"ex", "ey", "hx", "hy"}
    # nyquist halves per level: L0 4Hz, L1 2Hz, L2 1Hz
    assert levels[0]["frequency"].max().item() == pytest.approx(4.0)
    assert levels[1]["frequency"].max().item() == pytest.approx(2.0)
    assert levels[2]["frequency"].max().item() == pytest.approx(1.0)
    # a deeper level with the same window (samples) has fewer windows
    assert levels[2].sizes["time"] <= levels[0].sizes["time"]


def test_multi_level_merge_broadband():
    """estimate_tf_levels + merge_tf_levels give one sorted, de-duplicated
    broadband TF spanning the union of the per-level period ranges."""
    from emptyui_aurora._processing import estimate_tf_levels, merge_tf_levels

    runts = _synthetic_runts()
    levels = compute_fc_levels(runts, ["hx", "hy"], ["ex", "ey"],
                               num_decimations=3, factor=2, num_samples_window=128,
                               overlap=0.5, window_type="hamming")
    per = estimate_tf_levels(levels, ["hx", "hy"], ["ex", "ey"],
                             estimator_engine="Robust", periods_per_octave=4)
    assert list(per) == [0, 1, 2, 3]

    per_level_counts = [len(per[lv]["periods"]) for lv in per]
    P, Z = merge_tf_levels(per)
    assert np.all(np.diff(P) > 0)  # strictly ascending, no duplicate periods
    # each level spans [Fs/4, window/4] and overlaps its neighbours, so the
    # merge de-duplicates the shared octave grid to a continuous broadband set.
    assert len(P) < sum(per_level_counts)
    for i in range(len(per) - 1):
        assert per[i]["periods"][-1] > per[i + 1]["periods"][0]  # overlap
    assert P.min() == pytest.approx(per[0]["periods"].min())
    assert P.max() == pytest.approx(per[3]["periods"].max())
    assert np.all(np.isfinite(Z))
    assert Z.shape[0] == len(P)
    assert not np.allclose(Z, 0)
