"""Synthetic unit tests for the aurora processing core (no external data).
Error surfaces live in test_errors.py."""

from __future__ import annotations

import logging

import numpy as np
import pytest
from _synthetic import compute_fc_seconds, synthetic_runts, synthetic_source
from emptyui_aurora._processing import (
    build_tf,
    compute_fc_levels,
    estimate_tf,
)


def _synthetic_runts():
    """A fake ts.run-like object (duck-typed like an mth5 RunTS dataset)."""
    return synthetic_runts(synthetic_source(0))


def test_compute_fc_and_estimate_tf():
    runts = _synthetic_runts()
    fc = compute_fc_seconds(runts, ["hx", "hy"], ["ex", "ey"],
                    window_length_s=256, overlap=0.5, window_type="hamming")
    assert "frequency" in fc.dims and "time" in fc.dims
    assert set(fc.data_vars) == {"ex", "ey", "hx", "hy"}

    periods, Z, _ = estimate_tf(fc, ["hx", "hy"], ["ex", "ey"], periods_per_octave=8)
    assert periods.size > 1
    assert np.all(np.isfinite(Z))

    # crude check: recovered impedance finite and roughly non-trivial
    Zxy = np.median(np.abs(Z[:, 0, 1]))
    Zyx = np.median(np.abs(Z[:, 1, 0]))
    assert Zxy > 0
    assert Zyx > 0


def test_estimate_tf_returns_standard_errors():
    """estimate_tf returns per-element standard errors for both engines."""
    runts = _synthetic_runts()
    fc = compute_fc_seconds(runts, ["hx", "hy"], ["ex", "ey"],
                    window_length_s=256, overlap=0.5, window_type="hamming")
    for engine in ("Least Squares", "Robust"):
        _, Z, Zerr = estimate_tf(fc, ["hx", "hy"], ["ex", "ey"],
                                 estimator_engine=engine, periods_per_octave=8)
        assert Zerr.shape == Z.shape
        assert np.all(np.isfinite(Zerr))
        assert np.all(Zerr >= 0)
        # bands that produced an estimate carry a strictly positive error
        est = np.any(np.abs(Z) > 0, axis=(1, 2))
        assert np.all(Zerr[est] > 0)


def test_build_tf_carries_errors_and_station_identity():
    """build_tf stores element stds in transfer_function_error and the station
    identity in station/survey metadata (DESIGN.md §10.6)."""
    periods = np.array([2.0, 4.0])
    Z = np.array([[[1 + 0j, 0j], [0j, 2 + 0j]],
                  [[1 + 0j, 0j], [0j, 2 + 0j]]])
    Zerr = np.array([[0.1, 0.0], [0.0, 0.2]], dtype=float)
    Zerr = np.repeat(Zerr[None, :, :], 2, axis=0)
    tf = build_tf(periods, Z, ["bx", "by"], ["e1", "e2"],
                  channel_nomenclature="lemi12", Z_err=Zerr,
                  survey_id="bgs_37", station_id="NO46",
                  latitude=56.72, longitude=-2.85, elevation=120.0)
    assert tf.has_impedance
    err = tf.impedance_error
    assert np.allclose(err.sel(output="e1", input="bx").values, [0.1, 0.1])
    assert np.allclose(err.sel(output="e2", input="by").values, [0.2, 0.2])
    assert tf.station_metadata.id == "NO46"
    assert abs(float(tf.station_metadata.location.latitude) - 56.72) < 1e-9
    assert abs(float(tf.station_metadata.location.elevation) - 120.0) < 1e-9
    assert tf.survey_metadata.id == "bgs_37"


def test_build_tf_carries_tipper_errors():
    """A tipper-carrying TF stores the tipper row's errors too."""
    periods = np.array([2.0, 4.0])
    Z = np.zeros((2, 3, 2), dtype=complex)
    Z[:, 0, 0] = 1 + 0j
    Z[:, 2, 0] = 0.1 + 0.05j  # bz x hx (tipper row)
    Zerr = np.zeros((2, 3, 2), dtype=float)
    Zerr[:, 2, 0] = 0.01
    tf = build_tf(periods, Z, ["bx", "by"], ["e1", "e2", "bz"],
                  channel_nomenclature="lemi12", Z_err=Zerr)
    assert tf.has_tipper()
    terr = tf.tipper_error
    assert np.allclose(terr.sel(output="bz", input="bx").values, [0.01, 0.01])


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
    fc = compute_fc_seconds(runts, ["bx", "by"], ["e1", "e2", "bz"],
                    window_length_s=128, overlap=0.5, window_type="hamming")
    periods, Z, _ = estimate_tf(fc, ["bx", "by"], ["e1", "e2", "bz"],
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
    fc_h = compute_fc_seconds(runts, ["hx", "hy"], ["ex", "ey"],
                      window_length_s=256, overlap=0.5, window_type="hamming",
                      magnetic_unit="h")
    fc_b = compute_fc_seconds(runts, ["hx", "hy"], ["ex", "ey"],
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

    fc_b_auto = compute_fc_seconds(runts, ["bx", "by"], ["e1", "e2"],
                           window_length_s=256, overlap=0.5, window_type="hamming",
                           magnetic_unit="auto")
    fc_b_manual = compute_fc_seconds(runts, ["bx", "by"], ["e1", "e2"],
                             window_length_s=256, overlap=0.5, window_type="hamming",
                             magnetic_unit="b")
    np.testing.assert_allclose(fc_b_auto["bx"].values, fc_b_manual["bx"].values)
    assert fc_b_auto.attrs["magnetic_unit"] == "b"
    assert fc_b_auto.attrs["channel_nomenclature"] == "lemi12"

    # hx/hy naming (default nomenclature) means coil H: unchanged, unit 'h'
    runts_h = _synthetic_runts()
    fc_h_auto = compute_fc_seconds(runts_h, ["hx", "hy"], ["ex", "ey"],
                           window_length_s=256, overlap=0.5, window_type="hamming",
                           magnetic_unit="auto")
    fc_h_manual = compute_fc_seconds(runts_h, ["hx", "hy"], ["ex", "ey"],
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
    fc = compute_fc_seconds(runts, ["bx", "by"], ["e3", "e4"],
                    window_length_s=128, overlap=0.5, window_type="hamming")
    assert fc.attrs["channel_nomenclature"] == "lemi34"
    periods, Z, _ = estimate_tf(fc, ["bx", "by"], ["e3", "e4"],
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


def test_build_tf_returns_mt_metadata_tf():
    runts = _synthetic_runts()
    fc = compute_fc_seconds(runts, ["hx", "hy"], ["ex", "ey"],
                    window_length_s=256, overlap=0.5, window_type="hamming")
    periods, Z, _ = estimate_tf(fc, ["hx", "hy"], ["ex", "ey"])
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
    fc = compute_fc_seconds(runts, ["hx", "hy"], ["ex", "ey"],
                    window_length_s=256, overlap=0.5, window_type="hamming")
    periods, Z, _ = estimate_tf(fc, ["hx", "hy"], ["ex", "ey"],
                             estimator_engine=engine, remote_reference=rr,
                             periods_per_octave=8)
    assert periods.size > 1
    assert np.all(np.isfinite(Z))
    assert np.any(np.abs(Z) > 0)


def test_legacy_estimator_engine_names_still_accepted():
    """Old OLS/RME/RME_RR values keep loading previously saved workflows."""
    runts = _synthetic_runts()
    fc = compute_fc_seconds(runts, ["hx", "hy"], ["ex", "ey"],
                    window_length_s=256, overlap=0.5, window_type="hamming")
    for engine in ("OLS", "RME", "RME_RR"):
        rr = engine == "RME_RR"
        _, Z, _ = estimate_tf(fc, ["hx", "hy"], ["ex", "ey"],
                           estimator_engine=engine, remote_reference=rr,
                           periods_per_octave=8)
        assert np.all(np.isfinite(Z))


def test_remote_reference_uses_duplicated_dummy_station():
    """RME_RR with a 'dummy' remote station.

    We don't have two synchronized stations on disk, so we model the remote
    reference by duplicating the original data file — computing a second fc
    from a copy of the same time series and passing it as ref_fc. Identical
    data makes this a self-reference, but it exercises the full ref_fc path.
    """
    runts = _synthetic_runts()
    fc = compute_fc_seconds(runts, ["hx", "hy"], ["ex", "ey"],
                    window_length_s=256, overlap=0.5, window_type="hamming")
    # duplicate the raw data as a dummy remote station -> separate fc
    remote = compute_fc_seconds(runts, ["hx", "hy"], ["ex", "ey"],
                        window_length_s=256, overlap=0.5, window_type="hamming")
    periods, Z, _ = estimate_tf(fc, ["hx", "hy"], ["ex", "ey"],
                             estimator_engine="Robust", remote_reference=True,
                             ref_fc=remote, periods_per_octave=8)
    assert periods.size > 1
    assert np.all(np.isfinite(Z))
    assert np.any(np.abs(Z) > 0)

    # the estimator_engine must be honored: robust (not silently OLS)
    _, Z_ols, _ = estimate_tf(fc, ["hx", "hy"], ["ex", "ey"],
                           estimator_engine="Least Squares", periods_per_octave=8)
    assert not np.allclose(Z, Z_ols)


def test_remote_reference_without_remote_fc_warns(caplog):
    """RR requested but ref_fc absent: estimate against a self-reference AND
    warn loudly — the operator must never believe remote reference is in
    effect when it silently degenerated to plain RME (E2)."""
    runts = _synthetic_runts()
    fc = compute_fc_seconds(runts, ["hx", "hy"], ["ex", "ey"],
                    window_length_s=256, overlap=0.5, window_type="hamming")
    with caplog.at_level(logging.WARNING, logger="emptyui"):
        periods, Z, _ = estimate_tf(fc, ["hx", "hy"], ["ex", "ey"],
                                 estimator_engine="Robust",
                                 remote_reference=True,
                                 periods_per_octave=8)
    assert any("SELF-REFERENCE" in r.message for r in caplog.records)
    # the degenerate estimate still runs and is well-defined (equals RME)
    assert periods.size > 1
    assert np.all(np.isfinite(Z))


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
                             estimator_engine="Robust", periods_per_octave=4,
                             band_style="uniform")
    assert list(per) == [0, 1, 2, 3]

    per_level_counts = [len(per[lv]["periods"]) for lv in per]
    P, Z, _ = merge_tf_levels(per)
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


def _runts_with_channels(channels: dict[str, np.ndarray], sr: float = 8.0):
    """Build a fake ts.run-like object with the given named channels."""
    import xarray as xr

    n = len(next(iter(channels.values())))
    t = np.arange(n) / sr

    class FakeRunTS:
        sample_rate = sr

        def __init__(self):
            self.dataset = xr.Dataset(
                {k: ("time", v) for k, v in channels.items()},
                coords={"time": t},
            )

    return FakeRunTS()


def test_compute_fc_auto_detects_lemi_channels_including_tipper():
    """Empty input/output fields resolve from the data + nomenclature: lemi12
    names (bx,by / e1,e2) plus the vertical magnetic channel (bz) — the tipper
    is included by default when the run carries it."""
    rng = np.random.default_rng(3)
    n = 4096
    runts = _runts_with_channels({
        "bx": rng.standard_normal(n), "by": rng.standard_normal(n),
        "e1": rng.standard_normal(n), "e2": rng.standard_normal(n),
        "bz": rng.standard_normal(n),
    })
    fc = compute_fc_seconds(runts, "", "",
                    window_length_s=128, overlap=0.5, window_type="hamming")
    assert set(fc.data_vars) == {"bx", "by", "e1", "e2", "bz"}
    assert fc.attrs["input_channels"] == "bx,by"
    assert fc.attrs["output_channels"] == "e1,e2,bz"
    assert fc.attrs["channel_nomenclature"] == "lemi12"


def test_compute_fc_auto_drops_missing_vertical():
    """No vertical channel in the data -> impedance-only FC, no error."""
    rng = np.random.default_rng(4)
    n = 4096
    runts = _runts_with_channels({
        "bx": rng.standard_normal(n), "by": rng.standard_normal(n),
        "e1": rng.standard_normal(n), "e2": rng.standard_normal(n),
    })
    fc = compute_fc_seconds(runts, "", "",
                    window_length_s=128, overlap=0.5, window_type="hamming")
    assert set(fc.data_vars) == {"bx", "by", "e1", "e2"}
    assert fc.attrs["output_channels"] == "e1,e2"


def test_compute_fc_auto_tipper_only_when_no_electric():
    """Magnetic-only run (bx,by,bz) -> tipper-only FC (no impedance channels)."""
    rng = np.random.default_rng(5)
    n = 4096
    runts = _runts_with_channels({
        "bx": rng.standard_normal(n), "by": rng.standard_normal(n),
        "bz": rng.standard_normal(n),
    })
    fc = compute_fc_seconds(runts, "", "",
                    window_length_s=128, overlap=0.5, window_type="hamming")
    assert set(fc.data_vars) == {"bx", "by", "bz"}
    assert fc.attrs["output_channels"] == "bz"


def test_compute_fc_auto_phoenix_names():
    """phoenix123 data (h1,h2,e1,e2,h3) resolves to the actual names."""
    rng = np.random.default_rng(6)
    n = 4096
    runts = _runts_with_channels({
        "h1": rng.standard_normal(n), "h2": rng.standard_normal(n),
        "e1": rng.standard_normal(n), "e2": rng.standard_normal(n),
        "h3": rng.standard_normal(n),
    })
    fc = compute_fc_seconds(runts, "", "",
                    window_length_s=128, overlap=0.5, window_type="hamming")
    assert set(fc.data_vars) == {"h1", "h2", "e1", "e2", "h3"}
    assert fc.attrs["input_channels"] == "h1,h2"
    assert fc.attrs["output_channels"] == "e1,e2,h3"
    assert fc.attrs["channel_nomenclature"] == "phoenix123"
    # coil sensors: H already, no B->H conversion
    assert fc.attrs["magnetic_unit"] == "h"


def test_compute_fc_auto_standard_names():
    """Standard-named data with empty fields resolves to hx,hy / ex,ey."""
    runts = _synthetic_runts()
    fc = compute_fc_seconds(runts, "", "",
                    window_length_s=256, overlap=0.5, window_type="hamming")
    assert set(fc.data_vars) == {"hx", "hy", "ex", "ey"}
    assert fc.attrs["channel_nomenclature"] == "default"


def test_compute_fc_auto_case_insensitive_channel_names():
    """CSV-style uppercase channel names (Ex,Ey,Hx,Hy) resolve case-
    insensitively; the fc uses the exact names present in the data."""
    rng = np.random.default_rng(9)
    n = 4096
    runts = _runts_with_channels({
        "Ex": rng.standard_normal(n), "Ey": rng.standard_normal(n),
        "Hx": rng.standard_normal(n), "Hy": rng.standard_normal(n),
    })
    fc = compute_fc_seconds(runts, "", "",
                    window_length_s=128, overlap=0.5, window_type="hamming")
    assert set(fc.data_vars) == {"Ex", "Ey", "Hx", "Hy"}
    assert fc.attrs["input_channels"] == "Hx,Hy"
    assert fc.attrs["output_channels"] == "Ex,Ey"
    assert fc.attrs["channel_nomenclature"] == "default"



def test_nan_in_unused_auxiliary_channel_is_tolerated():
    """The NaN guard applies only to the channels that are actually STFT'd:
    a NaN in an auxiliary data_var (e.g. a LEMI temperature channel) must not
    hard-fail ComputeFC (2026-09-11 review fix)."""
    rng = np.random.default_rng(11)
    n = 4096
    aux = rng.standard_normal(n)
    aux[100] = np.nan  # NaN in a channel ComputeFC never touches
    runts = _runts_with_channels({
        "hx": rng.standard_normal(n), "hy": rng.standard_normal(n),
        "ex": rng.standard_normal(n), "ey": rng.standard_normal(n),
        "temperature": aux,
    })
    fc = compute_fc_seconds(runts, "hx,hy", "ex,ey",
                    window_length_s=128, overlap=0.5, window_type="hamming")
    assert set(fc.data_vars) == {"hx", "hy", "ex", "ey"}
    assert np.all(np.isfinite(fc["ex"].values))


def test_estimate_tf_no_banding_inherit_and_remote_reference_rule():
    """Manifest-level contracts (DESIGN.md §10.3 + §4 `enabled_when`):
    - Estimate TF declares NO chain-walking inherit of banding params from
      the fc chain (§24.9 — the sort is band-free; the estimator owns
      banding);
    - remote_reference declares its estimator condition instead of the
      frontend string-sniffing the engine value."""
    from emptyui_aurora.process import CoherenceSort, EstimateTF

    m = EstimateTF().manifest()
    assert not m["inherit"], "banding inherit from the sort must be gone"
    # the upstream sort indeed declares none of the banding params anymore
    sort_params = {p["name"] for p in CoherenceSort().manifest()["params"]}
    assert not ({"band_style", "periods_per_octave",
                 "min_bins_per_band", "t_short"} & sort_params)

    rr = next(p for p in m["params"] if p["name"] == "remote_reference")
    assert rr["enabled_when"] == {
        "param": "estimator_engine", "values": ["Robust", "RME", "RME_RR"],
    }


def test_compute_fc_levels_starved_cascade_stops(tmp_path):
    """Once a decimation level has fewer samples than one window, the
    cascade stops: deeper levels are strictly shorter, and decimating them
    anyway crashed scipy's decimate (padlen) on the full 3.5-day record
    (history §24.12 comparison run)."""
    from _synthetic import synthetic_runts, synthetic_source
    from emptyui_aurora._processing import compute_fc_levels

    runts = synthetic_runts(synthetic_source(31, n=600))
    levels = compute_fc_levels(runts, "hx,hy", "ex,ey", num_decimations=10,
                               num_samples_window=256, overlap=0.5,
                               window_type="boxcar")
    # level 0 fits (600 samples); decimating twice leaves 37 < 256 — the
    # cascade must stop there instead of crashing
    assert 0 in levels
    assert all(lv <= 2 for lv in levels)
