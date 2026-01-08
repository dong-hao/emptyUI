"""Coherence sort (EMTF-style) tests: single/two-FC semantics, trimming and
diagnostics. Error/boundary surfaces live in test_errors.py."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from _synthetic import synthetic_fc, synthetic_runts, synthetic_source
from emptyui.context import RunContext
from emptyui.store import DataStore
from emptyui_aurora._processing import (
    _multiple_coherence,
    coherence_sort_fc,
    estimate_tf,
)


def test_multiple_coherence_known_values():
    """Perfectly coherent E = Z·H scores ≈ 1; adding independent noise lowers it."""
    rng = np.random.default_rng(3)
    H = rng.standard_normal((2, 12)) + 1j * rng.standard_normal((2, 12))
    Z = np.array([[1 + 2j, 0.5 - 1j], [-0.5 + 0.3j, 2 + 0j]])
    E = Z @ H  # (2, 12)
    coh = _multiple_coherence(E, H)
    assert np.all(coh > 0.999)
    E_noisy = E + 5.0 * (rng.standard_normal((2, 12)) + 1j * rng.standard_normal((2, 12)))
    coh_noisy = _multiple_coherence(E_noisy, H)
    assert np.all(coh_noisy < coh)
    assert np.all(coh_noisy >= 0.0)
    assert np.all(coh_noisy <= 1.0)


def test_coherence_sort_drop_removes_noisy_windows():
    """Windows overlapping a noisy H segment are NaN'd (drop) and estimate_tf
    still produces a finite estimate (non-finite rows are skipped)."""
    source = synthetic_source(1)
    runts = synthetic_runts(source, h_noise_segment=(2000, 2600), seed=1)
    fc = synthetic_fc(runts, ["hx", "hy"], ["ex", "ey"])
    out, _, rows = coherence_sort_fc(fc, None, ["hx", "hy"], ["ex", "ey"],
                                     coherence_min=0.25, coherence_max=1.0,
                                     mode="drop", band_style="uniform",
                                     periods_per_octave=4, statistic="band")
    win = out["time"].values
    # fs = 8 Hz, window = 256 s -> scipy STFT times are window centres at
    # 128 s spacing; the noisy segment (samples 2000-2600 = 250-325 s) makes
    # the windows centred at 256/384 s incoherent, later ones are clean
    bad1 = int(np.argmin(np.abs(win - 256.0)))
    bad2 = int(np.argmin(np.abs(win - 384.0)))
    good = int(np.argmin(np.abs(win - 640.0)))
    assert bad1 != bad2
    # sorting applies only to the banded frequencies (the band grid's outer
    # edges); the noisy windows are dropped (NaN) in the bands where the
    # injected noise dominates the signal (short periods), clean windows stay
    # finite everywhere
    from emptyui_aurora._processing import _normalize_fc_layout, _period_bands
    _, edges = _period_bands(_normalize_fc_layout(out), 4)
    logf = np.log2(out["frequency"].values)
    banded = (logf <= edges[0]) & (logf > edges[-1])
    for ch in ("ex", "ey", "hx", "hy"):
        vals = out[ch].values
        assert np.isnan(vals[banded, bad1]).any()
        assert np.isnan(vals[banded, bad2]).any()
        assert np.all(np.isfinite(vals[banded, good]))
        assert np.all(np.isfinite(vals[~banded, bad1]))
    # estimate_tf skips the NaN rows without error
    _, Z, _ = estimate_tf(out, ["hx", "hy"], ["ex", "ey"],
                          periods_per_octave=8)
    assert np.all(np.isfinite(Z))
    # diagnostics recorded
    assert any(r["n_retained"] < r["n_windows"] for r in rows)


def test_two_fc_mode_uses_remote_reference():
    """Local E noisy, magnetics clean: single-FC mode drops (E-Hloc low) and
    two-FC mode is at least as strict (min of E-Hremote and Hloc-Hremote is
    bounded by the E noise), while the remote-FC copy carries exactly the
    same per-window treatment as the local copy (shared weight)."""
    source = synthetic_source(5)
    local_runts = synthetic_runts(source, e_noise=80.0, seed=5)
    remote_runts = synthetic_runts(source, e_noise=1.0, seed=6, with_e=False)
    local = synthetic_fc(local_runts, ["hx", "hy"], ["ex", "ey"])
    remote = synthetic_fc(remote_runts, ["hx", "hy"], ["hx", "hy"])

    # single-FC mode: E-Hloc coherence is low -> windows dropped
    s_out, _, _ = coherence_sort_fc(local, None, ["hx", "hy"], ["ex", "ey"],
                                    coherence_min=0.25, coherence_max=0.98,
                                    mode="drop", statistic="band")
    # two-FC mode: the conservative min drops at least the same windows
    t_out, r_out, rows = coherence_sort_fc(local, remote, ["hx", "hy"],
                                           ["ex", "ey"], coherence_min=0.25,
                                           coherence_max=0.98, mode="drop",
                                           statistic="band")
    assert np.isnan(s_out["ex"].values).any()
    assert np.isnan(t_out["ex"].values).any()
    assert (np.isnan(t_out["ex"].values).sum()
            >= np.isnan(s_out["ex"].values).sum())
    # the shared weight is baked into BOTH FCs: identical NaN masks
    assert np.array_equal(np.isnan(t_out["ex"].values),
                          np.isnan(r_out["hx"].values))
    # diagnostics expose both criteria
    assert all(r["median_coherence_transfer"] is not None for r in rows)


def test_two_fc_mode_uses_remote_h_for_signal_coherence():
    """Two-FC mode computes E/H coherence against remote H (not local H).

    Corrupt local magnetic FCs while keeping local E and remote H clean:
    single-FC mode (E on local H) reports low signal coherence, while
    two-FC mode (E on remote H) reports clearly higher signal coherence.
    """
    source = synthetic_source(12)
    local_runts = synthetic_runts(source, e_noise=2.0, seed=12)
    remote_runts = synthetic_runts(source, e_noise=2.0, seed=13, with_e=False)
    local = synthetic_fc(local_runts, ["hx", "hy"], ["ex", "ey"])
    remote = synthetic_fc(remote_runts, ["hx", "hy"], ["hx", "hy"])

    # Corrupt only the local magnetic predictors in FC space.
    rng = np.random.default_rng(123)
    bad_local = local.copy(deep=True)
    for ch in ("hx", "hy"):
        noise = (200.0 * rng.standard_normal(bad_local[ch].shape)
                 + 1j * 200.0 * rng.standard_normal(bad_local[ch].shape))
        bad_local[ch].values = bad_local[ch].values + noise

    _, _, rows_single = coherence_sort_fc(
        bad_local, None, ["hx", "hy"], ["ex", "ey"],
        coherence_min=0.25, coherence_max=0.98, mode="off",
    )
    _, _, rows_two = coherence_sort_fc(
        bad_local, remote, ["hx", "hy"], ["ex", "ey"],
        coherence_min=0.25, coherence_max=0.98, mode="off",
    )

    med_single = float(np.median([r["median_coherence_e"] for r in rows_single]))
    med_two = float(np.median([r["median_coherence_e"] for r in rows_two]))
    assert med_two > med_single + 0.2


def test_two_fc_mode_conservative_min():
    """Noisy local H: Hloc-Hremote is low, so two-FC mode (min) drops the
    windows even if E-Hremote looks fine — aurora's RME_RR signal covariance
    is unstable without local<->remote magnetic coherence."""
    source = synthetic_source(7)
    local_runts = synthetic_runts(source, h_noise_segment=(2000, 2600), seed=7)
    remote_runts = synthetic_runts(source, e_noise=1.0, seed=8, with_e=False)
    local = synthetic_fc(local_runts, ["hx", "hy"], ["ex", "ey"])
    remote = synthetic_fc(remote_runts, ["hx", "hy"], ["hx", "hy"])
    t_out, r_out, _ = coherence_sort_fc(local, remote, ["hx", "hy"],
                                        ["ex", "ey"], coherence_min=0.25,
                                        coherence_max=0.98, mode="drop",
                                        statistic="band")
    win = t_out["time"].values
    bad1 = int(np.argmin(np.abs(win - 256.0)))
    bad2 = int(np.argmin(np.abs(win - 384.0)))
    assert bad1 != bad2
    freqs = t_out["frequency"].values
    banded = (freqs > 0) & (freqs <= 2.0)
    # the noisy windows are dropped in the bands where the H noise dominates
    # and the same treatment is applied to the remote FC (shared weight)
    assert np.isnan(t_out["hx"].values[banded, bad1]).any()
    assert np.isnan(t_out["hx"].values[banded, bad2]).any()
    assert np.array_equal(np.isnan(t_out["hx"].values),
                          np.isnan(r_out["hx"].values))


def test_trim_removes_noisy_band():
    """Band-edge trimming NaNs the whole band when its retained-window median
    coherence stays below the lower bound."""
    source = synthetic_source(10)
    runts = synthetic_runts(source, h_noise_segment=(0, 8192), seed=10)
    fc = synthetic_fc(runts, ["hx", "hy"], ["ex", "ey"])
    out, _, rows = coherence_sort_fc(fc, None, ["hx", "hy"], ["ex", "ey"],
                                     coherence_min=0.5, coherence_max=1.0,
                                     mode="drop", trim=True,
                                     band_style="uniform",
                                     periods_per_octave=4, statistic="band")
    assert any(r["trimmed"] for r in rows)
    # the fully-trimmed bands are NaN across every window; bins outside the
    # banded range (DC, above the shortest band edge) are never touched
    assert np.isnan(out["ex"].values).any()
    from emptyui_aurora._processing import _normalize_fc_layout, _period_bands
    _, edges = _period_bands(_normalize_fc_layout(out), 4)
    logf = np.log2(out["frequency"].values)
    banded = (logf <= edges[0]) & (logf > edges[-1])
    assert np.all(np.isfinite(out["ex"].values[~banded]))
    # trimmed rows are all-NaN over their whole band; retained bands are finite
    from emptyui_aurora._processing import _normalize_fc_layout, _period_bands
    periods, edges = _period_bands(_normalize_fc_layout(out), 4)
    logf = np.log2(out["frequency"].values)
    # map each diagnostic row back to its band and check consistency
    for r in rows:
        p = r["period"]
        b = int(np.argmin(np.abs(np.array(periods) - p)))
        sel = (logf <= edges[b]) & (logf > edges[b + 1])
        bins = np.where(sel)[0]
        if r["trimmed"]:
            assert np.all(np.isnan(out["ex"].values[bins]))
        else:
            assert np.isfinite(out["ex"].values[bins]).any()


def test_node_execute_returns_table(tmp_path: Path):
    from emptyui_aurora.process import CoherenceSort

    ctx = RunContext(workspace=tmp_path, store=DataStore())
    fc = synthetic_fc(synthetic_runts(synthetic_source(11), seed=11), ["hx", "hy"], ["ex", "ey"])
    res = CoherenceSort().execute(ctx, fc, coherence_min=0.25,
                                  coherence_max=0.98, mode="weight")
    assert set(res["fc"].keys()) == {0}
    rows = json.loads(Path(res["table"]["path"]).read_text())
    assert rows and "period" in rows[0] and "n_retained" in rows[0]
    # weight mode keeps retained samples finite (fully down-weighted ones
    # are NaN, exactly like drop) — the good windows must survive
    assert np.isfinite(res["fc"][0]["ex"].values).any()



def test_node_band_params_match_estimate_tf(tmp_path: Path):
    """The sort's l2u banding must be identical to Estimate TF's when the
    same min_bins_per_band/t_short are passed — previously the node
    hard-coded min_bins_per_band=40 and had no t_short, silently banding
    differently from a customized estimator (2026-09-11 review fix)."""
    from emptyui_aurora._processing import compute_fc_levels, estimate_tf_levels
    from emptyui_aurora.process import CoherenceSort

    runts = synthetic_runts(synthetic_source(12), seed=12)
    levels = compute_fc_levels(runts, ["hx", "hy"], ["ex", "ey"],
                               num_decimations=2, factor=4,
                               num_samples_window=256, overlap=0.5,
                               window_type="hamming")
    ctx = RunContext(workspace=tmp_path, store=DataStore())
    res = CoherenceSort().execute(ctx, levels, periods_per_octave=4,
                                  band_style="l2u", min_bins_per_band=40.0,
                                  t_short=0.0)
    rows = json.loads(Path(res["table"]["path"]).read_text())
    sort_periods = {(r["decimation_level"], round(r["period"], 6)) for r in rows}

    per = estimate_tf_levels(levels, ["hx", "hy"], ["ex", "ey"],
                             estimator_engine="Robust", periods_per_octave=4,
                             band_style="l2u", min_bins_per_band=40.0,
                             t_short=0.0)
    est_periods = {(lv, round(float(p), 6))
                   for lv, e in per.items() for p in e["periods"]}
    assert sort_periods == est_periods

    # a head clip on the node removes the clipped bands from the diagnostics
    res2 = CoherenceSort().execute(
        RunContext(workspace=tmp_path, store=DataStore()), levels,
        periods_per_octave=4, band_style="l2u", min_bins_per_band=40.0,
        t_short=4.0)
    rows2 = json.loads(Path(res2["table"]["path"]).read_text())
    assert rows2 and all(r["period"] >= 4.0 for r in rows2)


def test_trim_disabled_with_mode_off(caplog):
    """mode="off" means no sorting at all: band-edge trimming must not NaN
    bands even when their median coherence is below coherence_min
    (regression: the trim block ran before the mode gate)."""
    source = synthetic_source(11)
    runts = synthetic_runts(source, h_noise_segment=(0, 8192), seed=11)
    fc = synthetic_fc(runts, ["hx", "hy"], ["ex", "ey"])
    out, _, rows = coherence_sort_fc(fc, None, ["hx", "hy"], ["ex", "ey"],
                                     coherence_min=0.5, coherence_max=1.0,
                                     mode="off", trim=True,
                                     band_style="uniform")
    assert not any(r["trimmed"] for r in rows)
    assert np.all(np.isfinite(out["ex"].values))
    # the trimmed-band warning is part of the drop/weight modes' contract
    out2, _, rows2 = coherence_sort_fc(fc, None, ["hx", "hy"], ["ex", "ey"],
                                       coherence_min=0.5, coherence_max=1.0,
                                       mode="drop", trim=True,
                                       band_style="uniform")
    assert any(r["trimmed"] for r in rows2)
    assert np.isnan(out2["ex"].values).any()


def test_apply_below_period_scope():
    """§24.5 P3: bands at/above apply_below_period pass through unsorted
    (bit-identical to the input FC) while bands below the threshold are
    sorted as usual — the head-scoped remedy."""
    source = synthetic_source(12)
    runts = synthetic_runts(source, h_noise_segment=(0, 8192), seed=12)
    fc = synthetic_fc(runts, ["hx", "hy"], ["ex", "ey"])
    out, _, rows = coherence_sort_fc(
        fc, None, ["hx", "hy"], ["ex", "ey"],
        coherence_min=0.5, coherence_max=1.0, mode="drop",
        band_style="uniform", apply_below_period=32.0)
    # bands above the threshold are untouched
    for r in rows:
        if r["period"] >= 32.0:
            assert r.get("unsorted"), r
    # find a banded period >= 32 s and check its bins are bit-identical
    from emptyui_aurora._processing import _normalize_fc_layout, _period_bands
    periods, edges = _period_bands(_normalize_fc_layout(out), 4)
    logf = np.log2(out["frequency"].values)
    checked = 0
    for r in rows:
        if not r.get("unsorted"):
            continue
        b = int(np.argmin(np.abs(np.array(periods) - r["period"])))
        sel = (logf <= edges[b]) & (logf > edges[b + 1])
        bins = np.where(sel)[0]
        if bins.size:
            np.testing.assert_array_equal(
                out["hx"].values[bins], fc["hx"].values[bins])
            checked += 1
    assert checked, "no unsorted band verified"
    # and at least one band below the threshold WAS sorted (NaNs present)
    short_bands = [r for r in rows if not r.get("unsorted")]
    assert short_bands, "expected bands below the threshold to be sorted"
    got_nan = False
    for r in short_bands:
        b = int(np.argmin(np.abs(np.array(periods) - r["period"])))
        sel = (logf <= edges[b]) & (logf > edges[b + 1])
        bins = np.where(sel)[0]
        if bins.size and np.isnan(out["hx"].values[bins]).any():
            got_nan = True
            break
    assert got_nan, "no band below the threshold was actually sorted"


def test_per_bin_statistic_known_values():
    """The per-bin window-ensemble coherence: perfectly coherent E = Z·H
    scores ≈ 1 at every (bin, window); added noise lowers it; the
    adjusted-R² correction keeps a pure-noise bin at ≈ 0 (the C2 bias is
    removed, unlike the band statistic)."""
    from emptyui_aurora._processing import _per_bin_coherence

    rng = np.random.default_rng(5)
    F, W = 6, 40
    H = rng.standard_normal((2, F, W)) + 1j * rng.standard_normal((2, F, W))
    Z = np.array([[1 + 2j, 0.5 - 1j], [-0.5 + 0.3j, 2 + 0j]])
    E = np.einsum("oi,ifw->ofw", Z, H)  # E_o = Σ_i Z[o,i]·H_i (exact fit)
    coh_e, coh_x = _per_bin_coherence(
        _ds(E[0], E[1], H[0], H[1]), None, ["hx", "hy"], ["ex", "ey"],
        ensemble_windows=8)
    assert coh_e.shape == (F, W)
    assert np.all(coh_e > 0.95)  # perfectly coherent, debiased
    assert np.all(coh_x == 1.0)  # single-FC mode
    # noise-dominated: E = pure independent noise → coh ≈ 0 (debias removes
    # the p/(W-1) sample bias that would otherwise inflate it to ~2/16)
    noise = (rng.standard_normal((2, F, W))
             + 1j * rng.standard_normal((2, F, W)))
    coh_n, _ = _per_bin_coherence(
        _ds(noise[0], noise[1], H[0], H[1]), None, ["hx", "hy"],
        ["ex", "ey"], ensemble_windows=8)
    assert np.all(coh_n < 0.5)


def _ds(hx, hy, ex, ey):
    import pandas as pd
    import xarray as xr

    t = pd.date_range("2024-01-01", periods=hx.shape[1], freq="s")
    f = np.arange(1, hx.shape[0] + 1) * 0.125
    return xr.Dataset(
        {"hx": (("frequency", "time"), hx), "hy": (("frequency", "time"), hy),
         "ex": (("frequency", "time"), ex), "ey": (("frequency", "time"), ey)},
        coords={"frequency": f, "time": t},
        attrs={"sample_rate": 8.0, "num_samples_window": 64,
               "overlap": 0.5, "window_type": "boxcar"})


def test_per_bin_sort_band_independent():
    """The per-bin weights must be identical under two different band
    tables (banding independence — the whole point of E7): sorting with
    bands_override=None (auto bands) vs a different override leaves the
    FC values identical (the band table only aggregates diagnostics)."""
    source = synthetic_source(14)
    runts = synthetic_runts(source, h_noise_segment=(0, 8192), seed=14)
    fc = synthetic_fc(runts, ["hx", "hy"], ["ex", "ey"])
    kw = {"coherence_min": 0.5, "coherence_max": 1.0, "mode": "drop",
          "statistic": "per_bin", "ensemble_windows": 8}
    out1, _, _ = coherence_sort_fc(fc, None, ["hx", "hy"], ["ex", "ey"],
                                   band_style="uniform", **kw)
    out2, _, _ = coherence_sort_fc(fc, None, ["hx", "hy"], ["ex", "ey"],
                                   band_style="l2u", **kw)
    np.testing.assert_array_equal(out1["ex"].values, out2["ex"].values)
    np.testing.assert_array_equal(out1["hx"].values, out2["hx"].values)
    # and the weights actually did something (noise-floor windows NaN'd)
    assert np.isnan(out1["ex"].values).any()


def test_per_bin_vs_band_mode_grammar():
    """Both statistics keep the NaN-row grammar: mode="drop" excludes
    low-coherence samples (NaN) in both flavors; the per-bin mode covers
    every bin (no band table in the weighting)."""
    source = synthetic_source(15)
    runts = synthetic_runts(source, h_noise_segment=(0, 8192), seed=15)
    fc = synthetic_fc(runts, ["hx", "hy"], ["ex", "ey"])
    out, _, rows = coherence_sort_fc(fc, None, ["hx", "hy"], ["ex", "ey"],
                                     coherence_min=0.5, coherence_max=1.0,
                                     mode="drop", band_style="uniform",
                                     statistic="per_bin")
    assert np.isnan(out["ex"].values).any()
    assert np.all(np.isfinite(out["ex"].values[~np.isnan(out["ex"].values)]))
    assert any(r["n_retained"] < r["n_windows"] for r in rows)
    # diagnostics still aggregate per band
    assert all("period" in r and "n_retained" in r for r in rows)


def test_per_bin_scope_leaves_tail_untouched():
    """apply_below_period with the per-bin statistic: bins at/above the
    threshold are bit-identical to the input; bins below may be sorted."""
    source = synthetic_source(16)
    runts = synthetic_runts(source, h_noise_segment=(0, 8192), seed=16)
    fc = synthetic_fc(runts, ["hx", "hy"], ["ex", "ey"])
    out, _, _ = coherence_sort_fc(
        fc, None, ["hx", "hy"], ["ex", "ey"], coherence_min=0.5,
        coherence_max=1.0, mode="drop", band_style="uniform",
        statistic="per_bin", apply_below_period=8.0)
    freqs = np.asarray(out["frequency"].values)
    tail = (freqs < 1.0 / 8.0) & (freqs > 0)  # period >= 8 s, DC excluded
    if tail.any():
        np.testing.assert_array_equal(out["hx"].values[tail],
                                      fc["hx"].values[tail])
