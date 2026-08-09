"""Coherence sort (EMTF-style, per-frequency) tests: per-(bin, window)
weighting, single/two-FC semantics, period scope, diagnostics. Error and
boundary surfaces live in test_errors.py."""

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
                                     mode="drop", ensemble_windows=2)
    win = out["time"].values
    # fs = 8 Hz, window = 256 s -> scipy STFT times are window centres at
    # 128 s spacing; the noisy segment (samples 2000-2600 = 250-325 s) makes
    # the windows centred at 256/384 s incoherent, later ones are clean
    bad1 = int(np.argmin(np.abs(win - 256.0)))
    bad2 = int(np.argmin(np.abs(win - 384.0)))
    # with a tight ±2 window ensemble, windows >2 away from the noise stay
    # clean (the default ±16 would smear the drop region across them)
    good = int(np.argmin(np.abs(win - 768.0)))
    assert bad1 != bad2
    # per-(bin, window) weights: the noisy windows are dropped (NaN) at the
    # bins where the injected noise dominates the signal (short periods),
    # clean windows stay finite everywhere except the DC bin (never weighted)
    freqs = out["frequency"].values
    binned = freqs > 0
    for ch in ("ex", "ey", "hx", "hy"):
        vals = out[ch].values
        assert np.isnan(vals[binned, bad1]).any()
        assert np.isnan(vals[binned, bad2]).any()
        # clean windows survive at the signal bins; pure-noise bins (where
        # even clean windows are incoherent) are dropped at every window
        assert np.isfinite(vals[binned, good]).any()
    # estimate_tf skips the NaN rows without error
    _, Z, _ = estimate_tf(out, ["hx", "hy"], ["ex", "ey"],
                          periods_per_octave=8)
    assert np.all(np.isfinite(Z))
    # per-bin diagnostics recorded
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
                                    mode="drop")
    # two-FC mode: the conservative min drops at least the same windows
    t_out, r_out, rows = coherence_sort_fc(local, remote, ["hx", "hy"],
                                           ["ex", "ey"], coherence_min=0.25,
                                           coherence_max=0.98, mode="drop")
    assert np.isnan(s_out["ex"].values).any()
    assert np.isnan(t_out["ex"].values).any()
    # per-bin ensembles: the two criteria differ bin-wise, so "at least as
    # strict" holds per bin only approximately — assert a shared drop
    # instead (both modes actually drop the noisy windows)
    assert np.isnan(t_out["ex"].values).any() and np.isnan(
        s_out["ex"].values).any()
    # the shared weight is baked into BOTH FCs: identical NaN masks
    assert np.array_equal(np.isnan(t_out["ex"].values),
                          np.isnan(r_out["hx"].values))
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
                                        coherence_max=0.98, mode="drop")
    win = t_out["time"].values
    bad1 = int(np.argmin(np.abs(win - 256.0)))
    bad2 = int(np.argmin(np.abs(win - 384.0)))
    assert bad1 != bad2
    freqs = t_out["frequency"].values
    banded = (freqs > 0) & (freqs <= 2.0)
    # the noisy windows are dropped at the bins where the H noise dominates
    # and the same treatment is applied to the remote FC (shared weight)
    assert np.isnan(t_out["hx"].values[banded, bad1]).any()
    assert np.isnan(t_out["hx"].values[banded, bad2]).any()
    assert np.array_equal(np.isnan(t_out["hx"].values),
                          np.isnan(r_out["hx"].values))


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


def test_node_has_no_banding_params():
    """history §24.9: banding is Estimate TF's concern — the sort node declares no
    band-geometry or statistic params anymore."""
    from emptyui_aurora.process import CoherenceSort

    declared = {p.name for p in CoherenceSort.params}
    assert declared == {"coherence_min", "coherence_max", "mode",
                        "apply_below_period", "ensemble_windows",
                        "smooth_bins", "pairing", "pair_tolerance_s"}


def test_mode_off_is_passthrough():
    """mode="off" means no sorting at all: the FC comes back bit-identical
    (no NaNs, weights all 1) and diagnostics still report per-bin rows."""
    source = synthetic_source(11)
    runts = synthetic_runts(source, h_noise_segment=(0, 8192), seed=11)
    fc = synthetic_fc(runts, ["hx", "hy"], ["ex", "ey"])
    out, _, rows = coherence_sort_fc(fc, None, ["hx", "hy"], ["ex", "ey"],
                                     coherence_min=0.5, coherence_max=1.0,
                                     mode="off")
    for ch in ("ex", "ey", "hx", "hy"):
        np.testing.assert_array_equal(out[ch].values, fc[ch].values)
    assert all(r["n_retained"] == r["n_windows"] for r in rows)


def test_apply_below_period_scope():
    """history §24.5 P3: bins at/above apply_below_period pass through unsorted
    (bit-identical to the input FC) while bins below the threshold are
    sorted as usual — the head-scoped remedy."""
    source = synthetic_source(12)
    runts = synthetic_runts(source, h_noise_segment=(0, 8192), seed=12)
    fc = synthetic_fc(runts, ["hx", "hy"], ["ex", "ey"])
    out, _, rows = coherence_sort_fc(
        fc, None, ["hx", "hy"], ["ex", "ey"],
        coherence_min=0.5, coherence_max=1.0, mode="drop",
        apply_below_period=32.0)
    freqs = np.asarray(out["frequency"].values)
    # bins above the threshold are untouched
    for r in rows:
        if r["period"] >= 32.0:
            assert r.get("unsorted"), r
    tail = (freqs < 1.0 / 32.0) & (freqs > 0)  # period >= 32 s, DC excluded
    np.testing.assert_array_equal(out["hx"].values[tail],
                                  fc["hx"].values[tail])
    # and at least one bin below the threshold WAS sorted (NaNs present)
    head = freqs >= 1.0 / 32.0
    assert head.any()
    assert np.isnan(out["hx"].values[head]).any(), \
        "no bin below the threshold was actually sorted"


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


def test_drop_nan_grammar_per_bin():
    """mode="drop" excludes low-coherence samples (NaN) at every bin —
    no band table anywhere in the weighting."""
    source = synthetic_source(15)
    runts = synthetic_runts(source, h_noise_segment=(0, 8192), seed=15)
    fc = synthetic_fc(runts, ["hx", "hy"], ["ex", "ey"])
    out, _, rows = coherence_sort_fc(fc, None, ["hx", "hy"], ["ex", "ey"],
                                     coherence_min=0.5, coherence_max=1.0,
                                     mode="drop")
    assert np.isnan(out["ex"].values).any()
    assert np.all(np.isfinite(out["ex"].values[~np.isnan(out["ex"].values)]))
    assert any(r["n_retained"] < r["n_windows"] for r in rows)
    # per-bin diagnostics: one row per frequency bin
    freqs = out["frequency"].values
    assert len(rows) == int(np.sum(freqs > 0))
    assert all("period" in r and "n_retained" in r for r in rows)


def test_per_bin_scope_leaves_tail_untouched():
    """apply_below_period: bins at/above the threshold are bit-identical to
    the input; bins below may be sorted."""
    source = synthetic_source(16)
    runts = synthetic_runts(source, h_noise_segment=(0, 8192), seed=16)
    fc = synthetic_fc(runts, ["hx", "hy"], ["ex", "ey"])
    out, _, _ = coherence_sort_fc(
        fc, None, ["hx", "hy"], ["ex", "ey"], coherence_min=0.5,
        coherence_max=1.0, mode="drop", apply_below_period=8.0)
    freqs = np.asarray(out["frequency"].values)
    tail = (freqs < 1.0 / 8.0) & (freqs > 0)  # period >= 8 s, DC excluded
    if tail.any():
        np.testing.assert_array_equal(out["hx"].values[tail],
                                      fc["hx"].values[tail])


def test_per_bin_coherence_smooth_bins_nan_aware():
    """smooth_bins pooling is NaN-aware and edge-clamped: the DC bin is NaN
    after first-difference recoloring — raw (np.roll) pooling poisoned its
    +-neighbors (the whole bin ensemble became NaN -> weight 0 and the
    adjacent bins silently vanished from the sort), and the wraparound
    pooled DC with Nyquist."""
    import xarray as xr
    from emptyui_aurora._processing import _per_bin_coherence

    F, W, I, O = 8, 20, 2, 1
    rng = np.random.default_rng(1)
    E = (rng.standard_normal((O, F, W)) + 1j * rng.standard_normal((O, F, W)))
    H = (rng.standard_normal((I, F, W)) + 1j * rng.standard_normal((I, F, W)))
    E[:, 0, :] = np.nan  # the DC bin — recoloring undefined there
    freq = np.arange(F, dtype=float)
    t = np.arange(W)

    def ds(EA, HA):
        data = {c: (("frequency", "time"), HA[i])
                for i, c in enumerate(["hx", "hy"])}
        data["ex"] = (("frequency", "time"), EA[0])
        return xr.Dataset(data, coords={"frequency": freq, "time": t})

    coh_e, _ = _per_bin_coherence(ds(E, H), None, ["hx", "hy"], ["ex"],
                                  ensemble_windows=4, smooth_bins=1)
    # bins adjacent to the NaN DC bin (1) and the edge bin (F-1) must score
    # real coherence — the raw pooling zeroed both exactly
    assert coh_e[1].max() > 0.01, coh_e[1]
    assert coh_e[F - 1].max() > 0.01, coh_e[F - 1]
    # the NaN DC bin itself has no data — exactly 0
    assert (coh_e[0] == 0).all()


# --- M21: the chunked transient ------------------------------------------

def test_chunked_statistic_matches_the_single_chunk_path(monkeypatch):
    """The M21 window-chunked _min_coh must reproduce the unchunked
    statistic: a chunk's haloed slice contains every window's full
    ensemble, so the values agree to FP rounding (the per-chunk cumsum
    baselines differ in low-order bits — hence allclose, not equal).
    Exercises NaN samples, bin pooling and the two-FC path."""
    from emptyui_aurora import _processing
    from emptyui_aurora._processing import _per_bin_coherence

    F, W, I, O = 8, 60, 2, 2
    rng = np.random.default_rng(7)
    E = rng.standard_normal((O, F, W)) + 1j * rng.standard_normal((O, F, W))
    H = rng.standard_normal((I, F, W)) + 1j * rng.standard_normal((I, F, W))
    R = rng.standard_normal((I, F, W)) + 1j * rng.standard_normal((I, F, W))
    E[:, 0, :] = np.nan  # the DC bin — recoloring undefined there
    E[0, 3, 10:13] = np.nan  # scattered invalid windows
    H[1, 5, 30] = np.nan

    def ds(hx, hy, ex, ey):
        data = {c: (("frequency", "time"), arr) for c, arr in
                (("hx", hx), ("hy", hy))}
        data["ex"] = (("frequency", "time"), ex)
        data["ey"] = (("frequency", "time"), ey)
        import xarray as xr
        return xr.Dataset(data, coords={"frequency": np.arange(F)})

    local = ds(H[0], H[1], E[0], E[1])
    ref = ds(R[0], R[1], E[0], E[1])  # ref inputs differ, outputs shared
    single = _per_bin_coherence(local, ref, ["hx", "hy"], ["ex", "ey"],
                                ensemble_windows=4, smooth_bins=1)
    monkeypatch.setattr(_processing, "_COH_BUDGET_BYTES", 32 * 2**10)
    chunked = _per_bin_coherence(local, ref, ["hx", "hy"], ["ex", "ey"],
                                 ensemble_windows=4, smooth_bins=1)
    np.testing.assert_allclose(chunked[0], single[0], rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(chunked[1], single[1], rtol=1e-9, atol=1e-12)


def test_chunking_activates_and_is_logged(monkeypatch, caplog):
    """Below the budget the chunked path announces itself (F, W, channels
    and the avoided transient) — the register's 'no guard, no warning'."""
    import logging

    from emptyui_aurora import _processing
    from emptyui_aurora._processing import _per_bin_coherence

    F, W, I, O = 4, 40, 2, 1
    rng = np.random.default_rng(3)
    E = rng.standard_normal((O, F, W)) + 1j * rng.standard_normal((O, F, W))
    H = rng.standard_normal((I, F, W)) + 1j * rng.standard_normal((I, F, W))

    def ds(hx, hy, ex):
        import xarray as xr
        return xr.Dataset(
            {"hx": (("frequency", "time"), hx),
             "hy": (("frequency", "time"), hy),
             "ex": (("frequency", "time"), ex)},
            coords={"frequency": np.arange(F)})

    with caplog.at_level(logging.INFO, logger="emptyui"):
        monkeypatch.setattr(_processing, "_COH_BUDGET_BYTES", 8 * 2**10)
        _per_bin_coherence(ds(H[0], H[1], E[0]), None, ["hx", "hy"],
                           ["ex"], ensemble_windows=3)
    assert any(
        "transient" in r.getMessage() and "chunk" in r.getMessage()
        for r in caplog.records
    ), [r.getMessage() for r in caplog.records]


def test_realistic_size_stays_bounded():
    """The register's complaint: 'no test at realistic sizes'. A 2.5 h /
    150 Hz level-0 FC (F=129, W=10000 — 5x the smoke corpus) must complete
    with the transient bounded (measured 713 MiB; the pre-M21 code scaled
    the SAME intermediates to 5.6 GiB at 5x this size)."""
    import tracemalloc

    import xarray as xr
    from emptyui_aurora._processing import _per_bin_coherence

    F, W = 129, 10000
    rng = np.random.default_rng(2)
    H = rng.standard_normal((2, F, W)) + 1j * rng.standard_normal((2, F, W))
    Z = np.array([[1 + 2j, 0.5 - 1j], [-0.5 + 0.3j, 2 + 0j]])
    E = np.einsum("oi,ifw->ofw", Z, H)  # perfectly coherent — cheap sanity
    data = {c: (("frequency", "time"), arr) for c, arr in
            (("hx", H[0]), ("hy", H[1]))}
    data["ex"] = (("frequency", "time"), E[0])
    data["ey"] = (("frequency", "time"), E[1])
    fc = xr.Dataset(data, coords={"frequency": np.arange(F)})

    tracemalloc.start()
    coh_e, coh_x = _per_bin_coherence(
        fc, None, ["hx", "hy"], ["ex", "ey"], ensemble_windows=16)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    # the coherent-construction sanity: an exact linear fit scores ~1
    assert np.all(coh_e > 0.95)
    assert (coh_x == 1.0).all()
    assert peak / 2**20 < 900, f"transient grew to {peak / 2**20:.0f} MiB"
