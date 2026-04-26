"""Tests for Merge FC (FC-database concatenation, history §24.5 P1 v2) and
the clock-zero window anchoring that makes per-run FCs pairable for remote
reference."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from _synthetic import synthetic_runts, synthetic_source
from emptyui_aurora._processing import (
    compute_fc_levels,
    estimate_tf,
    merge_fc_containers,
)


def _fc(times, freqs, chans=("hx", "hy", "ex", "ey"), sr=8.0, seed=0):
    """Small single-level fc Dataset with datetime64 windows."""
    import xarray as xr

    rng = np.random.default_rng(seed)
    data = {
        c: (("frequency", "time"),
            rng.standard_normal((len(freqs), len(times)))
            + 1j * rng.standard_normal((len(freqs), len(times))))
        for c in chans
    }
    return xr.Dataset(
        data_vars=data,
        coords={"frequency": np.asarray(freqs, dtype=float),
                "time": pd.to_datetime(list(times))},
        attrs={"sample_rate": sr, "num_samples_window": 64,
               "overlap": 0.5, "window_type": "boxcar",
               "decimation_level": 0},
    )


def _freqs(n=8, df=0.125):
    return [df * (k + 1) for k in range(n)]


def test_merge_fc_concat_sorted():
    """Two same-grid fcs with disjoint times: concatenated, time-sorted;
    attrs and channels come from the first input."""
    t1 = pd.date_range("2024-01-01", periods=3, freq="s")
    t2 = pd.date_range("2024-01-01T00:01:00", periods=4, freq="s")
    f = _freqs()
    a = _fc(t1, f, seed=1)
    b = _fc(t2, f, seed=2)
    merged = merge_fc_containers([a, b])
    assert sorted(merged) == [0]
    m = merged[0]
    assert m.sizes["time"] == 7
    assert (m["time"].values == np.concatenate([t1.values, t2.values])).all()
    assert list(m.data_vars) == list(a.data_vars)
    assert m.attrs["sample_rate"] == a.attrs["sample_rate"]
    np.testing.assert_array_equal(m["hx"].values[:, :3], a["hx"].values)
    np.testing.assert_array_equal(m["hx"].values[:, 3:], b["hx"].values)


def test_merge_fc_multilevel_union():
    """Level sets are unioned; a level missing in some input merges the
    rest (e.g. burst slots starving at deep levels)."""

    f = _freqs()
    t = pd.date_range("2024-01-01", periods=2, freq="s")
    a = {0: _fc(t, f, seed=3), 1: _fc(t, [f[0], f[1]], sr=2.0, seed=4)}
    b = {0: _fc(pd.date_range("2024-01-01T00:00:10", periods=2, freq="s"),
                f, seed=5)}
    a[1].attrs["sample_rate"] = 2.0
    merged = merge_fc_containers([a, b])
    assert sorted(merged) == [0, 1]
    assert merged[0].sizes["time"] == 4
    assert merged[1].sizes["time"] == 2  # only input a contributes


def test_merge_fc_intersects_frequency_grids(caplog):
    """Best-effort intersect: slightly drifted grids are intersected with a
    warning; non-intersecting grids error."""
    import logging


    f = _freqs()
    t = pd.date_range("2024-01-01", periods=2, freq="s")
    a = _fc(t, f, seed=6)
    drift = [x * (1 + 1e-9) for x in f]  # float drift — matched by tolerance
    b = _fc(pd.date_range("2024-01-01T00:00:10", periods=2, freq="s"),
            drift, seed=7)
    with caplog.at_level(logging.WARNING, logger="emptyui"):
        merged = merge_fc_containers([a, b])
    assert merged[0].sizes["frequency"] == len(f)  # all bins survive drift
    # disjoint grids: nothing survives
    c = _fc(pd.date_range("2024-01-01T00:00:20", periods=2, freq="s"),
            [10.0 + df for df in f], seed=8)
    with pytest.raises(ValueError, match="do not intersect"):
        merge_fc_containers([a, c])


def test_merge_fc_drops_missing_channels(caplog):
    """Channels missing in some input are dropped with a warning; an empty
    channel intersection errors."""
    import logging

    t = pd.date_range("2024-01-01", periods=2, freq="s")
    f = _freqs()
    a = _fc(t, f, seed=9)
    b = _fc(pd.date_range("2024-01-01T00:00:10", periods=2, freq="s"), f,
            chans=("hx", "hy", "ex"), seed=10)
    with caplog.at_level(logging.WARNING, logger="emptyui"):
        merged = merge_fc_containers([a, b])
    assert "ey" in caplog.text
    assert list(merged[0].data_vars) == ["hx", "hy", "ex"]
    with pytest.raises(ValueError, match="share no channels"):
        merge_fc_containers([_fc(t, f, seed=11),
                             _fc(t, f, chans=("hz",), seed=12)])


def test_merge_fc_mixed_rates_rekey_by_rate():
    """Levels are keyed by ACTUAL sample rate, not decimation index: a
    burst archive's rate families (whose cascade levels do not correspond
    by index) merge into consecutive int levels in DESCENDING rate order
 (history §24.8)."""
    t = pd.date_range("2024-01-01", periods=2, freq="s")
    f = _freqs()
    a = _fc(t, f, sr=8.0, seed=13)
    b = _fc(t, f, sr=2.0, seed=14)
    merged = merge_fc_containers([a, b])
    assert sorted(merged) == [0, 1]
    assert merged[0].attrs["sample_rate"] == 8.0
    assert merged[1].attrs["sample_rate"] == 2.0
    # single-rate inputs reproduce the input indices exactly
    merged2 = merge_fc_containers([a, a.isel(time=slice(0, 1))])
    assert sorted(merged2) == [0]


def test_merge_fc_requires_input():
    with pytest.raises(ValueError, match="no fc to merge"):
        merge_fc_containers([None, None])


def test_burst_mode_gapped_runs_rr(tmp_path):
    """Flagship §24.5 P1 v2 e2e: two burst-synchronized stations, two runs
    ~28 s apart (a wall-clock gap). Per-run FCs computed with
    clock_zero='epoch' share one absolute window grid; Merge FC
    concatenates them; the strict identical-grid check passes and RME_RR
    recovers the true impedance across the gap."""
    sr, _n, t, hx, hy = synthetic_source(21, n=768, sr=8.0, kind="white")
    schedule = (("2024-01-01T00:00:00", 0, 256),      # t = [0, 32) s
                ("2024-01-01T00:01:02", 512, 256))    # t = [64, 96) s
    local = {rid: synthetic_runts(
        (sr, m, t[i0:i0 + m], hx[i0:i0 + m], hy[i0:i0 + m]),
        z=(1.0, 3.0, -1.0, 4.0), e_noise=0.2, seed=21,
        start_datetime=start) for rid, (start, i0, m)
        in zip(("a", "b"), schedule)}
    remote = {rid: synthetic_runts(
        (sr, m, t[i0:i0 + m], hx[i0:i0 + m], hy[i0:i0 + m]),
        z=(1.0, 3.0, -1.0, 4.0), e_noise=0.2, seed=22,
        start_datetime=start) for rid, (start, i0, m)
        in zip(("a", "b"), schedule)}

    fc_kw = {"num_decimations": 0, "num_samples_window": 64, "overlap": 0.5,
             "window_type": "boxcar", "clock_zero": "epoch",
             "magnetic_unit": "h", "e_dipole_length_m": 0.001}
    fc_l = merge_fc_containers([
        compute_fc_levels(r, ["hx", "hy"], ["ex", "ey"], **fc_kw)
        for r in local.values()])
    fc_r = merge_fc_containers([
        compute_fc_levels(r, ["hx", "hy"], ["ex", "ey"], **fc_kw)
        for r in remote.values()])
    # the merged grids are identical across stations (the anchoring payoff)
    np.testing.assert_array_equal(fc_l[0]["time"].values, fc_r[0]["time"].values)
    # ...and span the gap (windows on both sides of [32, 64) s)
    centers = (fc_l[0]["time"].values - np.datetime64("2024-01-01")) \
        / np.timedelta64(1, "s")
    assert centers.min() < 32 and centers.max() > 64
    p, Z, _Z_err = estimate_tf(
        fc_l[0], ["hx", "hy"], ["ex", "ey"], estimator_engine="Robust",
        remote_reference=True, ref_fc=fc_r[0], band_style="l2u")
    assert p.size >= 1 and np.all(np.isfinite(Z))
    assert Z[0, 0, 1] == pytest.approx(3.0, rel=0.3)   # zxy
    assert Z[0, 1, 0] == pytest.approx(-1.0, rel=0.3)  # zyx
    # negative control: with clock stray (station B's runs start 3 s later —
    # a lost-GPS-lock clock, not uncommon on semi-permanent stations) the
    # unanchored grids drift apart and the strict check refuses to pair
    # them; note even WITH the anchor the stray drops B's first window
    # (coverage differs) — the future time-value-matching loosening covers
    # that case (history §24.5 P1 v2b)
    stray_remote = {rid: synthetic_runts(
        (sr, m, t[i0:i0 + m], hx[i0:i0 + m], hy[i0:i0 + m]),
        z=(1.0, 3.0, -1.0, 4.0), e_noise=0.2, seed=22,
        start_datetime=str((pd.Timestamp(start)
                            + pd.Timedelta(seconds=3)).isoformat()))
        for rid, (start, i0, m) in zip(("a", "b"), schedule)}
    fc_kw_noanchor = dict(fc_kw)
    fc_kw_noanchor["clock_zero"] = ""
    fc_l2 = merge_fc_containers([
        compute_fc_levels(r, ["hx", "hy"], ["ex", "ey"], **fc_kw_noanchor)
        for r in local.values()])
    fc_r2 = merge_fc_containers([
        compute_fc_levels(r, ["hx", "hy"], ["ex", "ey"], **fc_kw_noanchor)
        for r in stray_remote.values()])
    assert not np.array_equal(fc_l2[0]["time"].values, fc_r2[0]["time"].values)
    with pytest.raises(ValueError, match="time grid"):
        estimate_tf(fc_l2[0], ["hx", "hy"], ["ex", "ey"],
                    estimator_engine="Robust", remote_reference=True,
                    ref_fc=fc_r2[0], band_style="l2u")
