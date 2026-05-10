"""Coherence sort silent-drop behavior is now loud (2026-09-15).

The sort never removes frequency bins from the FC — it only down-weights
(√w) or NaNs per-(bin, window) samples; the FC grid is preserved exactly
(test_sort_preserves_fc_grid). But downstream, a band whose every sample
was fully down-weighted leaves the band regression with no usable rows:
estimate_tf skips it and merge_tf_levels drops the all-zero band —
silently, so the final TF just lacked period points with no explanation
(2026-09-15 user report on the burst workflow: deep levels with few
windows have their per-bin window-ensemble collapse to <= n_in segments,
the coherence statistic scores 0 there, weight mode NaNs every sample,
and the band vanishes from the merged TF). These tests pin the now-loud
grammar: warnings name the skipped periods.
"""

from __future__ import annotations

import logging
import sys

import numpy as np
import xarray as xr
from emptyui_aurora._processing import (
    coherence_sort_fcs,
    compute_fc_levels,
    estimate_tf_levels,
    merge_tf_levels,
)

sys.path.insert(0, "packages/emptyui-aurora/tests")
from test_magnetic_units import _field, _runts


def _container(seed=31):
    rng = np.random.default_rng(seed)
    hx, hy, ex, ey = _field(rng, n=16384, sr=8.0)
    runts = _runts(hx, hy, ex, ey)
    return compute_fc_levels(runts, ["hx", "hy"], ["ex", "ey"],
                             num_decimations=3, factor=4,
                             num_samples_window=256, overlap=0.5,
                             window_type="hamming",
                             prewhitening="first difference")


def test_sort_preserves_fc_grid():
    """The sort must never add/remove frequency bins or windows."""
    levels = _container()
    out, _ref, _rows = coherence_sort_fcs(
        levels, None, ["hx", "hy"], ["ex", "ey"], mode="weight")
    for lv in sorted(levels):
        assert np.array_equal(out[lv]["frequency"].values,
                              levels[lv]["frequency"].values)
        assert out[lv]["hx"].values.shape == levels[lv]["hx"].values.shape


def _fc_with_dead_band():
    """4-bin, 32-window fc whose band at T=2 s (f=0.5 Hz, bin 1) is all
    NaN — as if coherence sorting had down-weighted every sample there."""
    f = np.asarray([0.25, 0.5, 1.0, 2.0])  # ascending, like rfftfreq
    rng = np.random.default_rng(0)
    data = {}
    for ch in ("hx", "hy", "ex", "ey"):
        vals = (rng.normal(size=(4, 32)) + 1j * rng.normal(size=(4, 32)))
        data[ch] = (("frequency", "time"), vals)
    ds = xr.Dataset(data, coords={"frequency": f, "time": np.arange(32)})
    for ch in ("hx", "hy", "ex", "ey"):
        ds[ch].values[1, :] = np.nan
    bands = [(0.6, 0.4), (1.1, 0.9), (2.1, 1.9)]  # (f_hi, f_lo) pairs
    return ds, bands


def test_estimate_warns_on_dead_band(caplog):
    """A band whose samples are all NaN (sort down-weighted everything)
    is skipped — now with a warning naming the period."""
    ds, bands = _fc_with_dead_band()
    with caplog.at_level(logging.WARNING, logger="emptyui"):
        estimate_tf_levels({0: ds}, ["hx", "hy"], ["ex", "ey"],
                           estimator_engine="Robust", band_style="l2u",
                           bands_override={0: bands}, min_bins_per_band=0)
    msgs = [r.getMessage() for r in caplog.records
            if "had no usable samples" in r.getMessage()]
    assert msgs, "estimate must warn when a band has no usable samples"
    assert "T=2" in msgs[0]


def test_merge_warns_on_degenerate_bands(caplog):
    """All-zero bands dropped by merge_tf_levels now warn with periods."""
    per_level = {
        0: {"periods": np.array([1.0, 2.0, 4.0]),
            "Z": np.ones((3, 2, 2), dtype=complex),
            "Z_err": np.ones((3, 2, 2)),
            "periods_per_octave": 3},
        1: {"periods": np.array([4.0, 8.0]),
            "Z": np.zeros((2, 2, 2), dtype=complex),  # both degenerate
            "Z_err": np.zeros((2, 2, 2)),
            "periods_per_octave": 3},
    }
    with caplog.at_level(logging.WARNING, logger="emptyui"):
        P, _Z, _Zerr = merge_tf_levels(per_level)
    msgs = [r.getMessage() for r in caplog.records
            if "degenerate" in r.getMessage()]
    assert msgs, "merge must warn when dropping all-zero bands"
    assert "T=4" in msgs[0] and "T=8" in msgs[0]
    # the healthy level's periods survive
    assert 1.0 in P and 2.0 in P


def test_starved_deep_level_dropped_loudly(caplog):
    """End-to-end: on a starved deepest level (1 window) the per-bin
    ensemble collapses and the sort down-weights every sample — the band
    vanishes from the merged TF, but now with a warning instead of
    silently."""
    levels = _container()
    out, _ref, _rows = coherence_sort_fcs(
        levels, None, ["hx", "hy"], ["ex", "ey"],
        mode="weight", coherence_min=0.25, coherence_max=0.98)
    est = estimate_tf_levels(out, ["hx", "hy"], ["ex", "ey"],
                             estimator_engine="Robust", band_style="l2u",
                             min_bins_per_band=20.0)
    with caplog.at_level(logging.WARNING, logger="emptyui"):
        merge_tf_levels(est)
    msgs = [r.getMessage() for r in caplog.records
            if ("had no usable samples" in r.getMessage()
                or "degenerate" in r.getMessage())]
    assert msgs, "starved-level band loss must be reported, not silent"
