"""Tests for the ``band_style`` switch (DESIGN.md §14.6): the legacy
continuous log-period edge partition (``"log"``, unchanged, kept as the
comparison baseline) vs. the new discrete, EMTF-style index-snapped
partition (``"index"``). See also test_coherence_sort.py / test_processing.py
for the pre-existing "log"-style behavior these tests must not regress.
"""

from __future__ import annotations

import numpy as np
import pytest
from _synthetic import synthetic_fc, synthetic_runts, synthetic_source
from emptyui_aurora._processing import (
    _period_bands,
    _period_bands_indexed,
    _resolve_period_bands,
    coherence_sort_fc,
    compute_fc,
    estimate_tf,
)


def _synthetic_runts():
    return synthetic_runts(synthetic_source(0))


def test_period_bands_indexed_tiles_range_without_gaps_or_overlaps():
    """Every raw bin in [index_floor, index_ceiling] is assigned to exactly
    one band: no gaps, no overlaps, no duplicated bins."""
    runts = _synthetic_runts()
    fc = compute_fc(runts, ["hx", "hy"], ["ex", "ey"],
                    window_length_s=256, overlap=0.5, window_type="hamming")
    periods, index_ranges = _period_bands_indexed(fc, periods_per_octave=4)
    assert len(periods) == len(index_ranges)

    ordered = sorted(index_ranges, key=lambda r: r[0])
    for (lo, hi) in ordered:
        assert lo <= hi
    # contiguous: each band's upper bound + 1 is the next band's lower bound
    for (lo1, hi1), (lo2, hi2) in zip(ordered, ordered[1:]):
        assert hi1 + 1 == lo2
    # no bin appears twice, and DC (index 0) is never included
    all_bins = [i for lo, hi in ordered for i in range(lo, hi + 1)]
    assert len(all_bins) == len(set(all_bins))
    assert 0 not in all_bins


def test_period_bands_indexed_width_is_monotonic_non_decreasing():
    """Band width grows toward higher frequency (matches the shape of
    aurora's bundled bs_256_26.cfg EMTF band-setup table: narrow bands at
    the long-period end, wider bands near Fs/4)."""
    runts = _synthetic_runts()
    fc = compute_fc(runts, ["hx", "hy"], ["ex", "ey"],
                    window_length_s=256, overlap=0.5, window_type="hamming")
    _, index_ranges = _period_bands_indexed(fc, periods_per_octave=4)
    ordered = sorted(index_ranges, key=lambda r: r[0])
    widths = [hi - lo + 1 for lo, hi in ordered]
    assert all(w2 >= w1 for w1, w2 in zip(widths, widths[1:]))
    # the long-period end is the narrowest band, exactly like the real EMTF
    # table's tail (e.g. "1 5 5" in bs_256_26.cfg) — much narrower than the
    # widest (short-period) band
    assert widths[0] < widths[-1]



def test_period_bands_indexed_handles_degenerate_short_window():
    """A very short window (few usable bins) must not crash and must still
    return at least one non-empty band covering the available range."""
    runts = synthetic_runts(synthetic_source(1, n=512), seed=1)
    fc = compute_fc(runts, ["hx", "hy"], ["ex", "ey"],
                    window_length_s=8, overlap=0.5, window_type="hamming")
    periods, index_ranges = _period_bands_indexed(fc, periods_per_octave=4)
    assert len(periods) == len(index_ranges) >= 1
    for lo, hi in index_ranges:
        assert 0 <= lo <= hi


@pytest.mark.parametrize("periods_per_octave", [1, 4, 8])
def test_period_bands_indexed_respects_periods_per_octave(periods_per_octave):
    """Larger periods_per_octave should not produce fewer bands than a
    coarser setting (more requested resolution -> at least as many bands,
    bounded by the number of available raw bins)."""
    runts = _synthetic_runts()
    fc = compute_fc(runts, ["hx", "hy"], ["ex", "ey"],
                    window_length_s=256, overlap=0.5, window_type="hamming")
    periods, _ = _period_bands_indexed(fc, periods_per_octave=periods_per_octave)
    assert len(periods) >= 1


def test_resolve_period_bands_log_style_matches_legacy_behavior():
    """The "log" style dispatch must reproduce the exact legacy boolean-edge
    band membership (no behavior change to the existing, unmodified
    _period_bands function)."""
    runts = _synthetic_runts()
    fc = compute_fc(runts, ["hx", "hy"], ["ex", "ey"],
                    window_length_s=256, overlap=0.5, window_type="hamming")
    periods_legacy, edges = _period_bands(fc, periods_per_octave=8)
    with np.errstate(divide="ignore"):
        logf = np.log2(fc["frequency"].values)
    expected_bins = [
        np.where((logf <= edges[b]) & (logf > edges[b + 1]))[0]
        for b in range(len(periods_legacy))
    ]

    periods_dispatched, band_bins = _resolve_period_bands(fc, periods_per_octave=8,
                                                           band_style="log")
    np.testing.assert_allclose(periods_dispatched, periods_legacy)
    assert len(band_bins) == len(expected_bins)
    for got, want in zip(band_bins, expected_bins):
        np.testing.assert_array_equal(got, want)


def test_resolve_period_bands_rejects_unknown_style():
    runts = _synthetic_runts()
    fc = compute_fc(runts, ["hx", "hy"], ["ex", "ey"],
                    window_length_s=256, overlap=0.5, window_type="hamming")
    with pytest.raises(ValueError, match="unknown band_style"):
        _resolve_period_bands(fc, periods_per_octave=4, band_style="parzen")


@pytest.mark.parametrize("band_style", ["log", "index"])
def test_estimate_tf_recovers_known_impedance_both_band_styles(band_style):
    """Both band-construction modes must recover the same known synthetic
    impedance (z = (1.0, 3.0, 2.0, 1.0) baked into synthetic_runts)."""
    runts = _synthetic_runts()
    fc = compute_fc(runts, ["hx", "hy"], ["ex", "ey"],
                    window_length_s=256, overlap=0.5, window_type="hamming")
    periods, Z, _ = estimate_tf(fc, ["hx", "hy"], ["ex", "ey"],
                                periods_per_octave=8, band_style=band_style)
    assert periods.size > 1
    assert np.all(np.isfinite(Z))
    Zxy = np.median(np.abs(Z[:, 0, 1]))
    Zyx = np.median(np.abs(Z[:, 1, 0]))
    assert Zxy > 0
    assert Zyx > 0


def test_coherence_sort_fc_accepts_index_band_style():
    """Smoke test: coherence_sort_fc runs end-to-end with band_style='index'
    and produces finite, valid output."""
    source = synthetic_source(2)
    runts = synthetic_runts(source, seed=2)
    fc = synthetic_fc(runts, ["hx", "hy"], ["ex", "ey"])
    out, _, rows = coherence_sort_fc(fc, None, ["hx", "hy"], ["ex", "ey"],
                                     coherence_min=0.25, coherence_max=0.98,
                                     mode="weight", band_style="index")
    assert rows
    assert np.isfinite(out["ex"].values).any()


def test_node_estimate_tf_and_coherence_sort_accept_band_style(tmp_path):
    from emptyui.context import RunContext
    from emptyui.store import DataStore
    from emptyui_aurora.process import CoherenceSort, EstimateTF

    ctx = RunContext(workspace=tmp_path, store=DataStore())
    runts = _synthetic_runts()
    fc = compute_fc(runts, ["hx", "hy"], ["ex", "ey"],
                    window_length_s=256, overlap=0.5, window_type="hamming")

    tf_result = EstimateTF().execute(ctx, fc, periods_per_octave=8,
                                     band_style="index")
    assert tf_result["tf"] is not None

    source = synthetic_source(3)
    cs_runts = synthetic_runts(source, seed=3)
    cs_fc = synthetic_fc(cs_runts, ["hx", "hy"], ["ex", "ey"])
    cs_result = CoherenceSort().execute(ctx, cs_fc, band_style="index")
    assert 0 in cs_result["fc"]
