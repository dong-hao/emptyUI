"""Tests for the ``band_style`` switch (DESIGN.md §14.6): the legacy
continuous log-period edge partition (``"log"``, unchanged, kept as the
comparison baseline) vs. the new discrete, EMTF-style index-snapped
partition (``"index"``). See also test_coherence_sort.py / test_processing.py
for the pre-existing "log"-style behavior these tests must not regress.
"""

from __future__ import annotations

from itertools import pairwise

import numpy as np
import pytest
from _synthetic import synthetic_fc, synthetic_runts, synthetic_source
from emptyui_aurora._processing import (
    _container_band_edges,
    _emtf_band_table,
    _period_bands,
    _period_bands_indexed,
    _resolve_period_bands,
    coherence_sort_fc,
    compute_fc,
    compute_fc_levels,
    estimate_tf,
    estimate_tf_levels,
    merge_tf_levels,
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
    for (_, hi1), (lo2, _) in pairwise(ordered):
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
    assert all(w2 >= w1 for w1, w2 in pairwise(widths))
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


def _multi_level_fc():
    """Five cascading levels mimicking the reference long-period workflow
    (window 256, factor 4, ppo 3 — the configuration that exposed the
    zigzag defect on real BGS LEMI data)."""
    runts = synthetic_runts(synthetic_source(0, n=131072))
    return compute_fc_levels(runts, ["hx", "hy"], ["ex", "ey"],
                             num_decimations=4, factor=4,
                             num_samples_window=256, overlap=0.5,
                             window_type="hamming")


def test_index_bands_share_one_global_grid_across_levels():
    """Multi-level 'index' banding must derive band boundaries from ONE
    global frequency grid: every level's band labels are drawn from the same
    set, so overlapping bands carry identical labels across levels and
    merge_tf_levels' exact dedup fires (deepest wins)."""
    fc = _multi_level_fc()
    ppo = 3
    edges = _container_band_edges(fc, ppo)
    assert edges is not None
    assert np.all(np.diff(edges) < 0)  # strictly decreasing

    per_level = estimate_tf_levels(fc, ["hx", "hy"], ["ex", "ey"],
                                   periods_per_octave=ppo, band_style="index")
    for entry in per_level.values():
        # every label is the geometric mean of a consecutive global edge pair
        label_log = np.log2(1.0 / entry["periods"])
        edge_log = np.log2(edges)
        dist = np.abs(label_log[:, None]
                      - (edge_log[:-1] + edge_log[1:])[None, :] / 2.0)
        assert np.all(dist.min(axis=1) < 1e-9)


def test_index_merge_has_no_interleaved_duplicates():
    """Regression for the zigzag defect: with per-level bin-grid walks the
    merge's exact dedup never fired, so adjacent levels' overlapping bands
    interleaved as near-duplicate periods (adjacent gaps < one nominal band)
    — visibly alternating apparent resistivity. With the shared global grid
    the merged grid is the global band grid itself: adjacent periods differ
    by exactly one nominal band ratio."""
    fc = _multi_level_fc()
    ppo = 3
    per_level = estimate_tf_levels(fc, ["hx", "hy"], ["ex", "ey"],
                                   periods_per_octave=ppo, band_style="index")
    P, _, _ = merge_tf_levels(per_level)

    n_global = len(_container_band_edges(fc, ppo)) - 1
    n_estimated = sum(int((np.abs(np.asarray(e["Z"])).sum(axis=(1, 2)) > 0).sum())
                      for e in per_level.values())
    assert P.size == n_global          # one point per global band, no dups
    assert P.size < n_estimated        # overlaps were deduped (deepest wins)

    ratio = P[1:] / P[:-1]
    # merge keys are rounded to 8 log2 decimals — hence rtol 1e-6
    np.testing.assert_allclose(ratio[:-1], 2.0 ** (1.0 / ppo), rtol=1e-6)
    assert ratio[-1] <= 2.0 ** (1.0 / ppo)  # clamped final band is narrower


def test_index_and_log_merge_to_comparable_band_counts():
    """Sanity: after the global-grid fix, both styles merge to a similar
    number of bands (the index mode should no longer inflate the merged
    grid with interleaved duplicates)."""
    fc = _multi_level_fc()
    counts = {}
    for style in ("log", "index"):
        per_level = estimate_tf_levels(fc, ["hx", "hy"], ["ex", "ey"],
                                       periods_per_octave=3, band_style=style)
        P, _, _ = merge_tf_levels(per_level)
        counts[style] = P.size
    assert abs(counts["log"] - counts["index"]) <= 2


def test_container_band_edges_falls_back_on_missing_attrs():
    """Without sample_rate/window attrs the global grid cannot be derived
    and the dispatcher must fall back to the per-level walk (no crash)."""
    fc = _multi_level_fc()
    for fcl in fc.values():
        fcl.attrs.pop("sample_rate", None)
    assert _container_band_edges(fc, 3) is None
    per_level = estimate_tf_levels(fc, ["hx", "hy"], ["ex", "ey"],
                                   periods_per_octave=3, band_style="index")
    assert per_level


# --- band_style="emtf": disjoint EMTF/aurora-style table (DESIGN §14.7) ---

def test_emtf_table_is_disjoint_across_levels():
    """Every nominal band is owned by exactly one level: the union of all
    levels' frequency intervals contains no overlap (gaps between handovers
    are allowed and expected, like the classic table)."""
    fc = _multi_level_fc()
    for owner in ("deepest", "shallowest"):
        table = _emtf_band_table(fc, 3, owner=owner)
        iv = sorted((lo, hi) for pairs in table.values() for hi, lo in pairs)
        for a, b in pairwise(iv):
            assert b[0] >= a[1] - 1e-12  # no overlap


def test_emtf_table_owner_policies_differ():
    """deepest vs shallowest assign the overlap region to different levels:
    with the deepest policy the deepest level owns its full reliable range;
    with the shallowest policy the shallowest level does (classic EMTF)."""
    fc = _multi_level_fc()
    deep = _emtf_band_table(fc, 3, owner="deepest")
    shallow = _emtf_band_table(fc, 3, owner="shallowest")
    deepest_level = max(fc)
    shallowest_level = min(fc)
    assert len(deep[deepest_level]) > len(deep[shallowest_level])
    assert len(shallow[shallowest_level]) > len(shallow[deepest_level])
    # same total band count (both partition the same nominal span)
    assert sum(len(v) for v in deep.values()) == \
        sum(len(v) for v in shallow.values())


def test_emtf_estimate_evaluates_each_frequency_once():
    """band_style='emtf': merged point count == total owned bands (no
    duplicates to merge, no wasted estimates), and adjacent merged periods
    never interleave (no pair closer than ~half a nominal band)."""
    fc = _multi_level_fc()
    ppo = 3
    per_level = estimate_tf_levels(fc, ["hx", "hy"], ["ex", "ey"],
                                   periods_per_octave=ppo, band_style="emtf",
                                   band_owner="deepest")
    P, _, _ = merge_tf_levels(per_level)
    table = _emtf_band_table(fc, 3, owner="deepest")
    assert P.size == sum(len(v) for v in table.values())
    ratio = P[1:] / P[:-1]
    assert np.all(ratio > 1.05)  # no interleaved near-duplicates


def test_emtf_shallowest_table_matches_bs_256_26_shape():
    """For the bs_256_26 geometry (N=256, factor 4) the auto-generated
    shallowest-owner table reproduces the classic structure: ~3 bands/octave
    density, the head level owning its full usable range, deeper levels
    owning their bottom ~2 octaves (classic template bins 5..17), and
    single-bin tails."""
    import xarray as xr
    fs, N = 1.0, 256
    fake = {}
    for i in range(4):
        ds = xr.Dataset()
        ds.attrs = {"sample_rate": fs / 4 ** i, "num_samples_window": N}
        ds["frequency"] = np.arange(N // 2 + 1) * fs / 4 ** i / N
        fake[i] = ds
    table = _emtf_band_table(fake, 3, owner="shallowest")
    # head level owns its full range
    assert len(table[0]) == 12  # 4 octaves x 3
    # deeper levels own their bottom 2 octaves (classic bins 4..16 ≈ 5..17)
    for lv in (1, 2, 3):
        assert len(table[lv]) == 6
    # disjoint by construction, and tail bands shrink to 1-2 bins
    widths = []
    for lv, pairs in table.items():
        df = fs / 4 ** lv / N
        for hi, lo in pairs:
            w = int(hi / df) - int(np.ceil(lo / df)) + 1
            assert w >= 1
            widths.append((lv, w))
    tail = [w for lv, w in widths if lv > 0]
    assert max(tail) <= 4  # classic template's widest band is 14-17 = 4 bins
    assert min(tail) == 1


def test_emtf_owner_policies_recover_impedance(tmp_path):
    """Node-level smoke: EstimateTF runs end-to-end with band_style='emtf'
    under both owner policies and recovers the known synthetic impedance."""
    from emptyui.context import RunContext
    from emptyui.store import DataStore

    ctx = RunContext(workspace=tmp_path, store=DataStore())
    fc = _multi_level_fc()
    for owner in ("deepest", "shallowest"):
        per_level = estimate_tf_levels(fc, ["hx", "hy"], ["ex", "ey"],
                                       estimator_engine="Least Squares",
                                       periods_per_octave=3,
                                       band_style="emtf",
                                       band_owner=owner)
        P, Z, _ = merge_tf_levels(per_level)
        assert P.size >= 10
        assert np.all(np.isfinite(Z))
        assert np.median(np.abs(Z[:, 0, 1])) > 0

    from emptyui_aurora.process import EstimateTF
    result = EstimateTF().execute(ctx, fc, periods_per_octave=3,
                                  band_style="emtf", band_owner="shallowest")
    assert result["tf"] is not None


def test_emtf_rejects_unknown_owner_policy():
    fc = _multi_level_fc()
    with pytest.raises(ValueError, match="unknown band owner"):
        _emtf_band_table(fc, 3, owner="median")
