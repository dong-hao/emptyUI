"""Tests for the two band-construction schemes (DESIGN.md §14.8):
``band_style="emtf"`` (default) — the auto-derived, disjoint EMTF-style
table — and ``band_style="uniform"`` (formerly named ``"log"``) — the
uniform log2-grid partition with merge-based level combination. The retired
``"index"`` mode and the deepest-owner variant were removed after the
2026-09-09 assessment; see the history in DESIGN.md §14.6/§14.7.
"""

from __future__ import annotations

from itertools import pairwise

import numpy as np
import pytest
from _synthetic import synthetic_fc, synthetic_runts, synthetic_source
from emptyui_aurora._processing import (
    _emtf_band_table,
    _period_bands,
    _period_bands_indexed,
    _resolve_band_style,
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


def _multi_level_fc():
    """Five cascading levels mimicking the reference long-period workflow
    (window 256, factor 4, ppo 3 — the configuration that exposed the
    zigzag defect on real BGS LEMI data)."""
    runts = synthetic_runts(synthetic_source(0, n=131072))
    return compute_fc_levels(runts, ["hx", "hy"], ["ex", "ey"],
                             num_decimations=4, factor=4,
                             num_samples_window=256, overlap=0.5,
                             window_type="hamming")


# --- style resolution ---

def test_resolve_band_style():
    assert _resolve_band_style("emtf") == "emtf"
    assert _resolve_band_style("uniform") == "uniform"
    assert _resolve_band_style("log") == "uniform"  # legacy alias
    assert _resolve_band_style("LOG") == "uniform"  # case-insensitive
    assert _resolve_band_style(None) == "emtf"  # default
    with pytest.raises(ValueError, match="unknown band_style.*index"):
        _resolve_band_style("index")


# --- uniform scheme (formerly "log") ---

def test_resolve_period_bands_uniform_matches_legacy_log_behavior():
    """The "uniform" dispatch must reproduce the exact legacy boolean-edge
    band membership of the unmodified _period_bands function."""
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

    periods_dispatched, band_bins = _resolve_period_bands(
        fc, periods_per_octave=8, band_style="uniform")
    np.testing.assert_allclose(periods_dispatched, periods_legacy)
    assert len(band_bins) == len(expected_bins)
    for got, want in zip(band_bins, expected_bins):
        np.testing.assert_array_equal(got, want)


@pytest.mark.parametrize("periods_per_octave", [1, 4, 8])
def test_uniform_bands_respect_periods_per_octave(periods_per_octave):
    runts = _synthetic_runts()
    fc = compute_fc(runts, ["hx", "hy"], ["ex", "ey"],
                    window_length_s=256, overlap=0.5, window_type="hamming")
    periods, _ = _period_bands(fc, periods_per_octave=periods_per_octave)
    assert len(periods) >= 1


# --- emtf scheme ---

def test_period_bands_indexed_fallback_tiles_range_without_gaps():
    """The per-level fallback walk assigns every raw bin in
    [index_floor, index_ceiling] to exactly one band: no gaps, no overlaps,
    no duplicated bins."""
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


def test_period_bands_indexed_fallback_width_is_monotonic_non_decreasing():
    """Band width grows toward higher frequency (matches the shape of
    aurora's bundled bs_256_26.cfg EMTF band-setup table)."""
    runts = _synthetic_runts()
    fc = compute_fc(runts, ["hx", "hy"], ["ex", "ey"],
                    window_length_s=256, overlap=0.5, window_type="hamming")
    _, index_ranges = _period_bands_indexed(fc, periods_per_octave=4)
    ordered = sorted(index_ranges, key=lambda r: r[0])
    widths = [hi - lo + 1 for lo, hi in ordered]
    assert all(w2 >= w1 for w1, w2 in pairwise(widths))
    assert widths[0] < widths[-1]


def test_period_bands_indexed_fallback_handles_degenerate_short_window():
    """A very short window (few usable bins) must not crash and must still
    return at least one non-empty band covering the available range."""
    runts = synthetic_runts(synthetic_source(1, n=512), seed=1)
    fc = compute_fc(runts, ["hx", "hy"], ["ex", "ey"],
                    window_length_s=8, overlap=0.5, window_type="hamming")
    periods, index_ranges = _period_bands_indexed(fc, periods_per_octave=4)
    assert len(periods) == len(index_ranges) >= 1
    for lo, hi in index_ranges:
        assert 0 <= lo <= hi


def test_emtf_table_is_disjoint_across_levels():
    """Every nominal band is owned by exactly one level: the union of all
    levels' frequency intervals contains no overlap (gaps between handovers
    are allowed and expected, like the classic table)."""
    fc = _multi_level_fc()
    table = _emtf_band_table(fc, 3)
    iv = sorted((lo, hi) for pairs in table.values() for hi, lo in pairs)
    for a, b in pairwise(iv):
        assert b[0] >= a[1] - 1e-12  # no overlap


def test_emtf_ownership_is_shallowest_classic():
    """The shallowest policy (hard-coded, classic EMTF assignment): the head
    level owns its full usable range, deeper levels their bottom ~2 octaves
    (classic template bins 5..17), and the deepest level does not dominate
    the overlap region."""
    fc = _multi_level_fc()
    table = _emtf_band_table(fc, 3)
    shallowest, deepest = min(fc), max(fc)
    assert len(table[shallowest]) > len(table[deepest])


def test_emtf_estimate_evaluates_each_frequency_once():
    """band_style='emtf': merged point count == total owned bands (no
    duplicates to merge, no wasted estimates), and adjacent merged periods
    never interleave (no pair closer than ~half a nominal band)."""
    fc = _multi_level_fc()
    ppo = 3
    per_level = estimate_tf_levels(fc, ["hx", "hy"], ["ex", "ey"],
                                   periods_per_octave=ppo, band_style="emtf")
    P, _, _ = merge_tf_levels(per_level)
    table = _emtf_band_table(fc, ppo)
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
    table = _emtf_band_table(fake, 3)
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


def test_emtf_fallback_on_missing_attrs_does_not_crash():
    """Without sample_rate attrs the container-level table cannot be derived;
    the per-level fallback walk must still produce an estimate."""
    fc = _multi_level_fc()
    for fcl in fc.values():
        fcl.attrs.pop("sample_rate", None)
    per_level = estimate_tf_levels(fc, ["hx", "hy"], ["ex", "ey"],
                                   periods_per_octave=3, band_style="emtf")
    assert per_level


# --- estimation across both schemes ---

@pytest.mark.parametrize("band_style", ["uniform", "emtf"])
def test_estimate_tf_recovers_known_impedance_both_band_styles(band_style):
    """Both schemes must recover the same known synthetic impedance
    (z = (1.0, 3.0, 2.0, 1.0) baked into synthetic_runts)."""
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


def test_legacy_log_alias_matches_uniform():
    """band_style='log' (legacy name) must behave identically to 'uniform'."""
    fc = _multi_level_fc()
    kw = {"estimator_engine": "Least Squares", "periods_per_octave": 3}
    P_u, Z_u, _ = merge_tf_levels(
        estimate_tf_levels(fc, ["hx", "hy"], ["ex", "ey"],
                           band_style="uniform", **kw))
    P_l, Z_l, _ = merge_tf_levels(
        estimate_tf_levels(fc, ["hx", "hy"], ["ex", "ey"],
                           band_style="log", **kw))
    np.testing.assert_allclose(P_u, P_l)
    np.testing.assert_allclose(Z_u, Z_l)


def test_emtf_is_the_default_style():
    """Omitting band_style must use the emtf scheme (same merged grid as an
    explicit band_style='emtf' call)."""
    fc = _multi_level_fc()
    per_default = estimate_tf_levels(fc, ["hx", "hy"], ["ex", "ey"],
                                     periods_per_octave=3)
    per_emtf = estimate_tf_levels(fc, ["hx", "hy"], ["ex", "ey"],
                                  periods_per_octave=3, band_style="emtf")
    np.testing.assert_allclose(np.asarray(per_default[0]["periods"]),
                               np.asarray(per_emtf[0]["periods"]))


def test_coherence_sort_fc_accepts_both_styles():
    """Smoke: coherence_sort_fc runs end-to-end with both schemes."""
    source = synthetic_source(2)
    runts = synthetic_runts(source, seed=2)
    fc = synthetic_fc(runts, ["hx", "hy"], ["ex", "ey"])
    for style in ("uniform", "emtf"):
        out, _, rows = coherence_sort_fc(fc, None, ["hx", "hy"], ["ex", "ey"],
                                         coherence_min=0.25,
                                         coherence_max=0.98, mode="weight",
                                         band_style=style)
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

    # default (emtf) via node execute without band_style
    tf_result = EstimateTF().execute(ctx, fc, periods_per_octave=8)
    assert tf_result["tf"] is not None

    source = synthetic_source(3)
    cs_runts = synthetic_runts(source, seed=3)
    cs_fc = synthetic_fc(cs_runts, ["hx", "hy"], ["ex", "ey"])
    cs_result = CoherenceSort().execute(ctx, cs_fc, band_style="uniform")
    assert 0 in cs_result["fc"]
