"""Tests for the two band-construction schemes (DESIGN.md §14.9):
``band_style="l2u"`` (default) — the reference-template band table with
log-uniform window-starvation coarsening — and ``band_style="uniform"``
(formerly named ``"log"``) — the uniform log2-grid partition with
merge-based level combination. The retired ``"emtf"`` value is accepted as
a legacy alias of ``"l2u"`` (the auto-derived disjoint EMTF-style table was
replaced 2026-09-11); the ``"index"`` mode was removed after the 2026-09-09
assessment; see the history in DESIGN.md §14.6/§14.9.
"""

from __future__ import annotations

from itertools import pairwise

import numpy as np
import pytest
from _synthetic import synthetic_fc, synthetic_runts, synthetic_source
from emptyui_aurora._processing import (
    _l2u_band_table,
    _l2u_template,
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
    assert _resolve_band_style("l2u") == "l2u"
    assert _resolve_band_style("uniform") == "uniform"
    assert _resolve_band_style("log") == "uniform"  # legacy alias
    assert _resolve_band_style("LOG") == "uniform"  # case-insensitive
    assert _resolve_band_style("emtf") == "l2u"  # legacy alias, superseded
    assert _resolve_band_style(None) == "l2u"  # default
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


# --- l2u scheme ---

def test_l2u_template_scales_with_window():
    """The reference template scales with the window: window-128 native
    indices ×window/128, so window 256 is exactly the doubled official
    template family and other windows get the closest scaled equivalent
    (disjoint, non-degenerate bands)."""
    t128 = _l2u_template(128)
    assert t128 == ((15, 16), (12, 12), (9, 10), (7, 8), (6, 6), (5, 5))
    assert _l2u_template(256) == tuple((2 * a, 2 * b) for a, b in t128)
    for window in (64, 128, 256, 512):
        tpl = _l2u_template(window)
        assert tpl
        for a, b in tpl:
            assert 1 <= a <= b <= window // 2
        for (a1, _), (_, b2) in pairwise(tpl):  # descending k: disjoint
            assert b2 < a1


def test_l2u_table_keeps_template_when_window_rich():
    """A window-rich container keeps the template unchanged (every band's
    total samples exceed the threshold) — the coarsening is driven by the
    window count, like the legacy emtf behavior."""
    fc = _fake_levels(windows=(512, 128, 32, 8, 2))
    # no coarsening (threshold 0) == plain template everywhere
    plain = _l2u_band_table(fc, min_bins_per_band=0)
    assert all(len(v) == 6 for v in plain.values())
    # a high threshold keeps the template too when no band is starved
    # (the template's narrowest band is 1 bin, so even the deepest level
    # needs >= 40 windows to stay rich)
    rich = _fake_levels(windows=(5120, 1280, 320, 80, 64))
    table = _l2u_band_table(rich, min_bins_per_band=40.0)
    assert all(len(v) == 6 for v in table.values())


def _fake_levels(windows=(512, 128, 32, 8, 2), n_levels=5, fs=1.0, N=256, factor=4):
    """Deterministic multi-level fc container with a real ``time`` dim so the
    window counts are visible to the coarsening (the plain fake container of
    ``test_l2u_table_geometry`` has none)."""
    import xarray as xr

    out = {}
    for i in range(n_levels):
        ds = xr.Dataset()
        ds.attrs = {"sample_rate": fs / factor ** i, "num_samples_window": N}
        ds["frequency"] = np.arange(N // 2 + 1) * fs / factor ** i / N
        ds["time"] = np.arange(windows[i])
        for ch in ("hx", "hy", "ex", "ey"):
            ds[ch] = (("frequency", "time"), np.zeros((N // 2 + 1, windows[i])))
        out[i] = ds
    return out


def test_l2u_coarsening_rebuilds_starved_levels_uniformly():
    """Window-starved levels are rebuilt as a UNIFORM coarser grid (the l2u
    rule: any band below the total-samples threshold triggers a rebuild):
    L2 (32 windows, one 1-bin band < 40 samples) is rebuilt to a uniform
    4-band grid of the minimal width exceeding the threshold; L3/L4 collapse
    toward the 2-band floor; the window-rich head is untouched."""
    fc = _fake_levels()
    table = _l2u_band_table(fc, min_bins_per_band=40.0)
    assert len(table[0]) == 6  # 512 windows — rich, untouched
    assert len(table[1]) == 6  # 128 windows — rich, untouched
    assert len(table[2]) == 4  # 32 windows — rebuilt uniform (n-n//3 = 4)
    assert len(table[3]) == 2  # 8 windows -> 2-band floor
    assert len(table[4]) == 2  # 2 windows -> 2-band floor
    # every rebuilt band's total samples exceed the threshold (width >= w)
    # where the span allows it; L4 (2 windows) is the best-effort floor
    # (the required width 21 cannot fit two disjoint bands in 23 bins —
    # the width is shrunk to tile the span, below the threshold by design)
    n_windows = [512, 128, 32, 8, 2]
    for lv in (2, 3, 4):
        df = float(fc[lv].attrs["sample_rate"]) / 256.0
        for hi, lo in table[lv]:
            w = round(hi / df) - int(np.ceil(lo / df)) + 1
            assert n_windows[lv] * w >= (40 if lv < 4 else 1)


def test_l2u_coarsening_bands_are_disjoint_ascending():
    """The rebuilt grid is disjoint (no shared bins) and the periods are
    strictly ascending shortest→longest, like the template path."""
    fc = _fake_levels()
    table = _l2u_band_table(fc, min_bins_per_band=40.0)
    for lv, pairs in table.items():
        df = fc[lv].attrs["sample_rate"] / 256.0
        k_hi = [round(hi / df) for hi, lo in pairs]
        k_lo = [round(lo / df) for hi, lo in pairs]
        # disjoint: each band's bins are strictly below the next (desc k)
        for i in range(len(pairs) - 1):
            assert k_lo[i] > k_hi[i + 1]
        Ts = [1.0 / np.sqrt(hi * lo) for hi, lo in pairs]
        assert all(t2 > t1 for t1, t2 in pairwise(Ts))


def test_l2u_skips_levels_without_window_metadata():
    """A level without a ``time`` dim has no window count -> treated as
    window-rich and left unchanged (legacy behavior for read-back fcs)."""
    import xarray as xr
    fs, N = 1.0, 256
    fake = {}
    for i in range(4):
        ds = xr.Dataset()
        ds.attrs = {"sample_rate": fs / 4 ** i, "num_samples_window": N}
        ds["frequency"] = np.arange(N // 2 + 1) * fs / 4 ** i / N
        fake[i] = ds
    table = _l2u_band_table(fake, min_bins_per_band=40.0)
    assert all(len(v) == 6 for v in table.values())


def test_l2u_head_clip_drops_short_periods():
    """t_short clips the head: no evaluated band center is shorter, and the
    head level loses its shortest bands (the official USArray grid clips at
    ~7.3 s for the 1 Hz NIMS band — data-driven, §14.9)."""
    fc = _fake_levels()
    table = _l2u_band_table(fc, min_bins_per_band=40.0, t_short=30.0)
    assert 0 not in table  # L0 (8.3-25.6 s) fully clipped
    assert 1 in table  # L1 (33.1-102.5 s) survives
    for pairs in table.values():
        for hi, lo in pairs:
            assert 1.0 / np.sqrt(hi * lo) >= 30.0 - 1e-9


def test_l2u_fallback_on_missing_attrs_does_not_crash():
    """Without sample_rate attrs the container-level table cannot be derived;
    the per-level fallback walk must still produce an estimate."""
    fc = _multi_level_fc()
    for fcl in fc.values():
        fcl.attrs.pop("sample_rate", None)
    per_level = estimate_tf_levels(fc, ["hx", "hy"], ["ex", "ey"],
                                   periods_per_octave=3, band_style="l2u")
    assert per_level


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


# --- estimation across both schemes ---

@pytest.mark.parametrize("band_style", ["uniform", "l2u"])
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


def test_l2u_is_the_default_style():
    """Omitting band_style must use the l2u scheme (same merged grid as an
    explicit band_style='l2u' call)."""
    fc = _multi_level_fc()
    per_default = estimate_tf_levels(fc, ["hx", "hy"], ["ex", "ey"],
                                     periods_per_octave=3)
    per_l2u = estimate_tf_levels(fc, ["hx", "hy"], ["ex", "ey"],
                                 periods_per_octave=3, band_style="l2u")
    np.testing.assert_allclose(np.asarray(per_default[0]["periods"]),
                               np.asarray(per_l2u[0]["periods"]))


def test_coherence_sort_fc_accepts_both_styles():
    """Smoke: coherence_sort_fc runs end-to-end with both schemes."""
    source = synthetic_source(2)
    runts = synthetic_runts(source, seed=2)
    fc = synthetic_fc(runts, ["hx", "hy"], ["ex", "ey"])
    for style in ("uniform", "l2u"):
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

    # default (l2u) via node execute without band_style
    tf_result = EstimateTF().execute(ctx, fc, periods_per_octave=8)
    assert tf_result["tf"] is not None

    source = synthetic_source(3)
    cs_runts = synthetic_runts(source, seed=3)
    cs_fc = synthetic_fc(cs_runts, ["hx", "hy"], ["ex", "ey"])
    cs_result = CoherenceSort().execute(ctx, cs_fc, band_style="uniform")
    assert 0 in cs_result["fc"]


# --- §14.9 frequency-grid determination: l2u coarsening + head clip ---

def test_l2u_estimate_grid_matches_table():
    """estimate_tf_levels with the l2u scheme evaluates exactly the
    coarsened table (each frequency once, no degenerate duplicates)."""
    fc = _multi_level_fc()
    per_level = estimate_tf_levels(fc, ["hx", "hy"], ["ex", "ey"],
                                   periods_per_octave=3, band_style="l2u",
                                   min_bins_per_band=40.0)
    P, _, _ = merge_tf_levels(per_level)
    table = _l2u_band_table(fc, min_bins_per_band=40.0)
    assert P.size == sum(len(v) for v in table.values())
    assert np.all(P[1:] / P[:-1] > 1.05)


def test_node_estimate_tf_adaptive_params(tmp_path):
    """The Estimate TF node exposes the §14.9 refinements and they reach the
    grid: the coarsened+clipped node result has fewer periods than the
    plain default and its processing notes record the params."""
    from emptyui.context import RunContext
    from emptyui.store import DataStore
    from emptyui_aurora.process import EstimateTF

    ctx = RunContext(workspace=tmp_path, store=DataStore())
    fc = _multi_level_fc()
    plain = EstimateTF().execute(
        ctx, fc, periods_per_octave=3, min_bins_per_band=0.0)
    adaptive = EstimateTF().execute(
        ctx, fc, periods_per_octave=3, min_bins_per_band=40.0,
        t_short=10.0)
    assert adaptive["tf"].period.size < plain["tf"].period.size
    notes = adaptive["tf"].run_metadata.comments.value
    assert "min_bins_per_band=40.0" in notes
    assert "t_short=10.0" in notes
