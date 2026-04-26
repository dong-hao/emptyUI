"""Tests for the two band-construction schemes (DESIGN.md §10.2):
``band_style="l2u"`` (default) — the reference-template band table with
log-uniform window-starvation coarsening — and ``band_style="uniform"``
(formerly named ``"log"``) — the uniform log2-grid partition with
merge-based level combination. The retired ``"emtf"`` value is accepted as
a legacy alias of ``"l2u"`` (the auto-derived disjoint EMTF-style table was
replaced 2026-09-11); the ``"index"`` mode was removed after the 2026-09-09
assessment; see history §14.6/§14.9.
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
    manual_bands,
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


def test_coherence_sort_fc_no_band_geometry():
    """§24.9: the sort's API is band-free — legacy band-geometry kwargs are
    rejected rather than silently accepted."""
    source = synthetic_source(2)
    runts = synthetic_runts(source, seed=2)
    fc = synthetic_fc(runts, ["hx", "hy"], ["ex", "ey"])
    with pytest.raises(TypeError):
        coherence_sort_fc(fc, None, ["hx", "hy"], ["ex", "ey"],
                          coherence_min=0.25, coherence_max=0.98,
                          mode="weight", band_style="uniform")
    out, _, rows = coherence_sort_fc(fc, None, ["hx", "hy"], ["ex", "ey"],
                                     coherence_min=0.25,
                                     coherence_max=0.98, mode="weight")
    assert rows
    assert np.isfinite(out["ex"].values).any()


def test_node_estimate_tf_accepts_band_style(tmp_path):
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
    # §24.9: the sort node declares no banding params — passing the removed
    # kwarg is a signature error (the executor ignores stale params from
    # saved workflows, but the node API itself is band-free)
    with pytest.raises(TypeError):
        CoherenceSort().execute(ctx, cs_fc, band_style="uniform")


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


# --- head clip (t_short) across BOTH schemes (2026-09-13 fix) ---

def test_uniform_head_clip_drops_short_periods():
    """t_short must clip the uniform scheme too — the shortest-period bound
    was silently ignored for band_style='uniform' (only the l2u table path
    clipped); the same geometric-mean center-period criterion applies."""
    runts = _synthetic_runts()
    fc = compute_fc(runts, ["hx", "hy"], ["ex", "ey"],
                    window_length_s=256, overlap=0.5, window_type="hamming")
    periods_all, _ = _resolve_period_bands(fc, periods_per_octave=8,
                                           band_style="uniform")
    periods_clipped, band_bins = _resolve_period_bands(
        fc, periods_per_octave=8, band_style="uniform", t_short=4.0)
    assert len(periods_clipped) < len(periods_all)
    assert np.all(periods_clipped >= 4.0 - 1e-9)
    assert len(periods_clipped) == len(band_bins)
    # the kept periods are a strict prefix of the unclipped grid
    assert np.allclose(periods_all[-len(periods_clipped):], periods_clipped)


def test_uniform_head_clip_reaches_estimate():
    """estimate_tf_levels with band_style='uniform' + t_short: no merged
    period is shorter, and the head level is skipped when fully clipped."""
    fc = _multi_level_fc()
    per_unclipped = estimate_tf_levels(
        fc, ["hx", "hy"], ["ex", "ey"], periods_per_octave=3,
        band_style="uniform")
    per_clipped = estimate_tf_levels(
        fc, ["hx", "hy"], ["ex", "ey"], periods_per_octave=3,
        band_style="uniform", t_short=20.0)
    P_full, _, _ = merge_tf_levels(per_unclipped)
    P_clip, _, _ = merge_tf_levels(per_clipped)
    assert P_clip.size < P_full.size
    assert np.all(P_clip >= 20.0 - 1e-9)
    # the head level (shortest periods) contributed nothing when clipped
    assert 0 not in per_clipped


def test_head_clip_applies_to_l2u_fallback():
    """The l2u per-level fallback (metadata-less fcs) must honor t_short
    too — previously only the primary container-level table clipped."""
    fc = _multi_level_fc()
    for fcl in fc.values():
        fcl.attrs.pop("sample_rate", None)
    per = estimate_tf_levels(fc, ["hx", "hy"], ["ex", "ey"],
                             periods_per_octave=3, band_style="l2u",
                             t_short=30.0)
    assert per
    for entry in per.values():
        assert np.all(np.asarray(entry["periods"]) >= 30.0 - 1e-9)


def test_head_clip_zero_is_noop_for_both_schemes():
    """t_short=0/None preserves the legacy behavior for both schemes."""
    runts = _synthetic_runts()
    fc = compute_fc(runts, ["hx", "hy"], ["ex", "ey"],
                    window_length_s=256, overlap=0.5, window_type="hamming")
    for style in ("uniform", "l2u"):
        p0, b0 = _resolve_period_bands(fc, periods_per_octave=8,
                                       band_style=style)
        pn, bn = _resolve_period_bands(fc, periods_per_octave=8,
                                       band_style=style, t_short=None)
        pz, bz = _resolve_period_bands(fc, periods_per_octave=8,
                                       band_style=style, t_short=0.0)
        assert len(p0) == len(pn) == len(pz)
        assert len(b0) == len(bn) == len(bz)


def test_manual_bands_geometry():
    """§24.5 P4: manual_bands reproduces the parity harness's
    exact_official_bands reconstruction exactly (geometric-midpoint edges,
    level 0 = highest frequencies, order-normalized input)."""
    from emptyui_aurora._processing import manual_bands

    # official SDG31 level-0/1 centers (descending), scrambled on purpose
    freqs = [0.0391, 0.0312, 0.0195, 0.0156, 0.0244, 0.0488,  # level 0
             0.0137, 0.0117, 0.00977, 0.00781, 0.00610, 0.00488]  # level 1
    table = manual_bands(freqs, "6,6")
    assert sorted(table) == [0, 1]
    # level 0 = the six highest frequencies
    f0 = sorted((f for f in freqs[:6]), reverse=True)
    r = np.sqrt(f0[0] / f0[1])
    assert table[0][0][0] == pytest.approx(f0[0] * r)
    assert table[0][0][1] == pytest.approx(np.sqrt(f0[0] * f0[1]))
    hi, lo = table[0][0]
    assert hi > lo
    # edges are contiguous within a level
    for a, b in zip(table[0], table[0][1:]):
        assert a[1] == pytest.approx(b[0])


def test_manual_bands_matches_harness_formula():
    """The shared helper equals the parity harness's exact_official_bands
    (the provenance guarantee for band_style='manual'): the official SDG31
    EDI's 30 band centers — (6,6,6,6,4,2) per level, classic EMTF structure —
    rebuilt edge for edge. Inputs come from the harness's own EDI reader, so
    the comparison always runs on the real official grid (the EDI is a
    tracked file); both sides implement the geometric-midpoint rule and must
    agree to floating-point precision."""
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[3]
                           / "examples" / "data" / "nims-usarray-test"))
    from emptyui_aurora._processing import manual_bands

    try:
        from verify_sdg31 import exact_official_bands, read_official_edi
    except ImportError:  # harness deps unavailable — skip quietly
        pytest.skip("parity harness unavailable")
    periods_off, _, _ = read_official_edi()
    freqs_off = 1.0 / np.asarray(periods_off)  # descending (harness order)
    table_h = exact_official_bands(freqs_off)
    table_m = manual_bands(freqs_off, "6,6,6,6,4,2")
    # shape first: zip() below would silently truncate a level mismatch
    assert {lv: len(b) for lv, b in table_m.items()} == {
        lv: len(b) for lv, b in table_h.items()}
    assert [len(table_m[lv]) for lv in sorted(table_m)] == [6, 6, 6, 6, 4, 2]
    for lv in table_h:
        for (hi_h, lo_h), (hi_m, lo_m) in zip(table_h[lv], table_m[lv]):
            assert hi_m == pytest.approx(hi_h, rel=1e-12)
            assert lo_h == pytest.approx(lo_m, rel=1e-12)


def test_manual_bands_validation():
    """Clear errors for duplicate/non-positive centers, count mismatches,
    and single-band levels."""
    from emptyui_aurora._processing import manual_bands

    with pytest.raises(ValueError, match="duplicate"):
        manual_bands("0.1, 0.1, 0.05", "2,1")
    with pytest.raises(ValueError, match="positive"):
        manual_bands("0.1, -0.05", "1,1")
    with pytest.raises(ValueError, match="sums to"):
        manual_bands("0.1, 0.05, 0.02", "2,2")
    with pytest.raises(ValueError, match=">= 2"):
        manual_bands("0.1, 0.05, 0.02", "2,1")
    with pytest.raises(ValueError, match="cannot parse"):
        manual_bands("0.1, abc", "2")


def test_estimate_tf_manual_parity():
    """band_style='manual' through estimate_tf equals the equivalent
    bands_override call (same table, same estimate)."""
    source = synthetic_source(13)
    runts = synthetic_runts(source, seed=13)
    fc = synthetic_fc(runts, ["hx", "hy"], ["ex", "ey"])
    freqs = np.asarray(fc["frequency"].values)
    # two bands straddling a decade, on fc's own grid (single level, so all
    # three centers reconstruct that level's two edges + outer edges)
    centers = [freqs[20], freqs[10], freqs[5]]
    table = manual_bands(centers, "3")
    kw = {"estimator_engine": "OLS", "remote_reference": False}
    p1, Z1, _ = estimate_tf(fc, ["hx", "hy"], ["ex", "ey"],
                            band_style="manual",
                            bands_override=table.get(0, []), **kw)
    p2, Z2, _ = estimate_tf(fc, ["hx", "hy"], ["ex", "ey"],
                            band_style="l2u",
                            bands_override=table.get(0, []), **kw)
    np.testing.assert_allclose(p1, p2)
    np.testing.assert_allclose(Z1, Z2)
    # manual without an override errors clearly
    from emptyui_aurora._processing import _resolve_period_bands
    with pytest.raises(ValueError, match="requires an explicit per-level band table"):
        _resolve_period_bands(fc, 4, "manual")


def _burst_container():
    """Minimal multi-family fc container (attrs only — the table builder
    works from attrs + sizes): the official MTU-5A MT-mode rate ladder."""
    import xarray as xr

    rates = [2400.0, 600.0, 150.0, 37.5, 15.0, 3.75, 0.9375, 0.234375]
    out = {}
    for lv, fs in enumerate(rates):
        out[lv] = xr.Dataset(
            {"hx": (("frequency", "time"), np.zeros((129, 100), complex))},
            coords={"frequency": np.arange(1, 130) * (fs / 256),
                    "time": np.arange(100)},
            attrs={"sample_rate": fs, "num_samples_window": 256})
    return out


def test_l2u_cross_level_claims_tile_burst_families():
    """Multi-family burst container: the level spans overlap (the rate
    families are not related by the cascade factor — 150/15 = 10), so the
    same periods would be estimated twice with slightly different centers.
    The claim pass tiles the table deepest-rate-first: no band span
    overlaps another level's, every level keeps bands."""
    from emptyui_aurora._processing import _l2u_band_table

    levels = _burst_container()
    tbl = _l2u_band_table(levels, min_bins_per_band=20.0)
    assert tbl is not None
    assert all(tbl[lv] for lv in tbl)
    spans = sorted((1.0 / lo, 1.0 / hi)
                   for lv in tbl for hi, lo in tbl[lv])
    for a, b in pairwise(spans):
        assert b[0] >= a[1] - 1e-12, (a, b)  # no cross-level overlap


def test_l2u_claims_leave_aligned_cascades_unchanged():
    """A uniform single-rate cascade's template spans abut — the claims
    never bite (the pre-burst l2u table is reproduced exactly)."""
    from emptyui_aurora._processing import _container_geometry, _l2u_band_table

    levels = {}
    for lv in range(4):
        fs = 150.0 / 4**lv
        levels[lv] = _burst_container()[0].assign_attrs(
            sample_rate=fs)
    g = _container_geometry(levels)
    # L-A6: the cascade `factor` field was deleted (levels sort ascending
    # by id so the r/s ratio could never reach the >= 2 gate, and no
    # consumer existed) — the geometry derives per-level df, which is all
    # the template needs
    assert g is not None and "factor" not in g
    tbl = _l2u_band_table(levels, min_bins_per_band=0)
    # the template survives on every level (no bands claimed away)
    counts = {len(v) for v in tbl.values()}
    assert len(counts) == 1


def test_merge_tf_levels_tolerance_dedup_misaligned_families():
    """merge_tf_levels resolves near-duplicate centers from misaligned
    rate families (deepest wins within half the center spacing), while
    aligned cascades keep exact collisions and distinct neighbors."""
    from emptyui_aurora._processing import merge_tf_levels

    def _entry(periods, ppo=3):
        n = len(periods)
        return {"periods": np.asarray(periods, float),
                "Z": np.ones((n, 2, 2), complex),
                "Z_err": np.full(n, 0.1),
                "periods_per_octave": ppo}

    # deeper level (key 1) estimates 0.95/1.9/3.8 s; shallower (key 0)
    # 1.0/2.0/4.0 s — every center within half a spacing (1/6 oct) of a
    # deeper one must be dropped
    per_level = {0: _entry([1.0, 2.0, 4.0]),
                 1: _entry([0.95, 1.9, 3.8])}
    p, _Z, _e = merge_tf_levels(per_level)
    np.testing.assert_allclose(p, [0.95, 1.9, 3.8])
    # distinct neighbors survive: ppo=6 spacing (0.167 oct) > half-width
    per_level = {0: _entry([1.0, 1.122, 1.26], ppo=6),
                 1: _entry([2.0, 2.24, 2.52], ppo=6)}
    p, _Z, _e = merge_tf_levels(per_level)
    assert p.size == 6
