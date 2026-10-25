"""Plot Coherency node (docs/DESIGN.md §10.6 and §10.3): fc-direct per-frequency-bin
window-ensemble coherence — the exact statistic Coherence Sort applies
(shared `_per_bin_coherence` helper). No band geometry anywhere."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from _synthetic import synthetic_runts, synthetic_source
from emptyui.context import RunContext
from emptyui.store import DataStore
from emptyui_aurora._processing import (
    _normalize_fc_layout,
    compute_fc_levels,
)
from emptyui_aurora.plots import PlotCoherency


def _levels(seed=13):
    runts = synthetic_runts(synthetic_source(seed), seed=seed)
    return compute_fc_levels(runts, ["hx", "hy"], ["ex", "ey"],
                             num_decimations=2, factor=4,
                             num_samples_window=256, overlap=0.5,
                             window_type="boxcar")


def test_plot_matches_coherence_sort_numerically(tmp_path):
    """The plotted medians equal the sort's per-bin diagnostics (shared
    `_per_bin_coherence` helper — the plot is honest about the statistic
    the sort uses)."""
    levels = _levels()
    from emptyui_aurora._processing import (
        _per_bin_coherence,
    )
    from emptyui_aurora.process import CoherenceSort

    sample = next(iter(levels.values()))
    ins = [c.strip() for c in sample.attrs["input_channels"].split(",")]
    outs = [c.strip() for c in sample.attrs["output_channels"].split(",")]

    res = PlotCoherency().execute(
        RunContext(workspace=tmp_path, store=DataStore()), levels)
    assert Path(res["plot"]["path"]).stat().st_size > 1000

    # the sort's per-bin diagnostics carry the identical medians (shared
    # statistic — per (bin, window) window-ensemble coherence), checked on
    # one level bin-by-bin
    res2 = CoherenceSort().execute(
        RunContext(workspace=tmp_path, store=DataStore()), levels)
    rows = json.loads(Path(res2["table"]["path"]).read_text())
    assert rows, "sort diagnostics empty"

    lfc = _normalize_fc_layout(levels[0])
    coh_e, _coh_x = _per_bin_coherence(
        lfc, None, ins, outs, ensemble_windows=16)
    freq0 = lfc["frequency"].values
    rows_l0 = [r for r in rows if r["decimation_level"] == 0]
    idx = 0
    for i, fv in enumerate(freq0):
        if fv <= 0.0:
            continue  # DC bin — no diagnostics row
        row = rows_l0[idx]
        idx += 1
        med = (float(np.nanmedian(coh_e[i, :]))
               if np.isfinite(coh_e[i, :]).any() else None)
        assert row["period"] == pytest.approx(1.0 / fv)
        assert row["median_coherence_e"] == med


def test_node_renders_png_single_and_two_fc(tmp_path):
    levels = _levels()
    ctx = RunContext(workspace=tmp_path, store=DataStore())
    res = PlotCoherency().execute(ctx, levels)
    p = res["plot"]["path"]
    assert p.endswith(".png")
    assert len(Path(p).read_bytes()) > 1000

    # two-FC mode: a magnetic-only remote fc — renders with both criteria
    remote = synthetic_runts(synthetic_source(15, kind="white"),
                             with_e=False, seed=15)
    ref_levels = compute_fc_levels(remote, ["hx", "hy"], "",
                                   num_decimations=2, factor=4,
                                   num_samples_window=256, overlap=0.5,
                                   window_type="boxcar")
    res2 = PlotCoherency().execute(
        RunContext(workspace=tmp_path, store=DataStore()), levels, ref_fc=ref_levels)
    assert len(Path(res2["plot"]["path"]).read_bytes()) > 1000


def test_plot_has_no_banding_params():
    """history §24.9: no band-geometry params (or inherit-from-sort) on the plot —
    it renders the sort's per-bin statistic, not a band aggregation."""
    from emptyui_aurora.plots import PlotCoherency

    declared = {p.name for p in PlotCoherency.params}
    assert declared == {"coherence_min", "coherence_max",
                        "ensemble_windows", "smooth_bins", "sample_rate",
                        "variation"}
    assert not PlotCoherency().manifest()["inherit"]
    assert not hasattr(PlotCoherency, "inherit") or not getattr(
        PlotCoherency, "inherit", None)


def test_no_usable_bins_raises(tmp_path):
    """An fc with no usable electric outputs is refused, not plotted."""
    from _synthetic import synthetic_runts, synthetic_source

    remote = compute_fc_levels(
        synthetic_runts(synthetic_source(16), with_e=False, seed=16),
        ["hx", "hy"], "", num_decimations=2, factor=4,
        num_samples_window=256, overlap=0.5, window_type="boxcar")
    # drop the attrs the node uses to classify outputs — with no electric
    # channel in the container and no attrs hint, the node must refuse
    for fc in remote.values():
        fc.attrs.pop("output_channels", None)
    with pytest.raises(ValueError):
        PlotCoherency().execute(
            RunContext(workspace=tmp_path, store=DataStore()), remote)


def test_plot_coherency_two_fc_pairs_remote_grid(tmp_path):
    """The two-FC statistic applies to PAIRED windows — the remote fc is
    validated + paired exactly like Coherence Sort (shared helper): a
    different frequency grid crashes with an actionable error (not a raw
    broadcast error), and a same-grid shorter remote is trimmed to the
    matched windows instead of silently mixing windows."""
    from emptyui.context import RunContext
    from emptyui.store import DataStore
    from emptyui_aurora.plots import PlotCoherency

    levels = _levels(13)
    ctx = RunContext(workspace=tmp_path, store=DataStore())
    # (1) different frequency grid: actionable error, not einsum broadcast
    ref_bad = {lv: ds.isel(frequency=slice(0, ds.sizes["frequency"] - 1))
               for lv, ds in levels.items()}
    with pytest.raises(ValueError, match="different frequency grid"):
        PlotCoherency().execute(ctx, levels, ref_fc=ref_bad)
    # (2) same grids, fewer windows: intersect-trimmed, renders
    ref_fewer = {lv: ds.isel(time=slice(0, ds.sizes["time"] - 2))
                 for lv, ds in levels.items()}
    res = PlotCoherency().execute(ctx, levels, ref_fc=ref_fewer)
    assert res["plot"]["path"].endswith(".png")


# --- L18: the coherence map (time x frequency view of the sort's criterion)


def _two_fc_levels(seed=21, start_datetime=""):
    """Local + remote fc level containers (the two-FC map's inputs)."""
    from emptyui_aurora._processing import compute_fc_levels as cfl
    local = synthetic_runts(synthetic_source(seed), seed=seed,
                            start_datetime=start_datetime)
    ref = synthetic_runts(synthetic_source(seed + 100), e_noise=2.0,
                          seed=seed + 100, start_datetime=start_datetime)
    kw = {"num_decimations": 2, "factor": 4, "num_samples_window": 256,
          "overlap": 0.5, "window_type": "boxcar"}
    return cfl(local, ["hx", "hy"], ["ex", "ey"], **kw), \
        cfl(ref, ["hx", "hy"], ["ex", "ey"], **kw)


def test_coherence_map_matrix_is_the_sort_criterion():
    """The map's per-(bin, window) values are EXACTLY the sort's criterion:
    the conservative min of the E–remote-H signal and local↔remote
    transfer coherences (two-FC), computed over the intersect-paired
    windows — a dark region in the map is a window the sort down-weights."""
    from emptyui_aurora._processing import (
        _normalize_fc_layout,
        _pair_remote_fc,
        _per_bin_coherence,
    )
    from emptyui_aurora.plots import _coherence_map_matrix

    levels, ref_levels = _two_fc_levels()
    sample = _normalize_fc_layout(levels[0])
    ins = [c.strip() for c in sample.attrs["input_channels"].split(",")]
    outs = [c.strip() for c in sample.attrs["output_channels"].split(",")]

    coh, centres, paired_local = _coherence_map_matrix(
        levels[0], ref_levels[0], ins, outs, ensemble_windows=8,
        smooth_bins=0)

    # the reference: pair + statistic exactly like coherence_sort_fc
    lfc = _normalize_fc_layout(levels[0])
    rfc = _normalize_fc_layout(ref_levels[0])
    lfc2, rfc2 = _pair_remote_fc(lfc, rfc, pairing="intersect", tol_s=0.0)
    coh_e, coh_x = _per_bin_coherence(lfc2, rfc2, ins, outs,
                                      ensemble_windows=8, smooth_bins=0)
    np.testing.assert_array_equal(coh, np.minimum(coh_e, coh_x))
    # the map's time axis rides the PAIRED local grid
    assert centres.size == lfc2.sizes["time"]
    np.testing.assert_array_equal(
        np.asarray(paired_local["time"].values),
        np.asarray(lfc2["time"].values))


def test_coherence_map_renders_and_selects_tiers(tmp_path):
    """The node renders a PNG for both modes; the cascade-step selection
    follows the spectrogram's convention and errors actionably OOR."""
    from emptyui_aurora.plots import CoherenceMap

    levels, ref_levels = _two_fc_levels()
    ctx = RunContext(workspace=tmp_path, store=DataStore())

    res = CoherenceMap().execute(ctx, levels, ref_fc=ref_levels,
                                 decimation_level=0)
    assert Path(res["plot"]["path"]).stat().st_size > 1000

    res1 = CoherenceMap().execute(ctx, levels, ref_fc=None,
                                  decimation_level=-1)
    assert Path(res1["plot"]["path"]).stat().st_size > 1000

    with pytest.raises(ValueError, match="no decimation step 9"):
        CoherenceMap().execute(ctx, levels, decimation_level=9)


def test_coherence_map_window_centres():
    """The map's time axis is the fc's own window-centre labels, unchanged.

    H8: the fc `time` coordinate is ALREADY the scipy segment centre (L11;
    `compute_fc_levels` documents it, `_centre_to_start_s` subtracts
    N/(2 fs) to recover a start). The map used to add a second half-window,
    labelling window ENDS — a shift of half a window per level: 16 s at
    level 0 and 256 s at level 2 in this fixture, tens of minutes at the
    deepest production levels.
    """
    from emptyui_aurora._processing import _normalize_fc_layout
    from emptyui_aurora.plots import _coherence_map_matrix

    def _map(levels, ref_levels, lv):
        d = _normalize_fc_layout(levels[lv])
        ins = [c.strip() for c in d.attrs["input_channels"].split(",")]
        outs = [c.strip() for c in d.attrs["output_channels"].split(",")]
        return _coherence_map_matrix(
            levels[lv], ref_levels[lv], ins, outs, ensemble_windows=8,
            smooth_bins=0)

    # relative float-seconds fc (the fixture's default)
    levels, ref_levels = _two_fc_levels()
    _, centres, paired = _map(levels, ref_levels, 0)
    labels = np.asarray(paired["time"].values)
    np.testing.assert_array_equal(np.asarray(centres), labels)
    # the labels really are centres: the fixture starts at 0 s, so the
    # first label is half a window in, not at the record start.
    sr = float(paired.attrs["sample_rate"])
    nw = float(paired.attrs["num_samples_window"])
    assert float(labels[0]) == pytest.approx(nw / sr / 2)

    # absolute-time fc (the production path) — same no-shift contract
    dt_levels, dt_ref = _two_fc_levels(start_datetime="2024-01-01T00:00:00")
    _, dt_centres, dt_paired = _map(dt_levels, dt_ref, 0)
    assert np.asarray(dt_centres).dtype.kind == "M"
    np.testing.assert_array_equal(
        np.asarray(dt_centres), np.asarray(dt_paired["time"].values))

    # deepest level: a half-window shift here is minutes, so a regression
    # cannot hide inside a tolerance.
    deep = max(levels)
    _, deep_centres, deep_paired = _map(levels, ref_levels, deep)
    np.testing.assert_array_equal(
        np.asarray(deep_centres),
        np.asarray(deep_paired["time"].values))
    d = _normalize_fc_layout(levels[deep])
    deep_half = (float(d.attrs["num_samples_window"])
                 / float(d.attrs["sample_rate"]) / 2)
    assert deep_half >= 60  # the regression would shift by this much


def test_coherence_map_title_params_toggle():
    """print_params=False (the default) keeps the production title clean;
    True adds the provenance line (the Plot AppRes/Phase convention)."""
    from emptyui_aurora._processing import _normalize_fc_layout
    from emptyui_aurora.plots import _coherence_map_title

    levels, _ref = _two_fc_levels()
    paired = _normalize_fc_layout(levels[0])
    clean = _coherence_map_title(paired, 2, 15.0, 16, True, False)
    full = _coherence_map_title(paired, 2, 15.0, 16, True, True)
    assert clean.endswith("Coherence map")
    assert "ensemble" not in clean
    assert "ensemble ±16" in full and "step 2" in full
    assert "E–remote H" in full
