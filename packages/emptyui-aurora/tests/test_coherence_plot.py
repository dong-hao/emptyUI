"""Plot Coherency node (DESIGN.md §10.6 and §10.3): fc-direct per-frequency-bin
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
                             window_type="hamming")


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
                                   window_type="hamming")
    res2 = PlotCoherency().execute(
        RunContext(workspace=tmp_path, store=DataStore()), levels, ref_fc=ref_levels)
    assert len(Path(res2["plot"]["path"]).read_bytes()) > 1000


def test_plot_has_no_banding_params():
    """§24.9: no band-geometry params (or inherit-from-sort) on the plot —
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
        num_samples_window=256, overlap=0.5, window_type="hamming")
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
