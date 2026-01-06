"""Plot Coherency node (DESIGN §20.1): fc-direct per-band multiple
coherence — numerically identical to the Coherence Sort statistic (shared
`_band_coherence` helper)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from _synthetic import synthetic_fc, synthetic_runts, synthetic_source
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
    """The plotted medians equal the sort's diagnostics medians band by band
    (shared helper — the plot is honest about the statistic the sort uses)."""
    levels = _levels()
    from emptyui_aurora._processing import (
        _band_coherence,
        _bands_from_freq_pairs,
        _l2u_band_table,
    )
    from emptyui_aurora.process import CoherenceSort

    # recompute the plotted medians through the shared helper (with the
    # sort's degenerate-band cap, as the node applies it)
    bands = _l2u_band_table(levels, min_bins_per_band=40.0)
    sample = next(iter(levels.values()))
    ins = [c.strip() for c in sample.attrs["input_channels"].split(",")]
    outs = [c.strip() for c in sample.attrs["output_channels"].split(",")]
    helper_medians = {}
    for level, lfc in sorted(levels.items()):
        lfc = _normalize_fc_layout(lfc)
        periods, bins_list = _bands_from_freq_pairs(lfc, bands.get(level, []))
        for p, bins in zip(periods, bins_list):
            if bins.size == 0:
                continue
            coh_e, _ = _band_coherence(lfc, None, ins, outs, bins)
            if bins.size < len(ins) + 1:
                coh_e = np.minimum(coh_e, 0.98 - 1e-9)
            helper_medians[round(float(p), 6)] = float(np.median(coh_e))

    # the sort's diagnostics must carry the identical medians
    res2 = CoherenceSort().execute(
        RunContext(workspace=tmp_path, store=DataStore()), levels,
        periods_per_octave=4, band_style="l2u", min_bins_per_band=40.0,
        t_short=0.0)
    rows = json.loads(Path(res2["table"]["path"]).read_text())
    assert rows, "sort diagnostics empty"
    for r in rows:
        key = round(r["period"], 6)
        assert key in helper_medians
        assert helper_medians[key] == pytest.approx(r["median_coherence"])


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


def test_inherit_map_matches_estimate_tf():
    """Same inherit-from-sort declaration as Estimate TF (DESIGN §18.2)."""
    from emptyui_aurora.plots import PlotCoherency
    from emptyui_aurora.process import EstimateTF

    assert PlotCoherency().manifest()["inherit"] == EstimateTF().manifest()["inherit"]


def test_no_bands_raises(tmp_path):
    ctx = RunContext(workspace=tmp_path, store=DataStore())
    fc = synthetic_fc(synthetic_runts(synthetic_source(14), seed=14),
                      ["hx", "hy"], ["ex", "ey"])
    fc = _normalize_fc_layout(fc)
    # a window starved to nothing: t_short beyond the longest band clips all
    with pytest.raises(ValueError, match="no usable frequency bands"):
        PlotCoherency().execute(ctx, fc, periods_per_octave=4,
                                band_style="l2u", min_bins_per_band=0.0,
                                t_short=1e9)


def test_uniform_head_clip_raises_when_all_clipped(tmp_path):
    """The uniform scheme honors t_short (previously ignored): clipping
    every band must raise the same 'no usable frequency bands' error the
    l2u path raises — before the fix, uniform + t_short=1e9 silently
    plotted the unclipped head."""
    ctx = RunContext(workspace=tmp_path, store=DataStore())
    fc = synthetic_fc(synthetic_runts(synthetic_source(14), seed=14),
                      ["hx", "hy"], ["ex", "ey"])
    fc = _normalize_fc_layout(fc)
    with pytest.raises(ValueError, match="no usable frequency bands"):
        PlotCoherency().execute(ctx, fc, periods_per_octave=4,
                                band_style="uniform", min_bins_per_band=0.0,
                                t_short=1e9)
