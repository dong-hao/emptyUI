"""Default-surface regression tests.

History: the 2026-09-14 default flip (§24.5) set the per-bin statistic and
the 3-periods/octave / 20-min-bins SDG31-recipe values. The 2026-09-15
simplification (§24.9) removed banding from the sort entirely — the sort's
surface is now band-free and the estimator keeps the SDG31-recipe banding
defaults; the internal helpers (``_period_bands``, ``_l2u_band_table``,
``_resolve_period_bands``) keep their historical defaults, every call site
passes explicit values.
"""

from __future__ import annotations

import inspect

from emptyui_aurora._processing import (
    _per_bin_coherence,
    coherence_sort_fc,
    coherence_sort_fcs,
    estimate_tf,
    estimate_tf_levels,
)
from emptyui_aurora.plots import PlotCoherency
from emptyui_aurora.process import CoherenceSort, EstimateTF


def _param_defaults(node):
    return {p.name: p.default for p in node.params}


def test_coherence_sort_node_defaults():
    d = _param_defaults(CoherenceSort)
    assert d["ensemble_windows"] == 16
    assert d["smooth_bins"] == 0
    # §24.9: no banding or statistic params on the sort node
    for gone in ("statistic", "periods_per_octave", "band_style",
                 "min_bins_per_band", "t_short", "coherence_trim"):
        assert gone not in d


def test_estimate_tf_node_defaults():
    d = _param_defaults(EstimateTF)
    assert d["periods_per_octave"] == 3
    assert d["min_bins_per_band"] == 20.0


def test_plot_coherency_node_defaults():
    d = _param_defaults(PlotCoherency)
    assert d["ensemble_windows"] == 16
    assert d["smooth_bins"] == 0
    # §24.9: the plot renders the per-bin statistic — no banding params
    for gone in ("periods_per_octave", "band_style", "min_bins_per_band",
                 "t_short"):
        assert gone not in d


def test_low_level_api_defaults():
    sig = inspect.signature(coherence_sort_fc)
    assert sig.parameters["ensemble_windows"].default == 16
    assert "statistic" not in sig.parameters
    assert "band_style" not in sig.parameters
    sig = inspect.signature(coherence_sort_fcs)
    assert sig.parameters["ensemble_windows"].default == 16
    assert "statistic" not in sig.parameters
    assert "min_bins_per_band" not in sig.parameters
    sig = inspect.signature(estimate_tf)
    assert sig.parameters["periods_per_octave"].default == 3
    sig = inspect.signature(estimate_tf_levels)
    assert sig.parameters["periods_per_octave"].default == 3
    assert sig.parameters["min_bins_per_band"].default == 20.0
    sig = inspect.signature(_per_bin_coherence)
    assert sig.parameters["ensemble_windows"].default == 16
