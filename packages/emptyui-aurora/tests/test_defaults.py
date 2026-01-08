"""Regression test for the user-directed default flip (2026-09-14,
DESIGN.md §24.5): the per-bin coherence statistic is the default sort
statistic (EMTF-style, banding-independent), and the band-geometry
defaults move from the ablation-era values (4 periods/octave, 40 min
bins) to the SDG31-recipe values (3 periods/octave, 20 min bins) across
every user-facing surface — node manifests and the top-level low-level
APIs. Internal helpers (``_period_bands``, ``_l2u_band_table``,
``_resolve_period_bands``) keep their historical defaults; every call
site passes explicit values, so they never leak.
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
    assert d["statistic"] == "per_bin"
    assert d["ensemble_windows"] == 16
    assert d["periods_per_octave"] == 3
    assert d["min_bins_per_band"] == 20.0


def test_estimate_tf_node_defaults():
    d = _param_defaults(EstimateTF)
    assert d["periods_per_octave"] == 3
    assert d["min_bins_per_band"] == 20.0


def test_plot_coherency_node_defaults():
    d = _param_defaults(PlotCoherency)
    assert d["periods_per_octave"] == 3
    assert d["min_bins_per_band"] == 20.0


def test_low_level_api_defaults():
    sig = inspect.signature(coherence_sort_fc)
    assert sig.parameters["statistic"].default == "per_bin"
    assert sig.parameters["ensemble_windows"].default == 16
    assert sig.parameters["periods_per_octave"].default == 3
    sig = inspect.signature(coherence_sort_fcs)
    assert sig.parameters["statistic"].default == "per_bin"
    assert sig.parameters["ensemble_windows"].default == 16
    assert sig.parameters["periods_per_octave"].default == 3
    assert sig.parameters["min_bins_per_band"].default == 20.0
    sig = inspect.signature(estimate_tf)
    assert sig.parameters["periods_per_octave"].default == 3
    sig = inspect.signature(estimate_tf_levels)
    assert sig.parameters["periods_per_octave"].default == 3
    assert sig.parameters["min_bins_per_band"].default == 20.0
    sig = inspect.signature(_per_bin_coherence)
    assert sig.parameters["ensemble_windows"].default == 16
