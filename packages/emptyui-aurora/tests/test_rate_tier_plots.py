"""Burst-mode rate-tier plotting (DESIGN.md §10.6): the merged FC database is
rate-keyed; the plot nodes select a tier by its actual sample rate."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr


def _container(with_families: bool = True):
    """Minimal multi-family fc container — the MTU-5A ladder."""
    rng = np.random.default_rng(7)
    rates = [2400.0, 600.0, 150.0, 37.5, 15.0, 3.75, 0.9375, 0.234375]
    out = {}
    for lv, fs in enumerate(rates):
        out[lv] = xr.Dataset(
            {"hx": (("frequency", "time"),
                    rng.standard_normal((129, 100))
                    + 1j * rng.standard_normal((129, 100))),
             "ex": (("frequency", "time"),
                    rng.standard_normal((129, 100))
                    + 1j * rng.standard_normal((129, 100)))},
            coords={"frequency": np.arange(1, 130) * (fs / 256),
                    "time": np.arange(100)},
            attrs={"sample_rate": fs, "num_samples_window": 256,
                   "input_channels": "hx", "output_channels": "ex"})
    if with_families:
        for ds in out.values():
            ds.attrs["family_rates"] = [2400.0, 150.0, 15.0]
    return out


def test_auto_picks_lowest_native_family():
    from emptyui_aurora.plots import _resolve_rate_tiers

    rate, tiers = _resolve_rate_tiers(_container(), "auto")
    assert rate == 15.0  # the stream's native rate, not the deepest tier
    assert tiers == [4, 5, 6, 7]  # the family's whole cascade, native first


def test_auto_without_family_attr_is_legacy_all_levels():
    from emptyui_aurora.plots import _resolve_rate_tiers

    rate, tiers = _resolve_rate_tiers(_container(with_families=False), "auto")
    assert tiers == [0, 1, 2, 3, 4, 5, 6, 7]  # the filter is a no-op
    assert rate == 2400.0  # native rate first — legacy indices reproduced


def test_explicit_rate_selects_its_tier():
    from emptyui_aurora.plots import _resolve_rate_tiers

    for spec, rate in (("150", 150.0), ("37.5", 37.5), ("2400", 2400.0)):
        got, tiers = _resolve_rate_tiers(_container(), spec)
        assert got == rate
        assert len(tiers) == 1


def test_spectrogram_step_semantics_burst(tmp_path):
    """decimation_level indexes the cascade STEP within the selected
    family: 0 = the family's native tier (original data), 1 = one deeper,
    -1 = deepest; out-of-range steps reject naming the family's steps."""
    from emptyui.context import RunContext
    from emptyui.store import DataStore
    from emptyui_aurora.plots import PlotSpectrogramFC

    ctx = RunContext(workspace=tmp_path, store=DataStore())
    # auto family (15 Hz): step 0 = the native tier (level 4, <=7.5 Hz)
    art = PlotSpectrogramFC().execute(
        ctx, _container(), sample_rate="auto")["plot"]
    assert art is not None
    # step 1 = the 3.75 Hz tier
    art2 = PlotSpectrogramFC().execute(
        ctx, _container(), sample_rate="auto", decimation_level=1)["plot"]
    assert art2 is not None
    with pytest.raises(ValueError, match="decimation step"):
        PlotSpectrogramFC().execute(
            ctx, _container(), sample_rate="auto", decimation_level=9)


def test_unknown_rate_lists_available():
    from emptyui_aurora.plots import _resolve_rate_tiers

    with pytest.raises(ValueError, match="available rates"):
        _resolve_rate_tiers(_container(), "999")


def test_single_rate_keeps_every_level():
    from emptyui_aurora.plots import _resolve_rate_tiers

    # a uniform single-rate cascade: all levels share one rate
    levels = {}
    for lv in range(4):
        fs = 150.0 / 4**lv
        levels[lv] = xr.Dataset(
            {"hx": (("frequency", "time"), np.zeros((129, 100), complex))},
            coords={"frequency": np.arange(1, 130) * (fs / 256),
                    "time": np.arange(100)},
            attrs={"sample_rate": fs})
    _rate, tiers = _resolve_rate_tiers(levels, "auto")
    assert tiers == [0, 1, 2, 3]  # the filter is a no-op


def test_plot_coherency_burst_shows_only_selected_family(tmp_path):
    from emptyui.context import RunContext
    from emptyui.store import DataStore
    from emptyui_aurora.plots import PlotCoherency

    ctx = RunContext(workspace=tmp_path, store=DataStore())
    res = PlotCoherency().execute(ctx, _container(), sample_rate="150")
    assert res["plot"] is not None


def test_fc_summary_streams_rate_tiers():
    from emptyui.executor import _fc_summary

    out = _fc_summary(_container())
    assert out is not None
    assert out["kind"] == "fc"
    assert out["rates"] == sorted(out["rates"], reverse=True)
    assert 2400.0 in out["rates"] and 15.0 in out["rates"]
    assert _fc_summary({"not": "an fc"}) is None
