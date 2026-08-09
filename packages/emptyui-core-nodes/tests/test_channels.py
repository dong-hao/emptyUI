"""Channel resolution for core display nodes (no external data)."""

from __future__ import annotations

import numpy as np
from emptyui.context import RunContext
from emptyui.store import DataStore
from emptyui_core_nodes._channels import (
    auto_channel,
    infer_keyword,
    resolve_channels,
)
from emptyui_core_nodes.plots import PlotSpectrogramNode, PlotTimeSeriesNode
from emptyui_core_nodes.types import TimeSeries


def _ts(channels: dict[str, np.ndarray], sr: float = 8.0) -> TimeSeries:
    return TimeSeries(channels=channels, sample_rate=sr)


def test_resolve_empty_returns_all():
    ts = _ts({"Ex": np.zeros(16), "Ey": np.zeros(16), "Hx": np.zeros(16)})
    assert resolve_channels(list(ts.channels), "") == ["Ex", "Ey", "Hx"]


def test_resolve_case_insensitive_exact():
    ts = _ts({"Ex": np.zeros(16), "Ey": np.zeros(16)})
    assert resolve_channels(list(ts.channels), "ex,EY") == ["Ex", "Ey"]


def test_resolve_standard_names_through_lemi_nomenclature():
    ts = _ts({"e1": np.zeros(16), "e2": np.zeros(16),
              "bx": np.zeros(16), "by": np.zeros(16)})
    assert infer_keyword(list(ts.channels)) == "lemi12"
    assert resolve_channels(list(ts.channels), "ex,ey") == ["e1", "e2"]
    assert resolve_channels(list(ts.channels), "hx,hy") == ["bx", "by"]


def test_resolve_standard_names_through_phoenix_nomenclature():
    ts = _ts({"h1": np.zeros(16), "h2": np.zeros(16),
              "e1": np.zeros(16), "e2": np.zeros(16), "h3": np.zeros(16)})
    assert infer_keyword(list(ts.channels)) == "phoenix123"
    assert resolve_channels(list(ts.channels), "hx,hy") == ["h1", "h2"]
    assert resolve_channels(list(ts.channels), "ex,ey,hz") == ["e1", "e2", "h3"]


def test_auto_channel_prefers_electric():
    ts = _ts({"e1": np.zeros(16), "e2": np.zeros(16), "bz": np.zeros(16)})
    assert auto_channel(list(ts.channels)) == "e1"
    ts2 = _ts({"Ex": np.zeros(16), "Hx": np.zeros(16)})
    assert auto_channel(list(ts2.channels)) == "Ex"
    # no electric channels -> first channel
    ts3 = _ts({"bz": np.zeros(16), "bx": np.zeros(16)})
    assert auto_channel(list(ts3.channels)) == "bz"


def test_spectrogram_auto_channel_and_standard_name(tmp_path):
    """Empty channel picks the first electric channel; standard names resolve
    through the inferred nomenclature (LEMI e1/e2)."""
    rng = np.random.default_rng(11)
    n = 2048
    ts = _ts({"e1": rng.standard_normal(n), "e2": rng.standard_normal(n),
              "bx": rng.standard_normal(n), "by": rng.standard_normal(n)},
             sr=8.0)
    ctx = RunContext(workspace=tmp_path, store=DataStore())

    out = PlotSpectrogramNode().execute(ctx, ts, channel="", window_s=1.0)
    assert out["plot"]["path"].endswith("plot_spectrogram_e1_node.png")

    out2 = PlotSpectrogramNode().execute(ctx, ts, channel="ey", window_s=1.0)
    assert out2["plot"]["path"].endswith("plot_spectrogram_e2_node.png")


def test_timeseries_standard_names_resolve(tmp_path):
    rng = np.random.default_rng(12)
    n = 512
    ts = _ts({"e1": rng.standard_normal(n), "e2": rng.standard_normal(n),
              "bx": rng.standard_normal(n), "by": rng.standard_normal(n)},
             sr=8.0)
    ctx = RunContext(workspace=tmp_path, store=DataStore())
    out = PlotTimeSeriesNode().execute(ctx, ts, channels="ex,ey", seconds=10)
    assert out["plot"]["path"].endswith("plot_timeseries_node.png")


def test_fallback_maps_match_mt_metadata():
    """M5: the plugin-free _FALLBACK_MAPS mirrors mt_metadata's canonical
    CHANNEL_MAPS (v1.0.10) for standalone use — when the library IS
    importable the pin makes an upstream map change flag the fallback
    instead of drifting silently."""
    from emptyui_core_nodes import _channels

    maps = _channels._maps()
    assert maps is not _channels._FALLBACK_MAPS, (
        "mt_metadata importable — the fallback must not be in use")
    for kw, mapping in _channels._FALLBACK_MAPS.items():
        assert maps[kw] == mapping, f"stale fallback for {kw!r}"
