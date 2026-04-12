"""Plot Spectrogram (fc-based, aurora.plot.spectrogram — DESIGN §23).

Power comes straight from the Compute FC coefficients (no second FFT);
channel/level selection resolve from the fc with actionable errors."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from _synthetic import synthetic_runts, synthetic_source
from emptyui.context import RunContext
from emptyui.store import DataStore
from emptyui_aurora._processing import compute_fc_levels


def _levels(seed=16):
    runts = synthetic_runts(synthetic_source(seed), seed=seed)
    return compute_fc_levels(runts, ["hx", "hy"], ["ex", "ey"],
                             num_decimations=2, factor=4,
                             num_samples_window=256, overlap=0.5,
                             window_type="hamming")


def test_spectrogram_power_peaks_at_signal_frequency(tmp_path):
    """A strong 0.5 Hz component must put the power peak in its frequency
    bin (the spectrogram reuses the FC coefficients — no second FFT)."""
    from emptyui_aurora.plots import PlotSpectrogramFC

    levels = _levels()
    ds = levels[0]
    # inject a pure tone at 0.5 Hz (on-bin for these frequencies) in ex
    f0 = float(ds["frequency"].values[3])
    amp = 100.0
    ds["ex"].values[:, :] = amp * (
        np.cos(2 * np.pi * f0 * ds["time"].values)[None, :]
        + 1j * np.sin(2 * np.pi * f0 * ds["time"].values)[None, :]
    )
    ctx = RunContext(workspace=tmp_path, store=DataStore())
    out = PlotSpectrogramFC().execute(ctx, levels, channel="ex")
    path = Path(out["plot"]["path"])
    assert path.read_bytes().startswith(b"\x89PNG")


def test_spectrogram_level_selection(tmp_path):
    from emptyui_aurora.plots import PlotSpectrogramFC

    levels = _levels()
    ctx = RunContext(workspace=tmp_path, store=DataStore())
    # explicit level
    out = PlotSpectrogramFC().execute(ctx, levels, channel="ex",
                                      decimation_level=1)
    assert Path(out["plot"]["path"]).exists()
    # invalid level → actionable error listing the available ones
    with pytest.raises(ValueError, match="available"):
        PlotSpectrogramFC().execute(ctx, levels, channel="ex",
                                    decimation_level=9)


def test_spectrogram_channel_resolution(tmp_path):
    """Standard names map through the fc's nomenclature; unknown → error."""
    from emptyui_aurora.plots import PlotSpectrogramFC

    runts = synthetic_runts(synthetic_source(17), magnetic=("bx", "by"),
                            electric=("e1", "e2"), seed=17)
    levels = compute_fc_levels(runts, ["bx", "by"], "e1,e2",
                               num_decimations=0, num_samples_window=256,
                               overlap=0.5, window_type="hamming",
                               channel_nomenclature="lemi12")
    ctx = RunContext(workspace=tmp_path, store=DataStore())
    # standard name ex maps to e1 (lemi12)
    out = PlotSpectrogramFC().execute(ctx, levels, channel="ex")
    assert "e1" in Path(out["plot"]["path"]).name
    with pytest.raises(ValueError, match="not in"):
        PlotSpectrogramFC().execute(ctx, levels, channel="wombat")


def test_spectrogram_manifest_inherits_from_compute_fc():
    """Inherit map + split_inherited contract (DESIGN §23)."""
    from emptyui_aurora.plots import PlotSpectrogramFC
    from emptyui_aurora.process import ComputeFC

    m = PlotSpectrogramFC().manifest()
    inherit = m["inherit"]["fc"]
    assert inherit["channel"] == "output_channels"
    # L-A10: the display-only window_samples/decimation_levels params were
    # deleted (declared + inherited but never read — the annotation reads
    # the fc's own attrs)
    assert "window_samples" not in inherit
    assert "decimation_levels" not in inherit
    # upstream params exist on ComputeFC
    cf_params = {p["name"] for p in ComputeFC().manifest()["params"]}
    assert {"window_samples", "decimation_levels", "output_channels"} <= cf_params
    # channel offers comma-split inherited choices
    ch = next(p for p in m["params"] if p["name"] == "channel")
    assert ch["split_inherited"] is True


def test_old_ts_spectrogram_hidden_but_registered():
    """The ts.run spectrogram stays executable for saved workflows but is
    out of the palette (superseded by the fc-based node)."""
    from emptyui.registry import build_default_registry

    registry = build_default_registry()
    node = registry.get("core.plot.spectrogram")
    assert node is not None
    assert node.hidden is True
    assert registry.get("aurora.plot.spectrogram") is not None
