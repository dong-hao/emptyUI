"""Display nodes: render figures to `plot` outputs (DESIGN.md §3).

Display nodes emit a typed `plot` port (rendered-figure reference) consumed
by `Preview Plot` (inline viewer) and `Save Plot` (export) — ComfyUI's
Preview/Save Image pattern."""

from __future__ import annotations

from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from emptyui import Input, Node, Output, Param

from ._channels import auto_channel, resolve_channels
from .types import TimeSeries, coerce_run


def _select_channels(data: TimeSeries, spec: str) -> list[str]:
    """Resolve a channel spec against the data (empty = all). Standard
    component names (ex/ey/hx/hy/hz) are mapped through the inferred
    nomenclature; matching is case-insensitive."""
    return resolve_channels(list(data.channels), spec)


class PlotTimeSeriesNode(Node):
    type = "core.plot.timeseries"
    title = "Plot Time Series"
    description = "Stacked channel time-domain plot."
    category = "Display"
    inputs = (Input("data", "ts.run", label="time series"),)
    outputs = (Output("plot", "plot"),)
    params = (
        Param("channels", "string", label="Channels (comma-sep, empty=all)",
              default="", optional=True,
              description="Channel names to plot; empty = all channels. "
                          "Standard names (ex/ey/hx/hy/hz) are mapped through "
                          "the inferred channel nomenclature; matching is "
                          "case-insensitive."),
        Param("seconds", "number", label="Duration to plot (s)", default=None,
              optional=True),
    )

    def execute(self, ctx, data, channels: str = "",
                seconds=None) -> dict[str, Any]:
        data = coerce_run(data)
        names = _select_channels(data, channels)
        n = data.n_samples if seconds is None else min(
            data.n_samples, int(seconds * data.sample_rate))
        t = data.time_axis()[:n]

        fig, axes = plt.subplots(len(names), 1, figsize=(10, 1.8 * len(names)),
                                 sharex=True, squeeze=False)
        for ax, name in zip(axes[:, 0], names):
            ax.plot(t, data.channels[name][:n], lw=0.6)
            ax.set_ylabel(name)
        axes[-1, 0].set_xlabel("Time (s)")
        fig.suptitle("Time series")
        fig.tight_layout()

        path = ctx.workspace / "plot_timeseries.png"
        fig.savefig(path, dpi=110)
        plt.close(fig)
        artifact = ctx.publish_artifact(path, kind="image")
        return {"plot": artifact}


class PlotSpectrogramNode(Node):
    type = "core.plot.spectrogram"
    title = "Plot Spectrogram"
    description = "STFT spectrogram of one channel."
    category = "Display"
    inputs = (Input("data", "ts.run", label="time series"),)
    outputs = (Output("plot", "plot"),)
    params = (
        Param("channel", "string", label="Channel", default="",
              placeholder="auto (first electric channel)",
              description="Channel to spectrogram; empty = first electric "
                          "channel (or first channel). Standard names "
                          "(ex/ey/hx/hy/hz) are mapped through the inferred "
                          "channel nomenclature."),
        Param("window_s", "number", label="Window length (s)", default=10.0),
    )

    def execute(self, ctx, data, channel: str = "",
                window_s: float = 10.0) -> dict[str, Any]:
        data = coerce_run(data)
        from scipy.signal import spectrogram

        if str(channel or "").strip():
            names = resolve_channels(list(data.channels), channel)
        else:
            names = [auto_channel(list(data.channels))]
        channel = names[0]
        x = data.channels[channel]
        nperseg = max(8, int(window_s * data.sample_rate))

        f, t, sxx = spectrogram(x, fs=data.sample_rate, nperseg=nperseg)
        sxx_db = 10 * np.log10(sxx + 1e-24)

        fig, ax = plt.subplots(figsize=(10, 4))
        mesh = ax.pcolormesh(t, f, sxx_db, shading="auto", cmap="viridis")
        fig.colorbar(mesh, label="dB")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Frequency (Hz)")
        ax.set_title(f"Spectrogram — {channel}")
        fig.tight_layout()

        path = ctx.workspace / f"plot_spectrogram_{channel}.png"
        fig.savefig(path, dpi=110)
        plt.close(fig)
        artifact = ctx.publish_artifact(path, kind="image")
        return {"plot": artifact}


def register(registry) -> None:
    registry.add(PlotTimeSeriesNode)
    registry.add(PlotSpectrogramNode)
