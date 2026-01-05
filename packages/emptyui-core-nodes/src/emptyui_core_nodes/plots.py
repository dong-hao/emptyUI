"""Display nodes: render figures to `plot` outputs (DESIGN.md §3).

Display nodes emit a typed `plot` port (rendered-figure reference) consumed
by `Preview Plot` (inline viewer) and `Save Plot` (export) — ComfyUI's
Preview/Save Image pattern. Plot conventions follow DESIGN.md §15 (shared
style module ``_plotstyle.py``): 150 dpi, grid/tick conventions, absolute
datetime x-axis when the run carries a start time, channel unit hints, and
run/station identity in titles.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
from emptyui import Input, Node, Output, Param

from ._channels import auto_channel, resolve_channels
from ._plotstyle import DPI, apply_style, channel_unit_hint
from .types import TimeSeries, coerce_run, is_unknown_start

_log = logging.getLogger("emptyui")

#: minimum selectable time slot (DESIGN §23.4): a narrower window carries
#: no useful MT signal and usually indicates a slider/box slip
MIN_SLOT_S = 1.0


def _select_channels(data: TimeSeries, spec: str) -> list[str]:
    """Resolve a channel spec against the data (empty = all). Standard
    component names (ex/ey/hx/hy/hz) are mapped through the inferred
    nomenclature; matching is case-insensitive."""
    return resolve_channels(list(data.channels), spec)


def _time_axis(data: TimeSeries, n0: int, n1: int):
    """Return (x, label) for samples [n0, n1) — datetime axis when an
    absolute start is known, relative seconds otherwise."""
    sr = data.sample_rate
    if data.start_datetime:
        import pandas as pd

        t0 = pd.Timestamp(data.start_datetime) + pd.to_timedelta(
            data.start + n0 / sr, unit="s")
        x = t0 + pd.to_timedelta(np.arange(n1 - n0) / sr, unit="s")
        return x, "Time (UTC)"
    return data.start + n0 / sr + np.arange(n1 - n0) / sr, "Time (s)"


def _parse_bound(value: str, anchor: datetime | None, rec_len: float,
                 what: str, start_offset: float = 0.0) -> float:
    """Parse one time-selector bound into seconds-from-series-start.

    Accepts an ISO datetime (UTC; needs an absolute record start — measured
    from the record start, so ``start_offset`` is subtracted) or a bare
    number (relative seconds from the series start). Clamps into
    [0, rec_len] with a warning — out-of-range selection degrades, never
    rejects (DESIGN §23.4).
    """
    text = str(value).strip()
    try:
        rel = float(text)
    except ValueError:
        pass
    else:
        clamped = min(max(rel, 0.0), rec_len)
        if clamped != rel:
            _log.warning("%s=%.4g s falls outside the record (length %.3g s)"
                         " — clamped", what, rel, rec_len)
        return clamped
    if anchor is None:
        raise ValueError(
            f"{what}='{value}' is not a number and the record has no "
            "absolute start time — use relative seconds")
    try:
        dt = datetime.fromisoformat(text.replace(" ", "T")[:19])
    except ValueError as exc:
        raise ValueError(
            f"{what}='{value}' is not an ISO datetime or seconds number"
        ) from exc
    rel = (dt - anchor).total_seconds() - start_offset
    clamped = min(max(rel, 0.0), rec_len)
    if clamped != rel:
        _log.warning(
            "%s=%s falls outside the record (%s .. +%.0f s) — clamped",
            what, value, anchor.isoformat(sep=" ", timespec="seconds"),
            rec_len)
    return clamped


def _time_window(data: TimeSeries, start: str, end: str,
                 seconds) -> tuple[int, int]:
    """Resolve the time-selector params to sample indices [n0, n1).

    Guards (DESIGN §23.4): selector bounds are clamped into the record
    (warn, never reject), cannot cross, and an explicitly bounded slot must
    be > 1 s — that minimum applies to the time-selector params only, not
    to the legacy ``seconds`` duration (a 1 s waveform view of a 500 Hz run
    is legitimate). ``seconds`` caps the slot length from its start.
    """
    rec_len = data.n_samples / data.sample_rate
    anchor = None
    if data.start_datetime and not is_unknown_start(data.start_datetime):
        try:
            anchor = datetime.fromisoformat(data.start_datetime[:19])
        except ValueError:
            anchor = None
    elif data.start_datetime:
        _log.warning(
            "record starts at a placeholder epoch — time selector bounds "
            "are relative seconds, ISO values ignored")
    bounded = bool(str(start).strip() or str(end).strip())
    s0 = (_parse_bound(start, anchor, rec_len, "start", data.start)
          if str(start).strip() else 0.0)
    s1 = (_parse_bound(end, anchor, rec_len, "end", data.start)
          if str(end).strip() else rec_len)
    if seconds is not None:
        s1 = min(s1, s0 + float(seconds))
    if bounded and s1 - s0 <= MIN_SLOT_S:
        raise ValueError(
            f"selected time slot is {s1 - s0:.3g} s — the minimum slot is "
            f"> {MIN_SLOT_S:g} s (start={start or 'record start'}, "
            f"end={end or 'record end'}, record length {rec_len:.3g} s)")
    sr = data.sample_rate
    n0 = round(s0 * sr)
    n1 = round(s1 * sr)
    return max(0, min(n0, data.n_samples)), max(n0 + 1, min(n1, data.n_samples))


def _plot_title(data: TimeSeries, what: str) -> str:
    parts = [what]
    if data.station:
        parts.append(data.station)
    if data.run:
        parts.append(f"({data.run})")
    return " — ".join(parts)


class PlotTimeSeriesNode(Node):
    type = "core.plot.timeseries"
    title = "Plot Time Series"
    tooltip = "Plot the (stacked) channel time series as a PNG figure."
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
        Param("start", "string", label="Start (UTC)", default="",
              optional=True, time_bar="data",
              placeholder="record start",
              description="Slot start: ISO datetime (UTC, e.g. "
                          "2023-06-01 12:00:00) or relative seconds; empty "
                          "= record start. Driven by the time bar."),
        Param("end", "string", label="End (UTC)", default="",
              optional=True, time_bar="data",
              placeholder="record end",
              description="Slot end: ISO datetime (UTC) or relative "
                          "seconds; empty = record end. Driven by the "
                          "time bar."),
    )

    def execute(self, ctx, data, channels: str = "", seconds=None,
                start: str = "", end: str = "") -> dict[str, Any]:
        data = coerce_run(data)
        names = _select_channels(data, channels)
        n0, n1 = _time_window(data, start, end, seconds)
        x, xlabel = _time_axis(data, n0, n1)

        fig, axes = plt.subplots(len(names), 1, figsize=(10, 1.8 * len(names)),
                                 sharex=True, squeeze=False)
        for ax, name in zip(axes[:, 0], names):
            ax.plot(x, data.channels[name][n0:n1], lw=0.6)
            unit = channel_unit_hint(name)
            ax.set_ylabel(f"{name} ({unit})" if unit else name)
        apply_style(fig, axes[:, 0])
        axes[-1, 0].set_xlabel(xlabel)
        if data.start_datetime:
            locator = mdates.AutoDateLocator()
            axes[-1, 0].xaxis.set_major_locator(locator)
            axes[-1, 0].xaxis.set_major_formatter(
                mdates.ConciseDateFormatter(locator))
            fig.autofmt_xdate()
        fig.suptitle(_plot_title(data, "Time series"), fontsize=11,
                     fontweight="bold")
        fig.tight_layout(rect=(0, 0, 1, 0.96))

        path = ctx.artifact_path("plot_timeseries", ".png")
        fig.savefig(path, dpi=DPI, bbox_inches="tight",
                    metadata={"Software": "emptyUI"})
        plt.close(fig)
        artifact = ctx.publish_artifact(path, kind="image")
        return {"plot": artifact}


class PlotSpectrogramNode(Node):
    type = "core.plot.spectrogram"
    title = "Plot Spectrogram (TS)"
    tooltip = "Plot an STFT spectrogram of one time-series channel (kept for old workflows; the fc-based Plot Spectrogram is preferred)."
    description = (
        "STFT spectrogram of one time-series channel (in-line FFT). "
        "Superseded by aurora.plot.spectrogram, which reuses the Compute FC "
        "coefficients — hidden from the palette; saved workflows using it "
        "keep working."
    )
    hidden = True
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
        cbar = fig.colorbar(mesh, ax=ax)
        unit = channel_unit_hint(channel)
        cbar.set_label(f"Power (dB{', rel. ' + unit if unit else ''})")
        apply_style(fig, (ax,))
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Frequency (Hz)")
        ax.set_title(_plot_title(data, f"Spectrogram — {channel}"),
                     fontsize=10)
        fig.tight_layout()

        path = ctx.artifact_path(f"plot_spectrogram_{channel}", ".png")
        fig.savefig(path, dpi=DPI, bbox_inches="tight",
                    metadata={"Software": "emptyUI"})
        plt.close(fig)
        artifact = ctx.publish_artifact(path, kind="image")
        return {"plot": artifact}


def register(registry) -> None:
    registry.add(PlotTimeSeriesNode)
    registry.add(PlotSpectrogramNode)
