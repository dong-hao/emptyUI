"""Display nodes: render figures to `plot` outputs (DESIGN.md §3).

Display nodes emit a typed `plot` port (rendered-figure reference) consumed
by `Preview Plot` (inline viewer) and `Save Plot` (export) — ComfyUI's
Preview/Save Image pattern. Plot conventions follow DESIGN.md §10.6 (shared
style module ``_plotstyle.py``): 150 dpi, grid/tick conventions, absolute
datetime x-axis when the run carries a start time, channel unit hints, and
run/station identity in titles.
"""

from __future__ import annotations

from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
from emptyui import Input, Node, Output, Param

from ._channels import auto_channel, resolve_channels
from ._plotstyle import DPI, apply_style, channel_unit_hint
from ._timewindow import resolve_window_s
from .types import TimeSeries, coerce_run, is_run_collection


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


def _time_window(data: TimeSeries, start: str, end: str,
                 seconds) -> tuple[int, int]:
    """Resolve the time-selector params to sample indices [n0, n1).

    Guards live in the shared :mod:`._timewindow` (§23.4): bounds clamped
    into the record (warn, never reject), explicitly bounded slot > 1 s.
    The legacy ``seconds`` duration caps the slot from its start and keeps
    its old no-minimum behavior (a 1 s waveform view of a 500 Hz run is
    legitimate).
    """
    rec_len = data.n_samples / data.sample_rate
    s0, s1 = resolve_window_s(
        start, end, rec_len=rec_len, start_datetime=data.start_datetime,
        start_offset=data.start, enforce_min_slot=False)
    bounded = bool(str(start).strip() or str(end).strip())
    if seconds is not None:
        s1 = min(s1, s0 + float(seconds))
    if bounded and s1 - s0 <= 1.0:
        raise ValueError(
            f"selected time slot is {s1 - s0:.3g} s — the minimum slot is "
            f"> 1 s (start={start or 'record start'}, "
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
    description = (
        "Stacked channel time-domain plot. Accepts a single time series or "
        "a run collection: a `sample_rate` param selects the family to view "
        "(auto = the lowest rate present — the continuous stream, e.g. "
        "15 Hz Phoenix / 1 Hz LEMI/NIMS), and all runs of that family plot "
        "as segments on the shared time axis (gaps stay visible).")
    category = "Display"
    inputs = (Input("data", "ts.run", label="time series", collection_ok=True),)
    outputs = (Output("plot", "plot"),)
    params = (
        Param("channels", "string", label="Channels (comma-sep, empty=all)",
              default="", optional=True,
              description="Channel names to plot; empty = all channels. "
                          "Standard names (ex/ey/hx/hy/hz) are mapped through "
                          "the inferred channel nomenclature; matching is "
                          "case-insensitive."),
        Param("sample_rate", "string", label="Sample rate", default="auto",
              optional=True, suggest_from="data",
              placeholder="auto (lowest rate present)",
              description="On a run collection: which rate family to view "
                          "(auto = the lowest rate present — the continuous "
                          "stream, e.g. 15 Hz Phoenix / 1 Hz LEMI/NIMS; all "
                          "runs of that family plot as segments). A "
                          "single-run payload ignores this param."),
        Param("seconds", "number", label="Duration to plot (s)", default=None,
              optional=True),
        Param("start", "string", label="Start (UTC)", default="",
              optional=True, time_bar="data",
              placeholder="record start",
              description="Slot start: ISO datetime (UTC, e.g. "
                          "2023-06-01 12:00:00) or relative seconds; empty "
                          "= record start. Driven by the time bar. On a run "
                          "collection the window crops each run (clamped; "
                          "runs entirely outside are skipped)."),
        Param("end", "string", label="End (UTC)", default="",
              optional=True, time_bar="data",
              placeholder="record end",
              description="Slot end: ISO datetime (UTC) or relative "
                          "seconds; empty = record end. Driven by the "
                          "time bar. On a run collection the window crops "
                          "each run (clamped; runs entirely outside are "
                          "skipped)."),
    )

    @staticmethod
    def _family_members(collection, sample_rate: str):
        """Select one rate family's members from a run collection (Tier 1,
        backlog #21): returns ([(payload, extent_row)], chosen_rate, rates)."""
        extents = list(collection.extents)
        runs = list(collection.runs)
        by_rate: dict[float, list[int]] = {}
        for i, e in enumerate(extents):
            r = round(float(e.get("sample_rate", 0.0) or 0.0), 9)
            by_rate.setdefault(r, []).append(i)
        rates = sorted(by_rate)
        if not rates or rates[0] <= 0:
            raise ValueError(
                "the collection's extents carry no sample rates — cannot "
                "select a family to view")
        spec = str(sample_rate or "").strip().lower()
        if spec in ("", "auto"):
            hit = rates[0]
        else:
            try:
                want = round(float(spec), 9)
            except ValueError:
                raise ValueError(
                    f"sample_rate {sample_rate!r} is not a number — use a "
                    f"rate in Hz (available: "
                    f"{', '.join(f'{r:g}' for r in rates)}) or 'auto' for "
                    "the lowest rate present") from None
            hit = next((r for r in rates if r == want), None)
            if hit is None:
                closest = min(rates, key=lambda r: abs(r - want) / max(r, 1e-12))
                if abs(closest - want) / max(closest, 1e-12) < 1e-6:
                    hit = closest
            if hit is None:
                raise ValueError(
                    f"the collection holds no runs at {want:g} Hz "
                    f"(available rates: {', '.join(f'{r:g}' for r in rates)})"
                    " — set 'Sample rate' to one of them, or 'auto'")
        members = [(runs[i], extents[i])
                   for i in sorted(by_rate[hit],
                                   key=lambda j: str(extents[j].get("start", "")))]
        return members, hit, rates

    def execute(self, ctx, data, channels: str = "", sample_rate: str = "auto",
                seconds=None, start: str = "", end: str = "") -> dict[str, Any]:
        if is_run_collection(data):
            return self._execute_collection(
                ctx, data, channels, sample_rate, seconds, start, end)
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

    def _execute_collection(self, ctx, data, channels: str,
                            sample_rate: str, seconds, start: str,
                            end: str) -> dict[str, Any]:
        """Tier 1 (backlog #21): one rate family per figure — all runs of
        that family plot as segments on the shared time axis (NOT one plot
        per member: thousands of runs would overwhelm). The time bar
        crops each run (clamped; runs entirely outside are skipped)."""
        members, hit, rates = self._family_members(data, sample_rate)
        coerced = [coerce_run(payload) for payload, _ in members]
        # channel spec resolves against the first member (a family's runs
        # share components); per-member missing channels are skipped with
        # one aggregated warning
        names = _select_channels(coerced[0], channels)
        bounded = bool(str(start).strip() or str(end).strip())
        segments: list[tuple] = []  # (member_data, x, {name: arr})
        skipped_outside = 0
        for mdata in coerced:
            if bounded:
                try:
                    n0, n1 = _time_window(mdata, start, end, seconds)
                except ValueError:
                    skipped_outside += 1
                    continue
                if n1 <= n0:
                    skipped_outside += 1
                    continue
            else:
                n0, n1 = _time_window(mdata, start, end, seconds)
            x, _ = _time_axis(mdata, n0, n1)
            segments.append((mdata, x, {n: mdata.channels[n][n0:n1]
                                        for n in names
                                        if n in mdata.channels}))
        if skipped_outside:
            ctx.logger.warning(
                "%d run(s) of the %g Hz family fall outside the selected "
                "window — skipped", skipped_outside, hit)
        if not segments:
            raise ValueError(
                f"no {hit:g} Hz run falls within the selected window "
                f"(the family holds {len(coerced)} run(s)) — clear start/end "
                "or pick another 'Sample rate'")
        missing = [n for n in names
                   if all(n not in seg[2] for seg in segments)]
        if missing:
            ctx.logger.warning(
                "channels %s are missing in every %g Hz run — dropped",
                missing, hit)
        names = [n for n in names if n not in missing]
        if not names:
            raise ValueError(
                f"no plottable channel in the {hit:g} Hz runs "
                f"(spec {channels!r} matched none)")

        fig, axes = plt.subplots(len(names), 1,
                                 figsize=(10, 1.8 * len(names)),
                                 sharex=True, squeeze=False)
        for ax, name in zip(axes[:, 0], names):
            for mdata, x, chans in segments:
                ax.plot(x, chans[name], lw=0.6)
            unit = channel_unit_hint(name)
            ax.set_ylabel(f"{name} ({unit})" if unit else name)
        apply_style(fig, axes[:, 0])
        axes[-1, 0].set_xlabel("Time (UTC)")
        if segments[0][0].start_datetime:
            locator = mdates.AutoDateLocator()
            axes[-1, 0].xaxis.set_major_locator(locator)
            axes[-1, 0].xaxis.set_major_formatter(
                mdates.ConciseDateFormatter(locator))
            fig.autofmt_xdate()
        station = segments[0][0].station or ""
        title = " — ".join(
            p for p in ("Time series", station,
                        f"{len(segments)} run(s) @ {hit:g} Hz") if p)
        fig.suptitle(title, fontsize=11, fontweight="bold")
        fig.tight_layout(rect=(0, 0, 1, 0.96))

        path = ctx.artifact_path("plot_timeseries", ".png")
        fig.savefig(path, dpi=DPI, bbox_inches="tight",
                    metadata={"Software": "emptyUI"})
        plt.close(fig)
        artifact = ctx.publish_artifact(path, kind="image")
        ctx.progress(1.0, f"time series: {len(segments)} run(s) @ {hit:g} Hz")
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
        if is_run_collection(data):
            raise ValueError(
                "received a run collection — per-family viewing (sample-rate "
                "selection) is planned (M4 backlog): extract a single run to "
                "plot")
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


