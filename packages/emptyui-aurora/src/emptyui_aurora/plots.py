"""Display nodes for the frequency-domain QC products (DESIGN §20.1).

Plot Coherency renders the per-band multiple coherence of an fc — the
exact statistic Coherence Sort applies (shared `_band_coherence` helper) —
so the figure is a one-glance QC view of what the sort would do.
"""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np
from emptyui import Input, Node, Output, Param

from ._processing import (
    _band_coherence,
    _bands_from_freq_pairs,
    _l2u_band_table,
    _normalize_fc_layout,
    _resolve_band_style,
    _resolve_period_bands,
)
from .process import _fc_io_channels


class PlotCoherency(Node):
    type = "aurora.plot.coherency"
    title = "Plot Coherency"
    tooltip = "Per-band multiple coherence of an fc vs period (the Coherence Sort statistic)."
    description = (
        "Per-band multiple coherence vs period — the exact statistic "
        "Coherence Sort applies (per-window multiple coherence over each "
        "band's frequency-bin ensemble, DESIGN.md §20.1): the E–H signal "
        "coherence (remote H when a 'ref fc' is connected) and, two-FC, the "
        "local↔remote magnetic transfer coherence. Shaded guidance band at "
        "[coherence_min, coherence_max] (the sort's default bounds). This "
        "is a broadband QC heuristic, not EMTF's per-frequency window-"
        "ensemble coherence (DESIGN.md §16.2/§18.2)."
    )
    category = "Display"
    inputs = (
        Input("fc", "fc"),
        Input("ref_fc", "fc", label="ref fc (remote)", required=False),
    )
    outputs = (Output("plot", "plot"),)
    params = (
        Param("coherence_min", "number", label="Coherence min",
              default=0.25, min=0.0, max=1.0,
              description="Lower reference bound (shaded guidance band) — "
                          "the Coherence Sort default."),
        Param("coherence_max", "number", label="Coherence max",
              default=0.98, min=0.0, max=1.0,
              description="Upper reference bound — the Coherence Sort default."),
        Param("periods_per_octave", "integer", label="Periods per octave",
              default=3, min=1,
              description="Band geometry — inherited from an upstream "
                          "Coherence Sort when connected."),
        Param("band_style", "select", label="Band style",
              options=("l2u", "uniform"), default="l2u",
              description="Band construction scheme (DESIGN.md §14.9) — "
                          "inherited from an upstream Coherence Sort when "
                          "connected."),
        Param("min_bins_per_band", "number", label="Min bins per band",
              default=20.0, min=0,
              description="Window-starvation coarsening threshold (DESIGN.md "
                          "§14.9) — inherited from an upstream Coherence Sort."),
        Param("t_short", "number", label="Shortest period (s)",
              default=0.0, min=0,
              description="Head clip (DESIGN.md §14.9) — inherited from an "
                          "upstream Coherence Sort. 0 starts at the "
                          "template's shortest band."),
    )
    #: same inherit-from-sort declaration as Estimate TF (DESIGN §18.2): the
    #: plot must band exactly like the sort/estimator it QC's
    inherit: ClassVar[dict[str, dict[str, str]]] = {
        "fc": {"band_style": "band_style",
               "periods_per_octave": "periods_per_octave",
               "min_bins_per_band": "min_bins_per_band",
               "t_short": "t_short"},
    }
    deps: ClassVar[tuple[str, ...]] = ("matplotlib",)

    def execute(self, ctx, fc, ref_fc=None,
                coherence_min: float = 0.25, coherence_max: float = 0.98,
                periods_per_octave: int = 3,
                band_style: str = "l2u",
                min_bins_per_band: float = 20.0,
                t_short: float = 0.0) -> dict[str, Any]:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from emptyui_mth5.plotting import apply_style

        levels = fc if isinstance(fc, dict) else {0: fc}
        ref_levels = ref_fc if isinstance(ref_fc, dict) else (
            {0: ref_fc} if ref_fc is not None else None)
        sample = next(iter(levels.values()))
        ins, outs, _ = _fc_io_channels(sample)

        # banding identical to Coherence Sort / Estimate TF (shared helpers)
        style = _resolve_band_style(band_style)
        bands_override = None
        if style == "l2u":
            bands_override = _l2u_band_table(
                levels, min_bins_per_band=min_bins_per_band,
                t_short=t_short or None)

        rows: list[dict[str, Any]] = []
        for level, lfc in sorted(levels.items()):
            ref = None
            if ref_levels is not None:
                ref = (ref_levels.get(level) if isinstance(ref_levels, dict)
                       else ref_levels)
                if ref is None:
                    raise ValueError(
                        f"remote-reference fc has no decimation level {level} "
                        f"(has {sorted(ref_levels)}) — compute both stations' "
                        "FCs with the same decimation/window settings"
                    )
                ref = _normalize_fc_layout(ref)
            lfc = _normalize_fc_layout(lfc)
            if bands_override is not None:
                periods, band_bins_list = _bands_from_freq_pairs(
                    lfc, bands_override.get(level, []))
            else:
                periods, band_bins_list = _resolve_period_bands(
                    lfc, periods_per_octave, style, t_short=t_short or None)
            for b in range(len(periods)):
                band_bins = band_bins_list[b]
                if band_bins.size == 0:
                    continue
                coh_e, coh_x = _band_coherence(lfc, ref, ins, outs, band_bins)
                # degenerate-band cap: identical to Coherence Sort's policy —
                # a band with fewer bins than inputs+1 fits the regression
                # exactly, so its coherence is trivially 1.0 and the "coh ≈ 1
                # is suspicious" rule must not flag it (§16.2)
                if band_bins.size < len(ins) + 1:
                    cap = coherence_max - 1e-9
                    coh_e = np.minimum(coh_e, cap)
                    coh_x = np.minimum(coh_x, cap)
                rows.append({"period": float(periods[b]),
                             "coherence_signal": float(np.median(coh_e)),
                             "coherence_transfer": (float(np.median(coh_x))
                                                    if ref is not None else None)})

        if not rows:
            raise ValueError("no usable frequency bands to plot")

        rows.sort(key=lambda r: r["period"])
        period = np.array([r["period"] for r in rows])
        sig = np.array([r["coherence_signal"] for r in rows])
        xfer = np.array([r["coherence_transfer"] for r in rows])

        fig, ax = plt.subplots(figsize=(7, 4), dpi=150)
        apply_style(fig, (ax,))
        ax.axhspan(coherence_min, coherence_max, color="0.85", zorder=0,
                   label=f"sort bounds [{coherence_min:.2f}, {coherence_max:.2f}]")
        ax.plot(period, sig, "o-", color="#1f77b4", markersize=4,
                label="signal coherence (E–H)"
                if ref_fc is None else "signal coherence (E–remote H)")
        if ref_fc is not None:
            ax.plot(period, xfer, "s--", color="#d62728", markersize=4,
                    label="local↔remote transfer coherence")
        ax.set_xscale("log")
        ax.set_xlabel("Period (s)")
        ax.set_ylabel("Coherence$^2$")
        ax.set_ylim(-0.02, 1.02)
        ax.set_title(station_title_from_fc(sample))
        ax.legend(frameon=True, fontsize=8, loc="lower left")
        fig.tight_layout()

        path = ctx.artifact_path("coherency", ".png")
        fig.savefig(path)
        plt.close(fig)
        ctx.progress(1.0, f"coherency: {len(rows)} bands")
        return {"plot": ctx.publish_artifact(path, kind="image")}


class PlotSpectrogramFC(Node):
    type = "aurora.plot.spectrogram"
    title = "Plot Spectrogram"
    tooltip = "Spectrogram (power vs time/frequency) of one fc channel — reuses the Compute FC coefficients."
    description = (
        "Spectrogram of one channel computed from the fc's Fourier "
        "coefficients (10·log10|FC|² — no second FFT). The decimation level "
        "to display is selected per level id (−1 = deepest available); the "
        "channel resolves from the fc (standard names map via the "
        "nomenclature; empty = first electric channel). The upstream "
        "Compute FC's window length and channel lists are inherited "
        "(window/decimation annotation; channels offered as suggestions)."
    )
    category = "Display"
    inputs = (Input("fc", "fc"),)
    outputs = (Output("plot", "plot"),)
    params = (
        Param("channel", "string", label="Channel", default="",
              placeholder="auto (first electric channel)",
              split_inherited=True,
              description="Channel to spectrogram; empty = first electric "
                          "channel (or first channel). Standard names "
                          "(ex/ey/hx/hy/hz) are mapped through the fc's "
                          "channel nomenclature. Suggestions are split "
                          "from the upstream Compute FC channel params."),
        Param("decimation_level", "integer", label="Decimation level",
              default=-1, min=-1,
              description="Which decimation level to display: −1 (default) "
                          "= the deepest level in the connected fc; the "
                          "upstream decimation count is inherited for "
                          "orientation."),
        Param("window_samples", "integer", label="Window (samples)",
              default=None, optional=True,
              description="Display only — inherited from Compute FC; the "
                          "authoritative value is the fc's own metadata."),
        Param("decimation_levels", "integer", label="Decimation levels (upstream)",
              default=None, optional=True,
              description="Display only — inherited from Compute FC."),
    )
    inherit: ClassVar[dict[str, dict[str, str]]] = {
        "fc": {"channel": "output_channels",
               "window_samples": "window_samples",
               "decimation_levels": "decimation_levels"},
    }
    deps: ClassVar[tuple[str, ...]] = ("matplotlib",)

    def execute(self, ctx, fc, channel: str = "",
                decimation_level: int = -1,
                window_samples: int | None = None,
                decimation_levels: int | None = None) -> dict[str, Any]:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.dates as mdates
        import matplotlib.pyplot as plt
        from emptyui_core_nodes._channels import (
            auto_channel,
            is_electric_name,
            resolve_channels,
        )
        from emptyui_mth5.plotting import apply_style

        levels = fc if isinstance(fc, dict) else {0: fc}
        if decimation_level < 0:
            level = max(levels)
        else:
            level = int(decimation_level)
            if level not in levels:
                raise ValueError(
                    f"fc has no decimation level {level} (available: "
                    f"{sorted(levels)}) — set 'Decimation level' to one of "
                    "them, or −1 for the deepest"
                )
        ds = _normalize_fc_layout(levels[level])
        names = [str(c) for c in ds.data_vars]
        if str(channel or "").strip():
            spec = str(channel).strip().lower()
            # exact name (case-insensitive) first, then standard-name mapping
            low = {str(c).lower(): str(c) for c in names}
            channel = low.get(spec)
            if channel is None:
                mapped = resolve_channels(names, spec)
                channel = mapped[0] if mapped else None
            if channel is None:
                raise ValueError(
                    f"channel {spec!r} not in fc (channels: {names})")
        else:
            electric = [str(c) for c in ds.data_vars if is_electric_name(str(c))]
            base = [str(c) for c in ds.data_vars]
            channel = auto_channel(electric or base) or base[0]

        freq = ds["frequency"].values
        power = 10.0 * np.log10(np.abs(ds[channel].values) ** 2 + 1e-24)
        time = ds["time"].values
        time_is_dt = time.dtype.kind == "M"  # datetime64 (persisted fc)

        fig, ax = plt.subplots(figsize=(10, 4))
        mesh = ax.pcolormesh(time, freq, power, shading="auto", cmap="viridis")
        cbar = fig.colorbar(mesh, ax=ax)
        cbar.set_label("Power (dB)")
        apply_style(fig, (ax,))
        ax.set_yscale("log")
        ax.set_xlabel("Time" if time_is_dt else "Time (s)")
        if time_is_dt:
            ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(
                mdates.AutoDateLocator()))
        ax.set_ylabel("Frequency (Hz)")
        attrs_note = (
            f"level {level}, window {ds.attrs.get('num_samples_window', '?')} samples"
        )
        station = ds.attrs.get("station_id") or ""
        ax.set_title(
            (station + " — " if station else "")
            + f"Spectrogram — {channel} ({attrs_note})",
            fontsize=10,
        )
        fig.tight_layout()

        path = ctx.artifact_path(f"plot_spectrogram_{channel}", ".png")
        fig.savefig(path, dpi=150, bbox_inches="tight",
                    metadata={"Software": "emptyUI"})
        plt.close(fig)
        artifact = ctx.publish_artifact(path, kind="image")
        return {"plot": artifact}


def station_title_from_fc(fc) -> str:
    """Station title from an fc's identity attrs (§15 convention)."""
    station = fc.attrs.get("station_id") or ""
    survey = fc.attrs.get("survey_id") or ""
    parts = [p for p in (station, survey) if p]
    return " — ".join(parts) if parts else "Coherence"
