"""Display nodes for the frequency-domain QC products (docs/DESIGN.md §10.6).

Plot Coherency renders the per-frequency coherence of an fc — the exact
statistic Coherence Sort applies (shared `_per_bin_coherence` helper) —
so the figure is a one-glance QC view of what the sort would do.
"""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np
from emptyui import Input, Node, Output, Param

from ._processing import (
    _normalize_fc_layout,
    _pair_remote_fc,
    _per_bin_coherence,
)
from .process import _fc_io_channels


def _resolve_rate_tiers(levels: dict, sample_rate: str):
    """Resolve which decimation tier(s) to plot, burst-mode aware (history §20.1).

    A merged burst FC database is rate-keyed (`merge_fc_containers` re-keys
    to descending actual sample rates — the cascade indices of mixed-rate
    families do not correspond). Returns ``(family_rate, tier_ids)`` — the
    tier ids of ONE rate family, native rate first (descending tier rates):

    - ``sample_rate="auto"`` picks the lowest NATIVE rate family present
      (the continuous stream — the MT-transfer workhorse: 15 Hz Phoenix,
      1 Hz LEMI). The native families travel on the database's attrs
      (stamped at compute time); an fc WITHOUT sample_rate attrs RAISES
      (the docstring's "no-op for legacy fcs" path was unreachable —
      the code requires the metadata to build the rate map at all,
      L-A7).
    - an explicit value selects the tier(s) at that rate (the family
      cascade when the rate is a native family, the single tier when it
      is a derived one).

    Tier ids are sorted by descending tier rate, so ``decimation_level``
    indexes the CASCADE STEP within the family: 0 = the family's native
    tier (no decimation / original data), 1 = one decimation deeper, ...
    -1 = the deepest available step.
    """
    by_rate: dict[float, list] = {}
    for lv, ds in levels.items():
        r = round(float(getattr(ds, "attrs", {}).get("sample_rate", 0.0) or 0.0), 9)
        by_rate.setdefault(r, []).append(lv)
    if not by_rate or all(r <= 0 for r in by_rate):
        raise ValueError("fc carries no sample_rate metadata — cannot "
                         "select a rate tier")
    rates = sorted(by_rate, reverse=True)
    spec = str(sample_rate or "").strip().lower()
    if spec not in ("", "auto"):
        try:
            want = round(float(spec), 9)
        except ValueError:
            raise ValueError(
                f"sample_rate {sample_rate!r} is not a number — use a rate in "
                f"Hz (available: {', '.join(f'{r:g}' for r in rates)}) or "
                "'auto' for the lowest native rate present") from None
        hit = next((r for r in rates if r == want), None)
        if hit is None:
            closest = min(rates, key=lambda r: abs(r - want) / max(r, 1e-12))
            if abs(closest - want) / max(closest, 1e-12) < 1e-6:
                hit = closest
        if hit is None:
            raise ValueError(
                f"fc has no tier at {want:g} Hz (available rates: "
                f"{', '.join(f'{r:g}' for r in rates)}) — set 'Sample rate' to "
                "one of them, or 'auto' for the lowest native rate present")
        return hit, by_rate[hit]
    families = None
    for ds in levels.values():
        fam = getattr(ds, "attrs", {}).get("family_rates")
        if fam:
            try:
                fam = sorted((float(x) for x in fam), reverse=True)
            except (TypeError, ValueError):
                fam = None
            if fam:
                families = fam
                break
    if families is None:
        # legacy fc (per-level sample_rate attrs, no family_rates stamp —
        # e.g. a run-collection cascade of separately-computed ComputeFC
        # outputs): the
        # TESTED legacy contract is 'auto' = every level (indices =
        # cascade depth), titled with the native rate first
        ids = [lv for r in rates for lv in by_rate[r]]
        return rates[0], ids
    want_rate = min(families)
    lower_bound = next((f for f in families if f < want_rate), 0.0)
    ids = [lv for r in rates if lower_bound < r <= want_rate + 1e-12
           for lv in by_rate[r]]
    if not ids:
        hit = min(by_rate, key=lambda r: abs(r - want_rate))
        return hit, by_rate[hit]
    return want_rate, ids


class PlotCoherency(Node):
    type = "aurora.plot.coherency"
    title = "Plot Coherency"
    tooltip = "Per-frequency window-ensemble coherence of an fc vs period (the Coherence Sort statistic)."
    description = (
        "Per-frequency coherence vs period — the exact statistic "
        "Coherence Sort applies (EMTF-style window-ensemble coherence at "
        "each frequency bin, median across the sliding window ensemble, "
        "docs/DESIGN.md §10.6 / history §24.6): the E–H signal coherence (remote H when a "
        "'ref fc' is connected) and, two-FC, the local↔remote magnetic "
        "transfer coherence. Shaded guidance band at [coherence_min, "
        "coherence_max] (the sort's default bounds). Each decimation level "
        "plots as its own series."
    )
    category = "Display"
    terminal = True  # M13: manifest-declared run visibility
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
              default=1, min=0.0, max=1.0,
              description="Upper reference bound — the Coherence Sort default."),
        Param("ensemble_windows", "integer", label="Ensemble windows",
              default=16, min=1,
              description="Sliding window-ensemble half-width of the "
                          "per-bin statistic — match the upstream "
                          "Coherence Sort setting."),
        Param("smooth_bins", "integer", label="Frequency smoothing (bins)",
              default=0, min=0,
              description="Adjacent-bin pooling of the per-bin statistic — "
                          "match the upstream Coherence Sort setting."),
        Param("variation", "select", label="Variation band",
              default="iqr", options=("none", "iqr", "std"),
              description="The per-window spread behind the plotted median, "
                          "rendered as a shaded band: 'iqr' (default) = the "
                          "25–75 percentile of the sliding-ensemble "
                          "coherences per bin; 'std' = median ± 1 standard "
                          "deviation; 'none' = the bare median (the old "
                          "behavior)."),
        Param("sample_rate", "string", label="Sample rate", default="auto",
              optional=True, suggest_from="fc",
              placeholder="auto (lowest rate present)",
              description="Which rate family to plot on a merged burst FC "
                          "database (auto = the lowest rate present — the "
                          "continuous stream, e.g. 15 Hz Phoenix / 1 Hz "
                          "LEMI). A single-rate fc keeps every cascade "
                          "level; an fc without family-rate metadata (e.g. "
                          "merged separately-computed FCs) falls back to "
                          "plotting every level, titled with the native "
                          "rate. The connected fc's tiers are offered as "
                          "suggestions after a run."),
    )
    deps: ClassVar[tuple[str, ...]] = ("matplotlib",)

    def execute(self, ctx, fc, ref_fc=None,
                coherence_min: float = 0.25, coherence_max: float = 1.0,
                ensemble_windows: int = 16,
                smooth_bins: int = 0, variation: str = "iqr",
                sample_rate: str = "auto") -> dict[str, Any]:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from emptyui_mth5.plotting import apply_style

        levels = fc if isinstance(fc, dict) else {0: fc}
        all_levels = dict(levels)
        rate, tiers = _resolve_rate_tiers(levels, sample_rate)
        level_ids = set(tiers)
        levels = {lv: ds for lv, ds in levels.items() if lv in level_ids}
        ref_levels = ref_fc if isinstance(ref_fc, dict) else (
            {0: ref_fc} if ref_fc is not None else None)
        if ref_levels is not None:
            # match the remote's tier(s) by RATE — merged burst databases
            # key tiers by rate position, which differs between stations
            # with different rate families; legacy cascades match at their
            # shared cascade rates
            ref_by_rate: dict[float, list] = {}
            for lv, ds in ref_levels.items():
                r = round(float(getattr(ds, "attrs", {}).get(
                    "sample_rate", 0.0) or 0.0), 9)
                ref_by_rate.setdefault(r, []).append(lv)
            kept_rates = {round(float(getattr(ds, "attrs", {}).get(
                "sample_rate", 0.0) or 0.0), 9) for ds in levels.values()}
            missing = sorted(r for r in kept_rates if r not in ref_by_rate)
            if missing:
                raise ValueError(
                    f"remote-reference fc has no tier at "
                    f"{', '.join(f'{r:g}' for r in missing)} Hz "
                    f"(has {sorted(ref_levels)}) — compute both stations' "
                    "FCs with the same decimation/window settings")
            paired: dict = {}
            for r in kept_rates:
                for l_id, r_id in zip(
                        sorted(lv for lv, ds in all_levels.items()
                               if round(float(getattr(ds, "attrs", {}).get(
                                   "sample_rate", 0.0) or 0.0), 9) == r),
                        sorted(ref_by_rate[r])):
                    paired[l_id] = ref_levels[r_id]
            ref_levels = paired
        sample = next(iter(levels.values()))
        ins, outs, _ = _fc_io_channels(sample)

        rows: list[dict[str, Any]] = []

        def _band(vals, med):
            """The per-bin variation band behind the median (variation
            param): (lo, hi) or (None, None) when degenerate."""
            vals = vals[np.isfinite(vals)]
            if vals.size == 0:
                return None, None
            if variation == "std":
                sd = float(np.nanstd(vals))
                return med - sd, med + sd
            return (float(np.nanpercentile(vals, 25)),
                    float(np.nanpercentile(vals, 75)))

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
                lfc = _normalize_fc_layout(lfc)
                # the two-FC statistic applies to PAIRED windows — validate
                # + pair exactly like Coherence Sort does (shared helper):
                # unpaired grids either crash with a raw broadcast error or
                # silently compute the transfer coherence over mismatched
                # windows
                lfc, ref = _pair_remote_fc(
                    lfc, _normalize_fc_layout(ref),
                    pairing="intersect", tol_s=0.0, level=level)
            # the exact statistic the sort applies (shared helper, history §20.1)
            coh_e, coh_x = _per_bin_coherence(
                lfc, ref, ins, outs, ensemble_windows=ensemble_windows,
                smooth_bins=smooth_bins)
            freq = np.asarray(lfc["frequency"].values, dtype=float)
            for i, fv in enumerate(freq):
                if not np.isfinite(fv) or fv <= 0.0:
                    continue
                med_e = (float(np.nanmedian(coh_e[i, :]))
                         if np.isfinite(coh_e[i, :]).any() else None)
                med_x = (float(np.nanmedian(coh_x[i, :]))
                         if ref is not None and np.isfinite(coh_x[i, :]).any()
                         else None)
                lo_e, hi_e = _band(coh_e[i, :], med_e) if med_e is not None else (None, None)
                lo_x, hi_x = _band(coh_x[i, :], med_x) if med_x is not None else (None, None)
                rows.append({
                    "period": float(1.0 / fv),
                    "decimation_level": int(level),
                    "coherence_signal": med_e,
                    "coherence_transfer": med_x,
                    "signal_band": [lo_e, hi_e],
                    "transfer_band": [lo_x, hi_x],
                })

        if not rows:
            raise ValueError("no usable frequency bins to plot")

        # one series per decimation level (DESIGN §10.6): the cascades
        # overlap in period, so a single flattened series zigzags between
        # levels and interleaves their markers
        levels_order = sorted({r["decimation_level"] for r in rows})
        cmap = plt.get_cmap("tab10")
        series_color = {lv: cmap(i % 10) for i, lv in enumerate(levels_order)}
        signal_label = ("signal coherence (E–H)"
                        if ref_fc is None else "signal coherence (E–remote H)")

        fig, ax = plt.subplots(figsize=(7, 4), dpi=150)
        apply_style(fig, (ax,))
        ax.axhspan(coherence_min, coherence_max, color="0.85", zorder=0,
                   label=f"sort bounds [{coherence_min:.2f}, {coherence_max:.2f}]")
        for lv in levels_order:
            rows_lv = sorted((r for r in rows if r["decimation_level"] == lv),
                             key=lambda r: r["period"])
            period = np.array([r["period"] for r in rows_lv])
            sig = np.array([r["coherence_signal"] for r in rows_lv],
                           dtype=float)
            xfer = np.array([r["coherence_transfer"] for r in rows_lv],
                            dtype=float)

            def _arr(key, rows_lv=rows_lv):
                lo = np.array([r[key][0] if r[key][0] is not None else np.nan
                               for r in rows_lv], dtype=float)
                hi = np.array([r[key][1] if r[key][1] is not None else np.nan
                               for r in rows_lv], dtype=float)
                return lo, hi

            color = series_color[lv]
            if variation != "none":
                # per-bin spread behind the median, shaded (the whiskers of
                # hundreds of dense points would be unreadable)
                for y, key in ((sig, "signal_band"),
                               (xfer, "transfer_band")):
                    if not np.isfinite(y).any():
                        continue
                    lo, hi = _arr(key)
                    if not np.isfinite(lo).any() and not np.isfinite(hi).any():
                        continue
                    ax.fill_between(period, lo, hi, color=color, alpha=0.18,
                                    lw=0)
            ax.plot(period, sig, "o-", color=color, markersize=3,
                    label=f"level {lv} — {signal_label}")
            if ref_fc is not None:
                ax.plot(period, xfer, "s--", color=color, markersize=3,
                        label=f"level {lv} — local↔remote transfer coherence")
        ax.set_xscale("log")
        ax.set_xlabel("Period (s)")
        ax.set_ylabel("Coherence$^2$")
        ax.set_ylim(-0.02, 1.02)
        all_rates = {round(float(getattr(ds, "attrs", {}).get("sample_rate", 0) or 0), 9)
                     for ds in (fc.values() if isinstance(fc, dict) else [fc])}
        title = station_title_from_fc(sample)
        if len(all_rates) > 1:
            title += f" — {rate:g} Hz"
        ax.set_title(title)
        ax.legend(frameon=True, fontsize=8, loc="upper right")
        fig.tight_layout()

        path = ctx.artifact_path("coherency", ".png")
        fig.savefig(path)
        plt.close(fig)
        ctx.progress(1.0, f"coherency: {len(rows)} frequency bins")
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
        "nomenclature; empty = first electric channel)."
    )
    category = "Display"
    terminal = True  # M13: manifest-declared run visibility
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
              default=0, min=-1,
              description="Cascade step within the selected rate family: "
                          "0 (default) = the family's native tier (no "
                          "decimation / original data), 1..k = one decimation "
                          "deeper each, −1 = the deepest available step. On "
                          "a merged burst FC database the 'Sample rate' param "
                          "selects the rate family first."),
        Param("sample_rate", "string", label="Sample rate", default="auto",
              optional=True, suggest_from="fc",
              placeholder="auto (lowest rate present)",
              description="Which rate family to display on a merged burst "
                          "FC database (auto = the lowest rate present — "
                          "the continuous stream, e.g. 15 Hz Phoenix / 1 Hz "
                          "LEMI). The connected fc's tiers are offered as "
                          "suggestions after a run."),
    )
    # L-A10: the former display-only window_samples/decimation_levels
    # params were deleted — declared + inherited but never read (the
    # figure's annotation reads the fc's own authoritative attrs)
    inherit: ClassVar[dict[str, dict[str, str]]] = {
        "fc": {"channel": "output_channels"},
    }
    deps: ClassVar[tuple[str, ...]] = ("matplotlib",)

    def execute(self, ctx, fc, channel: str = "",
                decimation_level: int = 0, sample_rate: str = "auto",
                ) -> dict[str, Any]:
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
        # sample_rate selects the rate FAMILY; decimation_level indexes the
        # cascade step within it (0 = the family's native tier — original
        # data; 1..k deeper; -1 = deepest available step)
        rate, steps = _resolve_rate_tiers(levels, sample_rate)
        k = int(decimation_level)
        if k < 0:
            level = steps[-1]
        elif k < len(steps):
            level = steps[k]
        else:
            raise ValueError(
                f"fc has no decimation step {k} within the {rate:g} Hz "
                f"family (available steps: {len(steps)}, rates "
                f"{[round(float(levels[s].attrs['sample_rate']), 6) for s in steps]}"
                ") — lower 'Decimation level' or pick another 'Sample rate'")
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
            + (f", {rate:g} Hz" if rate > 0 else "")
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
    """Station title from an fc's identity attrs (history §15 convention)."""
    station = fc.attrs.get("station_id") or ""
    survey = fc.attrs.get("survey_id") or ""
    parts = [p for p in (station, survey) if p]
    return " — ".join(parts) if parts else "Coherence"


def _coherence_map_matrix(local_fc, ref_fc, ins, outs,
                          ensemble_windows: int = 16, smooth_bins: int = 0):
    """The coherence map's matrix: the Coherence Sort's exact criterion
    over (frequency bin, window) — docs/issues.md L18.

    Two-FC: the conservative min of the E–remote-H signal coherence and
    the local↔remote magnetic transfer coherence, computed over the
    intersect-paired windows (``_pair_remote_fc`` + ``_per_bin_coherence``,
    the sort's own helpers) — a dark region in the map is a window the
    sort down-weights. Single-FC: the E–H signal coherence.

    Returns ``(coh (F, W), window_centres, paired_local)`` — the time
    labels are the window CENTRES (start + half the window span, the
    settled L11 convention), derived from the fc's own window geometry.
    """
    import numpy as np

    local = _normalize_fc_layout(local_fc)
    ref = _normalize_fc_layout(ref_fc) if ref_fc is not None else None
    if ref is not None:
        local, ref = _pair_remote_fc(local, ref, pairing="intersect",
                                     tol_s=0.0)
    coh_e, coh_x = _per_bin_coherence(
        local, ref, ins, outs, ensemble_windows=ensemble_windows,
        smooth_bins=smooth_bins)
    coh = np.minimum(coh_e, coh_x) if ref is not None else coh_e

    t = np.asarray(local["time"].values)
    sr = float(getattr(local, "attrs", {}).get("sample_rate", 0) or 0)
    nw = float(getattr(local, "attrs", {}).get("num_samples_window", 0) or 0)
    if t.dtype.kind == "M":
        if nw > 0 and sr > 0:
            half = np.timedelta64(round(nw / sr / 2 * 1e9), "ns")
        elif t.size > 1:
            half = (t[1] - t[0]) // 2
        else:
            half = np.timedelta64(0, "ns")
        centres = t + half
    else:
        dt = (nw / sr) if (nw > 0 and sr > 0) else (
            float(np.median(np.diff(t))) if t.size > 1 else 0.0)
        centres = t + dt / 2
    return coh, centres, local


def _coherence_map_title(paired, step: int, family_rate: float,
                         ensemble_windows: int, two_fc: bool,
                         print_params: bool) -> str:
    """The map's title: station + 'Coherence map', plus the processing
    provenance line only when ``print_params`` (production images stay
    clean — the Plot AppRes/Phase convention)."""
    title = (station_title_from_fc(paired) + " — "
             if station_title_from_fc(paired) else "") + "Coherence map"
    if not print_params:
        return title
    mode_note = ("E–remote H / local↔remote min"
                 if two_fc else "E–H signal")
    tier_rate = float(getattr(paired, "attrs", {}).get("sample_rate", 0) or 0)
    return title + (f" (step {step} ({tier_rate:g} Hz) of the "
                    f"{family_rate:g} Hz family, window "
                    f"{paired.attrs.get('num_samples_window', '?')} samples"
                    f", ensemble ±{int(ensemble_windows)} — {mode_note})")


class CoherenceMap(Node):
    type = "aurora.plot.coherence_map"
    title = "Plot Coherence Map"
    tooltip = "Time x frequency map of the Coherence Sort's criterion — localizes WHEN a band is bad."
    description = (
        "Time-frequency map of COHERENCE instead of power (docs/issues.md "
        "L18): the exact per-(window, bin) statistic Coherence Sort "
        "applies — the E–H signal coherence, or two-FC the conservative "
        "min of the E–remote-H signal and the local↔remote magnetic "
        "transfer coherence over the intersect-paired windows. A dark "
        "region is a window the sort down-weights; a band that is dark "
        "only at certain times localizes the noise source (e.g. a remote "
        "station's daytime noise). The level follows the spectrogram's "
        "cascade-step convention; the ensemble semantics match the "
        "upstream sort's."
    )
    category = "Display"
    terminal = True  # M13: manifest-declared run visibility
    inputs = (
        Input("fc", "fc"),
        Input("ref_fc", "fc", label="ref fc (remote)", required=False),
    )
    outputs = (Output("plot", "plot"),)
    params = (
        Param("ensemble_windows", "integer", label="Ensemble windows",
              default=16, min=1,
              description="Sliding window-ensemble half-width — match the "
                          "upstream Coherence Sort setting."),
        Param("smooth_bins", "integer", label="Frequency smoothing (bins)",
              default=0, min=0,
              description="Adjacent-bin pooling — match the upstream "
                          "Coherence Sort setting."),
        Param("decimation_level", "integer", label="Decimation level",
              default=0, min=-1,
              description="Cascade step within the selected rate family: "
                          "0 (default) = the family's native tier, 1..k "
                          "one decimation deeper each, −1 = the deepest "
                          "available step."),
        Param("sample_rate", "string", label="Sample rate", default="auto",
              optional=True, suggest_from="fc",
              placeholder="auto (lowest rate present)",
              description="Which rate family to display on a merged burst "
                          "FC database (auto = the lowest rate present — "
                          "the continuous stream). The connected fc's "
                          "tiers are offered as suggestions after a run."),
        Param("print_params", "boolean", label="Print parameters on plot",
              default=False,
              description="Add the processing-provenance line (cascade "
                          "step, window size, ensemble, coherence type) "
                          "to the title — useful for algorithm "
                          "comparison; untick for production images."),
    )
    deps: ClassVar[tuple[str, ...]] = ("matplotlib",)

    def execute(self, ctx, fc, ref_fc=None, ensemble_windows: int = 16,
                smooth_bins: int = 0, decimation_level: int = 0,
                sample_rate: str = "auto", print_params: bool = False) -> dict[str, Any]:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.dates as mdates
        import matplotlib.pyplot as plt
        from emptyui_mth5.plotting import apply_style

        levels = fc if isinstance(fc, dict) else {0: fc}
        rate, steps = _resolve_rate_tiers(levels, sample_rate)
        k = int(decimation_level)
        if k < 0:
            lv = steps[-1]
        elif k < len(steps):
            lv = steps[k]
        else:
            raise ValueError(
                f"fc has no decimation step {k} within the {rate:g} Hz "
                f"family (available steps: {len(steps)}, rates "
                f"{[round(float(levels[s].attrs['sample_rate']), 6) for s in steps]}"
                ") — lower 'Decimation level' or pick another 'Sample rate'")
        ds = levels[lv]
        ins, outs, _ = _fc_io_channels(ds)

        ref = None
        if ref_fc is not None:
            ref_levels = ref_fc if isinstance(ref_fc, dict) else {0: ref_fc}
            # the remote's tier at the same family rate, matched like
            # PlotCoherency does (merged burst databases key tiers by
            # rate position, which differs between stations)
            ref_by_rate: dict[float, list] = {}
            for rl, rds in ref_levels.items():
                r = round(float(getattr(rds, "attrs", {}).get(
                    "sample_rate", 0.0) or 0.0), 9)
                ref_by_rate.setdefault(r, []).append(rl)
            r_rate = round(float(getattr(ds, "attrs", {}).get(
                "sample_rate", 0.0) or 0.0), 9)
            r_ids = ref_by_rate.get(r_rate)
            if not r_ids:
                raise ValueError(
                    f"remote-reference fc has no tier at {r_rate:g} Hz "
                    f"(has {sorted(ref_by_rate)}) — compute both "
                    "stations' FCs with the same decimation/window "
                    "settings")
            ref = ref_levels[r_ids[0]]

        coh, centres, paired = _coherence_map_matrix(
            ds, ref, ins, outs, ensemble_windows=ensemble_windows,
            smooth_bins=smooth_bins)

        # DC and nonpositive frequencies cannot live on a log axis
        freq = np.asarray(paired["frequency"].values, dtype=float)
        keep = np.isfinite(freq) & (freq > 0)
        coh = coh[keep, :]
        freq = freq[keep]

        time_is_dt = centres.dtype.kind == "M"
        fig, ax = plt.subplots(figsize=(10, 4))
        mesh = ax.pcolormesh(centres, freq, coh, shading="auto",
                             cmap="viridis", vmin=0.0, vmax=1.0)
        cbar = fig.colorbar(mesh, ax=ax)
        cbar.set_label("Coherence$^2$")
        apply_style(fig, (ax,))
        ax.set_yscale("log")
        ax.set_xlabel("Time" if time_is_dt else "Time (s)")
        if time_is_dt:
            ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(
                mdates.AutoDateLocator()))
        ax.set_ylabel("Frequency (Hz)")
        ax.set_title(_coherence_map_title(
            paired, k, rate, ensemble_windows, ref is not None,
            print_params), fontsize=10,)
        fig.tight_layout()

        path = ctx.artifact_path("coherence_map", ".png")
        fig.savefig(path, dpi=150, bbox_inches="tight",
                    metadata={"Software": "emptyUI"})
        plt.close(fig)
        ctx.progress(1.0, f"coherence map: {coh.shape[0]} bins x "
                          f"{coh.shape[1]} windows (level {lv})")
        return {"plot": ctx.publish_artifact(path, kind="image")}
