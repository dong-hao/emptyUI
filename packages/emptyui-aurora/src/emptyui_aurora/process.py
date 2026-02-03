"""aurora pipeline nodes: ComputeFC (TS→FC), EstimateTF (FC→TF), MergeTFLevels.

ComputeFC/EstimateTF split the aurora pipeline at the FC boundary (DESIGN.md
§3, §14). FC is passed in-memory between nodes and, when ``persist`` is on,
also written into the workspace MTH5's FC layer so it can be reused / merged.
Estimators are OLS / robust RME / remote-reference RME_RR (aurora-backed).
"""

from __future__ import annotations

from typing import Any, ClassVar

from emptyui import Input, Node, Output, Param
from emptyui_mth5.nomenclature import (
    KEYWORDS,
    classify_channels,
)

from ._processing import (
    build_tf,
    coherence_sort_fcs,
    compute_fc_levels,
    estimate_tf_levels,
    merge_fc_containers,
    merge_tf_levels,
)

_NOMENCLATURE_OPTIONS = ("auto",) + KEYWORDS


def _station_location(data):
    """(latitude, longitude, elevation) of the run's station, or (None,)*3."""
    try:
        loc = data.station_metadata.location
        return float(loc.latitude), float(loc.longitude), float(loc.elevation)
    except Exception:  # noqa: BLE001 — core TimeSeries carries no location
        return None, None, None


def _station_identity_attrs(data, survey: str, station: str, run: str) -> dict[str, Any]:
    """Primitive station-identity attrs to carry on an fc (DESIGN.md §15).

    The ts.run payload (mth5 RunTS) exposes survey/station/run metadata and the
    station location; only primitives are stored (xarray attrs must survive
    persistence), and missing values are omitted so downstream nodes can tell
    "unknown" apart from a default.
    """
    attrs: dict[str, Any] = {
        "survey_id": survey,
        "station_id": station,
        "run_id": run,
    }
    lat, lon, elev = _station_location(data)
    if lat is not None:
        attrs["station_latitude"] = lat
        attrs["station_longitude"] = lon
        attrs["station_elevation"] = elev
    return attrs

#: Nomenclature presets (actual channel names, mt_metadata CHANNEL_MAPS).
#: Each preset includes the vertical magnetic channel (bz/h3) so the tipper is
#: computed by default; ComputeFC drops it when the data has no vertical
#: channel (best-effort extension, see _processing._resolve_io_channels).
_NOMENCLATURE_PRESETS: dict[str, tuple[tuple[str, str], ...]] = {
    "lemi12": (("input_channels", "bx,by"), ("output_channels", "e1,e2,bz")),
    "lemi34": (("input_channels", "bx,by"), ("output_channels", "e3,e4,bz")),
    "musgraves": (("input_channels", "bx,by"), ("output_channels", "ex,ey,bz")),
    "phoenix123": (("input_channels", "h1,h2"), ("output_channels", "e1,e2,h3")),
    "default": (("input_channels", "hx,hy"), ("output_channels", "ex,ey,hz")),
}


class ComputeFC(Node):
    type = "aurora.process.compute_fc"
    title = "Compute FC"
    tooltip = "Windowed STFT: time series to Fourier coefficients over cascading decimation levels."
    description = (
        "Windowed STFT to Fourier coefficients (fc) over N cascading "
        "decimation levels (constant window in samples; sample rate halves by "
        "``factor`` per level). Specify input (horizontal magnetic) and output "
        "channel component names — outputs may be the electric pair (ex,ey) "
        "and/or a vertical magnetic channel (bz) for a geomagnetic transfer "
        "function (tipper), e.g. inputs bx,by / outputs e1,e2,bz. Empty input/"
        "output channels auto-detect from the data and its nomenclature — the "
        "vertical magnetic channel is included by default when present (tipper "
        "by default; impedance-only or tipper-only FCs are derived from what "
        "the run actually contains). The channel "
        "nomenclature (mt_metadata CHANNEL_MAPS: lemi12/lemi34/...) decides "
        "which names mean what — magnetic B vs H is derived from it, no manual "
        "switch needed. Persists each level's FC into the workspace MTH5 when "
        "an absolute start time is available."
    )
    category = "Processing"
    inputs = (Input("data", "ts.run", label="time series"),)
    outputs = (Output("fc", "fc"),)
    params = (
        Param("channel_nomenclature", "select", label="Channel nomenclature",
              options=_NOMENCLATURE_OPTIONS, default="auto",
              description="mt_metadata CHANNEL_MAPS keyword: maps standard "
                          "components (ex/ey/hx/hy/hz) to the instrument's "
                          "actual channel names and derives magnetic units",
              cascade={k: list(v) for k, v in _NOMENCLATURE_PRESETS.items()}),
        Param("input_channels", "string", label="Input channels", default="",
              placeholder="auto (detect from data)",
              description="Two horizontal magnetic channel names (e.g. 'bx,by' "
                          "or 'h1,h2'). Empty = resolve from the nomenclature "
                          "and the data (mt_metadata CHANNEL_MAPS)."),
        Param("output_channels", "string", label="Output channels", default="",
              placeholder="auto (detect from data)",
              description="Electric pair and/or vertical magnetic channel "
                          "(e.g. 'e1,e2' or 'e1,e2,bz' for a tipper). Empty = "
                          "resolve from the nomenclature and the data: the "
                          "electric pair plus the vertical magnetic channel "
                          "when present; a vertical-only run yields a "
                          "tipper-only FC."),
        Param("magnetic_unit", "select", label="Magnetic input",
              default="auto", options=("auto", "h", "b"),
              description="auto (default) = derived from the channel "
                          "nomenclature / names (bx/by → fluxgate B, hx/hy → "
                          "coil H); both are archived in nT and converted to "
                          "A/m (the fc always carries H — impedance in "
                          "ohm). An explicit 'h' means the data are already "
                          "in A/m (legacy/synthetic); 'b' forces the nT→A/m "
                          "conversion."),
        Param("e_dipole_length_m", "number", label="E dipole length (m)", default=None,
              optional=True,
              description="LEMI stores E as mV potential; divide by dipole length to get V/m"),
        Param("decimation_levels", "integer", label="Decimation levels", default=0,
              min=0,
              description="Number of cascading decimations (0 = none, native rate only)"),
        Param("factor", "integer", label="Decimation factor", default=4, min=2),
        Param("window_samples", "integer", label="Window (samples)", default=256, min=8,
              description="Constant window length in samples across all levels"),
        Param("overlap", "number", default=0.5, min=0.0, max=0.99,
              description="Window overlap fraction (< 1)"),
        Param("window_type", "select", options=("hamming", "hanning", "boxcar"),
              default="hamming",
              description="boxcar maximizes effective dof (best with "
                          "prewhitening at white-noise-limited scales); "
                          "hanning/hamming suppress line-spectra leakage "
                          "(day-scale windows)"),
        Param("window_type_deep", "select", label="Deep-level taper",
              default="", options=("", "hamming", "hanning", "boxcar"),
              optional=True,
              description="Taper override for the deep decimation levels "
                          "(see 'from level'): zero-endpoint tapers (hamming/"
                          "hanning) suppress diurnal-line leakage through the "
                          "day-scale windows; boxcar maximizes dof. Empty = "
                          "same as the base taper."),
        Param("deep_from_level", "integer", label="Deep taper from level",
              default=None, optional=True, min=0,
              description="Decimation level from which 'Deep-level taper' "
                          "applies (e.g. 3 with window 256 = windows longer "
                          "than ~1 h); empty = never (base taper everywhere)"),
        Param("clock_zero", "string", label="Clock zero",
              default="", optional=True,
              placeholder='"epoch" or ISO datetime; empty = off',
              description="Anchor window starts on a common absolute grid "
                          "t0 + k·step (head partial window dropped): "
                          "'epoch' (Unix epoch — automatically common to "
                          "all stations/runs) or an explicit ISO datetime. "
                          "Required for remote-reference pairing across "
                          "separately-computed per-run FCs (Merge FC, burst "
                          "mode); empty anchors at each run's start (legacy). "
                          "Needs an absolute start time."),
        Param("prewhitening", "select", label="Prewhitening", default="",
              options=("", "first difference"),
              description="Leakage-reduction prewhitening before the STFT: "
                          "'first difference' (aurora/EMTF-style) differences "
                          "each channel in time and recolors the FCs by the "
                          "exact inverse filter; off (default) keeps the "
                          "legacy emptyUI behavior."),
        Param("per_window_detrend", "select", label="Per-window detrend",
              default="", options=("", "linear"),
              description="Detrend each STFT window (linear) before "
                          "tapering — keeps window-level trends out of the "
                          "low-frequency bins (aurora's per-window default); "
                          "off (default) = global Detrend node only."),
        Param("reject_saturated", "boolean", label="Reject saturated windows",
              default=True,
              description="Zero-weight (NaN) every FC window whose flagged "
                          "sample fraction exceeds 'saturated max' (MTU-5A "
                          "block tags byte 15, §24.12 QC). No effect on "
                          "payloads without a 'saturated' mask."),
        Param("saturated_max", "number", label="Saturated max",
              default=0.01, min=0.0, max=1.0,
              enabled_when=("reject_saturated", (True,)),
              description="Flagged-fraction threshold: a window is "
                          "zero-weighted when more than this share of its "
                          "samples is flagged saturated. The original "
                          "any-overlap rule deleted the long-period tail "
                          "whenever saturation is frequent (a 73-minute "
                          "window almost always overlaps one clipped "
                          "sample out of 65536)."),
        Param("persist", "boolean", label="Persist to workspace", default=True,
              description="Write each level's FC into the workspace MTH5 FC layer"),
        Param("survey", "string", label="Survey id", default="", optional=True),
        Param("station", "string", label="Station id", default="", optional=True),
        Param("run", "string", label="Run id", default="", optional=True),
    )
    inherit: ClassVar[dict[str, dict[str, str]]] = {
        "data": {"survey": "survey", "station": "station", "run": "run",
                 "channel_nomenclature": "channel_nomenclature"},
    }

    def execute(self, ctx, data, input_channels: str = "",
                output_channels: str = "", channel_nomenclature: str = "auto",
                magnetic_unit: str = "auto",
                e_dipole_length_m=None,
                decimation_levels: int = 0, factor: int = 4,
                window_samples: int = 256,
                overlap: float = 0.5, window_type: str = "hamming",
                window_type_deep: str = "", deep_from_level=None,
                clock_zero: str = "",
                prewhitening: str = "", per_window_detrend: str = "",
                reject_saturated: bool = True,
                saturated_max: float = 0.01,
                persist: bool = True, survey: str = "", station: str = "",
                run: str = "") -> dict[str, Any]:
        # Channel names resolve inside compute_fc_levels (empty fields = auto
        # from nomenclature + data; tipper included when the vertical channel
        # is present, missing outputs dropped with a warning).
        levels = compute_fc_levels(
            data, input_channels, output_channels,
            num_decimations=decimation_levels, factor=factor,
            num_samples_window=window_samples, overlap=overlap, window_type=window_type,
            magnetic_unit=magnetic_unit,
            e_dipole_length_m=e_dipole_length_m,
            channel_nomenclature=channel_nomenclature,
            prewhitening=prewhitening,
            per_window_detrend=per_window_detrend,
            window_type_deep=window_type_deep,
            deep_from_level=deep_from_level,
            clock_zero=clock_zero,
            reject_saturated=reject_saturated,
            saturated_max=saturated_max,
        )
        from emptyui_mth5.persist import persist_fc_level, runts_context

        # Station identity travels with the fc (attrs) so EstimateTF can attach
        # it to the TF — the single point of truth for plot titles and export
        # headers (DESIGN.md §15).
        surv, stat, run_id = runts_context(data, survey, station, run)
        identity = _station_identity_attrs(data, surv, stat, run_id)
        for level, fc in levels.items():
            fc.attrs.update(identity)
        if persist:
            for level, fc in levels.items():
                if "datetime64" in str(fc.time.dtype):
                    persist_fc_level(ctx, fc, level, fc.attrs["sample_rate"],
                                     survey=surv, station=stat, run=run_id)
                else:
                    ctx.logger.warning(
                        f"FC level {level} has no absolute (datetime64) time axis — "
                        "skipping persistence."
                    )
                    break
        summary = ", ".join(
            f"L{lv}:{fc.sizes['frequency']}f×{fc.sizes['time']}w" for lv, fc in levels.items()
        )
        ctx.progress(1.0, f"computed FC levels [{summary}]")
        return {"fc": levels}


def _diurnal_prior_factor(start, use_prior: bool, prior_max: float,
                          local_utc_offset: float,
                          absolute_time: bool = True) -> float:
    """Soft diurnal prior for run QC (§24.13): the natural-signal envelope
    varies smoothly over the day (Szarka 1988; Garcia & Jones 2002), so a
    daytime run must be cleaner than a night run to pass the same score.
    The factor peaks at local noon (1 + (prior_max-1)) and is 1 at local
    midnight; callers divide their deviation threshold by it."""
    if not use_prior or not absolute_time or start in (None, 0):
        return 1.0
    import numpy as np
    import pandas as pd

    try:
        h = (pd.Timestamp(start).value / 1e9 / 3600.0
             + local_utc_offset) % 24.0
    except (TypeError, ValueError):
        return 1.0
    return 1.0 + (prior_max - 1.0) * 0.5 * (
        1.0 + np.cos(2 * np.pi * (h - 12.0) / 24.0))


class BuildFC(Node):
    type = "aurora.fc.build"
    title = "Build FC Database"
    tooltip = "Compute per-run FC cascades for every run of a station and merge them along time."
    description = (
        "Iterate the listed runs of one station, compute each run's "
        "cascading-decimation FC levels, and merge them along time (Merge "
        "FC's route) — the per-run cascade builder for archives with many "
        "runs: burst-mode recordings (scheduled high-rate slots + "
        "continuous low-rate streams, DESIGN §24.8) or gapped continuous "
        "acquisitions. A merged-time-series cascade cannot decimate across "
        "slot/run boundaries; per-run cascades are fine and starved deep "
        "levels are skipped with warnings. Clock zero defaults to 'epoch' "
        "here — staggered slot schedules make run-start anchoring useless "
        "and the absolute grid is what lets remote-reference slots pair "
        "across stations. FC persistence per run is available but off by "
        "default."
    )
    category = "Processing"
    inputs = (Input("mth5", "mth5"),)
    outputs = (Output("fc", "fc"),)
    params = (
        Param("survey", "string", label="Survey id", default="", optional=True),
        Param("station", "string", label="Station id", default="",
              optional=True, placeholder="auto (first available)"),
        Param("runs", "string", label="Runs", default="", optional=True,
              placeholder="e.g. sr2400_0001,... (empty = all runs)"),
        Param("input_channels", "string", default="", optional=True),
        Param("output_channels", "string", default="", optional=True),
        Param("channel_nomenclature", "select", label="Channel nomenclature",
              options=_NOMENCLATURE_OPTIONS, default="auto",
              description="mt_metadata CHANNEL_MAPS keyword: maps standard "
                          "components (ex/ey/hx/hy/hz) to the instrument's "
                          "actual channel names and derives magnetic units",
              cascade={k: list(v) for k, v in _NOMENCLATURE_PRESETS.items()}),
        Param("magnetic_unit", "select", label="Magnetic unit",
              options=("auto", "h", "b"), default="auto"),
        Param("e_dipole_length_m", "number", label="E dipole length (m)",
              default=0.0, optional=True, min=0,
              description="0 = take from the run metadata when present"),
        Param("decimation_levels", "integer", label="Decimation levels",
              default=0, min=0,
              description="0 = auto (decimate until fewer samples than one "
                          "window remain)"),
        Param("factor", "integer", label="Decimation factor", default=4, min=2),
        Param("window_samples", "integer", label="Window (samples)",
              default=256, min=8),
        Param("overlap", "number", default=0.5, min=0.0, max=0.99),
        Param("window_type", "select",
              options=("hamming", "hanning", "boxcar"), default="hamming"),
        Param("window_type_deep", "select", label="Deep-level taper",
              default="", options=("", "hamming", "hanning", "boxcar"),
              optional=True),
        Param("deep_from_level", "integer", label="Deep taper from level",
              default=None, optional=True, min=0),
        Param("clock_zero", "string", label="Clock zero", default="epoch",
              optional=True,
              placeholder='"epoch" or ISO datetime; empty = run starts',
              description="Defaults to 'epoch' (absolute grid): staggered "
                          "slot schedules make run-start anchoring useless "
                          "for burst archives, and the absolute grid is "
                          "what pairs remote-reference slots across "
                          "stations (DESIGN §24.8)."),
        Param("prewhitening", "select", label="Prewhitening", default="",
              options=("", "first difference")),
        Param("per_window_detrend", "select", label="Per-window detrend",
              default="", options=("", "linear")),
        Param("reject_saturated", "boolean", label="Reject saturated windows",
              default=True,
              description="Zero-weight (NaN) every FC window whose flagged "
                          "sample fraction exceeds 'saturated max' (MTU-5A "
                          "block tags byte 15, §24.12 QC). No effect on "
                          "payloads without a 'saturated' mask."),
        Param("saturated_max", "number", label="Saturated max",
              default=0.01, min=0.0, max=1.0,
              enabled_when=("reject_saturated", (True,)),
              description="Flagged-fraction threshold: a window is "
                          "zero-weighted when more than this share of its "
                          "samples is flagged saturated. The original "
                          "any-overlap rule deleted the long-period tail "
                          "whenever saturation is frequent (a 73-minute "
                          "window almost always overlaps one clipped "
                          "sample out of 65536)."),
        Param("persist", "boolean", label="Persist per-run FCs", default=False,
              description="Write each run's FC levels into the workspace "
                          "MTH5 keyed by run (aurora's FC-layer pattern) — "
                          "off by default; turn on once the procedure is "
                          "settled (DESIGN §24.8 decision (b))."),
        Param("regroup_contiguous", "boolean", label="Regroup contiguous runs",
              default=False,
              description="Cascade maximal contiguous spans of same-rate "
                          "runs as ONE payload (§24.13): hourly sessions "
                          "cannot form a level below ~0.23 Hz per slice, but "
                          "the underlying stream is unbroken — cascading the "
                          "kept contiguous span restores the long-period "
                          "tail. Built from ALL sessions: hour-level QC "
                          "lives in Select Runs, so spans never depend on "
                          "QC outcomes and each window is stamped with the "
                          "session it came from."),
    )

    def execute(self, ctx, mth5, survey: str = "", station: str = "",
                runs: str = "", input_channels: str = "",
                output_channels: str = "", channel_nomenclature: str = "auto",
                magnetic_unit: str = "auto", e_dipole_length_m=None,
                decimation_levels: int = 0, factor: int = 4,
                window_samples: int = 256, overlap: float = 0.5,
                window_type: str = "hamming", window_type_deep: str = "",
                deep_from_level=None, clock_zero: str = "epoch",
                prewhitening: str = "", per_window_detrend: str = "",
                reject_saturated: bool = True,
                saturated_max: float = 0.01,
                persist: bool = False,
                regroup_contiguous: bool = False) -> dict[str, Any]:
        from emptyui_mth5.runs import _resolve_run, _resolve_station, _resolve_survey

        survey_obj = _resolve_survey(mth5, survey)
        station_obj = _resolve_station(survey_obj, station)
        available = [str(x) for x in station_obj.run_summary["id"]]
        wanted = ([r.strip() for r in str(runs).split(",") if r.strip()]
                  if str(runs).strip() else available)
        missing = [r for r in wanted if r not in available]
        if missing:
            raise ValueError(
                f"run(s) {missing} not in station archive {available}")
        surv = str(survey_obj.metadata.id)
        stat = str(station_obj.metadata.id)

        # per-family decimation trim (DESIGN §24.8): on a mixed-rate burst
        # archive the continuous (lowest-rate) family carries the full
        # 'decimation_levels' cascade; higher-rate families decimate only
        # until their rate would reach the next family's native rate — the
        # lower family's cascade covers those frequencies, and same-rate
        # levels computed from different families would double-cover the
        # same acquisition windows. E.g. 2400/150/15 Hz with factor 4:
        # 2400 Hz slots decimate once (-> 600), 150 Hz slots once
        # (-> 37.5), the 15 Hz stream takes the full parameter.
        import numpy as np

        summary = station_obj.run_summary
        rate_of = dict(zip(summary["id"].astype(str),
                           summary["sample_rate"].astype(float)))
        family_rates = sorted({rate_of[r] for r in wanted})
        dec_of_run: dict[str, int] = {}
        for rid in wanted:
            r = rate_of[rid]
            lower = [x for x in family_rates if x < r * (1 - 1e-9)]
            if not lower:
                dec_of_run[rid] = decimation_levels
                continue
            d_max = max(0, int(np.ceil(np.log2(r / max(lower))
                                       / np.log2(factor))) - 1)
            dec_of_run[rid] = min(decimation_levels, d_max)

        dipole = e_dipole_length_m or None
        containers = []
        # fast per-run data path: reading channels via to_xarray and
        # wrapping in a duck payload avoids RunGroup.to_runts — whose
        # mt_metadata deepcopy/validation storm costs ~7 s per run (40x
        # the data read itself; DESIGN §24.8 performance note). The same
        # storm lives in the group .metadata properties (~20 s per access
        # on a burst archive) — access station metadata ONCE and derive
        # per-run rate/start from the channel attrs/coords.
        from types import SimpleNamespace

        import xarray as xr

        station_md = station_obj.metadata
        identity_base = _station_identity_attrs(
            SimpleNamespace(station_metadata=station_md), surv, stat, "")

        def _read_channels(run_obj):
            data_vars = {}
            sample_rate = 0.0
            start = None
            for name in run_obj.groups_list:
                da = run_obj.get_channel(name).to_xarray()
                sample_rate = float(da.attrs.get("sample_rate",
                                                 sample_rate))
                if start is None and "time" in da.coords:
                    start = da.coords["time"].values[0]
                data_vars[str(name)] = da
            return data_vars, sample_rate, start

        def _resolve_registry(data_vars):
            # §24.12: resolve the response-chain filter objects once from
            # the archive so the duck payload carries the calibration the
            # FC-level calibration divides by (to_xarray carries the refs,
            # but the objects live in /Survey/Filters)
            filter_names: set[str] = set()
            for da in data_vars.values():
                for ref in da.attrs.get("filters") or []:
                    try:
                        filter_names.add(ref["applied_filter"]["name"])
                    except (TypeError, KeyError):
                        continue
            registry = {}
            if filter_names:
                fg = survey_obj.filters_group
                for fn in sorted(filter_names):
                    try:
                        registry[fn] = fg.to_filter_object(fn)
                    except Exception:  # noqa: BLE001
                        ctx.logger.warning(
                            "filter '%s' referenced by the archive channels "
                            "could not be resolved — its response will not "
                            "be removed by Compute FC", fn)
            return registry

        # cascade spans: singletons by default; with regroup_contiguous,
        # maximal contiguous spans of same-rate runs (adjacency within two
        # samples) — the physically right processing unit for unbroken
        # acquisitions. Built from ALL sessions: hour-level QC lives in
        # Select Runs (§24.13), so spans never depend on QC outcomes and
        # the time series is read exactly once.
        import pandas as pd

        spans: list[list[str]] = []
        if not regroup_contiguous:
            spans = [[rid] for rid in wanted]
        else:
            per_rate: dict[float, list[str]] = {}
            for rid in wanted:
                per_rate.setdefault(rate_of[rid], []).append(rid)
            starts = dict(zip(summary["id"].astype(str),
                              pd.to_datetime(summary["start"])))
            ends = dict(zip(summary["id"].astype(str),
                            pd.to_datetime(summary["end"])))
            for rate in sorted(per_rate, reverse=True):
                ordered = sorted(per_rate[rate], key=lambda r: starts[r])
                cur: list[str] = []
                prev_end = None
                for rid in ordered:
                    start, end = starts[rid], ends[rid]
                    if prev_end is not None and (
                            start - prev_end).total_seconds() > 2.0 / rate:
                        spans.append(cur)
                        cur = []
                    cur.append(rid)
                    prev_end = end
                if cur:
                    spans.append(cur)

        for i, span in enumerate(spans, 1):
            members = []
            for rid in span:
                run_obj = _resolve_run(station_obj, rid)
                data_vars, sample_rate, start = _read_channels(run_obj)
                members.append((rid, data_vars, sample_rate, start))
            sample_rate = members[0][2]
            span_id = (span[0] if len(span) == 1
                       else f"sr{sample_rate:g}_span{i:04d}")
            if len(span) == 1:
                data_vars = members[0][1]
                start = members[0][3]
            else:
                # union of channels across members: the 'saturated' aux
                # mask exists only on runs with flagged samples — pad the
                # others with zeros (no saturation there) instead of
                # silently dropping the mask or crashing on concat
                keys = list(members[0][1])
                for mv in members[1:]:
                    for name in mv[1]:
                        if name not in keys:
                            keys.append(name)
                data_vars = {}
                for name in keys:
                    parts = []
                    for mv in members:
                        if name in mv[1]:
                            parts.append(mv[1][name])
                        else:
                            parts.append(
                                xr.zeros_like(next(iter(mv[1].values()))))
                    data_vars[name] = xr.concat(parts, dim="time")
                start = members[0][3]
            registry = _resolve_registry(data_vars)
            ds = xr.Dataset(data_vars)
            run_ts = SimpleNamespace(
                dataset=ds,
                sample_rate=sample_rate,
                start=start,
                filters=registry,
            )
            r = sample_rate
            lower = [x for x in family_rates if x < r * (1 - 1e-9)]
            if not lower:
                dec_of_span = decimation_levels
            else:
                d_max = max(0, int(np.ceil(np.log2(r / max(lower))
                                           / np.log2(factor))) - 1)
                dec_of_span = min(decimation_levels, d_max)
            levels = compute_fc_levels(
                run_ts, input_channels, output_channels,
                num_decimations=dec_of_span, factor=factor,
                num_samples_window=window_samples, overlap=overlap,
                window_type=window_type,
                magnetic_unit=magnetic_unit,
                e_dipole_length_m=dipole,
                channel_nomenclature=channel_nomenclature,
                prewhitening=prewhitening,
                per_window_detrend=per_window_detrend,
                window_type_deep=window_type_deep,
                deep_from_level=deep_from_level,
                clock_zero=clock_zero,
                reject_saturated=reject_saturated,
                saturated_max=saturated_max,
            )
            # the station identity travels in the attrs (DESIGN.md §15);
            # run_id is per-span, the rest comes from the hoisted station md
            identity = dict(identity_base, run_id=span_id)
            for level, fc in levels.items():
                fc.attrs.update(identity)
            if persist:
                from emptyui_mth5.persist import persist_fc_level

                for level, fc in levels.items():
                    if "datetime64" in str(fc.time.dtype):
                        persist_fc_level(ctx, fc, level,
                                         fc.attrs["sample_rate"],
                                         survey=surv, station=stat,
                                         run=span_id)
                    else:
                        ctx.logger.warning(
                            "run %s level %d has no absolute time axis — "
                            "skipping persistence", span_id, level)
                        break
            # §24.13: per-window run identity at SESSION granularity — the
            # merged FC database keeps only the first contributor's attrs,
            # so the QC nodes need an explicit coordinate to group windows
            # by run. Windows straddling an internal boundary follow the
            # session their start falls in; Select Runs drops rejected
            # sessions downstream, conservatively including the overlap
            # neighbors that picked up their samples through the cascade.
            if len(span) == 1:
                for level, fc in levels.items():
                    levels[level] = fc.assign_coords(run_id=(
                        "time", np.full(int(fc.sizes["time"]), span[0])))
            else:
                lens = [len(next(iter(mv[1].values()))) for mv in members]
                bnds = np.concatenate(
                    ([0.0], np.cumsum(lens)[:-1])) / sample_rate
                for level, fc in levels.items():
                    t = np.asarray(fc["time"].values)
                    if np.issubdtype(t.dtype, np.datetime64):
                        t0 = np.asarray(members[0][3],
                                        dtype="datetime64[ns]")
                        off = ((t - t0) / np.timedelta64(1, "s")).astype(float)
                    else:
                        off = t.astype(float)
                    k = np.minimum(
                        np.searchsorted(bnds[1:], off, side="right"),
                        len(span) - 1)
                    levels[level] = fc.assign_coords(run_id=(
                        "time", np.asarray(span)[k]))
            containers.append(levels)
            ctx.progress(
                (i) / len(spans),
                f"FC {i}/{len(spans)}: span {span_id} "
                f"({len(levels)} level(s), {len(span)} run(s))",
            )
        merged = merge_fc_containers(containers)
        n_win = sum(int(ds.sizes["time"]) for ds in merged.values())
        ctx.progress(
            1.0,
            f"FC database: {len(wanted)} run(s) in {len(spans)} span(s) -> "
            f"{len(merged)} level(s), {n_win} window(s)",
        )
        return {"fc": merged}


def _fc_io_channels(fc):
    """(input_channels, output_channels, keyword) from an fc's attrs.

    Falls back to classifying the channel names through the nomenclature when
    the attrs are absent (e.g. an fc read back from an mth5 archive).
    """
    kw = fc.attrs.get("channel_nomenclature") or None
    ins_raw = fc.attrs.get("input_channels")
    outs_raw = fc.attrs.get("output_channels")
    if ins_raw and outs_raw:
        ins = [c.strip() for c in ins_raw.split(",") if c.strip()]
        outs = [c.strip() for c in outs_raw.split(",") if c.strip()]
    else:
        ins, outs, _ = classify_channels([str(c) for c in fc.data_vars], kw)
    return ins, outs, kw


class CoherenceSort(Node):
    type = "aurora.process.coherence_sort"
    title = "Coherence Sort"
    tooltip = "Down-weight or drop low-coherence FC windows before TF estimation."
    description = (
        "EMTF-style coherence sort: down-weight (or drop) Fourier-coefficient "
        "windows whose per-frequency window-ensemble coherence falls outside "
        "[coherence_min, coherence_max] — low coherence = noise-dominated "
        "windows; coherence ≈ 1 can be an overfit/coherent-noise signature. "
        "With a remote 'ref fc', the criterion is the conservative minimum of "
        "the E–remote-H signal coherence and the local↔remote magnetic "
        "transfer coherence (remote reference replaces the local magnetic "
        "channels, so local E/H coherence is the wrong measure), and the "
        "single shared weight is applied to both FCs (exact weighted-LS "
        "semantics for RME_RR). The weights are per (frequency bin, window) "
        "and independent of band geometry: band construction is Estimate "
        "TF's concern (DESIGN §24.9)."
    )
    category = "Processing"
    inputs = (
        Input("fc", "fc"),
        Input("ref_fc", "fc", label="ref fc (remote)", required=False),
    )
    outputs = (
        Output("fc", "fc"),
        Output("ref_fc", "fc", label="ref fc (remote)"),
        Output("table", "table", label="coherence diagnostics"),
    )
    params = (
        Param("coherence_min", "number", label="Coherence min",
              default=0.25, min=0.0, max=1.0,
              description="Lower coherence bound (per-window coherence²). "
                          "Windows below are down-weighted/dropped."),
        Param("coherence_max", "number", label="Coherence max",
              default=0.98, min=0.0, max=1.0,
              description="Upper coherence bound; coherence ≈ 1 can be an "
                          "overfit/coherent-noise signature."),
        Param("mode", "select", label="Mode",
              options=("weight", "drop", "off"), default="weight",
              description="weight = smooth down-weighting outside the bounds "
                          "(rows scaled by √w); drop = exclude windows "
                          "outside the bounds (NaN); off = no sorting."),
        Param("apply_below_period", "number", label="Apply below period (s)",
              default=0.0, min=0, optional=True,
              description="Scope the sorting to the noise-floor head: bin "
                          "periods at/above this value pass through "
                          "unsorted (the deep tail is healthy and weighting "
                          "it only perturbs the estimate). 0 = sort all "
                          "bins."),
        Param("ensemble_windows", "integer", label="Ensemble windows",
              default=16, min=1,
              description="Sliding window-ensemble half-width for the "
                          "per-bin statistic: coherence at each (bin, "
                          "window) is estimated over ±this many windows "
                          "(center included). Larger = smoother/more dof."),
        Param("smooth_bins", "integer", label="Frequency smoothing (bins)",
              default=0, min=0,
              description="Pool ±this many adjacent bins into the per-bin "
                          "ensemble (a FIXED smoothing scale, not the "
                          "estimate's band table) — restores dof where "
                          "windows are few (deep levels)."),
    )

    def execute(self, ctx, fc, ref_fc=None,
                coherence_min: float = 0.25, coherence_max: float = 0.98,
                mode: str = "weight",
                apply_below_period: float = 0.0,
                ensemble_windows: int = 16,
                smooth_bins: int = 0) -> dict[str, Any]:
        import json

        levels = fc if isinstance(fc, dict) else {0: fc}
        sample = next(iter(levels.values()))
        ins, outs, _ = _fc_io_channels(sample)
        ref_levels = ref_fc if isinstance(ref_fc, dict) else (
            {0: ref_fc} if ref_fc is not None else None)
        out_levels, out_ref, rows = coherence_sort_fcs(
            levels, ref_levels, ins, outs,
            coherence_min=coherence_min, coherence_max=coherence_max,
            mode=mode, apply_below_period=apply_below_period or None,
            ensemble_windows=ensemble_windows, smooth_bins=smooth_bins,
        )
        path = ctx.artifact_path("coherence_sort", ".json")
        path.write_text(json.dumps(rows))
        artifact = ctx.publish_artifact(path, kind="table")
        retained = sum(r["n_retained"] for r in rows)
        total = sum(r["n_windows"] for r in rows)
        ctx.progress(
            1.0,
            f"coherence sort: {retained}/{total} samples kept "
            f"({len(rows)} frequency bins)",
        )
        result: dict[str, Any] = {"fc": out_levels, "table": artifact}
        if ref_fc is not None:
            result["ref_fc"] = out_ref
        return result


class SelectRuns(Node):
    type = "aurora.process.select_runs"
    title = "Select Runs"
    tooltip = "Drop noise-dominated runs before TF estimation (run-level QC)."
    description = (
        "Run-level QC (DESIGN §24.13, after Weckmann et al. 2005): score "
        "every run by its per-window impedance level in a reference band — "
        "the per-window |Z| from the run's own Fourier coefficients, "
        "compared against the station-wide clean level (a percentile of the "
        "runs' medians). Cultural noise inflates "
        "the electric channels, so a noise-dominated run deviates from the "
        "station's stable impedance level (both tails are checked: inflated "
        "and dead channels). The decision threshold carries a SOFT diurnal "
        "prior: daytime runs are held to a stricter deviation (the prior "
        "expects a solar plant nearby), but a measured-clean day still "
        "passes and a noisy night is still dropped — the clock shapes the "
        "expectation, it does not delete data. Requires an fc payload built "
        "by Build FC (per-window 'run_id' coordinate)."
    )
    category = "Processing"
    inputs = (Input("fc", "fc"),)
    outputs = (
        Output("fc", "fc"),
        Output("table", "table", label="run diagnostics"),
    )
    params = (
        Param("period_min", "number", label="Period min (s)", default=1.0,
              min=0.0,
              description="Reference-band lower period bound."),
        Param("period_max", "number", label="Period max (s)", default=10.0,
              min=0.0,
              description="Reference-band upper period bound."),
        Param("deviation_max", "number", label="Deviation max", default=2.0,
              min=1.0,
              description="A run whose median per-window impedance level in "
                          "the band exceeds the station-wide clean level by "
                          "more than this factor is dropped. Cultural noise "
                          "only ever inflates the electric channels, so the "
                          "check is one-sided."),
        Param("dead_max", "number", label="Dead max", default=2.0, min=1.0,
              description="Instrument-fault check, independent of the prior: "
                          "a run whose level falls below 1/this factor of the "
                          "clean level (dead or dropped-out channels) is "
                          "dropped."),
        Param("reference_percentile", "number", label="Reference percentile",
              default=50.0, min=0.0, max=100.0,
              description="Station-wide impedance level per bin = this "
                          "percentile over the runs' per-run medians. The "
                          "plain median (50) empirically reproduces the "
                          "official Phoenix TF best on the test corpus "
                          "(§24.13 sweep); lower percentiles select the "
                          "low-|Z| tail and bias yx up by ~+0.03 decades."),
        Param("use_prior", "boolean", label="Daytime soft prior",
              default=True,
              description="Scale the threshold by local time of day: the "
                          "diurnal natural-signal envelope (Szarka 1988; "
                          "Garcia & Jones 2002) is the prior — daytime runs "
                          "are judged up to 'prior max' times stricter, "
                          "night runs by the full threshold."),
        Param("prior_max", "number", label="Prior max", default=2.0, min=1.0,
              description="Strictness multiplier at local noon (1 = no "
                          "prior)."),
        Param("local_utc_offset", "number", label="Local UTC offset (h)",
              default=8.0, min=-12.0, max=14.0,
              description="Site timezone offset used by the prior (Tibet "
                          "= UTC+8, §24.11)."),
    )

    def execute(self, ctx, fc, period_min: float = 1.0,
                period_max: float = 10.0, deviation_max: float = 2.0,
                dead_max: float = 2.0, reference_percentile: float = 50.0,
                use_prior: bool = True, prior_max: float = 2.0,
                local_utc_offset: float = 8.0) -> dict[str, Any]:
        import json

        import numpy as np

        levels = fc if isinstance(fc, dict) else {0: fc}
        sample = next(iter(levels.values()))
        if "run_id" not in sample.coords:
            raise ValueError(
                "Select Runs needs an fc payload with a per-window "
                "'run_id' coordinate — rebuild it with Build FC")
        ins, outs, _ = _fc_io_channels(sample)
        absolute_time = np.issubdtype(
            np.asarray(sample["time"].values).dtype, np.datetime64)
        if use_prior and not absolute_time:
            ctx.logger.warning(
                "Select Runs: fc time axis is relative (no clock zero) — "
                "the diurnal soft prior is disabled")

        # per-level scoring (levels have different frequency axes, so the
        # in-band bin counts differ): per-run per-bin median -> station
        # clean-level percentile -> per-run up/down deviations; the run's
        # final scores are the medians over its levels
        per_run_up: dict[str, list[float]] = {}
        per_run_down: dict[str, list[float]] = {}
        run_start: dict[str, np.datetime64] = {}
        for ds in levels.values():
            f = np.asarray(ds["frequency"].values, dtype=float)
            band = (f >= 1.0 / period_max) & (f <= 1.0 / period_min)
            if not band.any():
                continue
            H = np.stack([np.asarray(ds[c].values) for c in ins])
            E = np.stack([np.asarray(ds[c].values) for c in outs])
            denom = np.sum(np.abs(H) ** 2, axis=0)
            num = np.einsum("ift,jft->jft", np.conj(H), E)
            zsq = np.mean(np.abs(num / denom) ** 2, axis=0)[band, :]
            runs = np.asarray(ds["run_id"].values)
            times = np.asarray(ds["time"].values)
            uniq = np.unique(runs)
            med = np.full((uniq.size, int(band.sum())), np.nan)
            for i, rid in enumerate(uniq):
                m = runs == rid
                med[i] = np.nanmedian(zsq[:, m], axis=1)
                if absolute_time:
                    t0 = times[m].min()
                    if str(rid) not in run_start or t0 < run_start[str(rid)]:
                        run_start[str(rid)] = t0
            # station-wide impedance level per bin: this percentile over the
            # runs' per-run medians — the clean level, not the median (noise
            # inflates; storms raise every run equally and cancel)
            ref = np.nanpercentile(med, reference_percentile, axis=0)
            with np.errstate(divide="ignore", invalid="ignore"):
                up = med / ref
                down = ref / med
            for i, rid in enumerate(uniq):
                per_run_up.setdefault(str(rid), []).append(
                    float(np.nanmedian(up[i][np.isfinite(up[i])])))
                per_run_down.setdefault(str(rid), []).append(
                    float(np.nanmedian(down[i][np.isfinite(down[i])])))
        if not per_run_up:
            raise ValueError(
                f"no fc frequency bins fall in the reference band "
                f"{period_min}-{period_max} s — widen it")

        rows = []
        for rid in sorted(per_run_up):
            up_d = [x for x in per_run_up[rid] if np.isfinite(x)]
            dn_d = [x for x in per_run_down[rid] if np.isfinite(x)]
            up = float(np.median(up_d)) if up_d else float("inf")
            dn = float(np.median(dn_d)) if dn_d else float("inf")
            pf = _diurnal_prior_factor(run_start.get(rid, 0), use_prior,
                                       prior_max, local_utc_offset,
                                       absolute_time=absolute_time)
            thr = deviation_max / pf
            dropped = up > thr or dn > dead_max
            rows.append({"run_id": rid, "n_levels": len(per_run_up[rid]),
                         "deviation": max(up, dn), "up": up, "down": dn,
                         "prior": float(pf),
                         "threshold": float(thr), "dropped": bool(dropped)})
        # kept = every run in the payload minus the explicitly dropped ones:
        # runs without in-band bins (e.g. a rate family whose cascade never
        # reaches the reference band) are not scored and must pass through
        dropped_ids = {r["run_id"] for r in rows if r["dropped"]}
        all_ids = set()
        for ds in levels.values():
            all_ids |= set(map(str, np.unique(ds["run_id"].values)))
        kept = all_ids - dropped_ids
        for r in rows:
            if r["dropped"]:
                ctx.logger.warning(
                    "Select Runs: dropping %s (up %.2f, down %.2f; "
                    "threshold %.2f, prior %.2f)", r["run_id"], r["up"],
                    r["down"], r["threshold"], r["prior"])
        # conservative guard: a window from a NEIGHBOR session that overlaps
        # a rejected one picked up its samples through the decimation
        # cascade — drop it too. Start-time attribution already puts right
        # straddlers in the rejected session; only left neighbors need the
        # extra cut (one window duration wide).
        dur = None
        try:
            dur = (float(sample.attrs["num_samples_window"])
                   / float(sample.attrs["sample_rate"]))
        except (KeyError, TypeError, ZeroDivisionError):
            dur = None
        n_extra = 0
        out = {}
        for lv, ds in levels.items():
            runs_arr = np.asarray(ds["run_id"].values)
            mask = np.isin(runs_arr, list(kept))
            if dropped_ids and dur:
                times = np.asarray(ds["time"].values)
                dt = np.issubdtype(times.dtype, np.datetime64)
                for rid in dropped_ids:
                    sel = runs_arr == rid
                    if not sel.any():
                        continue
                    t_lo = times[sel].min()
                    if dt:
                        delta = ((times - t_lo)
                                 / np.timedelta64(1, "s")).astype(float)
                    else:
                        delta = times.astype(float) - float(t_lo)
                    extra = (delta < 0) & (delta > -dur)
                    n_extra += int(extra.sum())
                    mask &= ~extra
            out[lv] = ds.isel(time=mask)
        if n_extra:
            ctx.logger.warning(
                "Select Runs: dropping %d overlap neighbor window(s) "
                "adjacent to rejected runs", n_extra)
        for ds in out.values():
            ds.attrs["runs_kept"] = ",".join(sorted(kept))
            ds.attrs["runs_dropped"] = ",".join(sorted(
                r["run_id"] for r in rows if r["dropped"]))
            ds.attrs["select_runs_band_s"] = f"{period_min}-{period_max}"
        path = ctx.artifact_path("select_runs", ".json")
        path.write_text(json.dumps(rows))
        artifact = ctx.publish_artifact(path, kind="table")
        ctx.progress(
            1.0,
            f"Select Runs: kept {len(kept)}/{len(rows)} run(s) in the "
            f"{period_min}-{period_max} s band",
        )
        return {"fc": out, "table": artifact}


class ResidualSort(Node):
    type = "aurora.process.residual_sort"
    title = "Residual Sort"
    tooltip = "Drop windows/runs whose data do not fit the first-pass TF model."
    description = (
        "Residual-based two-pass selection (DESIGN §24.13, the orthogonal "
        "complement of power and coherence screening): a first-pass TF "
        "(Estimate TF's 'tf levels') predicts each FC window's outputs; "
        "windows whose pooled unexplained-power fraction is excessive are "
        "dropped (NaN) and the TF is re-estimated. The score is per RUN "
        "(median over the run's windows, then median over the run's bands "
        "after per-band normalization against the station-wide median) — "
        "clean MT data sit near the noise floor of unexplained power; "
        "cultural-noise windows blow it up. This measures 'does this window "
        "fit the model' rather than 'is the floor high' — the right lever "
        "for diffuse, unpolarized noise. Cascadeable: chain Estimate TF -> "
        "Residual Sort -> Estimate TF, iterating if needed. Requires the fc "
        "payload's per-window 'run_id' coordinate (Build FC) and the "
        "'band_intervals' carried by the first pass's tf levels."
    )
    category = "Processing"
    inputs = (
        Input("fc", "fc"),
        Input("tf_levels", "tf.level", label="tf levels (first pass)"),
    )
    outputs = (
        Output("fc", "fc"),
        Output("table", "table", label="residual diagnostics"),
    )
    params = (
        Param("unexplained_max", "number", label="Unexplained max",
              default=2.0, min=1.0,
              description="A run whose median per-window unexplained-power "
                          "fraction exceeds the station-wide median by more "
                          "than this factor is dropped. One-sided by design: "
                          "a LOW fraction means the data fit the model well "
                          "(not dead channels — zero-power windows are "
                          "excluded as NaN), so there is no down check here; "
                          "'down' is published as a diagnostic only."),
        Param("period_min", "number", label="Period min (s)", default=0.0,
              min=0.0, optional=True,
              description="Scope the test to bands at/above this period "
                          "(0 = all bands)."),
        Param("period_max", "number", label="Period max (s)", default=0.0,
              min=0.0, optional=True,
              description="Scope the test to bands below this period "
                          "(0 = all bands)."),
        Param("use_prior", "boolean", label="Daytime soft prior",
              default=True,
              description="Scale the threshold by local time of day "
                          "(shared with Select Runs / Build FC QC)."),
        Param("prior_max", "number", label="Prior max", default=2.0,
              min=1.0, optional=True),
        Param("local_utc_offset", "number", label="Local UTC offset (h)",
              default=8.0, min=-12.0, max=14.0, optional=True),
    )

    def execute(self, ctx, fc, tf_levels, unexplained_max: float = 2.0,
                period_min: float = 0.0, period_max: float = 0.0,
                use_prior: bool = True, prior_max: float = 2.0,
                local_utc_offset: float = 8.0) -> dict[str, Any]:
        import json

        import numpy as np

        levels = fc if isinstance(fc, dict) else {0: fc}
        sample = next(iter(levels.values()))
        if "run_id" not in sample.coords:
            raise ValueError(
                "Residual Sort needs an fc payload with a per-window "
                "'run_id' coordinate — rebuild it with Build FC")
        ins, outs, _ = _fc_io_channels(sample)
        absolute_time = np.issubdtype(
            np.asarray(sample["time"].values).dtype, np.datetime64)

        # per (level, band): pooled per-window prediction error fraction
        per_run_band: dict[str, dict[int, float]] = {}
        run_start: dict[str, np.datetime64] = {}
        n_tests = 0
        n_intervals_seen = 0
        for lv, ds in levels.items():
            entry = (tf_levels or {}).get(lv)
            if not entry or not entry.get("band_intervals"):
                continue  # no band geometry — this level cannot be tested
            n_intervals_seen += len(entry["band_intervals"])
            f = np.asarray(ds["frequency"].values, dtype=float)
            H = np.stack([np.asarray(ds[c].values) for c in ins])  # (2,nf,nt)
            E = np.stack([np.asarray(ds[c].values) for c in outs])
            Zs = np.asarray(entry["Z"])  # (nbands, n_out, n_in)
            intervals = entry["band_intervals"]
            if Zs.shape[0] != len(intervals):
                raise ValueError(
                    f"Residual Sort: tf levels' Z rows ({Zs.shape[0]}) do "
                    f"not match the band intervals ({len(intervals)}) for "
                    f"level {lv}")
            runs = np.asarray(ds["run_id"].values)
            times = np.asarray(ds["time"].values)
            uniq = np.unique(runs)
            for b, (f_hi, f_lo) in enumerate(intervals):
                if f_hi is None or f_lo is None:
                    continue
                p_center = 1.0 / np.sqrt(float(f_hi) * float(f_lo))
                if period_min and p_center < period_min:
                    continue
                if period_max and p_center >= period_max:
                    continue
                bins = (f <= float(f_hi)) & (f >= float(f_lo))
                if not bins.any():
                    continue
                # pooled per-window vectors over the band's bins (saturated
                # NaN bins poison the sums -> excluded via the finite mask)
                hbar = np.nansum(H[:, bins, :], axis=1)  # (2, nt)
                ebar = np.nansum(E[:, bins, :], axis=1)
                e_pred = Zs[b] @ hbar  # (2, nt)
                resid = np.sum(np.abs(ebar - e_pred) ** 2, axis=0)
                power = np.sum(np.abs(ebar) ** 2, axis=0)
                frac = np.where(power > 0, resid / np.maximum(power, 1e-300),
                                np.nan)
                n_tests += 1
                for rid in uniq:
                    m = runs == rid
                    v = frac[m]
                    v = v[np.isfinite(v)]
                    if not v.size:
                        continue
                    per_run_band.setdefault(str(rid), {})[(lv, b)] = \
                        float(np.median(v))
                    if absolute_time:
                        t0 = times[m].min()
                        if str(rid) not in run_start or t0 < run_start[str(rid)]:
                            run_start[str(rid)] = t0
        if not per_run_band:
            if n_intervals_seen:
                ctx.logger.warning(
                    "Residual Sort: the period scope excluded every band — "
                    "returning the payload unfiltered")
                out = {lv: ds.copy() for lv, ds in levels.items()}
                for ds in out.values():
                    ds.attrs["residual_sort_tests"] = 0
                return {"fc": out}
            raise ValueError(
                "Residual Sort: no (level, band) test could be formed — "
                "check that the first pass carried 'band_intervals' and the "
                "period scope is not empty")

        # per band: station-wide median across runs; per run: median of the
        # normalized per-band scores over its tested bands
        band_scores: dict[int, list[float]] = {}
        for rid, bands in per_run_band.items():
            for key, val in bands.items():
                band_scores.setdefault(key, []).append(val)
        ref = {key: float(np.median(v)) for key, v in band_scores.items()}

        rows = []
        for rid in sorted(per_run_band):
            bands = per_run_band[rid]
            ratios = [bands[k] / ref[k] for k in bands
                      if ref.get(k, 0.0) > 0]
            raw = float(np.median(list(bands.values()))) if bands else float("nan")
            if ratios:
                score = float(np.median(ratios))
                dn = float(np.median([ref[k] / bands[k] for k in bands
                                      if bands[k] > 0 and ref.get(k, 0) > 0]))
            else:
                score, dn = float("inf"), float("inf")
            pf = _diurnal_prior_factor(run_start.get(rid, 0), use_prior,
                                       prior_max, local_utc_offset,
                                       absolute_time=absolute_time)
            thr = unexplained_max / pf
            # ONE-SIDED: low unexplained = good fit, never a drop reason
            # (the corpus showed 340 good-fit runs would be wrongly flagged
            # by an inherited down check)
            rows.append({"run_id": rid, "n_tests": len(bands),
                         "unexplained": score, "unexplained_frac": raw,
                         "down": dn, "prior": float(pf),
                         "threshold": float(thr), "dropped": bool(score > thr)})
        dropped_ids = {r["run_id"] for r in rows if r["dropped"]}
        all_ids = set()
        for ds in levels.values():
            all_ids |= set(map(str, np.unique(ds["run_id"].values)))
        kept = all_ids - dropped_ids
        for r in rows:
            if r["dropped"]:
                ctx.logger.warning(
                    "Residual Sort: dropping %s (unexplained %.2f, down %.2f; "
                    "threshold %.2f, prior %.2f)", r["run_id"], r["unexplained"],
                    r["down"], r["threshold"], r["prior"])
        out = {}
        for lv, ds in levels.items():
            mask = np.isin(np.asarray(ds["run_id"].values), list(kept))
            out[lv] = ds.isel(time=mask)
        for ds in out.values():
            ds.attrs["runs_kept"] = ",".join(sorted(kept))
            ds.attrs["runs_dropped"] = ",".join(sorted(dropped_ids))
            ds.attrs["residual_sort_tests"] = n_tests
        path = ctx.artifact_path("residual_sort", ".json")
        path.write_text(json.dumps(rows))
        artifact = ctx.publish_artifact(path, kind="table")
        ctx.progress(
            1.0,
            f"Residual Sort: kept {len(kept)}/{len(rows)} run(s) "
            f"({n_tests} level-band tests)",
        )
        return {"fc": out, "table": artifact}


class EstimateTF(Node):
    type = "aurora.process.estimate_tf"
    title = "Estimate TF"
    tooltip = "Estimate the impedance/tipper transfer function from Fourier coefficients (OLS or robust, optional remote reference)."
    description = (
        "Estimate a transfer function from Fourier coefficients, over every "
        "decimation level of the fc, and merge them into one broadband tf. "
        "Estimator: 'Least Squares' (aurora OLS) or 'Robust' (M-estimator). "
        "With 'Robust', tick 'Remote Reference' to use a second station's fc "
        "(on 'ref fc') as the remote reference; if none is connected the local "
        "channels are reused as a self-reference. No least-squares + "
        "remote-reference path by design. Also outputs per-level tfs ('tf levels')."
    )
    category = "Processing"
    inputs = (
        Input("fc", "fc"),
        Input("ref_fc", "fc", label="ref fc (remote)", required=False),
    )
    outputs = (
        Output("tf", "tf"),
        Output("tf_levels", "tf.level", label="tf levels", description="per decimation level"),
    )
    params = (
        Param("estimator_engine", "select", label="Estimator",
              options=("Robust", "Least Squares"), default="Robust"),
        Param("remote_reference", "boolean", label="Remote Reference",
              default=False,
              description="Robust remote reference (RME_RR). Only applies to "
                          "the 'Robust' estimator.",
              enabled_when=("estimator_engine", ("Robust", "RME", "RME_RR")),
              cascade={True: (("estimator_engine", "Robust"),)}),
        Param("periods_per_octave", "integer", label="Periods per octave",
              default=3, min=1,
              description="Log2 period bands per power-of-2 octave."),
        Param("band_style", "select", label="Band style",
              options=("l2u", "uniform", "manual"), default="l2u",
              description="Band construction scheme (DESIGN.md §14.9). "
                          "'l2u' (default) is the reference-template band "
                          "table with log-uniform window-starvation "
                          "coarsening: the same fixed band template at every "
                          "decimation level (the official EMTF grid's "
                          "structure), window-starved levels rebuilt as a "
                          "coarser uniform grid. 'uniform' is the uniform "
                          "log2-grid partition: overlapping levels estimate "
                          "the same nominal bands and the merge keeps the "
                          "deepest level. 'manual' uses an explicit "
                          "band-center table (see 'Band centers') — the "
                          "apples-to-apples/parity option."),
        Param("band_centers", "string", label="Band centers (Hz)",
              default="", optional=True,
              enabled_when=("band_style", ("manual",)),
              placeholder="e.g. 0.137, 0.109, 0.0856, ...",
              description="Flat list of band-center frequencies (Hz) for "
                          "manual mode — one per band across ALL levels, "
                          "highest frequency first (order normalized); "
                          "edges sit at geometric midpoints between "
                          "adjacent centers. Count must match "
                          "'Bands per level'."),
        Param("bands_per_level", "string", label="Bands per level",
              default="", optional=True,
              enabled_when=("band_style", ("manual",)),
              placeholder='e.g. "6,6,6,6,4,2"',
              description="Number of bands per decimation level (level 0 = "
                          "the highest frequencies); the sum must equal the "
                          "number of band centers. Every level needs >= 2 "
                          "centers (edges are reconstructed from geometric "
                          "midpoints)."),
        Param("min_bins_per_band", "number", label="Min bins per band",
              default=20.0, min=0,
              description="Window-starvation coarsening threshold "
                          "(DESIGN.md §14.9): a level where any band has "
                          "fewer than this many total samples (windows × "
                          "band width) is rebuilt as a coarser uniform grid "
                          "(n−⌊n/3⌋ bands, widened to exceed the threshold, "
                          "log-uniform spacing, no overlaps) — the official "
                          "USArray grid's coarse tail. 0 disables "
                          "coarsening."),
        Param("t_short", "number", label="Shortest period (s)",
              default=0.0, min=0,
              description="Head clip (DESIGN.md §14.9): bands shorter than "
                          "this are not evaluated. 0 starts at the "
                          "template's shortest band (the legacy behavior); "
                          "the official USArray grid clips at ~7.3 s for "
                          "the 1 Hz NIMS band (data-driven — noise exceeds "
                          "signal below 10-20 s)."),
    )

    def execute(self, ctx, fc, ref_fc=None,
                estimator_engine: str = "Robust",
                remote_reference: bool = False,
                periods_per_octave: int = 3,
                band_style: str = "l2u",
                min_bins_per_band: float = 20.0,
                t_short: float = 0.0,
                band_centers: str = "",
                bands_per_level: str = "") -> dict[str, Any]:
        # Accept either a single fc (treated as level 0) or a multi-level
        # container {level: fc}.
        levels = fc if isinstance(fc, dict) else {0: fc}
        sample_fc = next(iter(levels.values()))
        kw = sample_fc.attrs.get("channel_nomenclature") or None
        ins_raw = sample_fc.attrs.get("input_channels")
        outs_raw = sample_fc.attrs.get("output_channels")
        if ins_raw and outs_raw:
            ins = [c.strip() for c in ins_raw.split(",") if c.strip()]
            outs = [c.strip() for c in outs_raw.split(",") if c.strip()]
        else:
            # fc without attrs (e.g. read back from an mth5 archive): classify
            # the channels through the nomenclature instead of assuming names.
            ins, outs, _ = classify_channels(
                [str(c) for c in sample_fc.data_vars], kw
            )
        if not ins or not outs:
            raise ValueError(
                "could not determine input/output channels from the fc — "
                f"data_vars: {list(sample_fc.data_vars)}"
            )
        manual_bands = None
        if band_style == "manual":
            from emptyui_aurora._processing import manual_bands
            if not str(band_centers).strip() or not str(bands_per_level).strip():
                raise ValueError(
                    "band_style='manual' requires both 'band_centers' "
                    "(band-center frequencies in Hz) and 'bands_per_level' "
                    "(bands per decimation level, e.g. \"6,6,6,6,4,2\")")
            manual = manual_bands(band_centers, bands_per_level)
        else:
            manual = None
        per_level = estimate_tf_levels(
            levels, ins, outs,
            estimator_engine=estimator_engine,
            remote_reference=remote_reference,
            periods_per_octave=periods_per_octave,
            ref_levels=ref_fc,
            channel_nomenclature=kw,
            band_style=band_style,
            min_bins_per_band=min_bins_per_band,
            t_short=t_short,
            bands_override=manual,
        )
        periods, Z, Z_err = merge_tf_levels(per_level)
        offered = sum(len(e["periods"]) for e in per_level.values())
        notes = (
            f"estimator={estimator_engine}; remote_reference={remote_reference}; "
            f"levels={len(levels)}; "
            f"periods_per_octave={periods_per_octave}; band_style={band_style}; "
            f"min_bins_per_band={min_bins_per_band}; t_short={t_short}; "
            + (f"manual_bands={len(band_centers.split(','))} centers; "
               if manual else "")
            + f"nomenclature={kw or 'auto'}; "
            + f"band_periods_offered={offered}; merged_periods={len(periods)}"
        )
        tf = build_tf(
            periods, Z, ins, outs, channel_nomenclature=kw, Z_err=Z_err,
            survey_id=sample_fc.attrs.get("survey_id"),
            station_id=sample_fc.attrs.get("station_id"),
            run_id=sample_fc.attrs.get("run_id"),
            latitude=sample_fc.attrs.get("station_latitude"),
            longitude=sample_fc.attrs.get("station_longitude"),
            elevation=sample_fc.attrs.get("station_elevation"),
            start_datetime=sample_fc.attrs.get("start_datetime"),
            processing_notes=notes,
        )
        ctx.progress(
            1.0,
            f"estimated TF ({estimator_engine}) over {len(levels)} level(s), "
            f"{len(periods)}/{offered} merged periods",
        )
        return {"tf": tf, "tf_levels": per_level}

    #: Banding is Estimate TF's own concern (DESIGN §24.9): no inherit from
    #: an upstream Coherence Sort — the sort's per-(bin, window) weights are
    #: band-independent, so there is no banding parameter to auto-fill (and
    #: no duplicated banding param set on the sort node).


class MergeFC(Node):
    type = "aurora.fc.merge"
    title = "Merge FC"
    tooltip = "Concatenate Fourier coefficients from several runs along time, per decimation level."
    description = (
        "Merge Fourier coefficients from several runs of one station along "
        "the time axis (per decimation level) — the FC-database route for "
        "runs that must not be concatenated in the time domain: physically "
        "non-adjacent runs (gaps) or burst-mode acquisition (a merged time-"
        "series cascade cannot decimate across slot boundaries; per-run "
        "cascades are fine and starved deep levels are skipped). Inputs are "
        "matched per level: sample rates must agree (error), window "
        "settings should agree (warning), frequency grids are intersected "
        "and channels missing in some input are dropped (warnings). The "
        "FCs must share one absolute window grid for remote-reference "
        "pairing — compute with Compute FC's 'clock zero' anchor when the "
        "runs were computed separately."
    )
    category = "Processing"
    inputs = (
        Input("fc1", "fc"),
        Input("fc2", "fc", required=False),
        Input("fc3", "fc", required=False),
        Input("fc4", "fc", required=False),
    )
    outputs = (Output("fc", "fc"),)
    params = ()

    def execute(self, ctx, fc1, fc2=None, fc3=None, fc4=None) -> dict[str, Any]:
        merged = merge_fc_containers([fc1, fc2, fc3, fc4])
        n_win = sum(int(ds.sizes["time"]) for ds in merged.values())
        ctx.progress(
            1.0,
            f"merged FC: {len(merged)} level(s), {n_win} window(s)",
        )
        return {"fc": merged}


class MergeTFLevels(Node):
    type = "aurora.process.merge_tf_levels"
    title = "Merge TF Levels"
    tooltip = "Merge per-decimation-level TFs into one broadband transfer function."
    description = (
        "Merge per-decimation-level TFs (the 'tf levels' output of Estimate TF) "
        "into a single broadband transfer function, resolving duplicate periods."
    )
    category = "Processing"
    inputs = (Input("tf_levels", "tf.level", label="tf levels"),)
    outputs = (Output("tf", "tf"),)
    params = ()

    def execute(self, ctx, tf_levels) -> dict[str, Any]:
        first = next(iter(tf_levels.values()))
        ins = list(first["input_channels"])
        outs = list(first["output_channels"])
        kw = first.get("channel_nomenclature") or None
        periods, Z, Z_err = merge_tf_levels(tf_levels)
        tf = build_tf(
            periods, Z, ins, outs, channel_nomenclature=kw, Z_err=Z_err,
            survey_id=first.get("survey_id"),
            station_id=first.get("station_id"),
            run_id=first.get("run_id"),
            latitude=first.get("station_latitude"),
            longitude=first.get("station_longitude"),
            elevation=first.get("station_elevation"),
            start_datetime=first.get("start_datetime"),
        )
        ctx.progress(1.0, f"merged {len(tf_levels)} TF levels -> {len(periods)} periods")
        return {"tf": tf}


