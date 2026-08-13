"""aurora pipeline nodes: ComputeFC (TS→FC), EstimateTF (FC→TF), MergeTFLevels.

ComputeFC/EstimateTF split the aurora pipeline at the FC boundary (DESIGN.md
§3, history §14). FC is passed in-memory between nodes and, when ``persist`` is on,
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
    _MERGE_PROVENANCE_ATTRS,
    _centre_to_start_s,
    build_tf,
    coherence_sort_fcs,
    compute_fc_levels,
    estimate_tf_levels,
    merge_fc_containers,
    merge_tf_levels,
)

_NOMENCLATURE_OPTIONS = ("auto",) + KEYWORDS


def _attr_float(group_obj, key: str):
    """A float from an mth5 group's HDF5 attrs (dotted key), or None.

    The group ``.metadata`` properties rebuild the FULL hierarchy through
    pydantic on every access (~30-40 s on a burst archive, history §24.8) — the
    attrs carry the same scalars (location.*, time_period.*, ...) for free.
    """
    try:
        value = group_obj.hdf5_group.attrs[key]
        return float(value) if value is not None else None
    except (KeyError, TypeError, ValueError):
        return None


def _station_location(data):
    """(latitude, longitude, elevation) of the run's station, or (None,)*3."""
    try:
        loc = data.station_metadata.location
        return float(loc.latitude), float(loc.longitude), float(loc.elevation)
    except Exception:  # noqa: BLE001 — core TimeSeries carries no location
        return None, None, None


def _station_identity_attrs(data, survey: str, station: str, run: str) -> dict[str, Any]:
    """Primitive station-identity attrs to carry on an fc (DESIGN.md §10.6).

    The ts.run payload (mth5 RunTS) exposes survey/station/run metadata and the
    station location; only primitives are stored (xarray attrs must survive
    persistence), and missing values are omitted so downstream nodes can tell
    "unknown" apart from a default.
    """
    from emptyui_mth5.fc import FC_IDENTITY_KEYS

    attrs: dict[str, Any] = dict(
        zip(FC_IDENTITY_KEYS[:3], (survey, station, run)))
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


def _merge_provenance_attrs(data, logger) -> dict[str, Any]:
    """Merge Runs' gap provenance (M4 #2) carried by a ts.run payload.

    Copies ``merged_gap_intervals`` / ``merged_source_runs`` /
    ``merged_gap_method`` so the fc keeps the record of which intervals a
    merge fabricated — the remote-reference pairing warns from exactly these
    attrs. A payload carrying none contributes nothing (probe via ``dir()``
    like the rest of this module: mt_timeseries turns a missing attribute
    into NameError + an ERROR line); a payload that CAN carry them but
    cannot be read is logged — losing the provenance in silence is the very
    failure this copy prevents (E4: everything logs).
    """
    if data is None:
        return {}
    try:
        if "dataset" not in dir(data):
            return {}
        attrs = getattr(data.dataset, "attrs", None) or {}
        return {k: attrs[k] for k in _MERGE_PROVENANCE_ATTRS
                if attrs.get(k) is not None}
    except Exception as exc:  # noqa: BLE001 — duck-typed payloads
        logger.warning(
            "merge provenance: %s attrs could not be read (%s) — its "
            "merged_gap_intervals will not ride the fc, so the "
            "remote-reference gap warning stays silent for it",
            type(data).__name__, exc)
        return {}


def _tile_sessions(entries, session_seconds: float):
    """Tile runs into fixed-length QC sessions (§9.4) — the universal
    session-label mechanism, extracted from the retired Build FC
    (history §24.13).

    ``entries``: ``(rid, rate, start, end)`` with tz-aware or naive
    pandas-compatible bounds. Returns units
    ``(unit_id, rate, u_start, u_end, rid, lo, hi)`` — ``lo``/``hi`` are
    the exclusive-ish slice bounds (``hi`` trimmed half a sample so the
    boundary sample is not doubled by the inclusive crop). Sessions never
    cross run boundaries; the summary end is the last SAMPLE, so the
    tiled interval is ``[start, end + one sample)``.
    """
    import pandas as pd

    units = []
    for rid, rate, start, end in entries:
        half = pd.to_timedelta(0.5 / rate, unit="s")
        run_hi = end + pd.to_timedelta(1.0 / rate, unit="s")
        k = 0
        t = start
        while t < run_hi:
            t2 = min(t + pd.to_timedelta(session_seconds, unit="s"), run_hi)
            units.append((f"{rid}_s{k + 1:04d}", rate, t, t2, rid, t, t2 - half))
            t = t2
            k += 1
    return units


def _contiguous_spans(units):
    """Group units into cascade spans — the retired Build FC's criterion,
    verbatim: maximal contiguous same-rate spans (adjacency within two
    samples), deepest rate first; units without a parseable period each
    form their own span (they cannot be adjacency-tested).

    One guard beyond the original: OVERLAPPING units never join a span.
    Aligned collections crop every member to the same overlap — joining
    them would concatenate parallel views of the same samples and
    fabricate data. The original only tested the gap upper bound.
    """
    u_start = {u[0]: u[2] for u in units}
    u_end = {u[0]: u[3] for u in units}
    spans: list[list[str]] = []
    per_rate: dict[float, list] = {}
    for u in units:
        per_rate.setdefault(u[1], []).append(u[0])
    for rate in sorted(per_rate, reverse=True):
        datable = [r for r in per_rate[rate] if u_start[r] is not None]
        for r in per_rate[rate]:
            if u_start[r] is None:
                spans.append([r])
        ordered = sorted(datable, key=lambda r: u_start[r])
        cur: list[str] = []
        prev_end = None
        for uid in ordered:
            start, end = u_start[uid], u_end[uid]
            gap = (start - prev_end).total_seconds() if prev_end is not None else None
            if prev_end is not None and (
                    gap > 2.0 / rate or gap < 0.0):
                spans.append(cur)
                cur = []
            cur.append(uid)
            prev_end = end
        if cur:
            spans.append(cur)
    return spans


def _concat_span_channels(parts, span_id: str, logger) -> dict:
    """Concatenate a span's member channel dicts along time — the retired
    Build FC's union, verbatim: the ``saturated`` aux QC mask is
    zero-padded for members without it (no saturation there); a member
    missing a REAL channel is dropped from the span with a loud warning
    (fabricating an all-zero series would silently corrupt the estimate;
    ``merge_fc_containers`` drops such channels per run the same way).

    ``parts``: ``[(uid, {name: DataArray}), ...]`` in span order.
    """
    import xarray as xr

    keys = list(parts[0][1])
    for _, mv in parts[1:]:
        for name in mv:
            if name not in keys:
                keys.append(name)
    aux = {"saturated"}
    real_missing = [
        n for n in keys
        if n not in aux and any(n not in mv for _, mv in parts)]
    if real_missing:
        logger.warning(
            "span %s: run(s) lack channel(s) %s — dropped from "
            "the cascaded span (fabricating zeros would corrupt "
            "the estimate; process the runs separately or fix "
            "the ingestion)", span_id, real_missing)
        keys = [n for n in keys if n not in real_missing]
    data_vars = {}
    for name in keys:
        chans = []
        for _, mv in parts:
            if name in mv:
                chans.append(mv[name])
            else:
                # zeros_like copies the template channel's attrs
                # (incl. `filters` refs) — a fabricated channel
                # must not carry them (RunTS.calibrate divides
                # every channel by its response chain)
                pad = xr.zeros_like(next(iter(mv.values())))
                if name in aux:
                    pad.attrs = {"type": "auxiliary", "units": "1"}
                chans.append(pad)
        data_vars[name] = xr.concat(chans, dim="time")
    return data_vars


def _stamp_span_run_ids(levels, span, member_lens, sample_rate: float,
                        span_t0) -> None:
    """Per-window run identity at session granularity (history §24.13) —
    the retired Build FC's attribution, verbatim.

    A single-unit span stamps its unit id everywhere. A multi-unit span
    attributes each window to the unit its START falls in: the fc time
    coordinate is the scipy window CENTRE, so it is shifted back half a
    window of THIS level before the membership test (§10.4 — a straddling
    window follows the session its start falls in; Select Runs drops
    rejected sessions downstream, conservatively including the overlap
    neighbors that picked up their samples through the cascade).

    ``member_lens``: sample counts per member, span order; ``span_t0``:
    the span's first sample time (naive datetime64).
    """
    import numpy as np
    import pandas as pd

    if len(span) == 1:
        for level, fc in levels.items():
            levels[level] = fc.assign_coords(run_id=(
                "time", np.full(int(fc.sizes["time"]), span[0])))
        return
    bnds = np.concatenate(([0.0], np.cumsum(member_lens)[:-1])) / sample_rate
    for level, fc in levels.items():
        t = np.asarray(fc["time"].values)
        half_win = (float(fc.attrs.get("num_samples_window", 0)) / 2.0
                    / float(fc.attrs.get("sample_rate", 0) or sample_rate))
        if np.issubdtype(t.dtype, np.datetime64):
            t0 = span_t0
            if isinstance(t0, pd.Timestamp) and t0.tzinfo is not None:
                t0 = t0.tz_convert("UTC").tz_localize(None)
            t0 = np.asarray(t0, dtype="datetime64[ns]")
            off = ((t - t0) / np.timedelta64(1, "s")).astype(float) - half_win
        else:
            off = t.astype(float) - half_win
        k = np.minimum(
            np.searchsorted(bnds[1:], off, side="right"),
            len(span) - 1)
        levels[level] = fc.assign_coords(run_id=(
            "time", np.asarray(span)[k]))


class ComputeFC(Node):
    type = "aurora.process.compute_fc"
    # raw-data/FC-scale work: a generous own budget (issues.md M2)
    timeout = 3600.0
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
    inputs = (Input("data", "ts.run", label="time series",
                    accepts=("ts.runs",), collection_ok=True),)
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
        Param("decimation_levels", "integer", label="Decimation levels", default=4,
              min=0,
              description="Number of cascading decimations (0 = none, native rate only)"),
        Param("factor", "integer", label="Decimation factor", default=4, min=2),
        Param("window_samples", "integer", label="Window (samples)", default=256, min=8,
              description="Constant window length in samples across all levels"),
        Param("overlap", "number", default=0.5, min=0.0, max=0.99,
              description="Window overlap fraction (< 1)"),
        Param("window_type", "select", options=("hamming", "hanning", "boxcar"),
              default="boxcar",
              description="boxcar (default) maximizes effective dof — "
                          "pair with prewhitening (first difference) at "
                          "white-noise-limited scales; hanning/hamming "
                          "suppress line-spectra leakage (day-scale "
                          "windows)"),
        Param("window_type_deep", "select", label="Deep-level taper",
              default="same as shallow",
              options=("same as shallow", "hamming", "hanning", "boxcar"),
              optional=True,
              description="Taper override for the deep decimation levels "
                          "(see 'from level'): zero-endpoint tapers (hamming/"
                          "hanning) suppress diurnal-line leakage through the "
                          "day-scale windows; boxcar maximizes dof. 'Same as "
                          "shallow' keeps the base taper."),
        Param("deep_from_level", "integer", label="Deep taper from level",
              default=None, optional=True, min=0,
              description="Decimation level from which 'Deep-level taper' "
                          "applies (e.g. 3 with window 256 = windows longer "
                          "than ~1 h); empty = never (base taper everywhere)"),
        Param("clock_zero", "string", label="Clock zero",
              default="", optional=True,
              placeholder='"epoch" or ISO datetime; empty = off',
              description="Anchor windows on a common absolute grid "
                          "(labels are the actual sample times, within half "
                          "a sample of t0 + k·step; head partial window "
                          "dropped): "
                          "'epoch' (Unix epoch — automatically common to "
                          "all stations/runs) or an explicit ISO datetime. "
                          "Required for remote-reference pairing across "
                          "separately-computed per-run FCs (Merge FC, burst "
                          "mode); empty anchors at each run's start (legacy). "
                          "Needs an absolute start time."),
        Param("prewhitening", "select", label="Prewhitening",
              default="first difference",
              options=("none", "first difference"),
              description="Leakage-reduction prewhitening before the STFT "
                          "(default: 'first difference' — aurora/EMTF-style, "
                          "differences each channel in time and recolors the "
                          "FCs by the exact inverse filter; required to reap "
                          "boxcar's full dof). 'None' = off (the legacy "
                          "emptyUI behavior)."),
        Param("per_window_detrend", "select", label="Per-window detrend",
              default="none", options=("none", "linear"),
              description="Detrend each STFT window (linear) before "
                          "tapering — keeps window-level trends out of the "
                          "low-frequency bins (aurora's per-window default); "
                          "'none' (default) = global Detrend node only."),
        Param("reject_saturated", "boolean", label="Reject saturated windows",
              default=True,
              description="Zero-weight (NaN) every FC window whose flagged "
                          "sample fraction exceeds 'saturated max' (MTU-5A "
                          "block tags byte 15, history §24.12 QC). No effect on "
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
        Param("session_hours", "number", label="Session length (h)",
              default=0.0, min=0, optional=True,
              description="Cut every run of a collection into fixed-length "
                          "sessions (0 = off) before cascading — the "
                          "universal session-label mechanism (§9.4): "
                          "Select Runs / Residual Sort score and reject at "
                          "session granularity, so a noisy hour no longer "
                          "condemns an entire acquisition day (LEMI/NIMS "
                          "runs are whole days). The statistical unit is "
                          "FC windows per session: ~1 h at 1 Hz with "
                          "window 256 gives ~28 windows (the validated "
                          "MTU-5A stream recipe). Sessions are labelled "
                          "'{run id}_sNNNN' and never cross run "
                          "boundaries; a single ts.run input is treated as "
                          "a one-run collection. Use with 'Regroup "
                          "contiguous runs' to keep the deep levels "
                          "(per-session cascades starve them)."),
        Param("regroup_contiguous", "boolean",
              label="Regroup contiguous runs", default=False,
              description="Cascade maximal contiguous spans of same-rate "
                          "runs as ONE payload (history §24.13): hourly sessions "
                          "cannot form a level below ~0.23 Hz per slice, but "
                          "the underlying stream is unbroken — cascading the "
                          "kept contiguous span restores the long-period "
                          "tail. Built from ALL sessions: hour-level QC "
                          "lives in Select Runs, so spans never depend on "
                          "QC outcomes and each window is stamped with the "
                          "session it came from."),
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
                decimation_levels: int = 4, factor: int = 4,
                window_samples: int = 256, overlap: float = 0.5,
                window_type: str = "boxcar",
                window_type_deep: str = "", deep_from_level=None,
                clock_zero: str = "",
                prewhitening: str = "first difference",
                per_window_detrend: str = "",
                reject_saturated: bool = True,
                saturated_max: float = 0.01,
                session_hours: float = 0.0,
                regroup_contiguous: bool = False,
                persist: bool = True, survey: str = "", station: str = "",
                run: str = "") -> dict[str, Any]:
        if data is None:
            raise ValueError(
                "no time series input — check the upstream node ran and the "
                "edge targets this port")
        # Channel names resolve inside compute_fc_levels (empty fields = auto
        # from nomenclature + data; tipper included when the vertical channel
        # is present, missing outputs dropped with a warning).
        # NB: mt_timeseries objects convert missing-attribute AttributeError
        # into NameError AND log an ERROR line — probe via dir() (no
        # __getattr__ invocation) instead of try/getattr
        names = dir(data)
        extents = data.extents if "extents" in names else None
        is_collection = isinstance(extents, list) and "runs" in names
        sessioning = (
            (session_hours is not None and float(session_hours) > 0.0)
            or bool(regroup_contiguous))
        if not is_collection and sessioning:
            # §9.4: session QC works on the classic single-run flow too —
            # a lone ts.run is a one-run collection (its identity resolved
            # the same way the single route would)
            data = self._wrap_single_collection(
                ctx, data, survey, station, run)
            extents = data.extents
            is_collection = True
        if is_collection:
            try:
                return self._execute_collection(
                    ctx, data, input_channels, output_channels,
                    channel_nomenclature, magnetic_unit, e_dipole_length_m,
                    decimation_levels, factor, window_samples, overlap,
                    window_type, window_type_deep, deep_from_level, clock_zero,
                    prewhitening, per_window_detrend, reject_saturated,
                    saturated_max, session_hours, regroup_contiguous,
                    persist, survey, station, run)
            finally:
                # M3: a failure while persisting a member must not leak the
                # write-open workspace archive (the success path closes too)
                if persist:
                    from emptyui_mth5.persist import close_workspace_mth5
                    close_workspace_mth5(ctx)
        # L-A4 dipole fallback: the explicit param wins; otherwise the attr
        # Extract Run stamps from the archive's electric-channel metadata
        if e_dipole_length_m is None and "dataset" in dir(data):
            try:
                dl = float((data.dataset.attrs or {}).get(
                    "e_dipole_length_m", 0.0) or 0.0)
            except Exception:  # noqa: BLE001 — duck payloads
                dl = 0.0
            if dl > 0:
                e_dipole_length_m = dl
                ctx.logger.info(
                    "Compute FC: E dipole length %.6g m from the payload's "
                    "archived channel metadata", dl)
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
        from emptyui_mth5.persist import (
            close_workspace_mth5,
            get_workspace_mth5,
            persist_fc_level,
            runts_context,
        )

        # Station identity travels with the fc (attrs) so EstimateTF can attach
        # it to the TF — the single point of truth for plot titles and export
        # headers (DESIGN.md §10.6).
        surv, stat, run_id = runts_context(data, survey, station, run,
                                           logger=ctx.logger)
        identity = _station_identity_attrs(data, surv, stat, run_id)
        # M4 #2: Merge Runs' gap provenance (interpolated intervals) travels
        # with the fc so the remote-reference pairing can warn when local
        # windows fall inside fabricated remote samples.
        identity.update(_merge_provenance_attrs(data, ctx.logger))
        for level, fc in levels.items():
            fc.attrs.update(identity)
        if persist:
            wm = get_workspace_mth5(ctx)
            try:
                for level, fc in levels.items():
                    if "datetime64" in str(fc.time.dtype):
                        persist_fc_level(ctx, fc, level,
                                         fc.attrs["sample_rate"],
                                         survey=surv, station=stat,
                                         run=run_id, m=wm)
                    else:
                        ctx.logger.warning(
                            f"FC level {level} has no absolute (datetime64) time axis — "
                            "skipping persistence."
                        )
                        break
            finally:
                close_workspace_mth5(ctx)
        # per-window run identity (§9.3 universality): Select Runs / Residual
        # Sort key on the per-window 'run_id' coordinate — the classic
        # single-run flow (Extract -> Compute FC) now carries it too, so
        # session QC works without routing through Build FC. Whole runs get
        # their own id (all-or-nothing QC — the giant-run problem is what
        # Build FC's session_hours solves); stamped AFTER persistence so the
        # verified mth5 round trip is untouched.
        if run_id:
            import numpy as np

            for level, fc in levels.items():
                levels[level] = fc.assign_coords(run_id=(
                    "time", np.full(int(fc.sizes["time"]), run_id)))
        summary = ", ".join(
            f"L{lv}:{fc.sizes['frequency']}f×{fc.sizes['time']}w" for lv, fc in levels.items()
        )
        ctx.progress(1.0, f"computed FC levels [{summary}]")
        return {"fc": levels}

    @staticmethod
    def _wrap_single_collection(ctx, data, survey: str, station: str,
                                run: str):
        """Wrap one ts.run as a one-member collection so session QC
        (§9.4) works on the classic single-run flow — identity resolved
        exactly as the single route would resolve it."""
        from emptyui_mth5.persist import runts_context
        from emptyui_mth5.runs import RunCollection

        surv, stat, rid = runts_context(data, survey, station, run,
                                        logger=ctx.logger)
        return RunCollection(
            runs=[data],
            extents=[{"survey": surv, "station": stat, "run": rid,
                      "start": "", "end": "",
                      "sample_rate": float(getattr(data, "sample_rate", 0.0)
                                           or 0.0)}])

    def _execute_collection(
        self, ctx, collection, input_channels: str, output_channels: str,
        channel_nomenclature: str, magnetic_unit: str, e_dipole_length_m,
        decimation_levels: int, factor: int, window_samples: int,
        overlap: float, window_type: str, window_type_deep: str,
        deep_from_level, clock_zero: str, prewhitening: str,
        per_window_detrend: str, reject_saturated: bool,
        saturated_max: float, session_hours: float,
        regroup_contiguous: bool, persist: bool, survey: str, station: str,
        run: str,
    ) -> dict[str, Any]:
        """Per-run (or per-session) cascades merged along time, per level —
        the route the retired Build FC owned (DESIGN §9.3): 'time range'
        collections cascade member-wise, with optional §9.4 session
        tiling and contiguous-span regrouping."""
        import numpy as np
        import pandas as pd
        import xarray as xr
        from emptyui_mth5.persist import (
            get_workspace_mth5,
            persist_fc_level,
        )
        from emptyui_mth5.runs import _slice_window

        members = list(collection.runs)
        extents = list(collection.extents)
        if not members:
            raise ValueError(
                "run collection is empty — Extract Run's 'time range' "
                "selection matched no runs")
        if len(members) > 1 and not str(clock_zero).strip():
            ctx.logger.warning(
                "%d runs without clock_zero anchoring: each run's window "
                "grid anchors at its own start — remote-reference pairing "
                "across separately-computed runs needs clock_zero='epoch' "
                "(or an explicit ISO datetime)", len(members))
        # per-family decimation trim (history §24.8): on a mixed-rate collection the
        # lowest-rate family carries the full 'decimation_levels' cascade;
        # higher-rate families decimate only until their rate would reach
        # the next family's native rate.
        rates = [float(getattr(m, "sample_rate", 0.0) or 0.0)
                 for m in members]
        family_rates = sorted(set(rates))
        coll_prov = _merge_provenance_attrs(collection, ctx.logger)
        # L-A4 dipole fallback: the explicit param wins; otherwise the
        # attr Extract Run stamps from the archive's electric-channel
        # metadata rides the payload
        dipole = e_dipole_length_m or None
        if dipole is None:
            for m in members:
                try:
                    dl = float((m.dataset.attrs or {}).get(
                        "e_dipole_length_m", 0.0) or 0.0)
                except Exception:  # noqa: BLE001 — duck payloads
                    dl = 0.0
                if dl > 0:
                    dipole = dl
                    ctx.logger.info(
                        "Compute FC: E dipole length %.6g m from the "
                        "payload's archived channel metadata", dl)
                    break

        # member spans in absolute time: the payload's own time axis is the
        # ground truth of the data being cascaded; the extent rows are the
        # fallback. Units without an absolute axis pass through whole.
        def _member_bounds(i):
            m = members[i]
            try:
                t = np.asarray(m.dataset["time"].values)
            except Exception:  # noqa: BLE001 — duck payloads
                return None, None
            if not np.issubdtype(t.dtype, np.datetime64) or not len(t):
                return None, None
            # the LAST SAMPLE time (inclusive — the convention _tile_sessions
            # inherits from the run summary, which adds one sample itself)
            start = pd.Timestamp(t[0])
            end = pd.Timestamp(t[-1])
            if start.tzinfo is None:
                start = start.tz_localize("UTC")
            if end.tzinfo is None:
                end = end.tz_localize("UTC")
            return start, end

        session_seconds = float(session_hours or 0.0) * 3600.0
        units: list[tuple] = []
        for i in range(len(members)):
            ext = extents[i] if i < len(extents) else {}
            rid = str(ext.get("run", "") or f"member{i + 1}")
            m_start, m_end = _member_bounds(i)
            if m_start is None:
                if session_seconds > 0:
                    ctx.logger.warning(
                        "run %s has no absolute time axis — it cannot be "
                        "session-split and passes through whole", rid)
                units.append((rid, rates[i], None, None, rid, i, None, None))
                continue
            if session_seconds <= 0:
                units.append((rid, rates[i], m_start, m_end, rid, i,
                              None, None))
            else:
                for (uid, r, us, ue, _rid, lo, hi) in _tile_sessions(
                        [(rid, rates[i], m_start, m_end)], session_seconds):
                    units.append((uid, r, us, ue, rid, i, lo, hi))
        spans = (
            _contiguous_spans(units) if regroup_contiguous
            else [[u[0]] for u in units])
        unit_by_id = {u[0]: u for u in units}
        # one workspace handle for the whole execution (history §24.8): the
        # per-level open/close is quadratic — every close re-summarizes all
        # summary tables (the fc summary recursively walks every persisted
        # FC level)
        wm = get_workspace_mth5(ctx) if persist else None
        containers = []
        for i, span in enumerate(spans, 1):
            span_id = (span[0] if len(span) == 1
                       else f"sr{unit_by_id[span[0]][1]:g}_span{i:04d}")
            parts = []
            span_t0 = None
            for uid in span:
                _uid, rate, _u_start, _u_end, rid, mi, lo, hi = unit_by_id[uid]
                m = members[mi]
                if lo is None:
                    channels = {name: m.dataset[name]
                                for name in m.dataset.data_vars}
                else:
                    channels = _slice_window(
                        {name: m.dataset[name]
                         for name in m.dataset.data_vars}, lo, hi)
                    sample = next(iter(channels.values()))
                    if not sample.sizes.get("time", 0):
                        continue  # empty slice — nothing to contribute
                first = next(iter(channels.values()))
                t0 = np.asarray(first["time"].values)[0]
                if span_t0 is None:
                    span_t0 = t0
                parts.append((uid, channels, rate, t0))
            if not parts:
                raise ValueError(
                    f"span {span_id} is empty — its runs carry no samples")
            sample_rate = parts[0][2]
            if len(parts) == 1 and unit_by_id[span[0]][6] is None:
                # the whole-member fast path: the payload itself (its
                # filter registry and dataset attrs ride along untouched)
                run_ts = members[unit_by_id[span[0]][5]]
            else:
                data_vars = (
                    parts[0][1] if len(parts) == 1
                    else _concat_span_channels(
                        [(uid, ch) for uid, ch, _, _ in parts],
                        span_id, ctx.logger))
                filters = {}
                first_m = members[unit_by_id[span[0]][5]]
                if "filters" in dir(first_m):
                    filters = getattr(first_m, "filters", {}) or {}
                from types import SimpleNamespace

                run_ts = SimpleNamespace(
                    dataset=xr.Dataset(data_vars),
                    sample_rate=sample_rate,
                    start=span_t0,
                    filters=filters,
                )
            r = sample_rate
            dec_of_span = _family_decimation(
                r, family_rates, decimation_levels, factor)
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
            # station identity travels in the attrs (DESIGN.md §10.6) —
            # the first member's extent row names the survey/station; the
            # run key is the span id (the session id when tiled — the QC
            # label the per-window run_id coordinate carries)
            ext0 = extents[unit_by_id[span[0]][5]] if extents else {}
            surv = str(ext0.get("survey", "")) or survey
            stat = str(ext0.get("station", "")) or station
            identity = _station_identity_attrs(
                members[unit_by_id[span[0]][5]], surv, stat, span_id)
            # M4 #2: every contributing member's gap provenance (Align
            # Runs carried it through the crop), plus the collection's —
            # the fc must keep every interpolated interval, or the
            # remote-reference pairing stays silent about fabricated
            # samples.
            prov = dict(coll_prov)
            for uid in span:
                prov.update(_merge_provenance_attrs(
                    members[unit_by_id[uid][5]], ctx.logger))
            identity.update(prov)
            for level, fc in levels.items():
                fc.attrs.update(identity)
            if persist:
                for level, fc in levels.items():
                    if "datetime64" in str(fc.time.dtype):
                        persist_fc_level(
                            ctx, fc, level, fc.attrs["sample_rate"],
                            survey=surv, station=stat, run=span_id, m=wm)
                    else:
                        ctx.logger.warning(
                            "run %s level %d has no absolute time axis — "
                            "skipping persistence", span_id, level)
                        break
            # per-window run identity (§9.3/§9.4): QC nodes group windows by
            # run — session ids when tiled, the run id otherwise
            _stamp_span_run_ids(
                levels, span,
                [len(next(iter(ch.values()))) for _, ch, _, _ in parts],
                sample_rate, span_t0)
            containers.append(levels)
            ctx.progress(
                i / len(spans),
                f"FC {i}/{len(spans)}: {span_id} "
                f"({len(levels)} level(s), {len(span)} run(s))")
        merged = merge_fc_containers(containers)
        # the native rate families travel with the database (history §20.1): the
        # plot nodes' 'auto' sample-rate selection means the lowest FAMILY
        # (the continuous stream), not the deepest cascade tier — cascade
        # tiers extend below the stream's native rate. The database is a
        # plain dict; the attr rides every tier's Dataset attrs (so any
        # tier lookup finds it).
        for ds in merged.values():
            ds.attrs["family_rates"] = sorted(family_rates)
        n_win = sum(int(ds.sizes["time"]) for ds in merged.values())
        ctx.progress(
            1.0,
            f"run collection: {len(members)} run(s) in {len(spans)} "
            f"span(s) -> {len(merged)} level(s), {n_win} window(s)")
        return {"fc": merged}


def _diurnal_prior_factor(start, use_prior: bool, prior_max: float,
                          local_utc_offset: float,
                          absolute_time: bool = True) -> float:
    """Soft diurnal prior for run QC (history §24.13): the natural-signal envelope
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


def _family_decimation(r: float, family_rates: list[float],
                       decimation_levels: int, factor: int) -> int:
    """Per-family decimation trim (Build FC's rule, history §24.8).

    On a mixed-rate corpus the lowest-rate family carries the FULL
    ``decimation_levels`` cascade; a higher-rate family decimates only until
    its rate would reach the next family's native rate — the lower family's
    cascade covers those frequencies, and same-rate levels computed from
    different families would double-cover the same acquisition windows.
    E.g. 2400/150/15 Hz with factor 4: 2400 Hz slots decimate once (-> 600),
    150 Hz slots once (-> 37.5), the 15 Hz stream takes the full parameter.
    (M5: was computed inline at two sites that could drift apart.)
    """
    import numpy as np

    lower = [x for x in family_rates if x < r * (1 - 1e-9)]
    if not lower:
        return decimation_levels
    d_max = max(0, int(np.ceil(np.log2(r / max(lower))
                               / np.log2(factor))) - 1)
    return min(decimation_levels, d_max)


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
    # raw-data/FC-scale work: a generous own budget (issues.md M2)
    timeout = 3600.0
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
        "TF's concern (DESIGN.md §10.3)."
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
              default=1, min=0.0, max=1.0,
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
        Param("pairing", "select", label="Remote pairing",
              options=("intersect", "strict"), default="intersect",
              optional=True,
              description="How local and remote FC windows pair before "
                          "sorting (§9.3): 'intersect' (default) matches "
                          "window times, unmatched rows drop with a loud "
                          "report; 'strict' requires identical time "
                          "grids. Frequency grids stay strict either way."),
        Param("pair_tolerance_s", "number", label="Pair tolerance (s)",
              default=0.0, min=0, optional=True,
              description="Matching tolerance for window times (centre "
                          "labels, L11) "
                          "(0 = exact, or one sample of the level for "
                          "sub-sample record offsets). Raise for GPS-stray "
                          "clocks."),
    )

    def execute(self, ctx, fc, ref_fc=None,
                coherence_min: float = 0.25, coherence_max: float = 1.0,
                mode: str = "weight",
                apply_below_period: float = 0.0,
                ensemble_windows: int = 16,
                smooth_bins: int = 0,
                pairing: str = "intersect",
                pair_tolerance_s: float = 0.0) -> dict[str, Any]:
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
            pairing=pairing, tol_s=pair_tolerance_s,
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
    # raw-data/FC-scale work: a generous own budget (issues.md M2)
    timeout = 3600.0
    title = "Select Runs"
    tooltip = "Drop noise-dominated runs before TF estimation (run-level QC)."
    description = (
        "Run-level QC (DESIGN.md §10.4, after Weckmann et al. 2005): score "
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
                          "(history §24.13 sweep); lower percentiles select the "
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
                          "= UTC+8, history §24.11)."),
    )

    def execute(self, ctx, fc, period_min: float = 1.0,
                period_max: float = 10.0, deviation_max: float = 2.0,
                dead_max: float = 2.0, reference_percentile: float = 50.0,
                use_prior: bool = True, prior_max: float = 2.0,
                local_utc_offset: float = 8.0) -> dict[str, Any]:
        import json

        import numpy as np

        # the reference band converts to frequencies (1/period): a zero
        # bound would raise ZeroDivisionError mid-scoring (review 2026-09-24)
        if not period_min > 0 or not period_max > 0:
            raise ValueError(
                f"reference band bounds must be positive periods "
                f"(got {period_min}-{period_max} s)")
        levels = fc if isinstance(fc, dict) else {0: fc}
        sample = next(iter(levels.values()))
        if "run_id" not in sample.coords:
            raise ValueError(
                "Select Runs needs an fc payload with a per-window "
                "'run_id' coordinate — rebuild it with Build FC")
        ins, outs, _ = _fc_io_channels(sample)
        # L-A9: a magnetic-only fc (no electric outputs — or vice versa)
        # raised a raw numpy stack error; name the channels
        if not ins or not outs:
            raise ValueError(
                "Select Runs needs an fc with both input (horizontal "
                f"magnetic) and output channels — got inputs={ins}, "
                f"outputs={outs}; rebuild it with Build FC")
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
                    # L11 audit: the labels are window CENTRES — the run
                    # starts half a window before its first label, and the
                    # hour-of-day the prior scores is the DATA's
                    t0 = (times[m].min() - np.timedelta64(
                        round(_centre_to_start_s(ds)), "s"))
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
            # L-A3: a run with NO finite score (every level's in-band
            # medians NaN — e.g. all its windows coherence-dropped) has no
            # evidence of badness — pass it through per the contract; the
            # old inf-vs-threshold comparison silently DROPPED it
            unscored = not up_d and not dn_d
            up = float(np.median(up_d)) if up_d else float("inf")
            dn = float(np.median(dn_d)) if dn_d else float("inf")
            pf = _diurnal_prior_factor(run_start.get(rid, 0), use_prior,
                                       prior_max, local_utc_offset,
                                       absolute_time=absolute_time)
            thr = deviation_max / pf
            dropped = (up > thr or dn > dead_max) and not unscored
            rows.append({"run_id": rid, "n_levels": len(per_run_up[rid]),
                         "deviation": max(up, dn), "up": up, "down": dn,
                         "prior": float(pf),
                         "threshold": float(thr), "dropped": bool(dropped)})
            if unscored:
                ctx.logger.warning(
                    "Select Runs: run '%s' has no finite in-band score "
                    "(all its windows dropped or NaN) — passed through "
                    "unscored", rid)
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
        # cascade — drop it too. Start-time attribution (§10.4) already puts
        # right straddlers in the rejected session; only left neighbors need
        # the extra cut: windows whose START lies within one window duration
        # before the rejected session's first window start. The duration is
        # per level — deep levels have longer windows and pick up
        # correspondingly more rejected-session samples (a single shallow
        # level's duration under-covers the cascade).
        # L11 audit: the labels are window CENTRES, but this comparison is
        # same-level centre-vs-centre — the half-window shifts CANCEL, so
        # the label deltas are exactly the start deltas. Do NOT "fix" this
        # into a double shift.
        n_extra = 0
        out = {}
        for lv, ds in levels.items():
            runs_arr = np.asarray(ds["run_id"].values)
            mask = np.isin(runs_arr, list(kept))
            dur = None
            try:
                dur = (float(ds.attrs["num_samples_window"])
                       / float(ds.attrs["sample_rate"]))
            except (KeyError, TypeError, ZeroDivisionError):
                dur = None
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
            # ds.copy() first: isel shares the source's attrs dict, so the
            # QC stamps below would mutate the INPUT payload (§7.1
            # immutability contract — the input may be cached and shared
            # with other fan-out branches)
            out[lv] = ds.copy().isel(time=mask)
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
    # raw-data/FC-scale work: a generous own budget (issues.md M2)
    timeout = 3600.0
    title = "Residual Sort"
    tooltip = "Drop runs whose windows do not fit the first-pass TF model."
    description = (
        "Residual-based two-pass selection (DESIGN.md §10.4, the orthogonal "
        "complement of power and coherence screening): a first-pass TF "
        "(Estimate TF's 'tf levels') predicts each FC window's outputs; "
        "RUNS whose pooled unexplained-power fraction is excessive are "
        "dropped (all their windows) and the TF is re-estimated. The score "
        "is per RUN "
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
        # L-A9: a magnetic-only fc (no electric outputs — or vice versa)
        # raised a raw numpy stack error; name the channels
        if not ins or not outs:
            raise ValueError(
                "Residual Sort needs an fc with both input (horizontal "
                f"magnetic) and output channels — got inputs={ins}, "
                f"outputs={outs}; rebuild it with Build FC")
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
                        # L11 audit: labels are window centres — same
                        # correction as Select Runs' prior start
                        t0 = (times[m].min() - np.timedelta64(
                            round(_centre_to_start_s(ds)), "s"))
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
            # copy() first — isel shares attrs with the source (§7.1)
            out[lv] = ds.copy().isel(time=mask)
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
    # raw-data/FC-scale work: a generous own budget (issues.md M2)
    timeout = 3600.0
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
              # enabled_when deliberately lists the legacy engine vocabulary
              # (RME/RME_RR) alongside Robust: _resolve_estimator still
              # accepts it so previously saved workflows keep loading, and
              # the widget must stay editable there too.
              enabled_when=("estimator_engine", ("Robust", "RME", "RME_RR")),
              cascade={True: (("estimator_engine", "Robust"),)}),
        Param("periods_per_octave", "integer", label="Periods per octave",
              default=3, min=1,
              description="Log2 period bands per power-of-2 octave."),
        Param("band_style", "select", label="Band style",
              options=("l2u", "uniform", "manual"), default="l2u",
              description="Band construction scheme (DESIGN.md §10.2). "
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
                          "(DESIGN.md §10.2): a level where any band has "
                          "fewer than this many total samples (windows × "
                          "band width) is rebuilt as a coarser uniform grid "
                          "(n−⌊n/3⌋ bands, widened to exceed the threshold, "
                          "log-uniform spacing, no overlaps) — the official "
                          "USArray grid's coarse tail. 0 disables "
                          "coarsening."),
        Param("t_short", "number", label="Shortest period (s)",
              default=0.0, min=0,
              description="Head clip (DESIGN.md §10.2): bands shorter than "
                          "this are not evaluated. 0 starts at the "
                          "template's shortest band (the legacy behavior); "
                          "the official USArray grid clips at ~7.3 s for "
                          "the 1 Hz NIMS band (data-driven — noise exceeds "
                          "signal below 10-20 s)."),
        Param("pairing", "select", label="Remote pairing",
              options=("intersect", "strict"), default="intersect",
              optional=True,
              description="How local and remote FC windows pair "
                          "(§9.3): 'intersect' (default) matches window "
                          "times — unmatched rows drop with a loud "
                          "report (the SSMT2000 slot semantics: a slot "
                          "whose clocks cannot match is dropped, not "
                          "interpolated); frequency grids stay strict "
                          "either way (duty-cycle/window mismatch refuses "
                          "to work, as upstream). 'strict' requires "
                          "identical time grids (the Align Runs / parity "
                          "contract)."),
        Param("pair_tolerance_s", "number", label="Pair tolerance (s)",
              default=0.0, min=0, optional=True,
              description="Matching tolerance for window times (centre "
                          "labels, L11) "
                          "(0 = exact, or one sample of the level for "
                          "sub-sample record offsets; GPS-locked stations "
                          "on the epoch-anchored grid match exactly). Raise "
                          "for GPS-stray clocks: window labels shift by the "
                          "receiver's clock error; the closest pairs "
                          "still match one-to-one."),
    )

    def execute(self, ctx, fc, ref_fc=None,
                estimator_engine: str = "Robust",
                remote_reference: bool = False,
                periods_per_octave: int = 3,
                band_style: str = "l2u",
                min_bins_per_band: float = 20.0,
                t_short: float = 0.0,
                band_centers: str = "",
                bands_per_level: str = "",
                pairing: str = "intersect",
                pair_tolerance_s: float = 0.0) -> dict[str, Any]:
        # Accept either a single fc (treated as level 0) or a multi-level
        # container {level: fc}.
        levels = fc if isinstance(fc, dict) else {0: fc}
        sample_fc = next(iter(levels.values()))
        # M5: the SAME classification the QC nodes use — an inline copy here
        # diverged silently from _fc_io_channels and made the estimator
        # disagree with QC about which channels are inputs
        ins, outs, kw = _fc_io_channels(sample_fc)
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
            pairing=pairing,
            pair_tolerance_s=pair_tolerance_s,
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

    #: Banding is Estimate TF's own concern (DESIGN.md §10.3): no inherit from
    #: an upstream Coherence Sort — the sort's per-(bin, window) weights are
    #: band-independent, so there is no banding parameter to auto-fill (and
    #: no duplicated banding param set on the sort node).


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
        if not isinstance(tf_levels, dict) or not tf_levels:
            # L-A9: an empty dict raised a raw StopIteration from
            # next(iter(...)) — name the remedy
            raise ValueError(
                "Merge TF Levels received no per-level TFs — connect "
                "Estimate TF's 'tf levels' output")
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


