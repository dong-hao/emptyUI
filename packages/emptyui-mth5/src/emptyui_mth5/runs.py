"""Run extraction and channel split/merge nodes."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, ClassVar

import numpy as np
import pandas as pd
from emptyui import Input, Node, Output, Param
from emptyui_core_nodes._timewindow import resolve_window_s
from emptyui_core_nodes.types import (
    is_run_collection,
    is_unknown_start,
    warn_placeholder_start,
)
from mt_timeseries import RunTS

from .nomenclature import (
    KEYWORDS,
    actual_to_standard,
    infer_keyword,
    is_electric_name,
    is_magnetic_name,
    standard_name,
)

_log = logging.getLogger("emptyui")


@dataclass
class RunCollection:
    """A collection of run time series (Extract Run ``time range`` mode).

    Members are duck payloads (``.dataset`` xr.Dataset, ``.sample_rate``,
    ``.start`` — the same shape ``compute_fc_levels`` already consumes);
    :attr:`extents` carries each member's identity/extent row
    (survey/station/run/start/end/sample_rate) so downstream nodes can
    stamp per-run identity and the executor can summarize the collection
    for the time selector (history §23.4).
    """

    runs: list[Any] = field(default_factory=list)
    extents: list[dict[str, Any]] = field(default_factory=list)


def read_run_channels(run_obj):
    """Fast channel read for one archive run (the one TS->FC data path).

    Returns ``(data_vars, sample_rate, start)`` — channels are read
    straight from their HDF5 datasets and reassembled as xarray
    DataArrays. ``get_channel().to_xarray()`` pays mth5's mt_metadata
    validation storm per channel (~6 ms each, ~40x the data read itself;
    history §24.8 performance note) for metadata the duck payload never
    consumes — on a burst archive that made full-collection extraction
    metadata-bound (issues.md L21). The fast result is pinned bit-identical
    to the slow path (``test_read_run_channels_matches_slow_path``).
    """

    import h5py
    import xarray as xr
    from mt_timeseries.channel_ts import make_dt_coordinates

    data_vars = {}
    sample_rate = 0.0
    start = None
    for name in run_obj.groups_list:
        hdf5_ds = run_obj.hdf5_group.get(str(name))
        if not isinstance(hdf5_ds, h5py.Dataset):
            continue
        attrs = dict(hdf5_ds.attrs)
        filters = attrs.get("filters")
        if isinstance(filters, (str, bytes)):
            # the archive stores the applied-filter list as a JSON
            # string; mth5's validated metadata carries it as dicts
            try:
                attrs["filters"] = json.loads(filters)
            except ValueError:
                attrs["filters"] = []
        sr = float(attrs.get("sample_rate", 0.0) or 0.0)
        da = xr.DataArray(
            hdf5_ds[()],
            coords=[("time", make_dt_coordinates(
                attrs.get("time_period.start"), sr, hdf5_ds.shape[0]))],
            attrs=attrs,
        )
        sample_rate = sr or sample_rate
        if start is None and "time" in da.coords:
            start = da.coords["time"].values[0]
        data_vars[str(name)] = da
    if not sample_rate:
        # defensive: a division by zero deep in the STFT is worse than a
        # loud error naming the run
        raise ValueError(
            f"run '{_hdf5_id(run_obj)}' carries no sample_rate in "
            "its channel metadata — cannot build FCs")
    # time-base cross-check: a channel whose time period contradicts the
    # run's (e.g. a pre-fix artifact whose channel axes anchor on the TBL
    # STIM placeholder while the run carries the tag truth) silently mixes
    # time bases — warn loudly; re-ingest with the current reader
    run_start = str(
        run_obj.hdf5_group.attrs.get("time_period.start", "") or "")
    if run_start:
        for name, da in data_vars.items():
            ch_start = str(da.attrs.get("time_period.start", "") or "")
            if ch_start and ch_start[:19] != run_start[:19]:
                _log.warning(
                    "run '%s': channel '%s' time_period.start (%s) "
                    "contradicts the run's (%s) — the archive mixes time "
                    "bases (a stale pre-fix artifact?); re-ingest with the "
                    "current reader",
                    _hdf5_id(run_obj), name, ch_start, run_start)
    return data_vars, sample_rate, start


def _hdf5_id(group_obj) -> str:
    """The object's id from its HDF5 attrs — free, vs the ``.metadata``
    property's mt_metadata validation storm (history §24.8)."""
    return str(group_obj.hdf5_group.attrs.get("id", ""))


def resolve_filter_registry(survey_obj, data_vars) -> dict:
    """Resolve the response-chain filter objects once from the archive
    (history §24.12): the channel attrs carry filter *names*, the objects live in
    /Survey/Filters — a duck payload needs the objects for the FC-level
    calibration to divide by."""
    filter_names: set = set()
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
                _log.warning(
                    "filter '%s' referenced by the archive channels "
                    "could not be resolved — its response will not "
                    "be removed by Compute FC", fn)
    return registry


def _slice_window(data_vars: dict, lo, hi) -> dict:
    """Slice every channel's time axis to ``[lo, hi]`` (inclusive) when the
    run carries an absolute time axis; naive/aware bounds are reconciled
    against the archive's axis (the archive convention is UTC). Either
    bound may be None (L-M2: an end-only or start-only bound crops one
    side — xarray's slice accepts the open end)."""
    sample = next(iter(data_vars.values()))
    t_arr = np.asarray(sample["time"].values)
    if not np.issubdtype(t_arr.dtype, np.datetime64):
        return data_vars
    idx = pd.DatetimeIndex(t_arr)
    if idx.tz is None:  # naive archive, aware bounds
        blo = lo.tz_localize(None) if lo is not None else None
        bhi = hi.tz_localize(None) if hi is not None else None
    else:  # aware archive: localize naive bounds
        blo = lo.tz_localize("UTC") if (lo is not None and lo.tzinfo is None) else lo
        bhi = hi.tz_localize("UTC") if (hi is not None and hi.tzinfo is None) else hi
    if blo is None and bhi is None:
        return data_vars
    return {
        name: da.sel(time=slice(blo, bhi))
        for name, da in data_vars.items()}


class _CachedRunTS(RunTS):
    """RunTS with channel attribute caching (history §24.8).

    ``RunTS.__getattr__`` constructs a fresh ChannelTS on every attribute
    access (``run_metadata.copy()`` + validation ≈ 2 s per channel) — the
    bulk of the ``to_runts`` storm, paid again by every downstream node.
    Caching builds each channel once (first access) and hands the same
    object out afterwards; the cached channels are the real ``ChannelTS``
    with their response chains, so Calibrate/Split/Merge behave
    identically — Calibrate even returns a plain RunTS (its own
    construction path), which downstream nodes handle.
    """

    def __getattr__(self, name):
        try:
            keys = self.dataset.keys()
        except Exception:  # noqa: BLE001 — not constructed yet
            return super().__getattr__(name)
        if name in keys and not name.startswith("_"):
            cache = self.__dict__.setdefault("_channel_cache", {})
            if name not in cache:
                cache[name] = super().__getattr__(name)
            return cache[name]
        return super().__getattr__(name)

    def clear_channel_cache(self) -> None:
        """Drop the cached per-channel ChannelTS objects (history §24.8).

        ``attach_filters`` re-stamps a channel's dataset attrs (the
        applied-filter refs) in place; a cached ChannelTS resolved its
        response chain at first access — without this the stale chain wins
        and a later-attached filter is silently ignored (the executor's
        cache serves the same instance across runs/param edits)."""
        self.__dict__.pop("_channel_cache", None)


def _group_id(group_obj) -> str:
    """The group's id from its HDF5 path (cheap) — ``group_obj.metadata``
    is a property that rebuilds the FULL station/survey metadata through
    pydantic (all 145 runs of a burst archive ≈ 30-40 s, history §24.8)."""
    try:
        return str(group_obj.hdf5_group.name.split("/")[-1])
    except Exception:  # noqa: BLE001 — identity only, never blocks
        return ""


def _dipole_length_m(run_obj) -> float:
    """First positive electric-channel dipole length (issues.md L-A4).

    Read from the channel's HDF5 attrs (free — no mt_metadata storm);
    0.0 when the archive carries none. Compute FC falls back to this
    payload attr when its explicit param is unset, so externally-built
    archives with proper metadata get the correct mV -> V/m scaling
    without a manual parameter.
    """
    for comp in run_obj.groups_list:
        comp = str(comp)
        if not comp.startswith("e"):
            continue
        hdf5_ds = run_obj.hdf5_group.get(comp)
        attrs = getattr(hdf5_ds, "attrs", None)
        if attrs is None:  # a subgroup, or a name that isn't a channel
            continue
        try:
            dl = float(attrs.get("dipole_length") or 0.0)
        except (TypeError, ValueError):
            dl = 0.0
        if dl > 0:
            return dl
    return 0.0


def _wrap_runts(run_obj, station_obj, survey_obj, data_vars,
                sample_rate: float, registry: dict):
    """Build a real ``RunTS`` from the fast channel read (history §24.8).

    ``RunTS(dataset)`` construction is ~0.2 s vs ``to_runts``'s ~7-19 s
    mt_metadata deepcopy/validation storm. The archive's survey/station/
    run ids travel with the payload (fc identity), the resolved filter
    registry is attached as ``.filters`` — response chains resolve lazily
    from the channels' applied-filter refs (the same wiring ``to_runts``
    produces), so Calibrate/Split/Merge Channels and Compute FC all
    behave identically downstream.
    """
    import xarray as xr

    ds = xr.Dataset(data_vars)
    dl = _dipole_length_m(run_obj)
    if dl > 0:
        ds.attrs["e_dipole_length_m"] = dl
    runts = _CachedRunTS(
        ds,
        run_metadata={"id": _group_id(run_obj),
                      "sample_rate": sample_rate},
        station_metadata={"id": _group_id(station_obj)},
        survey_metadata={"id": _group_id(survey_obj)},
    )
    runts.filters = dict(registry)
    # NB: per-channel ChannelTS objects are NOT pre-built — RunTS resolves
    # response chains lazily from the dataset attrs + ``.filters`` registry
    # (``_get_channel_response``), and eager ``getattr(runts, name)`` would
    # pay the full ChannelTS-construction storm right here (history §24.8).
    return runts


def _run_duck_payload(survey_obj, run_obj, lo=None, hi=None):
    """One run as a duck payload (fast path), sliced to ``[lo, hi]`` when
    the run carries an absolute time axis."""
    import xarray as xr

    data_vars, sample_rate, _start = read_run_channels(run_obj)
    if lo is not None and hi is not None:
        data_vars = _slice_window(data_vars, lo, hi)
        sample = next(iter(data_vars.values()))
        if sample.sizes.get("time", 0):
            _start = sample["time"].values[0]
    filters = resolve_filter_registry(survey_obj, data_vars)
    ds = xr.Dataset(data_vars)
    dl = _dipole_length_m(run_obj)
    if dl > 0:
        ds.attrs["e_dipole_length_m"] = dl
    return SimpleNamespace(
        dataset=ds,
        sample_rate=sample_rate,
        start=_start,
        filters=filters,
    )


def _resolve_survey(mth5, survey: str):
    """Return a SurveyGroup object for ``survey`` (or the first available).

    Supports both the modern container (``mth5.surveys_group``,
    ``MasterSurveyGroup.groups_list`` = survey ids) and legacy single-survey
    v0.1.0 synthetic files (``mth5.survey_group`` is the survey itself).
    """
    master = getattr(mth5, "surveys_group", None)
    if master is not None:
        ids = master.groups_list or []
        if survey:
            if survey in ids:
                return master.get_survey(survey)
            raise ValueError(f"survey '{survey}' not in archive {ids}")
        if not ids:
            raise ValueError("archive has no surveys — set 'survey'")
        return master.get_survey(ids[0])
    legacy = getattr(mth5, "survey_group", None)
    if legacy is None:
        raise ValueError("archive has no survey group — set 'survey'")
    if survey and getattr(getattr(legacy, "metadata", None), "id", "") != survey:
        raise ValueError(f"survey '{survey}' not in archive")
    return legacy


def _resolve_station(survey_obj, station: str):
    """Return a StationGroup object for ``station`` (or the first available)."""
    stg = survey_obj.stations_group
    ids = stg.groups_list or []
    if station:
        if station in ids:
            return stg.get_station(station)
        raise ValueError(f"station '{station}' not in survey {ids}")
    if not ids:
        raise ValueError("survey has no stations — set 'station'")
    return stg.get_station(ids[0])


def _resolve_run(station_obj, run: str):
    """Return a Run object for ``run`` (or the first in the run summary)."""
    if str(run).strip():
        # direct HDF5 resolution — the run summary is a per-station walk
        # (~30 ms on a burst archive) and paying it per resolved run made
        # full-collection extraction metadata-bound (issues.md L21)
        name = str(run).strip()
        try:
            return station_obj.get_run(name)
        except Exception:  # noqa: BLE001
            ids = [str(x) for x in station_obj.run_summary["id"]]
            raise ValueError(f"run '{name}' not in station {ids}") from None
    summary = station_obj.run_summary
    ids = [str(x) for x in summary["id"]] if summary is not None and len(summary) else []
    if not ids:
        raise ValueError("station has no runs — set 'run'")
    return station_obj.get_run(ids[0])


def _utc_timestamp(value) -> pd.Timestamp:
    """Naive stamps are localized to UTC (the archive convention is UTC)."""
    ts = pd.Timestamp(str(value))
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")


def _range_bounds(start: str, end: str):
    """Parse the range-mode selection window: UTC ISO datetime or epoch
    seconds (the time bar writes ISO); naive stamps localize to UTC."""
    lo = _window_bounds(start, "")[0] if str(start).strip() else None
    hi = _window_bounds(end, "")[0] if str(end).strip() else None
    if lo is not None and hi is not None and hi <= lo:
        raise ValueError(
            f"selected time range is empty: end ({hi}) <= start ({lo})")
    return lo, hi


from .phoenix_mtu import _window_bounds


def _largest_runs_hint(summary, limit: int = 3) -> str:
    """The longest runs of a station summary — the auto-match hint in
    out-of-overlap errors (a burst archive's first summary row is a short
    slot; the continuous stream is what users usually want)."""
    try:
        spans = (pd.to_datetime(summary["end"]) - pd.to_datetime(summary["start"]))
        order = spans.sort_values(ascending=False).index[:limit]
        rows = summary.loc[order]
        return "; ".join(
            f"{r['id']} ({pd.Timestamp(r['start']):%Y-%m-%d %H:%M}.."
            f"{pd.Timestamp(r['end']):%Y-%m-%d %H:%M}, "
            f"{float(r['sample_rate']):g} Hz)" for _, r in rows.iterrows())
    except Exception:  # noqa: BLE001 — hint only, never blocks the error
        return ""


class ExtractRun(Node):
    type = "mth5.io.extract_run"
    # raw-data/FC-scale work: a generous own budget (issues.md M2)
    timeout = 3600.0
    title = "Extract Run"
    tooltip = "Extract one run's — or all runs' — time series from an MTH5 archive."
    description = (
        "Extract a run's time series as a RunTS ('run id' mode), or all "
        "runs intersecting a UTC time range as a run collection "
        "('time range' mode — the inclusive selection for burst-mode "
        "archives: the 150/2400 Hz slots and the continuous stream are "
        "each cropped to the window, so Compute FC can cascade and merge "
        "them per level exactly like Build FC does).")
    category = "Data I/O"
    inputs = (Input("mth5", "mth5"),)
    outputs = (
        Output("data", "ts.run", label="time series",
               description="one run ('run id' mode) or the whole run "
                           "collection ('time range' mode) — the shape "
                           "follows the selection"),
    )
    params = (
        Param("selection", "select", label="Selection",
              options=("time range", "run id"), default="time range",
              description="'time range' (default) extracts ALL runs whose "
                          "period intersects the UTC window (empty start/"
                          "end = every run of the station) — the inclusive "
                          "burst-mode selection; 'run id' extracts one run "
                          "(empty = the first in the run summary, warned "
                          "on multi-run stations)"),
        Param("survey", "string", label="Survey id", default=""),
        Param("station", "string", label="Station id", default=""),
        Param("run", "string", label="Run id", default="",
              enabled_when=("selection", ("run id",))),
        Param("sample_rates", "string", label="Sample rates (Hz)",
              default="", optional=True,
              enabled_when=("selection", ("time range",)),
              placeholder="e.g. 15,150 (empty = all families)",
              description="Comma list of sample rates to keep in 'time "
                          "range' mode; empty = every family (the "
                          "staggered burst families + the continuous "
                          "stream)"),
        Param("start", "string", label="Start (UTC)", default="", optional=True,
              time_bar="mth5", placeholder="run start",
              description="Slot start: ISO datetime (UTC, e.g. "
                          "2023-06-01 12:00:00) or seconds; empty = run "
                          "start. Driven by the time bar. Bare numbers are "
                          "MODE-RELATIVE (L-M1): 'run id' = seconds into "
                          "the run; 'time range' = epoch seconds (the bar "
                          "writes ISO in both — bare numbers come from "
                          "manual entry). In 'time range' mode this "
                          "selects the window: all runs intersecting it "
                          "are extracted."),
        Param("end", "string", label="End (UTC)", default="", optional=True,
              time_bar="mth5", placeholder="run end",
              description="Slot end: ISO datetime (UTC) or seconds; empty "
                          "= run end. Driven by the time bar. Bare numbers "
                          "are mode-relative: 'run id' = seconds into the "
                          "run; 'time range' = epoch seconds. In 'time "
                          "range' mode this selects the window."),
    )
    inherit: ClassVar[dict[str, dict[str, str]]] = {
        "mth5": {"survey": "survey_id", "station": "station_id",
                 "channel_nomenclature": "channel_nomenclature"},
    }

    def execute(self, ctx, mth5, selection: str = "time range",
                survey: str = "", station: str = "",
                run: str = "", sample_rates: str = "",
                start: str = "", end: str = "") -> dict[str, Any]:
        survey_obj = _resolve_survey(mth5, survey)
        station_obj = _resolve_station(survey_obj, station)
        if str(selection).strip() == "time range":
            # the single output port carries the collection (value follows
            # the selection — one-port convention, §4)
            coll = self._execute_range(
                ctx, survey_obj, station_obj, sample_rates, start, end)
            return {"data": coll["runs"]}
        run_obj = _resolve_run(station_obj, run)
        if not str(run).strip():
            # the silent first-row fallback is a burst-archive footgun
            # (the first summary row is a 16 s slot, not the continuous
            # stream the user wants) — degrade loudly (history §23.3)
            summary = station_obj.run_summary
            if summary is not None and len(summary) > 1:
                hint = _largest_runs_hint(summary)
                ctx.logger.warning(
                    "'run' param is empty — matched the first summary row "
                    f"'{run_obj.metadata.id}' of {len(summary)} runs; "
                    "longest: " + (hint or "n/a") + " — set 'run' "
                    "explicitly, or use 'time range' selection")
        start_dt = end_dt = None
        if str(start).strip() or str(end).strip():
            # time-selector guards (history §23.4): clamp into the run's extent.
            # The extent's time axis is used as-is — even a placeholder
            # epoch (1980/1970 synthetic/digitized record) is a valid
            # within-record selection axis (the bar shows the same axis).
            ext0 = ext1 = None
            tp = getattr(run_obj.metadata, "time_period", None)
            try:
                if tp and tp.start and tp.end:
                    ext0 = pd.Timestamp(tp.start)
                    ext1 = pd.Timestamp(tp.end)
            except (TypeError, ValueError):
                ext0 = ext1 = None
            if ext0 is None or ext1 is None or not ext1 > ext0:
                # fall back to the station's channel-derived run summary
                try:
                    row = station_obj.run_summary[
                        station_obj.run_summary["id"] == run_obj.metadata.id
                    ].iloc[0]
                    ext0 = pd.Timestamp(row["start"])
                    ext1 = pd.Timestamp(row["end"])
                except Exception:  # noqa: BLE001
                    ext0 = ext1 = None
            if ext0 is None or ext1 is None or not ext1 > ext0:
                ctx.logger.warning(
                    "run extent unknown (no time metadata) — start/end "
                    "passed through unclamped")
                # bare numbers are RELATIVE SECONDS per the node contract —
                # pd.to_datetime("600") would mis-parse them as epoch
                # nanoseconds; parse ISO only, error on anything else
                def _iso_or_none(text):
                    text = str(text).strip()
                    if not text:
                        return None
                    try:
                        return pd.Timestamp(text)
                    except (ValueError, TypeError) as exc:
                        raise ValueError(
                            f"time slot {text!r} is not an ISO datetime — "
                            "with no run extent, only ISO bounds (or empty) "
                            "are accepted (relative seconds need a known "
                            "record start)") from exc
                start_dt = _iso_or_none(start)
                end_dt = _iso_or_none(end)
            else:
                self._check_overlap(
                    ctx, start, end, ext0, ext1, station_obj,
                    str(run_obj.metadata.id))
                rec_len = (ext1 - ext0).total_seconds()
                s0, s1 = resolve_window_s(
                    start, end, rec_len=rec_len,
                    start_datetime=ext0.isoformat(),
                    allow_placeholder_axis=True,
                    what="time slot")
                start_dt = ext0 + pd.Timedelta(seconds=s0)
                end_dt = ext0 + pd.Timedelta(seconds=s1)
        data_vars, sample_rate, _start = read_run_channels(run_obj)
        # L-M2: crop when EITHER bound is set — the old start-only guard
        # silently dropped an end-only bound in the no-extent fallback
        # (the normal path clamps an empty start to the run start; the
        # fallback must match)
        if start_dt is not None or end_dt is not None:
            data_vars = _slice_window(data_vars, start_dt, end_dt)
        registry = resolve_filter_registry(survey_obj, data_vars)
        runts = _wrap_runts(run_obj, station_obj, survey_obj, data_vars,
                            sample_rate, registry)
        ctx.progress(1.0, f"extracted {_group_id(survey_obj)}/{_group_id(station_obj)}/{_group_id(run_obj)}")
        return {"data": runts}

    @staticmethod
    def _check_overlap(ctx, start: str, end: str, ext0, ext1,
                       station_obj, run_id: str) -> None:
        """An explicitly bounded window that misses the resolved run's
        extent entirely yields a 0 s clamp — reject with an actionable
        message naming the matched run (a burst archive's first summary
        row is a short slot; the continuous stream is what users want)."""
        if not (str(start).strip() or str(end).strip()):
            return
        e0 = ext0.tz_localize("UTC") if ext0.tzinfo is None else ext0
        e1 = ext1.tz_localize("UTC") if ext1.tzinfo is None else ext1

        def _abs_bound(text, fallback):
            """Bare numbers are RELATIVE seconds (resolve_window_s
            semantics); ISO/epoch via _window_bounds."""
            s = str(text).strip()
            if not s:
                return fallback
            try:
                rel = float(s)
            except ValueError:
                return _window_bounds(s, "")[0]
            return e0 + pd.Timedelta(seconds=rel)

        try:
            lo = _abs_bound(start, e0)
            hi = _abs_bound(end, e1)
            if lo is not None and hi is not None and lo < e1 and hi > e0:
                return  # overlaps
        except ValueError:
            return  # not ISO/epoch — resolve_window_s reports it properly
        hint = _largest_runs_hint(station_obj.run_summary)
        raise ValueError(
            f"requested window {lo}..{hi} does not overlap run "
            f"'{run_id}' ({e0:%Y-%m-%d %H:%M:%S}..{e1:%Y-%m-%d %H:%M:%S}) — "
            f"set the 'run' param (station '{station_obj.metadata.id}' "
            f"holds {len(station_obj.run_summary)} runs"
            + (f"; longest: {hint})" if hint else ")"))

    def _execute_range(self, ctx, survey_obj, station_obj, sample_rates,
                       start: str, end: str) -> dict[str, Any]:
        """'time range' mode: ALL runs of the station whose period
        intersects the UTC window (inclusive, burst-mode semantics), each
        loaded via the fast channel read and cropped to the window."""
        summary = station_obj.run_summary
        if summary is None or not len(summary):
            raise ValueError(
                f"station '{station_obj.metadata.id}' has no runs to select "
                "from — set 'survey'/'station' or check the archive")
        wanted = list(summary["id"].astype(str))
        rates = ([float(r.strip()) for r in str(sample_rates).split(",")
                  if r.strip()] if str(sample_rates).strip() else None)
        if rates is not None:
            rate_of = dict(zip(summary["id"].astype(str),
                               summary["sample_rate"].astype(float)))
            wanted = [r for r in wanted
                      if any(abs(rate_of[r] - x) <= 1e-6 for x in rates)]
            if not wanted:
                have = sorted(set(rate_of.values()))
                raise ValueError(
                    f"no runs match sample rates {rates} — the station "
                    f"carries rates {have} Hz")
        lo, hi = _range_bounds(start, end)
        kept, dropped = [], []
        for rid in wanted:
            row = summary[summary["id"].astype(str) == rid].iloc[0]
            try:
                rs, re_ = (_utc_timestamp(row["start"]),
                           _utc_timestamp(row["end"]))
            except (TypeError, ValueError):
                ctx.logger.warning(
                    "run %s has no parseable time period — dropped from "
                    "the range selection", rid)
                continue
            if (lo is not None and re_ <= lo) or (hi is not None and rs >= hi):
                dropped.append(rid)
                continue
            clo = rs if lo is None else max(lo, rs)
            chi = re_ if hi is None else min(hi, re_)
            kept.append((rid, rs, re_, clo, chi))
        if not kept:
            span = (f"{_utc_timestamp(summary['start'].min()):%Y-%m-%d %H:%M} .. "
                    f"{_utc_timestamp(summary['end'].max()):%Y-%m-%d %H:%M}")
            raise ValueError(
                f"no runs fall within the requested range "
                f"[{start or 'record start'}, {end or 'record end'}] — the "
                f"archive spans {span}")
        if dropped:
            ctx.logger.warning(
                "range [%s, %s]: dropped %d run(s) outside the window "
                "(%s kept)", start or "start", end or "end", len(dropped),
                len(kept))
        # identity from the HDF5 group names (free): the group .metadata
        # properties trigger the mt_metadata/HDF5 storm per access — ~20 s
        # each on a burst archive, ~250 s for the survey's all-station walk
        # (history §24.8 performance note) — never inside the run loop
        surv_id = survey_obj.hdf5_group.name.split("/")[-1]
        stat_id = station_obj.hdf5_group.name.split("/")[-1]
        members, extents = [], []
        registry_cache: dict[tuple, dict] = {}
        for i, (rid, rs, re_, clo, chi) in enumerate(kept, 1):
            run_obj = _resolve_run(station_obj, rid)
            payload = _run_duck_payload(survey_obj, run_obj, lo=clo, hi=chi)
            # history §24.12: the response-chain filters are the same few objects
            # across members of one survey — resolve each name-set once and
            # assign it to EVERY member (issues.md L23: the fresh per-member
            # registry from _run_duck_payload used to survive on all but the
            # first member of each name-set, so downstream saw 4784 distinct
            # filter objects instead of a handful and the FC response cache
            # never hit)
            names = tuple(sorted(getattr(payload, "filters", {}) or {}))
            if names:
                if names not in registry_cache:
                    registry_cache[names] = resolve_filter_registry(
                        survey_obj, payload.dataset)
                payload.filters = registry_cache[names]
            members.append(payload)
            extents.append({
                "survey": surv_id,
                "station": stat_id,
                "run": rid,
                "start": clo.isoformat(),
                "end": chi.isoformat(),
                "sample_rate": float(payload.sample_rate),
            })
            ctx.progress(i / len(kept), f"extracted {rid} ({i}/{len(kept)})")
        rates_seen = sorted({e["sample_rate"] for e in extents})
        ctx.progress(
            1.0,
            f"range [{start or 'start'}, {end or 'end'}]: {len(kept)} "
            f"run(s) at rates {[f'{r:g}' for r in rates_seen]} Hz")
        return {"runs": RunCollection(runs=members, extents=extents)}


class SplitChannels(Node):
    type = "mth5.runs.split_channels"
    title = "Split Channels"
    tooltip = "Split a run into standard ex/ey/hx/hy/hz channel outputs."
    description = (
        "Split a run time series into individual channels. Channel names are "
        "resolved through the mt_metadata channel nomenclature (LEMI e1/e2 or "
        "e3/e4, bx/by/bz, ...) onto the standard ex/ey/hx/hy/hz outputs."
    )
    category = "Processing"
    inputs = (Input("data", "ts.run", label="time series"),)
    outputs = (Output("ex", "ts.channel"), Output("ey", "ts.channel"),
               Output("hx", "ts.channel"), Output("hy", "ts.channel"),
               Output("hz", "ts.channel"))
    params = (
        Param("channel_nomenclature", "select", label="Channel nomenclature",
              options=("auto",) + KEYWORDS, default="auto",
              description="mt_metadata CHANNEL_MAPS keyword; auto infers it "
                          "from the actual channel names"),
    )

    def execute(self, ctx, data,
                channel_nomenclature: str = "auto") -> dict[str, Any]:
        if is_run_collection(data):
            raise ValueError(
                "Split Channels requires a single run — Extract Run 'time "
                "range' produces a run collection; extract one run ('run id' "
                "mode) for channel tooling")
        names = _channel_names(data)
        kw = (channel_nomenclature if channel_nomenclature != "auto"
              else infer_keyword(names))
        out = {}
        for comp in ("ex", "ey", "hx", "hy", "hz"):
            actual = standard_name(kw, comp)
            out[comp] = _get_channel(data, actual) if actual in names else None
        return out


def _payload(data):
    """(channels {name: np.ndarray}, sample_rate) from a ts.run payload.

    Accepts an mth5 RunTS (``.dataset`` + ``.sample_rate``) or a core
    TimeSeries (``.channels`` dict + ``.sample_rate``). Raises TypeError for
    anything else — the shared interpretation primitive for the run nodes.
    """
    if hasattr(data, "dataset") and not isinstance(data, dict):  # mth5 RunTS
        ds = data.dataset
        ch = {str(k): np.asarray(v.values, dtype=float) for k, v in ds.data_vars.items()}
        return ch, float(data.sample_rate)
    if hasattr(data, "channels") and hasattr(data, "sample_rate"):  # core TimeSeries
        ch = {str(k): np.asarray(v, dtype=float) for k, v in data.channels.items()}
        return ch, float(data.sample_rate)
    raise TypeError(f"cannot interpret ts.run value of type {type(data).__name__}")


def _channel_names(data) -> list[str]:
    """Actual channel names of a ts.run payload (RunTS or core TimeSeries)."""
    return list(_payload(data)[0])


def _get_channel(data, name: str):
    """Return a ChannelTS for ``name`` (mth5 RunTS) or a ChannelTS wrapper for
    a core TimeSeries channel. Uses exact-name lookup only — never getattr on
    a component that is absent (mt_timeseries raises NameError)."""
    if hasattr(data, "dataset") and not isinstance(data, dict):
        ch = getattr(data, name)
        # carry the survey identity: mt_timeseries' RunTS.__getattr__ copies
        # run/station metadata but never the survey, so a Split→Merge round
        # trip persisted survey "0" (§3)
        try:
            if str(getattr(ch.survey_metadata, "id", "") or "") in ("", "0"):
                sid = getattr(data, "survey_metadata", None)
                if sid is not None:
                    ch.survey_metadata = sid
        except Exception as exc:  # noqa: BLE001 — identity only, never blocks
            _log.debug("_get_channel: survey identity carry-over skipped (%s)",
                       exc)
        return ch
    from emptyui_core_nodes.types import channel_calibration
    from mt_metadata.timeseries.filtered import AppliedFilter
    from mt_metadata.timeseries.filters import ChannelResponse
    from mt_timeseries import ChannelTS

    arr = _payload(data)[0][name]
    ch = ChannelTS("electric" if is_electric_name(name) else "magnetic",
                   data=arr)
    ch.component = name
    pairs = channel_calibration(data).get(str(name)) or []
    if pairs:
        # keep the response chain on the wrapper so Merge Channels rebuilds
        # a RunTS that still carries it (history §24.12)
        ch.channel_response = ChannelResponse(
            filters_list=[obj for obj, _ in pairs])
        ch.channel_metadata.filters = [
            AppliedFilter(name=str(obj.name), applied=bool(applied), stage=i)
            for i, (obj, applied) in enumerate(pairs, 1)
        ]
    return ch


class MergeChannels(Node):
    type = "mth5.runs.merge_channels"
    title = "Merge Channels"
    tooltip = "Combine channel time series back into one run time series."
    description = "Combine channel time series into a run time series."
    category = "Processing"
    inputs = (
        Input("ex", "ts.channel", required=False),
        Input("ey", "ts.channel", required=False),
        Input("hx", "ts.channel", required=False),
        Input("hy", "ts.channel", required=False),
        Input("hz", "ts.channel", required=False),
    )
    outputs = (Output("data", "ts.run", label="time series"),)
    params = (
        Param("start", "string", label="Start (UTC)", default="", optional=True,
              time_bar="ex,ey,hx,hy,hz", placeholder="run start",
              description="Window start: ISO datetime (UTC) or seconds "
                          "from the run start; empty = run start. Driven "
                          "by the time bar (extent from the first "
                          "connected channel)."),
        Param("end", "string", label="End (UTC)", default="", optional=True,
              time_bar="ex,ey,hx,hy,hz", placeholder="run end",
              description="Window end: ISO datetime (UTC) or seconds from "
                          "the run start; empty = run end."),
    )

    def execute(self, ctx, ex=None, ey=None, hx=None, hy=None, hz=None,
                start: str = "", end: str = "") -> dict[str, Any]:
        from mth5.timeseries import RunTS

        channels = [c for c in (ex, ey, hx, hy, hz) if c is not None]
        if not channels:
            raise ValueError("at least one channel required")
        # mt_timeseries merges the channels' station/run ids but never the
        # survey — carry it explicitly when a channel carries one (the Split
        # outputs do, §3)
        survey_md = None
        for c in channels:
            sid = getattr(c, "survey_metadata", None)
            if sid is not None and str(getattr(sid, "id", "") or "") not in ("", "0"):
                survey_md = sid
                break
        if survey_md is not None:
            runts = RunTS(array_list=channels, survey_metadata=survey_md)
        else:
            runts = RunTS(array_list=channels)
        if str(start).strip() or str(end).strip():
            # time-selector window (history §23.4): crop the merged run; all
            # channels share the run's time base
            ch, sr = _payload(runts)
            n = max(len(v) for v in ch.values())
            _, _, abs_start, rel_start = _run_time_info(runts)
            anchor_iso = ""
            if abs_start is not None:
                anchor_iso = (abs_start + pd.Timedelta(seconds=rel_start)
                              ).isoformat()
            s0, s1 = resolve_window_s(
                start, end, rec_len=n / sr, start_datetime=anchor_iso,
                what="time slot")
            i0 = round(s0 * sr)
            n_common = round(s1 * sr) - i0
            start_iso = (anchor_iso and (pd.Timestamp(anchor_iso)
                                         + pd.Timedelta(seconds=s0)
                                         ).isoformat()) or ""
            runts = _crop_to_runts(runts, ch, sr, i0, n_common, start_iso)
        return {"data": runts}


def _utc(ts: pd.Timestamp) -> pd.Timestamp:
    """Normalize a timestamp to timezone-aware UTC.

    mth5 RunTS starts are tz-aware (GPS time) while LoadCsv produces naive
    ISO stamps; mixing the two in Align Runs' overlap comparison raises a
    raw TypeError. Naive stamps are assumed UTC (the repo-wide convention,
    cf. ``parse_iso_utc``) and localized here.
    """
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def _run_time_info(data):
    """-> (channels {name: np.ndarray}, sample_rate, abs_start | None, rel_start).

    ``abs_start`` is a timezone-aware UTC pandas Timestamp when the payload
    carries a real absolute start (mth5 RunTS GPS time index or a core
    TimeSeries with ``start_datetime``; naive stamps are assumed UTC).
    Placeholder epochs (mt_timeseries' 1980 default and the Unix 1970
    default — common in digitized paper records) are treated as "no
    absolute time", with a warning — never a rejection (old records keep
    processing; see emptyui_core_nodes.types.UNKNOWN_EPOCH_STARTS).
    """
    ch, sr = _payload(data)
    if hasattr(data, "dataset") and not isinstance(data, dict):  # mth5 RunTS
        abs_start = None
        try:
            ts = pd.Timestamp(str(getattr(data, "start", None)))
            if is_unknown_start(ts):
                warn_placeholder_start(
                    getattr(data, "station_metadata", None) and
                    str(data.station_metadata.id) or "run",
                    ts.isoformat(),
                )
            else:
                abs_start = _utc(ts)
        except (TypeError, ValueError):
            pass
        return ch, sr, abs_start, 0.0
    rel = float(getattr(data, "start", 0.0) or 0.0)
    sd = getattr(data, "start_datetime", "") or ""
    abs_start = None
    if sd:
        try:
            ts = pd.Timestamp(sd)
            if is_unknown_start(ts):
                warn_placeholder_start(
                    str(getattr(data, "station", "") or "record"),
                    ts.isoformat(),
                )
            else:
                abs_start = _utc(ts)
        except (TypeError, ValueError):
            pass
    return ch, sr, abs_start, rel


def _channel_units_from(source):
    """Best-effort per-channel unit strings from a ts.run payload.

    mth5 RunTS archives carry units in the dataset variable attrs
    (``nanoTesla`` / ``milliVolt per kilometer``); core/synthetic payloads
    usually carry none. Align Runs must preserve them: Compute FC reads the
    units to decide the nT→A/m conversion (history §16.3) — dropping them
    silently returns the impedance to wrong units.
    """
    units: dict[str, str] = {}
    if hasattr(source, "dataset"):
        for name in source.dataset.data_vars:
            u = source.dataset[name].attrs.get("units")
            if u:
                units[str(name)] = str(u)
    if not units:
        # core TimeSeries payloads carry the units dict coerced from the
        # RunTS attrs at coerce_run (core filter nodes preserve it)
        try:
            units = getattr(source, "units", None) or {}
        except (AttributeError, NameError):
            units = {}
    return units


def _crop_to_runts(source, channels, sample_rate, start_idx, n_common, start_iso):
    """Rebuild a ts.run payload cropped to [start_idx, start_idx+n_common).

    Both aligned outputs are normalized to an mth5 ``RunTS`` (regardless of
    input type) so downstream consumers — aurora compute_fc_levels and the core
    display/filter nodes via ``coerce_run`` — accept them. The absolute start
    is stamped so downstream FC time grids are identical for both runs.
    """
    from emptyui_core_nodes.types import channel_calibration
    from mt_metadata.timeseries.filtered import AppliedFilter
    from mt_metadata.timeseries.filters import ChannelResponse
    from mt_timeseries import ChannelTS
    from mth5.timeseries import RunTS

    units = _channel_units_from(source)
    calibration = channel_calibration(source)
    pieces = []
    for name, arr in channels.items():
        seg = np.asarray(arr, dtype=float)[start_idx:start_idx + n_common].copy()
        ctype = "electric" if is_electric_name(name) else (
            "magnetic" if is_magnetic_name(name) else "auxiliary"
        )
        ch = ChannelTS(ctype, data=seg, start=start_iso or None,
                       sample_rate=float(sample_rate))
        ch.component = name
        u = units.get(str(name))
        if u:
            ch.channel_metadata.units = u
        pairs = calibration.get(str(name)) or []
        if pairs:
            # carry the response chain across the crop (history §24.12): objects
            # via channel_response (RunTS registers them), refs via channel
            # metadata (the archive round trip restores them). The refs are
            # set AFTER channel_response — the setter would clobber flags.
            ch.channel_response = ChannelResponse(
                filters_list=[obj for obj, _ in pairs])
            ch.channel_metadata.filters = [
                AppliedFilter(name=str(obj.name), applied=bool(applied),
                              stage=i)
                for i, (obj, applied) in enumerate(pairs, 1)
            ]
        pieces.append(ch)
    kwargs = {}
    for attr in ("run_metadata", "station_metadata", "survey_metadata"):
        meta = getattr(source, attr, None)
        if meta is not None:
            kwargs[attr] = meta
    out = RunTS(array_list=pieces, **kwargs)
    return out


class AlignRuns(Node):
    type = "mth5.runs.align"
    # raw-data/FC-scale work: a generous own budget (issues.md M2)
    timeout = 3600.0
    title = "Align Runs"
    tooltip = "Crop two time series to their common acquisition overlap (remote-reference prerequisite)."
    description = (
        "Crop two time series to their common acquisition overlap so their "
        "Fourier coefficients can be paired — the first step of remote-"
        "reference processing. Run collections (Extract Run 'time range' "
        "mode) are aligned member-wise: every member of both sides is "
        "cropped to the common overlap of the two collections' union spans, "
        "members entirely outside are dropped with a warning, and extent "
        "rows are updated to the cropped spans. Both runs must share the "
        "same sample rate (aurora hard-errors on mixed rates; collections "
        "are warned — the FC pairing refuses real mismatches). The overlap "
        "is computed from absolute start/end times (GPS); when no absolute "
        "times are available the relative offsets are used with a warning "
        "(single runs only). Both outputs are normalized to an mth5 RunTS "
        "over the identical interval — run Compute FC on both with "
        "identical window/decimation settings."
    )
    category = "Processing"
    inputs = (
        # collection_ok (issues.md M24): Extract Run 'time range' payloads
        # (burst-mode corpora) align member-wise — the frontend's wiring
        # check gates ts.runs payloads on this flag (DESIGN §4)
        Input("data1", "ts.run", label="time series 1", collection_ok=True),
        Input("data2", "ts.run", label="time series 2", collection_ok=True),
    )
    outputs = (
        Output("data1", "ts.run", label="time series 1"),
        Output("data2", "ts.run", label="time series 2"),
    )
    params = (
        Param("start", "string", label="Start (UTC)", default="", optional=True,
              time_bar="data1,data2", placeholder="overlap start",
              description="Window start within the aligned overlap: ISO "
                          "datetime (UTC) or seconds from the overlap "
                          "start; empty = overlap start. Driven by the "
                          "time bar (extent = overlap of both inputs)."),
        Param("end", "string", label="End (UTC)", default="", optional=True,
              time_bar="data1,data2", placeholder="overlap end",
              description="Window end within the aligned overlap: ISO "
                          "datetime (UTC) or seconds from the overlap "
                          "start; empty = overlap end."),
    )

    def execute(self, ctx, data1, data2, start: str = "",
                end: str = "") -> dict[str, Any]:
        # §9.3: Extract Run 'time range' mode emits run collections (Compute
        # FC / Calibrate consume them member-wise) — Align Runs crops BOTH
        # sides to their common overlap member-wise (a single run acts as a
        # 1-member collection; its output kind is preserved).
        if is_run_collection(data1) or is_run_collection(data2):
            return _align_collections(ctx, data1, data2, start, end)
        ch1, sr1, a1, r1 = _run_time_info(data1)
        ch2, sr2, a2, r2 = _run_time_info(data2)
        if abs(sr1 - sr2) > 1e-9 * max(sr1, sr2, 1.0):
            raise ValueError(
                f"runs have different sample rates: {sr1} vs {sr2} Hz — "
                "remote reference requires identical sampling"
            )
        sr = sr1
        n1 = max((len(v) for v in ch1.values()), default=0)
        n2 = max((len(v) for v in ch2.values()), default=0)
        if n1 < 1 or n2 < 1:
            raise ValueError("runs have no samples")
        # mixed channel lengths (the LEMI row-indexing pathology this module
        # documents): the crop slices each channel independently, so a short
        # channel silently yields a ragged RunTS — warn loudly
        for tag, ch in (("input 1", ch1), ("input 2", ch2)):
            lens = {len(v) for v in ch.values()}
            if len(lens) > 1:
                ctx.logger.warning(
                    "%s carries channels of unequal length (%s) — the "
                    "aligned RunTS will be ragged (upstream defect, likely "
                    "row-indexed LEMI data)", tag,
                    sorted(lens))

        if a1 is not None and a2 is not None:
            t0 = max(a1, a2)
            t1 = min(
                a1 + pd.Timedelta(seconds=n1 / sr),  # exclusive end
                a2 + pd.Timedelta(seconds=n2 / sr),
            )
            if t1 <= t0:
                raise ValueError(
                    f"runs do not overlap: [{a1}, ...] vs [{a2}, ...]"
                )
            n_common = max(1, int((t1 - t0).total_seconds() * sr))
            i1 = round((t0 - a1).total_seconds() * sr)
            i2 = round((t0 - a2).total_seconds() * sr)
            start_iso = t0.isoformat()
        else:
            if a1 is None and a2 is None:
                ctx.logger.warning(
                    "no absolute start times — aligning by relative offsets"
                )
            else:
                ctx.logger.warning(
                    "one run lacks an absolute start — aligning by relative offsets"
                )
            r0 = max(r1, r2)
            r_end = min(r1 + n1 / sr, r2 + n2 / sr)  # exclusive end
            if r_end <= r0:
                raise ValueError("runs do not overlap (relative time axes)")
            n_common = max(1, int((r_end - r0) * sr))
            i1 = round((r0 - r1) * sr)
            i2 = round((r0 - r2) * sr)
            start_iso = ""

        if i1 + n_common > n1:
            n_common = n1 - i1
        if i2 + n_common > n2:
            n_common = n2 - i2
        if n_common < 1:
            raise ValueError("runs have no overlapping samples")

        # time-selector window (history §23.4): one slot within the aligned overlap
        # (both outputs must keep the identical interval). Absolute ISO
        # bounds anchor at the overlap start when the runs carry GPS time;
        # without GPS time only relative seconds are accepted (the shared
        # resolver raises otherwise).
        n_overlap = n_common  # the natural overlap, before the window crop
        if str(start).strip() or str(end).strip():
            rec_len = n_common / sr
            s0, s1 = resolve_window_s(
                start, end, rec_len=rec_len, start_datetime=start_iso or "",
                what="time slot")
            off = round(s0 * sr)
            i1 += off
            i2 += off
            n_common = round(s1 * sr) - off
            if start_iso:
                start_iso = (pd.Timestamp(start_iso)
                             + pd.Timedelta(seconds=s0)).isoformat()
        if n_common < 1:
            raise ValueError("runs have no overlapping samples")

        out1 = _crop_to_runts(data1, ch1, sr, i1, n_common, start_iso)
        out2 = _crop_to_runts(data2, ch2, sr, i2, n_common, start_iso)
        # M4 #2: merged-run provenance (interpolated gap intervals) travels
        # across the crop so Compute FC can stamp it on the fc attrs and the
        # remote-reference pairing can warn about fabricated remote samples.
        _carry_merge_provenance(data1, out1)
        _carry_merge_provenance(data2, out2)
        _warn_merge_gaps_in_overlap(
            ctx, (data1, data2), (out1, out2),
            (start_iso, n_common, sr))
        # history §24.3 E1b: aligning against a partial remote silently discards
        # local data (the official SDG31 comparison lost 29% of the local
        # run this way) — warn when the overlap leaves a large share of an
        # input out.
        _warn_partial_coverage(ctx, (n1, n2), n_overlap)
        _warn_if_uncorrelated(ctx, out1, out2, sr, n_common)
        ctx.progress(
            1.0,
            f"aligned runs: {n_common} samples ({n_common / sr:.1f} s) overlap",
        )
        return {"data1": out1, "data2": out2}


def _member_id(member, ext) -> str:
    """Identity for a collection member in warnings (the extent row's run id,
    else the payload type)."""
    if ext and str(ext.get("run", "")).strip():
        return str(ext["run"])
    return type(member).__name__


def _member_span(member, ext):
    """(start, end) aware-UTC Timestamps for a collection member.

    The member's own dataset time axis is the truth; the extents row
    (Extract Run's identity/extent stamp) is the fallback. Returns
    ``(None, None)`` when neither carries an absolute axis (relative axes
    cannot be cropped to an absolute overlap).
    """
    import numpy as np

    ds = getattr(member, "dataset", None)
    if ds is not None and not isinstance(ds, dict):
        try:
            t = np.asarray(ds["time"].values)
            # datetime64, or the object array a tz-aware index exposes
            if (np.issubdtype(t.dtype, np.datetime64) or t.dtype == object) \
                    and t.size:
                idx = pd.DatetimeIndex(t)
                return _utc(idx[0]), _utc(idx[-1])
        except (TypeError, ValueError, KeyError):  # degenerate, fall through
            pass
    if ext:
        try:
            s = pd.Timestamp(str(ext.get("start", "")))
            e = pd.Timestamp(str(ext.get("end", "")))
            if not pd.isna(s) and not pd.isna(e):
                return _utc(s), _utc(e)
        except (TypeError, ValueError):
            pass
    return None, None


def _align_collections(ctx, data1, data2, start: str, end: str) -> dict:
    """Collection-aware alignment (§9.3): crop both sides to their common
    overlap member-wise.

    Extract Run's 'time range' mode emits run collections (Compute FC /
    Calibrate consume them member-wise); Align Runs crops every member of
    both sides to the common overlap of the two collections' union spans.
    Members entirely outside the overlap are DROPPED with a named warning
    (they cannot contribute pairable FC windows); partially-overlapping
    members are cropped and their extent rows updated, so the time bars
    and downstream summaries stay truthful. A single run acts as a
    1-member collection whose output kind is preserved as a single RunTS.

    Sample rates are WARNED about (a burst collection legitimately carries
    mixed rates) — the FC pairing refuses real mismatches with its own
    actionable error. Absolute spans are required: relative-offset
    alignment is only defined for the single-run path.
    """
    sides = []
    for tag, value in (("time series 1", data1), ("time series 2", data2)):
        if is_run_collection(value):
            members = list(value.runs)
            extents = list(value.extents)
            if not members:
                raise ValueError(
                    f"{tag} is an empty run collection — Extract Run's "
                    "'time range' selection matched no runs")
            rows = [(m, extents[i] if i < len(extents) else None)
                    for i, m in enumerate(members)]
            sides.append(("collection", tag,
                          [(m, e) + _member_span(m, e) for m, e in rows]))
        else:
            # single run: span from _run_time_info (handles a core
            # TimeSeries' start_datetime, which _member_span cannot)
            ch, sr, a, _r = _run_time_info(value)
            n = max((len(v) for v in ch.values()), default=0)
            if n < 1:
                raise ValueError(f"{tag} has no samples")
            e = (a + pd.Timedelta(seconds=n / sr)) if a is not None else None
            sides.append(("single", tag, [(value, None, a, e)]))

    def _rates(rows):
        out = set()
        for m, _e, _s, _e2 in rows:
            try:
                r = float(_payload(m)[1])
            except Exception:  # noqa: BLE001, S112 — degenerate member, skip
                continue
            if r > 0:
                out.add(round(r, 9))
        return out

    ra, rb = _rates(sides[0][2]), _rates(sides[1][2])
    if ra and rb and ra != rb:
        ctx.logger.warning(
            "align: the two inputs carry different sample-rate sets (%s vs "
            "%s Hz) — remote-reference pairing needs identical sampling; "
            "the FC pairing will refuse mismatched frequency grids",
            ", ".join(f"{r:g}" for r in sorted(ra)),
            ", ".join(f"{r:g}" for r in sorted(rb)))

    def _bounds(rows):
        spans = [(s, e) for _m, _e, s, e in rows if s is not None]
        if not spans:
            return None
        return min(s for s, _e in spans), max(e for _s, e in spans)

    ba, bb = _bounds(sides[0][2]), _bounds(sides[1][2])
    if ba is None or bb is None:
        missing = sides[0][1] if ba is None else sides[1][1]
        raise ValueError(
            f"{missing} carries no absolute time span — Align Runs on run "
            "collections needs the archive start/end times (Extract Run "
            "stamps them); relative-offset alignment is only defined for "
            "the single-run path")
    t0 = max(ba[0], bb[0])
    t1 = min(ba[1], bb[1])
    if t1 <= t0:
        raise ValueError(
            f"the two inputs do not overlap: [{ba[0]:%Y-%m-%d %H:%M}, "
            f"{ba[1]:%Y-%m-%d %H:%M}] vs [{bb[0]:%Y-%m-%d %H:%M}, "
            f"{bb[1]:%Y-%m-%d %H:%M}] — check the stations' coverage")

    # history §23.4 window params narrow the overlap (both outputs identical)
    if str(start).strip() or str(end).strip():
        s0, s1 = resolve_window_s(
            start, end, rec_len=(t1 - t0).total_seconds(),
            start_datetime=t0.isoformat(), what="time slot")
        w0 = t0 + pd.Timedelta(seconds=s0)
        w1 = t0 + pd.Timedelta(seconds=s1)
    else:
        w0, w1 = t0, t1

    out = {}
    for idx, (kind, _tag, rows) in enumerate(sides, 1):
        members, extents = [], []
        n_before = n_after = 0
        for m, ext, s, e in rows:
            if s is None or e is None:
                ctx.logger.warning(
                    "align: dropping input %d run '%s' — no absolute time "
                    "span (cannot crop to the overlap)", idx,
                    _member_id(m, ext))
                continue
            lo, hi = max(w0, s), min(w1, e)
            if hi <= lo:
                ctx.logger.warning(
                    "align: dropping input %d run '%s' — outside the "
                    "common overlap [%s, %s]", idx, _member_id(m, ext),
                    t0.isoformat(), t1.isoformat())
                continue
            ch, sr, abs_start, _rel = _run_time_info(m)
            if abs_start is None:
                ctx.logger.warning(
                    "align: dropping input %d run '%s' — its payload "
                    "carries no absolute start (placeholder epoch?)",
                    idx, _member_id(m, ext))
                continue
            i0 = max(0, round((lo - abs_start).total_seconds() * sr))
            # hi is END-INCLUSIVE (the dataset axis's last sample, or
            # Extract Run's `sel`-crop stamp): the sample count from lo
            # through hi is (hi−lo)·sr + 1 — the previous exclusive
            # arithmetic dropped each fully-covered member's final sample
            n = max(0, round((hi - lo).total_seconds() * sr) + 1)
            n_avail = max((len(v) for v in ch.values()), default=0)
            n = max(0, min(n, n_avail - i0))
            if n < 1:
                ctx.logger.warning(
                    "align: dropping input %d run '%s' — degenerate after "
                    "the overlap crop", idx, _member_id(m, ext))
                continue
            # issues.md L22: collection members stay duck payloads —
            # rebuilding a RunTS per member paid ~70 ms of mt_metadata
            # validation each (5993 ChannelTS + 1196 RunTS = 84 s of the
            # 98 s align on the one-day RR corpus); the crop is an isel
            # on the member's dataset with every attr intact. Singles and
            # RunTS members keep the rebuild (the single's output kind is
            # a normalized RunTS by contract; a RunTS member's response
            # chains live on the object).
            if kind == "single" or isinstance(m, RunTS):
                run_out = _crop_to_runts(m, ch, sr, i0, n, lo.isoformat())
            else:
                cropped = m.dataset.isel(time=slice(i0, i0 + n)).copy()
                run_out = SimpleNamespace(
                    dataset=cropped,
                    sample_rate=float(sr),
                    start=(cropped["time"].values[0]
                           if cropped.sizes.get("time", 0) else lo),
                    filters=getattr(m, "filters", None),
                )
            _carry_merge_provenance(m, run_out)
            members.append(run_out)
            new_ext = dict(ext or {})
            new_ext["start"] = lo.isoformat()
            # match Extract Run's END-INCLUSIVE extents convention
            new_ext["end"] = (lo + pd.Timedelta(
                seconds=(n - 1) / sr)).isoformat()
            new_ext["sample_rate"] = float(sr)
            extents.append(new_ext)
            n_before += n_avail
            n_after += n
        if not members:
            raise ValueError(
                f"no runs of input {idx} fall within the common overlap "
                f"[{t0:%Y-%m-%d %H:%M}, {t1:%Y-%m-%d %H:%M}] — nothing to "
                "align")
        if n_before and n_after < 0.9 * n_before:
            ctx.logger.warning(
                "alignment discards %.0f%% of input %d (%d of %d samples; "
                "runs outside the overlap are dropped) — for remote "
                "reference, restrict both stations to the same window "
                "(Build FC start/end) or extract matching time ranges",
                100 * (1 - n_after / n_before), idx, n_after, n_before)
        if kind == "single":
            out[f"data{idx}"] = members[0]
        else:
            out[f"data{idx}"] = RunCollection(runs=members, extents=extents)

    # M4 #2 / history §24.3 E1b: the collection route is Extract Run's DEFAULT
    # output — it must run the same warning chain as the single-run body
    # (gap provenance inside the overlap, cross-station coherence). Partial
    # coverage is already reported by the per-side drop warning above (the
    # same 90% threshold, with the route's own remedy).
    sr_overlap = next(iter(sorted(ra or rb)), None)
    if sr_overlap is not None:
        n_ov = max(1, round((w1 - w0).total_seconds() * sr_overlap))
        _warn_merge_gaps_in_overlap(
            ctx, (data1, data2), (out["data1"], out["data2"]),
            (w0.isoformat(), n_ov, sr_overlap))
        if ra and ra == rb and len(ra) == 1:
            _warn_if_uncorrelated(
                ctx, out["data1"], out["data2"], sr_overlap, n_ov)

    ctx.progress(
        1.0,
        f"aligned runs: [{t0:%Y-%m-%d %H:%M}, {t1:%Y-%m-%d %H:%M}] overlap "
        f"({(t1 - t0).total_seconds():.1f} s)")
    return out


def _merge_gap_attrs(payload):
    """(gap_intervals, source_runs) provenance attrs from a merged ts.run
    payload (Merge Runs), or (None, None) on payloads without them.

    A run collection — Extract Run's default 'time range' output — reports
    the union of its members' provenance: every member bridged by
    interpolation must reach the overlap/coherence warnings (M4 #2), not
    only the first one.
    """
    if is_run_collection(payload):
        gaps: list = []
        runs: list = []
        for member in payload.runs:
            g, r = _merge_gap_attrs(member)
            _union_append(gaps, g)
            _union_append(runs, r)
        return (gaps or None), (runs or None)
    ds = getattr(payload, "dataset", None)
    attrs = getattr(ds, "attrs", None) or {}
    return attrs.get("merged_gap_intervals"), attrs.get("merged_source_runs")


def _union_append(dst: list, values) -> None:
    """Append every element of ``values`` to ``dst`` once, keyed by its
    string form (provenance entries are lists/arrays of stamps, which do
    not hash)."""
    if values is None:
        return
    seen = {str(v) for v in dst}
    for v in values:
        if str(v) not in seen:
            seen.add(str(v))
            dst.append(v)


def _carry_merge_provenance(source, out) -> None:
    """Copy Merge Runs' gap provenance onto a cropped output payload (the
    crop rebuilds the RunTS and loses dataset attrs)."""
    gaps, runs = _merge_gap_attrs(source)
    if gaps is None and runs is None:
        return
    try:
        if gaps is not None:
            out.dataset.attrs["merged_gap_intervals"] = gaps
            gm = (getattr(source, "dataset", None).attrs
                  or {}).get("merged_gap_method")
            if gm is not None:
                out.dataset.attrs["merged_gap_method"] = gm
        if runs is not None:
            out.dataset.attrs["merged_source_runs"] = runs
    except Exception as exc:  # noqa: BLE001 — duck-typed payloads without dataset attrs
        _log.debug("_carry_merge_provenance: attrs not recorded (%s)", exc)


def _warn_merge_gaps_in_overlap(ctx, inputs, outputs, overlap) -> None:
    """M4 #2: warn when the aligned overlap includes interpolated gap(s).

    Merge Runs bridges inter-run gaps by interpolation — the samples inside
    those intervals are fabricated. When the aligned overlap (the interval
    both stations will be FC-computed over) includes such an interval, the
    local windows falling inside it pair with fabricated remote samples
    (silent degradation — the time grids still match, so no error fires).
    """
    start_iso, n_common, sr = overlap
    if not start_iso or n_common < 1:
        return  # relative alignment: gap intervals are not interpretable
    for tag, src, out in zip(("input 1", "input 2"), inputs, outputs):
        gaps, _runs = _merge_gap_attrs(src)
        if gaps is None or len(gaps) == 0:
            continue
        try:
            t0 = _utc(pd.Timestamp(start_iso))
            t1 = t0 + pd.Timedelta(seconds=n_common / sr)
        except Exception:  # noqa: BLE001 — malformed stamps degrade to no-op
            return
        hit = []
        for g in gaps:
            try:
                gs, ge = _utc(pd.Timestamp(g[0])), _utc(pd.Timestamp(g[1]))
            except Exception as exc:  # noqa: BLE001
                _log.warning(
                    "merge provenance: malformed gap interval %r (%s) — "
                    "skipping it, the overlap gap report is only partial",
                    g, exc)
                continue
            lo, hi = max(gs, t0), min(ge, t1)
            if hi > lo:
                hit.append((lo.isoformat(), hi.isoformat()))
        if not hit:
            continue
        n_hit = 0
        for lo, hi in hit:
            a = max(pd.Timestamp(lo), t0)
            b = min(pd.Timestamp(hi), t1)
            n_hit += max(0, int((b - a).total_seconds() * sr))
        share = min(1.0, n_hit / max(1, n_common))
        ctx.logger.warning(
            "%s was merged across acquisition gap(s) %s — %.0f%% of the "
            "aligned overlap falls inside interpolated gap(s) (%d of %d "
            "samples): windows over them pair with fabricated "
            "(interpolated) remote samples. Prefer Merge FC (Fourier "
            "coefficients concatenate without fabrication) or exclude the "
            "gap intervals via the time-selector window.",
            tag, ", ".join(f"[{a}, {b})" for a, b in hit), 100 * share,
            n_hit, n_common,
        )


def _warn_partial_coverage(ctx, lengths, n_common) -> None:
    """Warn when the overlap crop discards a large share of an input.

    The E1b failure mode: a local run aligned against a partial remote is
    cropped to the remote's extent — local data is silently dropped and the
    estimate quietly loses the coverage the remote cannot supply.
    """
    for i, n in enumerate(lengths, start=1):
        if n > 0 and n_common < 0.9 * n:
            ctx.logger.warning(
                "alignment discards %.0f%% of input %d (%d of %d samples) "
                "— the other run starts/ends inside it; for remote "
                "reference, merge the remote station's runs first (Merge "
                "Runs) so alignment sees the station's full coverage "
                "(history §24.3 E1b)",
                100 * (1 - n_common / n), i, n - n_common, n,
            )


def _aligned_channel_names(payload) -> list[str]:
    """Channel names of an aligned output payload — a single ts.run, or a
    run collection (Extract Run's 'time range' route aligns collections)."""
    if is_run_collection(payload):
        names: list[str] = []
        for member in payload.runs:
            for key in _payload(member)[0]:
                if str(key) not in names:
                    names.append(str(key))
        return names
    if hasattr(payload, "dataset") and not isinstance(payload, dict):
        return [str(c) for c in payload.dataset.data_vars]
    return [str(c) for c in payload.channels]


def _aligned_channel_values(payload, name) -> np.ndarray:
    """Samples of one channel of an aligned output payload; collection
    members are concatenated in time order (Align Runs cropped every member
    to the same overlap, so they tile it)."""
    if is_run_collection(payload):
        blocks = []
        for member in payload.runs:
            ch = _payload(member)[0]
            if name in ch:
                blocks.append(ch[name])
        if not blocks:
            return np.zeros(0, dtype=float)
        return np.concatenate(blocks)
    if hasattr(payload, "dataset") and not isinstance(payload, dict):
        return np.asarray(payload.dataset[name].values, dtype=float)
    return np.asarray(payload.channels[name], dtype=float)


#: M7: the coherence diagnostic is sized in seconds, not in bins. The band is
#: the long periods that set a remote reference's quality (64-1024 s).
_COHERENCE_PERIOD_S = (64.0, 1024.0)
#: The series is decimated to this analysis rate before the spectrum: the
#: band's Nyquist (1/2 Hz) is far above 1/64 Hz, and a 150 Hz or 2400 Hz pair
#: then resolves the band the same way an 8 Hz pair does.
_COHERENCE_ANALYSIS_HZ = 1.0
#: Segment length, in seconds: two of the longest band period.
_COHERENCE_SEG_S = 2048.0
#: Shortest aligned overlap that holds the longest band period.
_COHERENCE_MIN_OVERLAP_S = _COHERENCE_PERIOD_S[1]


def _warn_if_uncorrelated(ctx, out1, out2, sr, n_common) -> None:
    """Diagnostic: warn when the stations' horizontal magnetic fields are not
    coherent at long periods after alignment.

    A clock offset or an un-filled data gap (mt_io's LEMI reader indexes by
    row, not by timestamp!) breaks station alignment and would silently
    degrade a remote-reference estimate — this catches that before the user
    looks at a rough TF. Both output shapes are handled: a single ts.run and
    a run collection (its members concatenated in time order).
    """
    names1 = _aligned_channel_names(out1)
    names2 = _aligned_channel_names(out2)
    # horizontal magnetic channels only — the vertical (hz) field is
    # naturally weakly coherent between stations (source-field geometry),
    # so checking it would false-alarm on healthy station pairs.
    # Names are resolved through the nomenclature (hx->hx, bx->hx for
    # lemi12, ...), NOT matched literally: LEMI data keeps its bx/by names,
    # and the guard was built precisely for that pair (2026-09-11 review).
    # Each station's keyword is inferred from its own channels (handles a
    # standard-named local site paired with a LEMI-named remote).
    std1 = {actual_to_standard(infer_keyword(names1), c): c for c in names1}
    std2 = {actual_to_standard(infer_keyword(names2), c): c for c in names2}
    # each station is indexed with ITS OWN actual channel name — the two
    # stations may use different naming (standard local + LEMI-named remote)
    mags = [s for s in ("hx", "hy") if s in std1 and s in std2]
    if not mags:
        # M7: every skip is logged; the check is optional, so info level
        ctx.logger.info(
            "station coherence check skipped: no horizontal magnetic channel "
            "present on both stations")
        return
    x = _aligned_channel_values(out1, std1[mags[0]])
    y = _aligned_channel_values(out2, std2[mags[0]])
    n = int(min(x.size, y.size, n_common))
    # the check needs an overlap that holds the longest band period (sized in
    # seconds, so it is the same at every station rate)
    if n / float(sr) < _COHERENCE_MIN_OVERLAP_S:
        ctx.logger.info(
            "station coherence check skipped: %.0f s of aligned overlap, "
            "need %.0f s to cover the longest band period",
            n / float(sr), _COHERENCE_MIN_OVERLAP_S)
        return
    x, y = x[:n], y[:n]
    from scipy.signal import coherence, decimate

    q = max(1, round(float(sr) / _COHERENCE_ANALYSIS_HZ))
    rate = float(sr) / q
    if q > 1:
        x = decimate(x, q, zero_phase=True)
        y = decimate(y, q, zero_phase=True)
    # a quarter of the record as the segment cap leaves 7-8 segments at 50%
    # overlap, so the coherence estimate is not biased high by a few segments
    nperseg = min(round(_COHERENCE_SEG_S * rate), x.size // 4)
    f, C = coherence(x, y, fs=rate, nperseg=nperseg, noverlap=nperseg // 2)
    longest, shortest = _COHERENCE_PERIOD_S[1], _COHERENCE_PERIOD_S[0]
    band = (f >= 1.0 / longest) & (f <= 1.0 / shortest)
    if not band.any():
        ctx.logger.info(
            "station coherence check skipped: no frequency bin in the "
            "%.0f-%.0f s band for a %.0f s segment",
            shortest, longest, nperseg / rate)
        return
    c = float(np.nanmedian(C[band]))
    if c < 0.5:
        ctx.logger.warning(
            "aligned stations are weakly coherent at long periods "
            "(median %.2f for %s) — remote reference will be degraded; "
            "check for clock offsets or un-filled data gaps between the sites",
            c, mags[0],
        )
    else:
        ctx.logger.info("station coherence (%s, %.0f-%.0f s): %.2f",
                        mags[0], shortest, longest, c)
