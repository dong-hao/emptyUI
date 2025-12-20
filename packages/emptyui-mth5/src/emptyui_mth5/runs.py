"""Run extraction and channel split/merge nodes."""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np
import pandas as pd
from emptyui import Input, Node, Output, Param

from .nomenclature import (
    KEYWORDS,
    infer_keyword,
    is_electric_name,
    is_magnetic_name,
    standard_name,
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
    summary = station_obj.run_summary
    ids = [str(x) for x in summary["id"]] if summary is not None and len(summary) else []
    if run:
        if run in ids:
            return station_obj.get_run(run)
        raise ValueError(f"run '{run}' not in station {ids}")
    if not ids:
        raise ValueError("station has no runs — set 'run'")
    return station_obj.get_run(ids[0])


class ExtractRun(Node):
    type = "mth5.io.extract_run"
    title = "Extract Run"
    tooltip = "Extract one run's time series from an MTH5 archive."
    description = "Extract a run's time series as a RunTS."
    category = "Data I/O"
    inputs = (Input("mth5", "mth5"),)
    outputs = (Output("data", "ts.run", label="time series"),)
    params = (
        Param("survey", "string", label="Survey id", default=""),
        Param("station", "string", label="Station id", default=""),
        Param("run", "string", label="Run id", default=""),
        Param("start", "string", label="Start (ISO)", default="", optional=True),
        Param("end", "string", label="End (ISO)", default="", optional=True),
    )
    inherit: ClassVar[dict[str, dict[str, str]]] = {
        "mth5": {"survey": "survey_id", "station": "station_id",
                 "channel_nomenclature": "channel_nomenclature"},
    }

    def execute(self, ctx, mth5, survey: str = "", station: str = "",
                run: str = "", start: str = "", end: str = "") -> dict[str, Any]:
        survey_obj = _resolve_survey(mth5, survey)
        station_obj = _resolve_station(survey_obj, station)
        run_obj = _resolve_run(station_obj, run)
        start_dt = pd.to_datetime(start) if start else None
        end_dt = pd.to_datetime(end) if end else None
        runts = run_obj.to_runts(start=start_dt, end=end_dt)
        ctx.progress(1.0, f"extracted {survey_obj.metadata.id}/{station_obj.metadata.id}/{run_obj.metadata.id}")
        return {"data": runts}


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
        return getattr(data, name)
    from mt_timeseries import ChannelTS

    arr = _payload(data)[0][name]
    ch = ChannelTS("electric" if is_electric_name(name) else "magnetic",
                   data=arr)
    ch.component = name
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
    params = ()

    def execute(self, ctx, ex=None, ey=None, hx=None, hy=None, hz=None) -> dict[str, Any]:
        from mth5.timeseries import RunTS

        channels = [c for c in (ex, ey, hx, hy, hz) if c is not None]
        if not channels:
            raise ValueError("at least one channel required")
        runts = RunTS(array_list=channels)
        return {"data": runts}


def register(registry) -> None:
    registry.add(ExtractRun)
    registry.add(SplitChannels)
    registry.add(MergeChannels)
    registry.add(AlignRuns)


def _run_time_info(data):
    """-> (channels {name: np.ndarray}, sample_rate, abs_start | None, rel_start).

    ``abs_start`` is a pandas Timestamp when the payload carries a real
    absolute start (mth5 RunTS GPS time index or a core TimeSeries with
    ``start_datetime``). mt_timeseries uses the 1980 epoch for unknown starts,
    which is treated as "no absolute time".
    """
    ch, sr = _payload(data)
    if hasattr(data, "dataset") and not isinstance(data, dict):  # mth5 RunTS
        abs_start = None
        try:
            ts = pd.Timestamp(str(getattr(data, "start", None)))
            if ts != pd.Timestamp("1980-01-01T00:00:00"):
                abs_start = ts
        except (TypeError, ValueError):
            pass
        return ch, sr, abs_start, 0.0
    rel = float(getattr(data, "start", 0.0) or 0.0)
    sd = getattr(data, "start_datetime", "") or ""
    abs_start = pd.Timestamp(sd) if sd else None
    return ch, sr, abs_start, rel


def _channel_units_from(source):
    """Best-effort per-channel unit strings from a ts.run payload.

    mth5 RunTS archives carry units in the dataset variable attrs
    (``nanoTesla`` / ``milliVolt per kilometer``); core/synthetic payloads
    usually carry none. Align Runs must preserve them: Compute FC reads the
    units to decide the nT→A/m conversion (DESIGN §16.4) — dropping them
    silently returns the impedance to wrong units.
    """
    units: dict[str, str] = {}
    if hasattr(source, "dataset"):
        for name in source.dataset.data_vars:
            u = source.dataset[name].attrs.get("units")
            if u:
                units[str(name)] = str(u)
    return units


def _crop_to_runts(source, channels, sample_rate, start_idx, n_common, start_iso):
    """Rebuild a ts.run payload cropped to [start_idx, start_idx+n_common).

    Both aligned outputs are normalized to an mth5 ``RunTS`` (regardless of
    input type) so downstream consumers — aurora compute_fc and the core
    display/filter nodes via ``coerce_run`` — accept them. The absolute start
    is stamped so downstream FC time grids are identical for both runs.
    """
    from mt_timeseries import ChannelTS
    from mth5.timeseries import RunTS

    units = _channel_units_from(source)
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
        pieces.append(ch)
    kwargs = {}
    for attr in ("run_metadata", "station_metadata", "survey_metadata"):
        meta = getattr(source, attr, None)
        if meta is not None:
            kwargs[attr] = meta
    return RunTS(array_list=pieces, **kwargs)


class AlignRuns(Node):
    type = "mth5.runs.align"
    title = "Align Runs"
    tooltip = "Crop two time series to their common acquisition overlap (remote-reference prerequisite)."
    description = (
        "Crop two time series to their common acquisition overlap so their "
        "Fourier coefficients can be paired — the first step of remote-"
        "reference processing. Both runs must share the same sample rate "
        "(aurora hard-errors on mixed rates). The overlap is computed from "
        "absolute start/end times (GPS); when no absolute times are available "
        "the relative offsets are used with a warning. Both outputs are "
        "normalized to an mth5 RunTS over the identical interval — run "
        "Compute FC on both with identical window/decimation settings."
    )
    category = "Processing"
    inputs = (
        Input("data1", "ts.run", label="time series 1"),
        Input("data2", "ts.run", label="time series 2"),
    )
    outputs = (
        Output("data1", "ts.run", label="time series 1"),
        Output("data2", "ts.run", label="time series 2"),
    )
    params = ()

    def execute(self, ctx, data1, data2) -> dict[str, Any]:
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

        out1 = _crop_to_runts(data1, ch1, sr, i1, n_common, start_iso)
        out2 = _crop_to_runts(data2, ch2, sr, i2, n_common, start_iso)
        _warn_if_uncorrelated(ctx, out1, out2, sr, n_common)
        ctx.progress(
            1.0,
            f"aligned runs: {n_common} samples ({n_common / sr:.1f} s) overlap",
        )
        return {"data1": out1, "data2": out2}


def _warn_if_uncorrelated(ctx, out1, out2, sr, n_common) -> None:
    """Diagnostic: warn when the stations' horizontal magnetic fields are not
    coherent at long periods after alignment.

    A clock offset or an un-filled data gap (mt_io's LEMI reader indexes by
    row, not by timestamp!) breaks station alignment and would silently
    degrade a remote-reference estimate — this catches that before the user
    looks at a rough TF.
    """
    names1 = {str(c) for c in out1.dataset.data_vars}
    names2 = {str(c) for c in out2.dataset.data_vars}
    # horizontal magnetic channels only — the vertical (hz) field is
    # naturally weakly coherent between stations (source-field geometry),
    # so checking it would false-alarm on healthy station pairs.
    mags = [c for c in ("hx", "hy") if c in names1 and c in names2]
    if not mags:
        return
    from scipy.signal import coherence

    nperseg = min(4096, max(256, int(n_common // 4)))
    x = np.asarray(out1.dataset[mags[0]].values, dtype=float)
    y = np.asarray(out2.dataset[mags[0]].values, dtype=float)
    f, C = coherence(x, y, fs=float(sr), nperseg=nperseg, noverlap=nperseg // 2)
    band = (f >= 1.0 / 1024.0) & (f <= 1.0 / 64.0)
    if not band.any():
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
        ctx.logger.info("station coherence (%s, 64-1024 s): %.2f", mags[0], c)
