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


def _channel_names(data) -> list[str]:
    """Actual channel names of a ts.run payload (RunTS or core TimeSeries)."""
    if hasattr(data, "dataset") and not isinstance(data, dict):  # mth5 RunTS
        return [str(c) for c in data.dataset.data_vars]
    if hasattr(data, "channels"):
        ch = data.channels
        if isinstance(ch, dict):
            return [str(k) for k in ch]
        return [str(c) for c in ch]
    raise TypeError(
        f"cannot interpret ts.run value of type {type(data).__name__}"
    )


def _get_channel(data, name: str):
    """Return a ChannelTS for ``name`` (mth5 RunTS) or a ChannelTS wrapper for
    a core TimeSeries channel. Uses exact-name lookup only — never getattr on
    a component that is absent (mt_timeseries raises NameError)."""
    if hasattr(data, "dataset") and not isinstance(data, dict):
        return getattr(data, name)
    from mt_timeseries import ChannelTS

    arr = np.asarray(data.channels[name], dtype=float)
    ch = ChannelTS("electric" if is_electric_name(name) else "magnetic",
                   data=arr)
    ch.component = name
    return ch


class MergeChannels(Node):
    type = "mth5.runs.merge_channels"
    title = "Merge Channels"
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
