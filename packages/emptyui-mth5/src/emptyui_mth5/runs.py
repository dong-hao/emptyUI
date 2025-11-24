"""Run extraction and channel split/merge nodes."""

from __future__ import annotations

from typing import Any, ClassVar

import pandas as pd
from emptyui import Input, Node, Output, Param


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
        "mth5": {"survey": "survey_id", "station": "station_id"},
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
    description = "Split a run time series into individual channels."
    category = "Processing"
    inputs = (Input("data", "ts.run", label="time series"),)
    outputs = (Output("ex", "ts.channel"), Output("ey", "ts.channel"),
               Output("hx", "ts.channel"), Output("hy", "ts.channel"),
               Output("hz", "ts.channel"))
    params = ()

    def execute(self, ctx, data) -> dict[str, Any]:
        out = {}
        for comp in ("ex", "ey", "hx", "hy", "hz"):
            chan = getattr(data, comp, None)
            out[comp] = chan
        return out


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
        merged = channels[0]
        for c in channels[1:]:
            merged = merged + c
        runts = RunTS(dataset=merged.dataset)
        return {"data": runts}


def register(registry) -> None:
    registry.add(ExtractRun)
    registry.add(SplitChannels)
    registry.add(MergeChannels)
