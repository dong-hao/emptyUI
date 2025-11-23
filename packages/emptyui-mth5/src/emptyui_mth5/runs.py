"""Run extraction and channel split/merge nodes."""

from __future__ import annotations

from typing import Any

import pandas as pd
from emptyui import Input, Node, Output, Param


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

    def execute(self, ctx, mth5, survey: str = "", station: str = "",
                run: str = "", start: str = "", end: str = "") -> dict[str, Any]:
        survey_obj = mth5.surveys_group.get_survey(survey) if survey else None
        if survey_obj is None:
            survey_obj = mth5.surveys_group.surveys_list[0] if survey_obj is None else None
        station_obj = survey_obj.stations_group.get_station(station)
        if run:
            run_obj = station_obj.get_run(run)
        else:
            run_id = station_obj.run_summary.iloc[0]["id"]
            run_obj = station_obj.get_run(run_id)
        start_dt = pd.to_datetime(start) if start else None
        end_dt = pd.to_datetime(end) if end else None
        runts = run_obj.to_runts(start=start_dt, end=end_dt)
        ctx.progress(1.0, f"extracted {survey}/{station}/{run_obj.metadata.id}")
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
