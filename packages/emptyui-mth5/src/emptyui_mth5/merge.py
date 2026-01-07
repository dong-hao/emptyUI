"""Merge Runs node (DESIGN.md §3 'Merging multiple runs'; §24.5 P1 rebuild).

Combines several acquisition runs of one station (e.g. SDH28 'b' + 'c') into
a single ``ts.run`` so a downstream Align Runs sees the station's full
coverage — the parity recipe's dominant short-period factor (DESIGN
§24.3 E1b: aligning against a partial remote cropped away ~29% of the local
run). The run list comes from the node parameters (like Extract Run's ids);
the dead ``selection``-payload path was removed (its producer node was
never delivered, M4-deferred).

Physical merge bridges inter-run gaps by interpolation (``gap_method``) —
fabricated samples stay limited to the gap (~0.2% of windows for typical
run gaps). The production-grade alternative — concatenating Fourier
coefficients per decimation level (aurora ``merge_stfts``; no sample
fabrication, the gap simply produces no windows) — is the planned 'Merge
FC' node (§24.5 P1 v2).
"""

from __future__ import annotations

from typing import Any

from emptyui import Input, Node, Output, Param

from .runs import _resolve_run, _resolve_station, _resolve_survey


def merge_runts(run_ts_list, gap_method: str = "slinear",
                new_run_id: str = "merged"):
    """Physically concatenate mth5 RunTS objects (in order) into one series.

    Runs must share the sample rate; inter-run gaps are bridged with
    ``gap_method`` interpolation (mt_timeseries ``RunTS.merge``). The merged
    run's id is set to ``new_run_id`` (station/survey metadata come from the
    first run). Returns the merged RunTS.
    """
    if not run_ts_list:
        raise ValueError("no runs to merge")
    rates = {float(rt.sample_rate) for rt in run_ts_list}
    if len(rates) > 1:
        raise ValueError(
            f"cannot merge runs of mixed sample rates: {sorted(rates)} — "
            "all runs of one station must share the sampling")
    merged = run_ts_list[0]
    for rt in run_ts_list[1:]:
        merged = merged.merge(rt, gap_method=gap_method)
    try:
        merged.run_metadata.id = new_run_id
    except Exception:  # noqa: BLE001, S110 — duck-typed payloads without run metadata
        pass
    return merged


class MergeRuns(Node):
    type = "mth5.runs.merge"
    title = "Merge Runs"
    tooltip = "Combine several acquisition runs of one station into a single time series."
    description = (
        "Combine several acquisition runs of one station (e.g. runs 'b,c') "
        "into a single time series so a downstream Align Runs sees the "
        "station's full coverage — remote-reference alignment against a "
        "partial remote otherwise crops away local data (DESIGN §24.3 E1b). "
        "Runs are concatenated in the given order; inter-run gaps are "
        "bridged by interpolation (gap_method). Resolve ids like Extract "
        "Run: empty survey/station fall back to the first available."
    )
    category = "Processing"
    inputs = (Input("mth5", "mth5"),)
    outputs = (Output("merged_run", "ts.run", label="merged time series"),)
    params = (
        Param("survey", "string", label="Survey id", default="", optional=True),
        Param("station", "string", label="Station id", default="",
              optional=True, placeholder="auto (first available)"),
        Param("runs", "string", label="Runs", default="", optional=True,
              placeholder="e.g. b,c (empty = all runs)",
              description="Comma-separated run ids in acquisition order "
                          "(e.g. 'b,c'); empty = every run of the station "
                          "in archive order."),
        Param("gap_method", "select", label="Gap method",
              options=("slinear", "linear", "nearest", "zero"),
              default="slinear", optional=True,
              description="Bridging interpolation across inter-run gaps."),
        Param("new_run_id", "string", label="New run id", default="merged",
              optional=True),
    )

    def execute(self, ctx, mth5, survey: str = "", station: str = "",
                runs: str = "", gap_method: str = "slinear",
                new_run_id: str = "merged") -> dict[str, Any]:
        survey_obj = _resolve_survey(mth5, survey)
        station_obj = _resolve_station(survey_obj, station)
        available = [str(x) for x in station_obj.run_summary["id"]]
        wanted = ([r.strip() for r in str(runs).split(",") if r.strip()]
                  if str(runs).strip() else available)
        missing = [r for r in wanted if r not in available]
        if missing:
            raise ValueError(
                f"run(s) {missing} not in station archive {available}")
        run_ts_list = []
        for run_id in wanted:
            run_obj = _resolve_run(station_obj, run_id)
            run_ts_list.append(run_obj.to_runts())
        merged = merge_runts(run_ts_list, gap_method=gap_method,
                             new_run_id=new_run_id)
        ctx.progress(
            1.0,
            f"merged {len(wanted)} run(s) ({', '.join(wanted)}): "
            f"{merged.dataset.sizes['time']} samples",
        )
        return {"merged_run": merged}
