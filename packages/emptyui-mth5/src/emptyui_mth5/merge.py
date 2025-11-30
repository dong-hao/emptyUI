"""Merge Runs node (DESIGN.md §3 'Merging multiple runs').

Two modes:
- FC-level (default): produce a multi-row KernelDataset so aurora merges STFTs.
- Physical: concatenate raw RunTS via mt_timeseries RunTS.merge() and persist.
"""

from __future__ import annotations

from typing import Any

from emptyui import Input, Node, Output, Param

from .runs import _resolve_run, _resolve_station, _resolve_survey


class MergeRuns(Node):
    type = "mth5.runs.merge"
    title = "Merge Runs"
    description = (
        "Combine multiple runs for a single TF estimate. FC-level mode keeps "
        "multiple rows in the KernelDataset so aurora concatenates STFTs (no "
        "sample fabrication); physical mode concatenates raw time series "
        "(interpolates gaps — opt-in). Note: the FC-level 'selection' mode "
        "currently has no producer node in the default pipeline "
        "(superseded by Extract Run → Compute FC; plugin-tier/M4)."
    )
    category = "Processing"
    inputs = (
        Input("selection", "selection"),
        Input("mth5", "mth5", required=False),
    )
    outputs = (
        Output("selection", "selection"),
        Output("merged_run", "ts.run", label="merged time series",
               description="physical mode only"),
    )
    params = (
        Param("mode", "select", label="Mode",
              options=("fc", "physical"), default="fc"),
        Param("gap_method", "select", label="Gap method (physical)",
              options=("slinear", "linear", "nearest", "zero"),
              default="slinear", optional=True),
        Param("new_run_id", "string", label="New run id (physical)",
              default="merged", optional=True),
    )

    def execute(self, ctx, selection, mth5=None, mode: str = "fc",
                gap_method: str = "slinear", new_run_id: str = "merged") -> dict[str, Any]:
        if mode == "fc":
            # KernelDataset already carries multiple rows; validate constraints
            # (uniform sample rate) and pass through.
            rates = selection["sample_rate"].unique() if "sample_rate" in selection else []
            if rates.size > 1:
                raise ValueError(
                    f"cannot merge runs of mixed sample rates: {rates.tolist()} "
                    "(aurora processes one native rate only)"
                )
            ctx.progress(1.0, f"FC merge ready: {len(selection)} run rows")
            return {"selection": selection}

        # physical mode: concat raw series across rows
        run_ts_list = []
        for _, row in selection.iterrows():
            survey_obj = _resolve_survey(mth5, str(row.get("survey", "")))
            station_obj = _resolve_station(survey_obj, str(row.get("station", "")))
            run_obj = _resolve_run(station_obj, str(row.get("run", "")))
            run_ts_list.append(
                run_obj.to_runts(
                    start=row.get("start"), end=row.get("end"),
                )
            )
        if not run_ts_list:
            raise ValueError("no runs to merge")
        merged = run_ts_list[0]
        for rt in run_ts_list[1:]:
            merged = merged.merge(rt, gap_method=gap_method)
        ctx.progress(1.0, f"physical merge: {len(run_ts_list)} runs")
        return {"selection": selection, "merged_run": merged}


def register(registry) -> None:
    registry.add(MergeRuns)
