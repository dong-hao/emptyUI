"""Run-selection node: archive + criteria -> mth5 KernelDataset (mt.selection)."""

from __future__ import annotations

from typing import Any

from emptyui import Input, Node, Output, Param


class SelectRuns(Node):
    type = "aurora.select.runs"
    title = "Select Runs"
    description = (
        "Build an mth5 KernelDataset (mt.selection) from an archive by "
        "survey/station, optionally restricted to a time window. Supports "
        "multiple runs per station for a single TF estimate (FC-level merge)."
    )
    category = "Processing"
    inputs = (Input("mth5", "mth5"),)
    outputs = (Output("selection", "selection"),)
    params = (
        Param("survey", "string", default=""),
        Param("station", "string", default=""),
        Param("start", "string", label="Start (ISO)", default="", optional=True),
        Param("end", "string", label="End (ISO)", default="", optional=True),
        Param("input_channels", "string", default="hx,hy", optional=True),
        Param("output_channels", "string", default="ex,ey,hz", optional=True),
    )

    def execute(self, ctx, mth5, survey: str = "", station: str = "",
                start: str = "", end: str = "",
                input_channels: str = "hx,hy",
                output_channels: str = "ex,ey,hz") -> dict[str, Any]:
        from mth5.processing import KernelDataset

        kds = KernelDataset()
        kds.from_mth5s(
            mth5_paths=[mth5.filename],
            survey=[survey] if survey else None,
            station=[station] if station else None,
        )
        if start or end:
            import pandas as pd

            kds.start = pd.to_datetime(start) if start else kds.start
            kds.end = pd.to_datetime(end) if end else kds.end
        ctx.progress(1.0, f"selected {len(kds)} run rows")
        return {"selection": kds}


def register(registry) -> None:
    registry.add(SelectRuns)
