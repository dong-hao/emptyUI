"""TF analysis plots (Display nodes) — publication-quality style (DESIGN.md §15).

The rendering logic lives in :mod:`emptyui_mth5.plotting` (USArray
conventions: log10 Period × log10 ρ_a, phase 0–90° fold for off-diagonal,
tipper Re/Im with ylim [-1, 1], error bars, ○ XY / ✕ YX markers, equal
per-decade scale, site + LON/LAT title).
"""

from __future__ import annotations

from typing import Any

from emptyui import Input, Node, Output, Param

from .plotting import plot_impedance_response, plot_tipper


class PlotAppResPhase(Node):
    type = "mth5.plot.appres_phase"
    title = "Plot AppRes & Phase"
    description = (
        "Apparent resistivity (top) and phase (bottom) vs period for the "
        "four impedance components, derived from a transfer function. "
        "USArray-style: log10 period and resistivity, off-diagonal phase "
        "folded into (0, 90°), error bars, colorblind-safe markers."
    )
    category = "Display"
    inputs = (Input("tf", "tf", label="tf"),)
    outputs = (Output("plot", "plot"),)
    params = (
        Param("mode", "radio", label="Components", default="off-diagonal",
              options=("off-diagonal", "diagonal"),
              description="off-diagonal = xy/yx; diagonal = xx/yy"),
    )

    def execute(self, ctx, tf, mode: str = "off-diagonal") -> dict[str, Any]:
        if not tf.has_impedance:
            raise ValueError("transfer function has no impedance data")
        path = ctx.workspace / "plot_appres_phase.png"
        plot_impedance_response(tf, mode, path)
        artifact = ctx.publish_artifact(path, kind="image")
        return {"plot": artifact}


class PlotTipper(Node):
    type = "mth5.plot.tipper"
    title = "Plot Tipper"
    description = (
        "Geomagnetic transfer function (tipper) real and imaginary parts vs "
        "period for Tzx and Tzy, derived from a transfer function that "
        "includes a vertical magnetic output (e.g. 'e1,e2,bz'). The tipper is "
        "dimensionless (Bz/H or Hz/H); normal tipper magnitude does not "
        "exceed 1, so the y-axis is fixed to [-1, 1]."
    )
    category = "Display"
    inputs = (Input("tf", "tf", label="tf"),)
    outputs = (Output("plot", "plot"),)
    params = ()

    def execute(self, ctx, tf) -> dict[str, Any]:
        if not tf.has_tipper():
            raise ValueError(
                "transfer function has no tipper data (no vertical magnetic "
                "output — compute the FC with e.g. 'e1,e2,bz' outputs)"
            )
        path = ctx.workspace / "plot_tipper.png"
        plot_tipper(tf, path)
        artifact = ctx.publish_artifact(path, kind="image")
        return {"plot": artifact}


def register(registry) -> None:
    registry.add(PlotAppResPhase)
    registry.add(PlotTipper)
