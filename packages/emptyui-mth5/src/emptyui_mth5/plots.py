"""TF analysis plots (Display nodes) — publication-quality style (DESIGN.md §10.6).

The rendering logic lives in :mod:`emptyui_mth5.plotting` (USArray
conventions: log10 Period × log10 ρ_a, phase 0–90° fold for off-diagonal,
tipper Re/Im with ylim [-1, 1], error bars, ○ XY / ✕ YX markers, equal
per-decade scale, site + LON/LAT title).
"""

from __future__ import annotations

from typing import Any

from emptyui import Input, Node, Output, Param

from .plotting import plot_impedance_response, plot_phase_tensor, plot_tipper


class PlotAppResPhase(Node):
    type = "mth5.plot.appres_phase"
    title = "Plot AppRes & Phase"
    tooltip = "Plot apparent resistivity and phase vs period from a transfer function."
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
        Param("rho_ylim", "string", label="ρ_a y-limits", default="auto",
              placeholder="auto or min,max (e.g. 5,40000)",
              description="Pin the apparent-resistivity y-axis to a fixed "
                          "range 'min,max' (e.g. '5,40000') to match an "
                          "external reference figure; 'auto' (default) fits "
                          "whole decades around the data."),
        Param("print_params", "boolean", label="Print parameters on plot",
              default=False,
              description="Print the processing-provenance line (run "
                          "comment) under the figure — useful for "
                          "algorithm comparison; untick for production "
                          "images."),
    )

    def execute(self, ctx, tf, mode: str = "off-diagonal",
                rho_ylim: str = "auto", print_params: bool = False) \
            -> dict[str, Any]:
        if not tf.has_impedance:
            raise ValueError("transfer function has no impedance data")
        limits = None
        if str(rho_ylim).strip() not in ("", "auto"):
            try:
                parts = [float(x) for x in str(rho_ylim).split(",")]
            except ValueError as exc:
                raise ValueError(
                    f"rho_ylim '{rho_ylim}' is not 'auto' or 'min,max'"
                ) from exc
            if len(parts) != 2 or parts[0] <= 0 or parts[1] <= parts[0]:
                raise ValueError(
                    f"rho_ylim '{rho_ylim}' is not 'auto' or 'min,max' "
                    "(min > 0, max > min)"
                )
            limits = (parts[0], parts[1])
        path = ctx.artifact_path("plot_appres_phase", ".png")
        plot_impedance_response(tf, mode, path, rho_ylim=limits,
                                show_params=print_params)
        artifact = ctx.publish_artifact(path, kind="image")
        return {"plot": artifact}


class PlotTipper(Node):
    type = "mth5.plot.tipper"
    title = "Plot Tipper"
    tooltip = "Plot the geomagnetic transfer function (tipper) real & imaginary parts vs period."
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
    params = (
        Param("tipper_ylim", "string", label="Tz y-limits", default="auto",
              placeholder="auto or min,max (e.g. -1,1)",
              description="Pin the tipper y-axis to a fixed range 'min,max' "
                          "on both Re/Im panels; 'auto' (default) uses "
                          "[-1, 1] and expands only when the data "
                          "legitimately exceeds 1 (|T| > 1 near 2D/3D "
                          "conductivity contrasts)."),
        Param("print_params", "boolean", label="Print parameters on plot",
              default=False,
              description="Print the processing-provenance line (run "
                          "comment) under the figure — useful for "
                          "algorithm comparison; untick for production "
                          "images."),
    )

    def execute(self, ctx, tf, tipper_ylim: str = "auto",
                print_params: bool = True) -> dict[str, Any]:
        if not tf.has_tipper():
            raise ValueError(
                "transfer function has no tipper data (no vertical magnetic "
                "output — compute the FC with e.g. 'e1,e2,bz' outputs)"
            )
        limits = None
        if str(tipper_ylim).strip() not in ("", "auto"):
            try:
                parts = [float(x) for x in str(tipper_ylim).split(",")]
            except ValueError as exc:
                raise ValueError(
                    f"tipper_ylim '{tipper_ylim}' is not 'auto' or 'min,max'"
                ) from exc
            if len(parts) != 2 or parts[1] <= parts[0]:
                raise ValueError(
                    f"tipper_ylim '{tipper_ylim}' is not 'auto' or 'min,max' "
                    "(max > min)"
                )
            limits = (parts[0], parts[1])
        path = ctx.artifact_path("plot_tipper", ".png")
        plot_tipper(tf, path, ylim=limits, show_params=print_params)
        artifact = ctx.publish_artifact(path, kind="image")
        return {"plot": artifact}


class PlotPhaseTensor(Node):
    type = "mth5.plot.phase_tensor"
    title = "Plot Phase Tensor"
    tooltip = "Plot the phase tensor (Caldwell ellipses colored by skew) vs period."
    description = (
        "Phase tensor vs period (Caldwell et al. 2004): the ellipse of "
        "Φ = Re(Z)⁻¹·Im(Z) at each period — orientation = principal "
        "azimuth, axis ratio from the singular values, fill color = skew "
        "β (colorbar; β ≠ 0 indicates 3D structure; galvanic-distortion "
        "free). Needs the complete 2x2 impedance. MVP: no PT error bars "
        "(Z_err → PT covariance propagation is nontrivial, DESIGN.md §10.6)."
    )
    category = "Display"
    inputs = (Input("tf", "tf", label="tf"),)
    outputs = (Output("plot", "plot"),)
    params = (
        Param("print_params", "boolean", label="Print parameters on plot",
              default=False,
              description="Print the processing-provenance line (run "
                          "comment) under the figure — useful for "
                          "algorithm comparison; untick for production "
                          "images."),
    )

    def execute(self, ctx, tf, print_params: bool = True) -> dict[str, Any]:
        if not tf.has_impedance:
            raise ValueError("transfer function has no impedance data")
        path = ctx.artifact_path("plot_phase_tensor", ".png")
        plot_phase_tensor(tf, path, show_params=print_params)
        artifact = ctx.publish_artifact(path, kind="image")
        return {"plot": artifact}
