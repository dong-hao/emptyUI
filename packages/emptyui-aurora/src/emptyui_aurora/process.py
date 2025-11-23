"""aurora pipeline nodes: ComputeFC (TS→FC), EstimateTF (FC→TF), MergeTFLevels.

ComputeFC/EstimateTF split the aurora pipeline at the FC boundary (DESIGN.md
§3). FC is passed in-memory between nodes (first pass); archive FC persistence
is a follow-up. Estimator is first-pass LSQ (robust RME/RME_RR is planned).
"""

from __future__ import annotations

from typing import Any

from emptyui import Input, Node, Output, Param

from ._processing import build_tf, compute_fc, estimate_tf


class ComputeFC(Node):
    type = "aurora.process.compute_fc"
    title = "Compute FC"
    description = (
        "Windowed STFT from time series to Fourier coefficients (fc). "
        "Specify input/output channel component names (LEMI: bx,by / e1,e2)."
    )
    category = "Processing"
    inputs = (
        Input("data", "ts.run", label="time series"),
        Input("bands", "bands", required=False),
    )
    outputs = (Output("fc", "fc"),)
    params = (
        Param("input_channels", "string", label="Input channels", default="hx,hy"),
        Param("output_channels", "string", label="Output channels", default="ex,ey"),
        Param("magnetic_unit", "radio", label="Magnetic input", default="h",
              options=("h", "b"),
              description="h = coil sensor (already H); b = fluxgate (B converted to H)"),
        Param("e_dipole_length_m", "number", label="E dipole length (m)", default=None,
              optional=True,
              description="LEMI stores E as mV potential; divide by dipole length to get V/m"),
        Param("window_length", "number", label="Window length (s)", default=4096.0),
        Param("overlap", "number", default=0.5),
        Param("window_type", "select", options=("hamming", "hanning"), default="hamming"),
    )

    def execute(self, ctx, data, bands=None, input_channels: str = "hx,hy",
                output_channels: str = "ex,ey", magnetic_unit: str = "h",
                e_dipole_length_m=None,
                window_length: float = 4096.0,
                overlap: float = 0.5, window_type: str = "hamming") -> dict[str, Any]:
        ins = [c.strip() for c in input_channels.split(",") if c.strip()]
        outs = [c.strip() for c in output_channels.split(",") if c.strip()]
        if len(ins) != 2 or len(outs) != 2:
            raise ValueError("need exactly two input and two output channels")
        fc = compute_fc(data, ins, outs, window_length, overlap, window_type,
                        magnetic_unit=magnetic_unit,
                        e_dipole_length_m=e_dipole_length_m)
        ctx.progress(1.0, f"computed FC: {fc.sizes['frequency']} freq x {fc.sizes['time']} windows")
        return {"fc": fc}


class EstimateTF(Node):
    type = "aurora.process.estimate_tf"
    title = "Estimate TF"
    description = (
        "Estimate a transfer function from Fourier coefficients via "
        "cross-power regression (first-pass LSQ)."
    )
    category = "Processing"
    inputs = (
        Input("fc", "fc"),
        Input("selection", "selection", required=False),
        Input("bands", "bands", required=False),
    )
    outputs = (Output("tf", "tf"),)
    params = (
        Param("estimator_engine", "select", options=("RME_RR", "RME", "OLS"),
              default="RME_RR"),
        Param("remote_reference", "boolean", default=False),
        Param("periods_per_decade", "integer", default=8, min=1),
    )

    def execute(self, ctx, fc, selection=None, bands=None,
                estimator_engine: str = "RME_RR",
                remote_reference: bool = False,
                periods_per_decade: int = 8) -> dict[str, Any]:
        ins = [c.strip() for c in (fc.attrs.get("input_channels") or "hx,hy").split(",")]
        outs = [c.strip() for c in (fc.attrs.get("output_channels") or "ex,ey").split(",")]
        periods, Z = estimate_tf(
            fc, ins, outs,
            estimator_engine=estimator_engine,
            remote_reference=remote_reference,
            periods_per_decade=periods_per_decade,
        )
        tf = build_tf(periods, Z, ins, outs)
        ctx.progress(1.0, f"estimated TF over {len(periods)} periods")
        return {"tf": tf}


class MergeTFLevels(Node):
    type = "aurora.process.merge_tf_levels"
    title = "Merge TF Levels"
    description = "Merge per-decimation-level TFs into a single transfer function."
    category = "Processing"
    inputs = (Input("tf_levels", "tf.level", label="tf level"),)
    outputs = (Output("tf", "tf"),)
    params = ()

    def execute(self, ctx, tf_levels) -> dict[str, Any]:
        raise NotImplementedError(
            "MergeTFLevels: multi-decimation-level merge is a follow-up; "
            "ComputeFC currently produces a single-level fc."
        )


def register(registry) -> None:
    registry.add(ComputeFC)
    registry.add(EstimateTF)
    registry.add(MergeTFLevels)
