"""aurora pipeline nodes: ComputeFC (TS→FC), EstimateTF (FC→TF), MergeTFLevels.

ComputeFC/EstimateTF split the aurora pipeline at the FC boundary (DESIGN.md
§3, §14). FC is passed in-memory between nodes and, when ``persist`` is on,
also written into the workspace MTH5's FC layer so it can be reused / merged.
Estimators are OLS / robust RME / remote-reference RME_RR (aurora-backed).
"""

from __future__ import annotations

from typing import Any

from emptyui import Input, Node, Output, Param

from ._processing import (
    build_tf,
    compute_fc_levels,
    estimate_tf_levels,
    merge_tf_levels,
)


class ComputeFC(Node):
    type = "aurora.process.compute_fc"
    title = "Compute FC"
    description = (
        "Windowed STFT to Fourier coefficients (fc) over N cascading "
        "decimation levels (constant window in samples; sample rate halves by "
        "``factor`` per level). Specify input/output channel component names "
        "(LEMI: bx,by / e1,e2). Persists each level's FC into the workspace "
        "MTH5 when an absolute start time is available."
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
        Param("decimation_levels", "integer", label="Decimation levels", default=0,
              min=0,
              description="Number of cascading decimations (0 = none, native rate only)"),
        Param("factor", "integer", label="Decimation factor", default=4, min=2),
        Param("window_samples", "integer", label="Window (samples)", default=256, min=8,
              description="Constant window length in samples across all levels"),
        Param("overlap", "number", default=0.5),
        Param("window_type", "select", options=("hamming", "hanning"), default="hamming"),
        Param("persist", "boolean", label="Persist to workspace", default=True,
              description="Write each level's FC into the workspace MTH5 FC layer"),
        Param("survey", "string", default="", optional=True),
        Param("station", "string", default="", optional=True),
        Param("run", "string", default="", optional=True),
    )

    def execute(self, ctx, data, bands=None, input_channels: str = "hx,hy",
                output_channels: str = "ex,ey", magnetic_unit: str = "h",
                e_dipole_length_m=None,
                decimation_levels: int = 0, factor: int = 4,
                window_samples: int = 256,
                overlap: float = 0.5, window_type: str = "hamming",
                persist: bool = True, survey: str = "", station: str = "",
                run: str = "") -> dict[str, Any]:
        ins = [c.strip() for c in input_channels.split(",") if c.strip()]
        outs = [c.strip() for c in output_channels.split(",") if c.strip()]
        if len(ins) != 2 or len(outs) != 2:
            raise ValueError("need exactly two input and two output channels")
        levels = compute_fc_levels(
            data, ins, outs, num_decimations=decimation_levels, factor=factor,
            num_samples_window=window_samples, overlap=overlap, window_type=window_type,
            magnetic_unit=magnetic_unit, e_dipole_length_m=e_dipole_length_m,
        )
        if persist:
            from emptyui_mth5.persist import persist_fc_level, runts_context

            surv, stat, run_id = runts_context(data, survey, station, run)
            for level, fc in levels.items():
                if "datetime64" in str(fc.time.dtype):
                    persist_fc_level(ctx, fc, level, fc.attrs["sample_rate"],
                                     survey=surv, station=stat, run=run_id)
                else:
                    ctx.logger.warning(
                        f"FC level {level} has no absolute (datetime64) time axis — "
                        "skipping persistence."
                    )
                    break
        summary = ", ".join(
            f"L{lv}:{fc.sizes['frequency']}f×{fc.sizes['time']}w" for lv, fc in levels.items()
        )
        ctx.progress(1.0, f"computed FC levels [{summary}]")
        return {"fc": levels}


class EstimateTF(Node):
    type = "aurora.process.estimate_tf"
    title = "Estimate TF"
    description = (
        "Estimate a transfer function from Fourier coefficients, over every "
        "decimation level of the fc, and merge them into one broadband tf. "
        "Estimator: 'Least Squares' (aurora OLS) or 'Robust' (M-estimator). "
        "With 'Robust', tick 'Remote Reference' to use a second station's fc "
        "(on 'ref fc') as the remote reference; if none is connected the local "
        "channels are reused as a self-reference. No least-squares + "
        "remote-reference path by design. Also outputs per-level tfs ('tf levels')."
    )
    category = "Processing"
    inputs = (
        Input("fc", "fc"),
        Input("ref_fc", "fc", label="ref fc (remote)", required=False),
        Input("selection", "selection", required=False),
        Input("bands", "bands", required=False),
    )
    outputs = (
        Output("tf", "tf"),
        Output("tf_levels", "tf.level", label="tf levels", description="per decimation level"),
    )
    params = (
        Param("estimator_engine", "select", label="Estimator",
              options=("Robust", "Least Squares"), default="Robust"),
        Param("remote_reference", "boolean", label="Remote Reference",
              default=False,
              description="Robust remote reference (RME_RR). Only applies to "
                          "the 'Robust' estimator."),
        Param("periods_per_octave", "integer", label="Periods per octave",
              default=4, min=1,
              description="Log2 period bands per power-of-2 octave."),
    )

    def execute(self, ctx, fc, ref_fc=None, selection=None, bands=None,
                estimator_engine: str = "Robust",
                remote_reference: bool = False,
                periods_per_octave: int = 4) -> dict[str, Any]:
        # Accept either a single fc (treated as level 0) or a multi-level
        # container {level: fc}.
        levels = fc if isinstance(fc, dict) else {0: fc}
        sample_fc = next(iter(levels.values()))
        ins = [c.strip() for c in (sample_fc.attrs.get("input_channels") or "hx,hy").split(",")]
        outs = [c.strip() for c in (sample_fc.attrs.get("output_channels") or "ex,ey").split(",")]
        per_level = estimate_tf_levels(
            levels, ins, outs,
            estimator_engine=estimator_engine,
            remote_reference=remote_reference,
            periods_per_octave=periods_per_octave,
            ref_levels=ref_fc,
        )
        periods, Z = merge_tf_levels(per_level)
        tf = build_tf(periods, Z, ins, outs)
        ctx.progress(
            1.0,
            f"estimated TF ({estimator_engine}) over {len(levels)} level(s), "
            f"{len(periods)} merged periods",
        )
        return {"tf": tf, "tf_levels": per_level}


class MergeTFLevels(Node):
    type = "aurora.process.merge_tf_levels"
    title = "Merge TF Levels"
    description = (
        "Merge per-decimation-level TFs (the 'tf levels' output of Estimate TF) "
        "into a single broadband transfer function, resolving duplicate periods."
    )
    category = "Processing"
    inputs = (Input("tf_levels", "tf.level", label="tf levels"),)
    outputs = (Output("tf", "tf"),)
    params = ()

    def execute(self, ctx, tf_levels) -> dict[str, Any]:
        first = next(iter(tf_levels.values()))
        ins = list(first["input_channels"])
        outs = list(first["output_channels"])
        periods, Z = merge_tf_levels(tf_levels)
        tf = build_tf(periods, Z, ins, outs)
        ctx.progress(1.0, f"merged {len(tf_levels)} TF levels -> {len(periods)} periods")
        return {"tf": tf}


def register(registry) -> None:
    registry.add(ComputeFC)
    registry.add(EstimateTF)
    registry.add(MergeTFLevels)
