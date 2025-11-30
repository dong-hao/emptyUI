"""Table-producing Display nodes (DESIGN.md §3, `table` port type)."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
from emptyui import Input, Node, Output


def _as_dataframe(value):
    """Return a pandas DataFrame from a selection payload."""
    df = getattr(value, "df", None)  # mth5 KernelDataset -> .df
    if df is not None and hasattr(df, "to_dict"):
        return df
    if hasattr(value, "to_dict"):
        return value
    raise TypeError(
        f"cannot interpret selection payload of type {type(value).__name__} as a table"
    )


class ViewSelection(Node):
    type = "mth5.table.selection"
    title = "View Selection"
    tooltip = "Render the run selection (mth5 KernelDataset) as a table."
    description = (
        "Render the run selection (mth5 KernelDataset) as a table inside a "
        "View Table node."
    )
    category = "Display"
    inputs = (Input("selection", "selection"),)
    outputs = (Output("table", "table"),)
    params = ()

    def execute(self, ctx, selection) -> dict[str, Any]:
        df = _as_dataframe(selection)
        records = df.to_dict(orient="records")
        path = ctx.artifact_path("selection_table", ".json")
        path.write_text(json.dumps(records))
        artifact = ctx.publish_artifact(path, kind="table")
        ctx.progress(1.0, f"selection table: {len(records)} rows")
        return {"table": artifact}


class ViewTF(Node):
    type = "mth5.table.tf"
    title = "View TF"
    tooltip = "Render a transfer function's apparent resistivity/phase values as a table."
    description = (
        "Render a transfer function as a table: period, apparent resistivity "
        "and phase for the off-diagonal (zxy, zyx) components."
    )
    category = "Display"
    inputs = (Input("tf", "tf"),)
    outputs = (Output("table", "table"),)
    params = ()

    def execute(self, ctx, tf) -> dict[str, Any]:
        Z = tf.dataset["transfer_function"]
        period = np.asarray(Z.coords["period"], dtype=float)
        omega = 2.0 * np.pi / period

        def _rho_phase(out, inp):
            z = Z.sel(output=out, input=inp).values.astype(complex)
            rho = (np.abs(z) ** 2) / (1.2566370614359172e-6 * omega)
            phase = np.angle(z, deg=True)
            return rho, phase

        # nomenclature-aware accessors: tf.ex/ey/hx/hy are the *actual*
        # instrument channel names of this TF
        rho_xy, ph_xy = _rho_phase(tf.ex, tf.hy)
        rho_yx, ph_yx = _rho_phase(tf.ey, tf.hx)
        records = [
            {
                "period_s": round(float(p), 5),
                "rho_xy_ohm_m": round(float(rx), 4),
                "rho_yx_ohm_m": round(float(ry), 4),
                "phase_xy_deg": round(float(px), 2),
                "phase_yx_deg": round(float(py), 2),
            }
            for p, rx, ry, px, py in zip(period, rho_xy, ph_xy, rho_yx, ph_yx)
        ]
        path = ctx.artifact_path("tf_table", ".json")
        path.write_text(json.dumps(records))
        artifact = ctx.publish_artifact(path, kind="table")
        ctx.progress(1.0, f"TF table: {len(records)} periods")
        return {"table": artifact}


def register(registry) -> None:
    registry.add(ViewSelection)
    registry.add(ViewTF)
