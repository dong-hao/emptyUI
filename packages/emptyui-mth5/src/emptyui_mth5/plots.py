"""TF analysis plots (Display nodes)."""

from __future__ import annotations

from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from emptyui import Input, Node, Output, Param

MU0 = 4.0 * np.pi * 1e-7


class PlotAppResPhase(Node):
    type = "mth5.plot.appres_phase"
    title = "Plot AppRes & Phase"
    description = (
        "Apparent resistivity (top) and phase (bottom) vs period for the "
        "four impedance components, derived from a transfer function."
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

        Z = tf.dataset["transfer_function"]
        if not tf.has_impedance:
            raise ValueError("transfer function has no impedance data")

        if mode == "off-diagonal":
            components = [("zxy", "ex", "hy"), ("zyx", "ey", "hx")]
        else:
            components = [("zxx", "ex", "hx"), ("zyy", "ey", "hy")]

        period = np.asarray(Z.coords["period"], dtype=float)
        omega = 2.0 * np.pi / period

        fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
        for label, out, inp in components:
            if out not in Z.coords["output"].values or inp not in Z.coords["input"].values:
                continue
            z = Z.sel(output=out, input=inp).values.astype(complex)
            rho = (np.abs(z) ** 2) / (MU0 * omega)
            phase = np.angle(z, deg=True)
            axes[0].loglog(period, rho, ".-", label=label)
            axes[1].semilogx(period, phase, ".-", label=label)

        axes[0].set_ylabel("Apparent resistivity ($\\Omega\\cdot$m)")
        axes[0].legend()
        axes[0].grid(True, which="both", alpha=0.3)
        axes[1].set_ylabel("Phase (deg)")
        axes[1].set_xlabel("Period (s)")
        axes[1].legend()
        axes[1].grid(True, which="both", alpha=0.3)
        fig.suptitle(f"MT apparent resistivity & phase — {mode}")
        fig.tight_layout()

        path = ctx.workspace / "plot_appres_phase.png"
        fig.savefig(path, dpi=110)
        plt.close(fig)
        artifact = ctx.publish_artifact(path, kind="image")
        return {"plot": artifact}


def register(registry) -> None:
    registry.add(PlotAppResPhase)
