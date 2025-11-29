"""Final experiment: lemi.chn calibration modes vs baseline (short periods).

Uses the production calibration framework (emptyui_mth5.calibration) — the
FAP parser and filter attachment are no longer duplicated here.
"""
import sys
from pathlib import Path

sys.path.insert(0, "packages/emptyui-mth5/src")
sys.path.insert(0, "packages/emptyui-aurora/src")
sys.path.insert(0, "packages/emptyui/src")

import tempfile

import numpy as np

from emptyui.context import RunContext
from emptyui.store import DataStore
from emptyui_mth5.calibration import attach_filters, build_fap_filters, parse_fap
from emptyui_mth5.lemi import LoadLemi
from emptyui_mth5.runs import ExtractRun
from emptyui_aurora.process import ComputeFC, EstimateTF

DATA_DIR = Path("examples/data/bgs-lemi-424")
NO46 = DATA_DIR / "lemi_txt" / "NO46"
CHN = DATA_DIR / "lemi.chn"


def show(tf, title):
    MU0 = 4 * np.pi * 1e-7
    P = tf.dataset["transfer_function"].coords["period"].values
    print(f"--- {title} ---")
    for label, out_c, in_c in (("zxy", tf.ex, tf.hy), ("zyx", tf.ey, tf.hx)):
        z = tf.dataset["transfer_function"].sel(output=out_c, input=in_c).values
        rho = np.abs(z) ** 2 / (MU0 * (2 * np.pi / P))
        ph = np.degrees(np.angle(z))
        for i in range(len(P)):
            if P[i] <= 32:
                print(f"{label} T={P[i]:7.2f}s rho={rho[i]:9.3f} phase={ph[i]:8.2f}")


def estimate(ctx, run):
    fc = ComputeFC().execute(ctx, run, input_channels="bx,by",
                             output_channels="e1,e2,bz",
                             channel_nomenclature="lemi12", window_samples=256,
                             decimation_levels=0, persist=False)["fc"]
    return EstimateTF().execute(ctx, fc, estimator_engine="Robust")["tf"]


def main():
    out = Path(tempfile.mkdtemp())
    ctx = RunContext(workspace=out, store=DataStore())
    m = LoadLemi().execute(ctx, data_path=str(NO46), survey_id="bgs",
                           station_id="NO46", mth5_filename="no46.mth5",
                           save_path=str(out), channel_nomenclature="lemi12")["mth5"]
    run = ExtractRun().execute(ctx, m, survey="bgs", station="NO46")["data"]
    filters = build_fap_filters(parse_fap(CHN), extend_flat=False)

    show(estimate(ctx, run), "baseline")

    r2 = ExtractRun().execute(ctx, m, survey="bgs", station="NO46")["data"]
    attach_filters(r2, filters)
    show(estimate(ctx, r2.calibrate()), "calibrated (filter extrapolates below 0.25 Hz)")

    r3 = ExtractRun().execute(ctx, m, survey="bgs", station="NO46")["data"]
    attach_filters(r3, build_fap_filters(parse_fap(CHN), extend_flat=True))
    show(estimate(ctx, r3.calibrate()), "calibrated (flat below 0.25 Hz)")


if __name__ == "__main__":
    main()
