"""Diagnose short-period collapse: coherence + E/H ratio vs period."""
import sys
from pathlib import Path

sys.path.insert(0, "packages/emptyui-mth5/src")
sys.path.insert(0, "packages/emptyui-aurora/src")
sys.path.insert(0, "packages/emptyui/src")

import tempfile

import numpy as np

from emptyui.context import RunContext
from emptyui.store import DataStore
from emptyui_mth5.lemi import LoadLemi
from emptyui_mth5.runs import ExtractRun
from emptyui_aurora.process import ComputeFC

DATA_DIR = Path("examples/data/bgs-lemi-424")
NO46 = DATA_DIR / "lemi_txt" / "NO46"


def main():
    out = Path(tempfile.mkdtemp())
    ctx = RunContext(workspace=out, store=DataStore())
    m = LoadLemi().execute(ctx, data_path=str(NO46), survey_id="bgs",
                           station_id="NO46", mth5_filename="no46.mth5",
                           save_path=str(out), channel_nomenclature="lemi12")["mth5"]
    run = ExtractRun().execute(ctx, m, survey="bgs", station="NO46")["data"]
    fc = ComputeFC().execute(ctx, run, input_channels="bx,by",
                             output_channels="e1,e2,bz",
                             channel_nomenclature="lemi12", window_samples=256,
                             decimation_levels=0, persist=False)["fc"][0]

    f = fc["frequency"].values
    mu0 = 4 * np.pi * 1e-7
    print("  T(s)   f(Hz)    |Hy|    |E1|   |E1|/|Hy|  coh2(E1,Hy)  |Hx|    |E2|   coh2(E2,Hx)")
    for T in (4.0, 5.66, 8.0, 11.31, 16.0, 22.63, 32.0, 45.25, 64.0):
        f_t = 1.0 / T
        i = np.argmin(np.abs(f - f_t))
        if abs(f[i] - f_t) / f_t > 0.15:
            continue
        hy = fc["by"].values[i]
        e1 = fc["e1"].values[i]
        hx = fc["bx"].values[i]
        e2 = fc["e2"].values[i]
        c1 = np.abs(np.mean(e1 * np.conj(hy))) ** 2 / (
            np.mean(np.abs(e1) ** 2) * np.mean(np.abs(hy) ** 2))
        c2 = np.abs(np.mean(e2 * np.conj(hx))) ** 2 / (
            np.mean(np.abs(e2) ** 2) * np.mean(np.abs(hx) ** 2))
        print(f"{T:6.2f} {f[i]:7.4f} {np.mean(np.abs(hy)):9.3f} "
              f"{np.mean(np.abs(e1)):8.2f} "
              f"{np.mean(np.abs(e1))/np.mean(np.abs(hy)):10.3f} "
              f"{c1:10.3f} {np.mean(np.abs(hx)):9.3f} "
              f"{np.mean(np.abs(e2)):8.2f} {c2:10.3f}")


if __name__ == "__main__":
    main()
