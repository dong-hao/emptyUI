"""Isolate short-period problem: B-B coherence vs E-B, lag, raw spectra."""
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

DATA_DIR = Path("examples/data/bgs-lemi-424")
NO46 = DATA_DIR / "lemi_txt" / "NO46"


def main():
    out = Path(tempfile.mkdtemp())
    ctx = RunContext(workspace=out, store=DataStore())
    m = LoadLemi().execute(ctx, data_path=str(NO46), survey_id="bgs",
                           station_id="NO46", mth5_filename="no46.mth5",
                           save_path=str(out), channel_nomenclature="lemi12")["mth5"]
    run = ExtractRun().execute(ctx, m, survey="bgs", station="NO46")["data"]
    ds = run.dataset
    fs = float(run.sample_rate)
    n = len(ds["by"].values)
    print("n samples:", n, "start:", run.start)

    # 1) cross-correlation e1 vs by (detrended), lags -20..20 s
    e1 = ds["e1"].values - np.mean(ds["e1"].values)
    by = ds["by"].values - np.mean(ds["by"].values)
    bx = ds["bx"].values - np.mean(ds["bx"].values)
    e2 = ds["e2"].values - np.mean(ds["e2"].values)
    maxlag = 20
    xc = np.correlate(e1, by, mode="full")[len(e1) - 1 - maxlag: len(e1) + maxlag]
    xc /= np.sqrt(np.sum(e1 ** 2) * np.sum(by ** 2))
    peak = np.argmax(np.abs(xc)) - maxlag
    print(f"peak |xc|(e1,by) = {np.max(np.abs(xc)):.3f} at lag {peak}s")
    xc2 = np.correlate(e2, bx, mode="full")[len(e2) - 1 - maxlag: len(e2) + maxlag]
    xc2 /= np.sqrt(np.sum(e2 ** 2) * np.sum(bx ** 2))
    print(f"peak |xc|(e2,bx) = {np.max(np.abs(xc2)):.3f} at lag {np.argmax(np.abs(xc2)) - maxlag}s")

    # 2) Welch coherence: bx-by (both magnetic) vs e1-by
    from scipy.signal import welch, csd, coherence

    nperseg = 4096
    f, cxx = coherence(bx, by, fs=fs, nperseg=nperseg, noverlap=nperseg // 2)
    f2, ceb = coherence(e1, by, fs=fs, nperseg=nperseg, noverlap=nperseg // 2)
    f3, ceb2 = coherence(e2, bx, fs=fs, nperseg=nperseg, noverlap=nperseg // 2)
    print("\n  T(s)   coh(bx,by)  coh(e1,by)  coh(e2,bx)")
    for T in (4.0, 5.66, 8.0, 11.31, 16.0, 22.63, 32.0, 45.25, 64.0):
        i = np.argmin(np.abs(f - 1.0 / T))
        print(f"{T:6.2f} {cxx[i]:11.3f} {ceb[i]:11.3f} {ceb2[i]:11.3f}")

    # 3) raw spectra levels
    f, pbx = welch(bx, fs=fs, nperseg=nperseg, noverlap=nperseg // 2)
    _, pe1 = welch(e1, fs=fs, nperseg=nperseg, noverlap=nperseg // 2)
    print("\n  T(s)   PSD bx (nT^2/Hz)  PSD e1 ((mV/km)^2/Hz)")
    for T in (4.0, 8.0, 16.0, 32.0, 64.0):
        i = np.argmin(np.abs(f - 1.0 / T))
        print(f"{T:6.2f} {pbx[i]:16.4e} {pe1[i]:16.4e}")


if __name__ == "__main__":
    main()
