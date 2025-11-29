"""Empirical test: does applying lemi.chn calibration fix short-period rho?"""
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
CHN = DATA_DIR / "lemi.chn"


def parse_chn(path):
    """Parse the LEMI .chn calibration (MOD,ARG per channel)."""
    lines = Path(path).read_text().splitlines()
    header = lines[0].split()
    ntf, nch = int(header[1]), int(header[2])
    channels = [c.strip("'") for c in lines[1].split()]
    rows = []
    for line in lines[2:2 + ntf]:
        parts = line.split()
        freq = float(parts[0])
        vals = {}
        for i, ch in enumerate(channels):
            mod = float(parts[1 + 2 * i])
            arg = float(parts[2 + 2 * i])
            vals[ch] = mod * np.exp(1j * np.deg2rad(arg))
        rows.append((freq, vals))
    return rows, channels


def response_at(rows, f, channel):
    """Interpolate (log-f, complex) response; clamp beyond table ends."""
    fs = np.array([r[0] for r in rows])
    vs = np.array([r[1][channel] for r in rows])
    if f <= fs[0]:
        return vs[0]
    if f >= fs[-1]:
        return vs[-1]
    return np.exp(np.interp(np.log(f), np.log(fs), np.log(np.abs(vs)))) * \
        np.exp(1j * np.interp(np.log(f), np.log(fs), np.unwrap(np.angle(vs))))


def main():
    out = Path(tempfile.mkdtemp())
    ctx = RunContext(workspace=out, store=DataStore())
    m = LoadLemi().execute(ctx, data_path=str(NO46), survey_id="bgs",
                           station_id="NO46", mth5_filename="no46.mth5",
                           save_path=str(out), channel_nomenclature="lemi12")["mth5"]
    run = ExtractRun().execute(ctx, m, survey="bgs", station="NO46")["data"]
    ds = run.dataset
    fs = float(run.sample_rate)
    print("channels:", list(ds.data_vars), "fs:", fs)

    rows, channels = parse_chn(CHN)
    print("chn freqs (Hz?):", [r[0] for r in rows])

    # Cross-spectral Z from raw data at short periods (Welch, 256s window)
    from scipy.signal import welch, csd

    nperseg = 256
    f_w, pxx = welch(ds["e1"].values, fs=fs, nperseg=nperseg, noverlap=nperseg // 2)
    f_w, pxy = csd(ds["e1"].values, ds["by"].values, fs=fs, nperseg=nperseg,
                   noverlap=nperseg // 2)
    f_w, pyy = welch(ds["by"].values, fs=fs, nperseg=nperseg, noverlap=nperseg // 2)
    f_w, pyx = csd(ds["e2"].values, ds["bx"].values, fs=fs, nperseg=nperseg,
                   noverlap=nperseg // 2)
    f_w, pxx2 = welch(ds["bx"].values, fs=fs, nperseg=nperseg, noverlap=nperseg // 2)
    mu0 = 4 * np.pi * 1e-7
    print("\nT(s)   |Zxy| rho_raw  rho_freqcal  rho_percal | phase_raw  phase_freqcal")
    for T in (4.0, 5.66, 8.0, 11.31, 16.0, 22.63, 32.0, 45.25, 64.0):
        f = 1.0 / T
        i = np.argmin(np.abs(f_w - f))
        z1 = pxy[i] / pyy[i]  # e1 x by
        rho = np.abs(z1) ** 2 / (mu0 * 2 * np.pi * f_w[i])
        ph = np.degrees(np.angle(z1))
        # calibration corrections (response = measured/true -> true = meas/resp)
        h = response_at(rows, f_w[i], "#HY")
        e = response_at(rows, f_w[i], "#EX")
        corr_freq = (h / e) ** 2  # rho correction if first col = Hz
        h_p = response_at(rows, 1.0 / f_w[i], "#HY")
        e_p = response_at(rows, 1.0 / f_w[i], "#EX")
        corr_per = (h_p / e_p) ** 2  # rho correction if first col = period s
        print(f"{T:6.1f} {np.abs(z1):8.3f} {rho:9.3f} {rho * corr_freq:12.3f} "
              f"{rho * corr_per:12.3f} | {ph:9.2f} "
              f"{ph - np.degrees(np.angle(e / h)):9.2f}")


if __name__ == "__main__":
    main()
