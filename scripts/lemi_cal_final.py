"""Final experiment: lemi.chn calibration modes vs baseline (short periods)."""
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
from emptyui_aurora.process import ComputeFC, EstimateTF

DATA_DIR = Path("examples/data/bgs-lemi-424")
NO46 = DATA_DIR / "lemi_txt" / "NO46"
CHN = DATA_DIR / "lemi.chn"
MAP = {"#HX": "bx", "#HY": "by", "#HZ": "bz", "#EX": "e1", "#EY": "e2"}


def parse_chn(path):
    lines = Path(path).read_text().splitlines()
    ntf, nch = int(lines[0].split()[1]), int(lines[0].split()[2])
    channels = [c.strip("'") for c in lines[1].split()]
    rows = []
    for line in lines[2:2 + ntf]:
        parts = line.split()
        rows.append((float(parts[0]),
                     {ch: (float(parts[1 + 2 * i]), float(parts[2 + 2 * i]))
                      for i, ch in enumerate(channels)}))
    return rows


def attach(run, rows, extend_flat=False):
    """Attach per-channel FAP filters to the run (mt_timeseries style)."""
    from mt_metadata.timeseries.filters import FrequencyResponseTableFilter

    freqs = np.array([r[0] for r in rows])
    filters = dict(run.filters)
    for comp, ch in MAP.items():
        mods = np.array([r[1][comp][0] for r in rows])
        args = np.array([r[1][comp][1] for r in rows])
        if extend_flat:
            freqs_ext = np.concatenate([[freqs[0] / 1e3], freqs])
            mods = np.concatenate([[mods[0]], mods])  # clamp (flat) below table
            args = np.concatenate([[0.0], args])
            name = f"lemi424_{ch}_cal"
            fap = FrequencyResponseTableFilter(
                frequencies=freqs_ext, amplitudes=mods, phases=args,
                instrument_type="flux gate magnetometer",
                units_in="nanoTesla" if ch.startswith("b") else "mV/km",
                units_out="nanoTesla" if ch.startswith("b") else "mV/km",
                gain=1.0, sequence_number=1)
        else:
            name = f"lemi424_{ch}_cal"
            fap = FrequencyResponseTableFilter(
                frequencies=freqs, amplitudes=mods, phases=args,
                instrument_type="flux gate magnetometer",
                units_in="nanoTesla" if ch.startswith("b") else "mV/km",
                units_out="nanoTesla" if ch.startswith("b") else "mV/km",
                gain=1.0, sequence_number=1)
        fap.name = name
        filters[name] = fap
        run.dataset[ch].attrs["filter.name"] = [name]
    run.filters = filters
    return run


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
    rows = parse_chn(CHN)

    show(estimate(ctx, run), "baseline")

    r2 = ExtractRun().execute(ctx, m, survey="bgs", station="NO46")["data"]
    attach(r2, rows, extend_flat=False)
    show(estimate(ctx, r2.calibrate()), "calibrated (filter extrapolates below 0.25 Hz)")

    r3 = ExtractRun().execute(ctx, m, survey="bgs", station="NO46")["data"]
    attach(r3, rows, extend_flat=True)
    show(estimate(ctx, r3.calibrate()), "calibrated (flat below 0.25 Hz)")


if __name__ == "__main__":
    main()
