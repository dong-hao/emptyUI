"""CLB coupling comparison (DESIGN §24.12): two TF images from the real
MTU-5A corpus (1690618B), one per CLB handling variant.

Background: the box CLB table carries a first-order high-pass with -3 dB
corner ~5 mHz (E) / ~4.1 mHz (H) — the documented MTU AC-coupling filter.
The recording is DC-coupled (TBL ACDC=0; the ex channel carries a ~-140
mV/km self-potential offset that could not survive AC coupling). Two
variants:

- "asis"      : use the CLB response as-is (user's reading: the CLB already
                includes the deployed coupling).
- "dropped"   : neutralize the CLB below its per-channel -3 dB corner
                (amp 1, phase 0 — the DC-coupled input is flat there).

Both variants share everything else: night window 2026-06-18 11:00-23:00
UTC, all three rates, BuildFC (window 256, factor 4, 8 decimation levels,
clock zero epoch), per-bin coherence sort (defaults), self-reference
Estimate TF (defaults).

Run:  .venv/bin/python test/verification/clb_coupling_comparison.py
Outputs: test/verification/plot_appres_phase_asis.png / _dropped.png
"""

from __future__ import annotations

import time
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

OUT = Path("test/verification")
DATA = Path("examples/data/phoenix-mtu-test")
WIN = ("2026-06-18T11:00:00", "2026-06-18T23:00:00")


def _corner_from_table(t):
    """-3 dB corner of a channel table (fallback: the documented 0.005 Hz)."""
    a, f = t.amplitudes, t.frequencies
    cross = np.where((a[:-1] < 1 / np.sqrt(2)) & (a[1:] >= 1 / np.sqrt(2)))[0]
    if not cross.size:
        return 0.005
    return float(np.interp(1 / np.sqrt(2),
                           [a[cross[0]], a[cross[0] + 1]],
                           [f[cross[0]], f[cross[0] + 1]]))


def _neutralize_clb(cal):
    for num, t in cal.channels.items():
        fc = _corner_from_table(t)
        below = t.frequencies < fc
        t.amplitudes = np.where(below, 1.0, t.amplitudes)
        t.phases_deg = np.where(below, 0.0, t.phases_deg)
    return cal


def ingest(tag: str, neutralize: bool):
    import emptyui_mth5.phoenix_calibration as pc

    orig = pc.parse_asc_calibration

    if neutralize:
        def patched(path):
            cal = orig(path)
            return _neutralize_clb(cal) if cal.kind == "CLB" else cal

        pc.parse_asc_calibration = patched
    try:
        from emptyui_mth5.phoenix_mtu import read_phoenix_mtu

        t0 = time.time()
        res = read_phoenix_mtu(DATA, save_path=OUT,
                               mth5_filename=f"mtu_clb_{tag}.mth5",
                               start=WIN[0], end=WIN[1])
        print(f"[{tag}] ingest {time.time() - t0:.1f} s, "
              f"{len(res['runs'])} runs, cal="
              f"{ {c: len(v) for c, v in res['calibration'].items()} }")
    finally:
        pc.parse_asc_calibration = orig


def process(archive_path: Path, tag: str, decimation_levels: int = 8):
    from emptyui.context import RunContext
    from emptyui.store import DataStore
    from emptyui_aurora.process import BuildFC, CoherenceSort, EstimateTF
    from emptyui_mth5.plots import PlotAppResPhase
    from mth5.mth5 import MTH5

    ctx = RunContext(workspace=OUT, store=DataStore(), node_id=tag)
    m = MTH5()
    m.open_mth5(str(archive_path), "r")
    try:
        t0 = time.time()
        fc = BuildFC().execute(ctx, m, decimation_levels=decimation_levels)["fc"]
        print(f"[{tag}] BuildFC {time.time() - t0:.1f} s: " + ", ".join(
            f"L{k}:{v.sizes['frequency']}fx{v.sizes['time']}w"
            for k, v in sorted(fc.items())))
        t0 = time.time()
        sorted_fc = CoherenceSort().execute(ctx, fc=fc)["fc"]
        print(f"[{tag}] sort {time.time() - t0:.1f} s")
        t0 = time.time()
        est = EstimateTF().execute(ctx, fc=sorted_fc, remote_reference=False)
        print(f"[{tag}] estimate {time.time() - t0:.1f} s")
        tf = est["tf"]
        PlotAppResPhase().execute(ctx, tf, print_params=True)
        png = OUT / f"plot_appres_phase_{tag}.png"
        print(f"[{tag}] plot -> {png}")
        # numeric table for the variant comparison
        z = tf.impedance.data if hasattr(tf.impedance, "data") else np.asarray(tf.impedance)
        per = np.asarray(tf.period)
        mu0 = 4e-7 * np.pi
        w = 2 * np.pi / per
        rho = np.abs(z) ** 2 / (mu0 * w[:, None, None])
        import json

        rows = []
        for i, p in enumerate(per):
            rows.append({
                "period_s": float(p),
                "rho_xy": float(rho[i, 0, 1]), "rho_yx": float(rho[i, 1, 0]),
                "phi_xy": float(np.degrees(np.angle(z[i, 0, 1]))),
                "phi_yx": float(np.degrees(np.angle(z[i, 1, 0]))),
            })
        (OUT / f"clb_comparison_{tag}.json").write_text(json.dumps({"tag": tag, "rows": rows}, indent=1))
        print(f"[{tag}] metrics -> {OUT / f'clb_comparison_{tag}.json'} ({len(rows)} periods)")
    finally:
        m.close_mth5()


if __name__ == "__main__":
    import sys

    full = "--full" in sys.argv
    if full:
        globals()["WIN"] = ("", "")  # the entire 3.5-day recording
        print("FULL-record mode: no ingest window")
    ingest("asis", False)
    ingest("dropped", True)
    process(OUT / "mtu_clb_asis.mth5", "asis",
            decimation_levels=10 if full else 8)
    process(OUT / "mtu_clb_dropped.mth5", "dropped",
            decimation_levels=10 if full else 8)
