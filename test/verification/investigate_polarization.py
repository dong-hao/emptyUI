"""§24.13 Stage 1: do strongly-polarized magnetic noise features even exist
in this region? Compute Weckmann-style polarization + single-event cluster
features from the sessions FC payload and dump per-run diagnostics.

Features per FC window (QC band 0.1-1.0 Hz, pooled across bins):
  S_hh = sum_k h h^H  (2x2 Hermitian)  ->  P_w = tr, alpha_B = major axis
  Z_w  = (sum_k e h^H) @ inv(S_hh)     (single-event band-averaged impedance)
Per run (stream hours + sr150 slots):
  pol_conc_win : |sum_w P_w exp(2i alpha_B)| / sum_w P_w  (coherent-noise
                 signature; natural H has no preferred direction)
  pol_dir      : dominant polarization angle (deg, mod 180)
  pol_conc_bin : median over in-band bins of the per-bin weighted resultant
  spread       : median over components of median_w |Z_w - med| / |med|
  up / down    : impedance-level deviation (the existing score)

Run:  .venv/bin/python test/verification/investigate_polarization.py
Outputs: test/verification/polarization_features.json + PNG plots
"""

from __future__ import annotations

import json
import time
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

OUT = Path("test/verification")
MU0 = 4e-7 * np.pi
TAG = "asis_full_sel"
P_MIN, P_MAX = 1.0, 10.0  # QC band


def build_fc():
    import pandas as pd  # noqa: F401
    from emptyui.context import RunContext
    from emptyui.store import DataStore
    from emptyui_aurora.process import BuildFC
    from mth5.mth5 import MTH5

    ctx = RunContext(workspace=OUT, store=DataStore(), node_id=TAG)
    m = MTH5()
    m.open_mth5(str(OUT / f"mtu_clb_{TAG}.mth5"), "r")
    try:
        t0 = time.time()
        fc = BuildFC().execute(ctx, m, decimation_levels=10)["fc"]
        print(f"BuildFC {time.time() - t0:.1f} s", flush=True)
        return fc
    finally:
        m.close_mth5()


def features(fc):

    from emptyui_aurora.process import _fc_io_channels

    rows = []
    # per-window angle samples for the rose plot
    rose = []
    for ds in fc.values():
        f = np.asarray(ds["frequency"].values, dtype=float)
        band = (f >= 1.0 / P_MAX) & (f <= 1.0 / P_MIN)
        if not band.any():
            continue
        ins, outs, _ = _fc_io_channels(ds)
        # the impedance/polarization analysis uses the two horizontal E
        # channels (hz is a tipper output here)
        outs = [c for c in outs if c.startswith("e")]
        if len(ins) != 2 or len(outs) != 2:
            continue
        H = np.stack([np.asarray(ds[c].values) for c in ins])  # (2,nf,nt)
        E = np.stack([np.asarray(ds[c].values) for c in outs])
        hb, eb = H[:, band, :], E[:, band, :]  # (2,nk,nt)
        finite = np.isfinite(hb[0]) & np.isfinite(hb[1]) \
            & np.isfinite(eb[0]) & np.isfinite(eb[1])  # (nk,nt)
        # per-bin polarization (axial angle mod pi)
        cross_k = hb[0] * np.conj(hb[1])  # (nk,nt)
        a_k = 0.5 * np.arctan2(2 * np.real(cross_k),
                               np.abs(hb[0]) ** 2 - np.abs(hb[1]) ** 2)
        p_k = np.abs(hb[0]) ** 2 + np.abs(hb[1]) ** 2  # (nk,nt)
        ph = np.nan_to_num(np.exp(2j * a_k) * p_k)
        p_kz = np.nan_to_num(p_k)
        # pooled per-window matrices
        # NOTE: both operands must share the frequency index label —
        # einsum("akt,blt->abt") would sum k and l independently (outer
        # product of the bin sums, rank-1 by construction)
        S_hh = np.einsum("akt,bkt->abt", hb, np.conj(hb))  # (2,2,nt)
        S_eh = np.einsum("ikt,jkt->ijt", eb, np.conj(hb))  # (2,2,nt)
        ok = finite.all(axis=0)  # windows with all bins finite
        P_w = np.real(S_hh[0, 0] + S_hh[1, 1])
        det = np.real(S_hh[0, 0] * S_hh[1, 1]
                      - S_hh[0, 1] * np.conj(S_hh[0, 1]))
        # near-degenerate windows (zero power or parallel h vectors in the
        # band) cannot be inverted — exclude from the single-event solve
        ok &= (P_w > 0) & (det > 1e-12 * P_w ** 2)
        A = S_hh.transpose(2, 0, 1)
        B = S_eh.transpose(2, 0, 1)
        Z = np.zeros_like(A)
        if ok.any():
            with np.errstate(all="ignore"):
                Z[ok] = np.linalg.solve(A[ok], B[ok])
        a_w = 0.5 * np.arctan2(2 * np.real(S_hh[0, 1]),
                               np.real(S_hh[0, 0] - S_hh[1, 1]))
        # cluster spread over the good windows
        good = ok & np.isfinite(Z).all(axis=(1, 2))
        runs = np.asarray(ds["run_id"].values)
        times = np.asarray(ds["time"].values)
        for rid in np.unique(runs):
            m_ = runs == rid
            idx = np.where(m_)[0]
            # per-run weighted polarization resultant + dominant angle
            pw = np.nan_to_num(P_w[idx])
            vec = np.nansum(pw * np.exp(2j * np.nan_to_num(a_w[idx])))
            r_w = float(abs(vec) / max(pw.sum(), 1e-30))
            theta = 0.5 * np.angle(vec) if pw.sum() > 0 else float("nan")
            # per-run per-bin weighted resultant (this run's windows only)
            r_run = (np.abs(ph[:, m_].sum(axis=1))
                     / np.maximum(p_kz[:, m_].sum(axis=1), 1e-30))
            # cluster spread over this run's good windows
            gg = good[idx]
            Zg = Z[idx[gg]]
            if Zg.shape[0] >= 4:
                sp = []
                for c in ((0, 1), (1, 0)):
                    zc = Zg[:, c[0], c[1]]
                    med = np.median(zc)
                    sp.append(np.median(np.abs(zc - med))
                              / max(abs(med), 1e-30))
                spread = float(np.median(sp))
            else:
                spread = float("nan")
            rows.append({
                "run_id": str(rid), "rate": float(ds.attrs["sample_rate"]),
                "n_win": int(m_.sum()),
                "pol_conc_bin": float(np.nanmedian(r_run)),
                "pol_conc_win": r_w,
                "pol_dir_deg": float(np.degrees(theta)) % 180.0
                if np.isfinite(theta) else float("nan"),
                "spread": spread,
                "first_time": str(times[idx[0]]) if idx.size else "",
            })
            # rose samples: good windows' pooled angles
            for w in idx[gg]:
                rose.append({"run_id": str(rid),
                             "alpha_deg": float(np.degrees(a_w[w])) % 180.0,
                             "power": float(P_w[w]),
                             "time": str(times[w])})
    return rows, rose


def enrich(rows):
    """Add hour-of-day and the existing impedance-level score context."""
    import pandas as pd

    # impedance-level context from the FC stage is recomputed cheaply here
    for r in rows:
        t = pd.Timestamp(r["first_time"])
        r["local_hour"] = ((t.value / 1e9 / 3600.0 + 8.0) % 24.0)
        r["utc_hour"] = (t.value / 1e9 / 3600.0) % 24.0
        r["family"] = r["run_id"].split("_")[0]
    return rows


if __name__ == "__main__":
    fc = build_fc()
    rows, rose = features(fc)
    rows = enrich(rows)
    (OUT / "polarization_features.json").write_text(
        json.dumps({"runs": rows, "rose": rose}, indent=1))
    print(f"wrote polarization_features.json: {len(rows)} runs, "
          f"{len(rose)} window angles")
    # quick console summary
    import collections

    by_fam = collections.defaultdict(list)
    for r in rows:
        by_fam[r["family"]].append(r)
    for fam, rs in sorted(by_fam.items()):
        pw = [r["pol_conc_win"] for r in rs if np.isfinite(r["pol_conc_win"])]
        sp = [r["spread"] for r in rs if np.isfinite(r["spread"])]
        night = [r for r in rs if r["local_hour"] < 7 or r["local_hour"] >= 19]
        day = [r for r in rs if 7 <= r["local_hour"] < 19]
        pwn = [r["pol_conc_win"] for r in night if np.isfinite(r["pol_conc_win"])]
        pwd = [r["pol_conc_win"] for r in day if np.isfinite(r["pol_conc_win"])]
        q = lambda v: np.percentile(v, 50) if v else float("nan")
        print(f"{fam}: n={len(rs)} pol_conc med={q(pw):.2f} "
              f"night={q(pwn):.2f} day={q(pwd):.2f} | spread med={q(sp):.2f}")
