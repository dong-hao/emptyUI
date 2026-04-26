"""SDG31 long-period investigation (history §24, E2 + tail probes).

Baseline: the E1b working recipe — boxcar + first-difference PW, window 128
samples constant across levels, factor 4, 5 decimations, exact official
band table, −5.7° rotation, Robust RME_RR, full remote (SDH28 b+c merged).

Focus: the tail mismatches (T > ~2 ks). Level structure of the tail:
level 4 (rate 1/256 Hz, window 32768 s ≈ 9.1 h) covers 1024–8192 s with
~50 windows; level 5 (rate 1/1024 Hz, window 131072 s ≈ 1.52 d) covers
4096–32768 s with only ~13 windows. The diurnal (Sq) magnetic variation
lives at 86 ks — inside EVERY level-4/5 window — so window-level trends
are a physical contamination candidate here (E2: per-window detrend, which
aurora applies by default and emptyUI does not).

Probes:
1. taper × per-window-detrend ablation on the tail (boxcar/hamming/hann ×
   detr off/linear);
2. level-resolved tail estimates (level 4 vs level 5 separately matched
   against the official periods they cover) — separates "deepest-wins
   merge choice" from "level-5 estimate quality";
3. official σ at the tail rows.

Run:  .venv/bin/python examples/data/nims-usarray-test/long_period_sdg31.py
"""

from __future__ import annotations

import json
import tempfile

import numpy as np
from emptyui.context import RunContext
from emptyui.registry import build_default_registry
from emptyui.store import DataStore
from emptyui_aurora._processing import (
    compute_fc_levels,
    estimate_tf,
    merge_tf_levels,
)
from emptyui_mth5.nomenclature import classify_channels
from parity_sdg31 import STFT_BASE, summarize_split
from verify_sdg31 import (
    DATA_DIR,
    MU0,
    ROTATION_DEG,
    _to_si,
    exact_official_bands,
    load_calibrated,
    read_official_edi,
    rotate_tensor,
    tensor_compare,
)

CONV = 1e-6 * MU0 / 1e-9
TAIL_MIN_S = 1000.0


def chain(local, remote, bands, ins, outs, *, taper="boxcar",
          detr="", levels=None, sort=None, coh_min=0.7):
    """Recipe chain with ablation knobs; returns (periods, Z, Z_err)."""
    kw = {**STFT_BASE, "window_type": taper, "prewhitening": "first difference",
          "per_window_detrend": detr}
    fc_l = compute_fc_levels(local, "", "", **kw)
    fc_r = compute_fc_levels(remote, "", "", **kw)
    if sort:
        from emptyui_aurora._processing import coherence_sort_fc
        for lv in sorted(fc_l):
            fc_l[lv], fc_r[lv], _ = coherence_sort_fc(
                fc_l[lv], fc_r.get(lv), ins, outs, coherence_min=coh_min,
                coherence_max=1.0, mode="weight")
    per = {}
    for lv in sorted(fc_l):
        if levels is not None and lv not in levels:
            continue
        p, Z, Ze = estimate_tf(
            fc_l[lv], ins, outs, estimator_engine="Robust",
            remote_reference=True, periods_per_octave=3,
            ref_fc=fc_r.get(lv), band_style="l2u",
            bands_override=bands.get(lv, []))
        per[lv] = {"periods": p, "Z": Z, "Z_err": Ze,
                   "input_channels": ins, "output_channels": outs,
                   "channel_nomenclature": "default"}
    periods, Z, Z_err = merge_tf_levels(per)
    return periods, Z[:, :2, :2], Z_err[:, :2, :2]


def tail_table(periods, Z, periods_off, zo, label):
    """Per-period deltas for official rows above TAIL_MIN_S."""
    Zc = rotate_tensor(Z, ROTATION_DEG)
    rows = tensor_compare(periods, Zc, periods_off, zo)
    print(f"  {label}")
    for r in rows:
        if r["period_official"] < TAIL_MIN_S:
            continue
        print(f"    T={r['period_official']:>9.1f}  d_rho xy/yx "
              f"{r['delta_log10_rho_xy']:+.3f}/{r['delta_log10_rho_yx']:+.3f}  "
              f"d_phi xy/yx {r['delta_phi_deg_xy']:+.1f}/{r['delta_phi_deg_yx']:+.1f}")
    return rows


def main():
    reg = build_default_registry()
    periods_off, zo, var_o = read_official_edi()
    bands = exact_official_bands(1.0 / np.asarray(periods_off))

    with tempfile.TemporaryDirectory() as ws:
        ctx = RunContext(workspace=ws, store=DataStore())
        print("loading + calibrating + aligning (full remote b+c) ...")
        local = load_calibrated(reg, ctx, "SDG31", "b")
        remote = load_calibrated(reg, ctx, "SDH28", "b").merge(
            load_calibrated(reg, ctx, "SDH28", "c"), gap_method="slinear")
        al = reg.get("mth5.runs.align").execute(ctx, local, remote)
        L, R = al["data1"], al["data2"]
        print(f"  aligned span: {L.dataset.sizes['time']} samples "
              f"({L.dataset.sizes['time'] / 86400:.2f} d)")

        # level geometry: window duration and window count per deep level
        print("\nlevel geometry (recipe, window 128 samples):")
        for lv in sorted(compute_fc_levels(L, "", "", **{**STFT_BASE,
                "window_type": "boxcar",
                "prewhitening": "first difference"}).keys()):
            rate = 1.0 / 4 ** lv
            n_win = L.dataset.sizes["time"] / 4 ** lv
            nseg = int((n_win - 64) // 64) + 1
            print(f"  level {lv}: rate {rate:.6g} Hz, window {128 / rate:.0f} s"
                  f" ({128 / rate / 3600:.1f} h), ~{nseg} windows")

        from emptyui_aurora._processing import compute_fc_levels as cfl
        kw0 = {**STFT_BASE, "window_type": "boxcar",
               "prewhitening": "first difference"}
        s0 = next(iter(cfl(L, "", "", **kw0).values()))
        ins, outs, _ = classify_channels(
            [str(c) for c in s0.data_vars],
            s0.attrs.get("channel_nomenclature", "default"))

        results = {"protocol": {**STFT_BASE, "window_type": "boxcar",
                                "prewhitening": "first difference",
                                "per_window_detrend": "off",
                                "bands": "exact", "rotation_deg": ROTATION_DEG,
                                "remote": "SDH28 b+c"},
                   "variants": {}}

        def keep(label, rows):
            results["variants"][label] = {"stats": summarize_split(rows, label)}

        # ---- 1. taper x per-window-detrend ablation ---------------------
        for taper, detr in (("boxcar", ""), ("boxcar", "linear"),
                            ("hamming", ""), ("hamming", "linear"),
                            ("hanning", "")):
            periods, Z, _Ze = chain(L, R, bands, ins, outs, taper=taper,
                                    detr=detr)
            rows = tail_table(periods, Z, periods_off, zo,
                              f"{taper} + PW" + (f" + detr({detr})" if detr else ""))
            keep(f"{taper}_detr-{detr or 'off'}", rows)

        # ---- 2. level-resolved tail estimates ---------------------------
        print("\nlevel-resolved tail (which level owns each official period):")
        for lv in (4, 5):
            periods, Z, _Ze = chain(L, R, bands, ins, outs, levels={lv})
            Zc = rotate_tensor(Z, ROTATION_DEG)
            rows = tensor_compare(periods, Zc, periods_off, zo)
            print(f"  level {lv} (window {128 * 4 ** lv / 3600:.1f} h):")
            for r in rows:
                print(f"    T={r['period_official']:>9.1f}  d_rho "
                      f"{r['delta_log10_rho_xy']:+.3f}/{r['delta_log10_rho_yx']:+.3f}  "
                      f"d_phi {r['delta_phi_deg_xy']:+.1f}/{r['delta_phi_deg_yx']:+.1f}")
            keep(f"level{lv}_only", rows)

        # ---- 3. official sigma at the tail ------------------------------
        print("\nofficial sigma at tail rows:")
        for i, p in enumerate(periods_off):
            if p < TAIL_MIN_S:
                continue
            sxy = np.sqrt(var_o["xy"][i]) * CONV
            zoff = np.abs(zo["zxy"][i]) * _to_si(1.0)
            print(f"  T={p:>9.1f}  sigma_off/|Z_off| = {sxy / zoff:.3f}")

        out = DATA_DIR / "sdg31_long_period_metrics.json"
        out.write_text(json.dumps(results, indent=1, default=float))
        print(f"\nmetrics -> {out}")


if __name__ == "__main__":
    main()
