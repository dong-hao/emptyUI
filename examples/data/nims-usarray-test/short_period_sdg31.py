"""SDG31 short-period gap investigation (DESIGN §24, E1 follow-up).

Baseline: boxcar + first-difference prewhitening (the E1-adopted starting
point), window 128, factor 4, 5 decimations, exact official band table,
fixed −5.7° rotation, Robust RME_RR.

Deficit region: the first three official periods (7.3 / 9.1 / 11.6 s) —
where the fluxgate noise floor is close to the signal level — plus two
borderline tail bands (~7.3 ks, ~11.9–18.7 ks).

Established here (2026-09-13):

1. The official product's σ (EDI .VAR) at the short periods is SMALL
   (σ/|Z| = 0.053 at 7.3 s) while the un-cleaned estimate honestly reports
   σ/|Z| ≈ 0.3–0.9 — the gap is a real selection/statistics difference,
   not within-error scatter.
2. Self-reference is hopeless below ~50 s (ρ underestimated by up to
   6.4 decades — the fluxgate noise floor dominates local H): the remote
   reference is doing enormous work; the user's noise hypothesis is
   confirmed quantitatively.
3. Window 64 does NOT fix the short periods (9.1 s gets worse by orders of
   magnitude) — consistent with EMTF's decset.cfg using 128-sample windows
   at ALL levels. Window reduction is not the classical lever.
4. **The dominant factor is remote-data coverage**: the official tag is
   SDG31b_H28coh (RUNLIST=SDG31a SDG31b, REMOTESITE=SDH28, coherence-
   processed by the EMTF FCU). Aligning against SDH28 run b only crops
   away the last ~29% of SDG31-b (SDH28-b ends 3 days earlier); the FCU
   uses all SDH28 data (runs b+c). With the full remote:
   - 9.1 s and 11.6 s are FIXED with no sorting at all (d_rho ≤ 0.07,
     d_phi ≤ 3.2°);
   - 7.3 s is fixed by mild coherence selection (weight, coherence_min=0.7:
     d_rho −0.046/−0.050, d_phi +3.7/−1.3 — 119/13348 windows retained);
   - the tail outliers shrink (11.9 ks: Δlog₁₀ρ +0.125 → +0.074) but mild
     sorting slightly regresses the deepest tail bands — selection belongs
     at the noise-floor end, not the tail.
5. Level-0 sort diagnostics (E–remote-H multiple coherence): 0.32/0.49/0.52
   at 7.3/9.1/11.6 s — the E field is only ~30–50% predictable from the
   remote H there; the official product's σ convention at noise-floor
   periods is far more optimistic than the data support (E5 must reconcile
   error conventions).

Run:  .venv/bin/python examples/data/nims-usarray-test/short_period_sdg31.py
Outputs: printed tables + sdg31_short_period_metrics.json
"""

from __future__ import annotations

import json
import tempfile

import numpy as np
from emptyui.context import RunContext
from emptyui.registry import build_default_registry
from emptyui.store import DataStore
from emptyui_aurora._processing import (
    coherence_sort_fc,
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
HEAD_MAX_S = 15.0   # the noise-floor head (the deficit region)
SHORT_MAX_S = 60.0  # per-period report covers the level-0 bands


def sorted_levels(fc_l, fc_r, ins, outs, bands, mode, coherence_min):
    """Coherence-sort every level on the SAME exact band table."""
    out_l, out_r, diag = {}, {}, {}
    for lv in sorted(fc_l):
        l_out, r_out, rows = coherence_sort_fc(
            fc_l[lv], fc_r.get(lv), ins, outs,
            coherence_min=coherence_min, coherence_max=1.0, mode=mode)
        out_l[lv] = l_out
        out_r[lv] = r_out
        diag[lv] = rows
    return out_l, out_r, diag


def run_variant(local, remote, bands, ins, outs, *, window=None, sort=None,
                coherence_min=0.25, remote_reference=True):
    """One variant -> (periods, Z, Z_err, level-0 diagnostics)."""
    window = window or STFT_BASE["num_samples_window"]
    kw = {**STFT_BASE, "num_samples_window": window,
          "window_type": "boxcar", "prewhitening": "first difference"}
    fc_l = compute_fc_levels(local, "", "", **kw)
    fc_r = compute_fc_levels(remote, "", "", **kw)
    diag = None
    if sort is not None:
        fc_l, fc_r, diag = sorted_levels(fc_l, fc_r, ins, outs, bands,
                                         sort, coherence_min)
    per = {}
    for lv in sorted(fc_l):
        p, Z, Ze = estimate_tf(
            fc_l[lv], ins, outs, estimator_engine="Robust",
            remote_reference=remote_reference, periods_per_octave=3,
            ref_fc=fc_r.get(lv) if remote_reference else None,
            band_style="l2u", bands_override=bands.get(lv, []))
        per[lv] = {"periods": p, "Z": Z, "Z_err": Ze,
                   "input_channels": ins, "output_channels": outs,
                   "channel_nomenclature": "default"}
    periods, Z, Z_err = merge_tf_levels(per)
    return periods, Z[:, :2, :2], Z_err[:, :2, :2], diag


def short_table(periods, Z, periods_off, zo, label):
    """Per-period deltas for the official rows up to SHORT_MAX_S."""
    Zc = rotate_tensor(Z, ROTATION_DEG)
    rows = tensor_compare(periods, Zc, periods_off, zo)
    print(f"  {label}")
    for r in rows:
        if r["period_official"] > SHORT_MAX_S:
            break
        print(f"    T={r['period_official']:>6.1f}  d_rho xy/yx "
              f"{r['delta_log10_rho_xy']:+.3f}/{r['delta_log10_rho_yx']:+.3f}  "
              f"d_phi xy/yx {r['delta_phi_deg_xy']:+.1f}/{r['delta_phi_deg_yx']:+.1f}")
    return rows


def main():
    reg = build_default_registry()
    periods_off, zo, var_o = read_official_edi()
    bands = exact_official_bands(1.0 / np.asarray(periods_off))

    print("official sigma (EDI .VAR, SI) at short periods:")
    for i, p in enumerate(periods_off[:5]):
        sxy = np.sqrt(var_o["xy"][i]) * CONV
        zoff = np.abs(zo["zxy"][i]) * _to_si(1.0)
        print(f"  T={p:>6.1f}  sigma_off/|Z_off| = {sxy / zoff:.3f}")

    with tempfile.TemporaryDirectory() as ws:
        ctx = RunContext(workspace=ws, store=DataStore())
        print("\nloading + calibrating + aligning ...")
        local = load_calibrated(reg, ctx, "SDG31", "b")
        remote_b = load_calibrated(reg, ctx, "SDH28", "b")
        remote_c = load_calibrated(reg, ctx, "SDH28", "c")
        remote_full = remote_b.merge(remote_c, gap_method="slinear")
        print(f"  remote b: {remote_b.dataset.sizes['time']} samples, "
              f"c: {remote_c.dataset.sizes['time']}, "
              f"b+c: {remote_full.dataset.sizes['time']}")

        al = reg.get("mth5.runs.align").execute(ctx, local, remote_full)
        local_al, remote_al = al["data1"], al["data2"]

        kw0 = {**STFT_BASE, "window_type": "boxcar",
               "prewhitening": "first difference"}
        fc0 = compute_fc_levels(local_al, "", "", **kw0)
        s0 = next(iter(fc0.values()))
        ins, outs, _ = classify_channels(
            [str(c) for c in s0.data_vars],
            s0.attrs.get("channel_nomenclature", "default"))

        results = {"protocol": {**STFT_BASE, "window_type": "boxcar",
                                "prewhitening": "first difference",
                                "bands": "exact", "rotation_deg": ROTATION_DEG},
                   "variants": {}}

        def keep(label, rows, diag0=None):
            results["variants"][label] = {"stats": summarize_split(rows, label)}
            if diag0 is not None:
                results["variants"][label]["level0_diagnostics"] = diag0

        # ---- 1. baseline with b-only remote (the E1 configuration) ------
        al_b = reg.get("mth5.runs.align").execute(ctx, local, remote_b)
        periods, Z, _Ze, _ = run_variant(al_b["data1"], al_b["data2"], bands,
                                        ins, outs)
        rows = short_table(periods, Z, periods_off, zo,
                           "baseline (remote b only, no sort)")
        keep("remote_b_only", rows)

        # ---- 2. sort diagnostics on the full remote ---------------------
        _, _, diag = sorted_levels(
            {0: fc0[0]}, {0: compute_fc_levels(remote_al, "", "", **kw0)[0]},
            ins, outs, bands, "weight", 0.25)
        print("\n  level-0 sort diagnostics (median coh per band):")
        print(f"  {'T(s)':>7} {'n_win':>6} {'coh_E':>7} {'coh_xfer':>9}")
        for r in diag[0]:
            print(f"  {r['period']:>7.2f} {r['n_windows']:>6} "
                  f"{r['median_coherence_e']:>7.3f} "
                  f"{(r['median_coherence_transfer'] or np.nan):>9.3f}")
        results["level0_diagnostics"] = diag[0]

        # ---- 3. full remote coverage (the official data selection) ------
        periods, Z, _Ze, _ = run_variant(local_al, remote_al, bands, ins, outs)
        rows = short_table(periods, Z, periods_off, zo,
                           "remote b+c, no sort")
        keep("remote_bc_no_sort", rows)

        # ---- 4. coherence sorting on the full remote --------------------
        for mode, cmin in (("weight", 0.7), ("drop", 0.5), ("weight", 0.25)):
            periods, Z, _Ze, diag0 = run_variant(local_al, remote_al, bands,
                                                ins, outs, sort=mode,
                                                coherence_min=cmin)
            rows = short_table(periods, Z, periods_off, zo,
                               f"remote b+c, sort {mode} (coh_min={cmin})")
            keep(f"remote_bc_sort_{mode}_{cmin}", rows, diag0)

        # ---- 5. self-reference (quantifies the fluxgate noise floor) ----
        periods, Z, _Ze, _ = run_variant(local_al, remote_al, bands, ins, outs,
                                         remote_reference=False)
        rows = short_table(periods, Z, periods_off, zo,
                           "self-reference (RR off)")
        keep("self_ref", rows)

        # ---- 6. window 64 (reference — not the classical lever) ---------
        periods, Z, _Ze, _ = run_variant(local_al, remote_al, bands, ins, outs,
                                         window=64)
        rows = short_table(periods, Z, periods_off, zo, "window 64 (reference)")
        keep("window64", rows)

        out = DATA_DIR / "sdg31_short_period_metrics.json"
        out.write_text(json.dumps(results, indent=1, default=float))
        print(f"\nmetrics -> {out}")


if __name__ == "__main__":
    main()
