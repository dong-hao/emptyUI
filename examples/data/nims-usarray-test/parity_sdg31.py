"""SDG31 physics-parity ablation (DESIGN §24): which classical-EMTF FC-stage
factor closes the gap to the official TF?

Fixed protocol (§24.1 — nothing varies but the stated factor):
- band table reconstructed EXACTLY from the official EDI frequencies
  (bands_override; the l2u grid difference must not confound physics);
- fixed −5.7° true-north rotation (documented frame rotation, §16.3);
- Robust (RME_RR) estimator with the remote reference, aurora defaults;
- STFT base: window 128 samples, factor 4, 5 decimations, 50% overlap.

Ablation grid (E1, §24.3): taper ∈ {hamming, boxcar, hann} × prewhitening
∈ {off, first difference}. Metrics: paired |Δlog₁₀ρ| / |Δφ| (folded) at the
matched official periods, split into short-period [15, 100) s (the leakage
-limited target) and the reliable tail [100, 18 725] s; plus error-bar
agreement (our σ_Z vs the official EDI `.VAR` — E5's first metric).

Run:  .venv/bin/python examples/data/nims-usarray-test/parity_sdg31.py
Outputs (next to the EDI): sdg31_parity_metrics.json, and a printed table.
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
from verify_sdg31 import (
    DATA_DIR,
    MU0,
    ROTATION_DEG,
    _row_dict,
    exact_official_bands,
    load_calibrated,
    read_official_edi,
    rotate_tensor,
    tensor_compare,
)

RELIABLE_MIN_S = 15.0
SPLIT_S = 100.0  # short-period focus split
GATE_RHO_DECADES = 0.1
GATE_PHASE_DEG = 10.0

STFT_BASE = {
    "num_samples_window": 128, "num_decimations": 5, "factor": 4, "overlap": 0.5,
}
TAPERS = ("hamming", "boxcar", "hann")
PWS = ("", "first difference")


def summarize_split(rows, label):
    """Paired delta statistics on the matched grid, split into the
    noise-floor head [official start, 15) s, the short-period target
    [15, 100) s, and the reliable tail [100, 18 725] s."""
    out = {"label": label}
    for name, sel in (
            ("head", lambda r: r["period_official"] < RELIABLE_MIN_S),
            ("short_period", lambda r: RELIABLE_MIN_S <= r["period_official"] < SPLIT_S),
            ("reliable_tail", lambda r: SPLIT_S <= r["period_official"]),
            ("reliable_full", lambda r: r["period_official"] >= RELIABLE_MIN_S),
    ):
        group = [r for r in rows if sel(r)]
        if not group:
            out[name] = None
            continue
        dr = np.abs([r["delta_log10_rho_xy"] for r in group]
                    + [r["delta_log10_rho_yx"] for r in group])
        dp = np.abs([r["delta_phi_deg_xy"] for r in group]
                    + [r["delta_phi_deg_yx"] for r in group])
        out[name] = {
            "n": len(group),
            "median_abs_log10_rho": float(np.median(dr)),
            "p90_abs_log10_rho": float(np.percentile(dr, 90)),
            "median_abs_phase_deg": float(np.median(dp)),
            "p90_abs_phase_deg": float(np.percentile(dp, 90)),
            "pass_rho": float(np.mean(dr <= GATE_RHO_DECADES)),
            "pass_phase": float(np.mean(dp <= GATE_PHASE_DEG)),
        }
    return out


def error_agreement(idx, Z_err, var_o):
    """Our σ_Z (std of the complex element) vs the official EDI σ = sqrt(.VAR),
    per off-diagonal component at matched periods (E5 metric)."""
    out = {}
    for (r, c), name in (((0, 1), "xy"), ((1, 0), "yx")):
        ours = Z_err[idx, r, c]
        official = np.sqrt(var_o[name][[i for i in idx]])  # variance -> std
        # official VAR is in (mV/km/nT)² — same SI conversion as the tensor
        official = official * 1e-6 * MU0 / 1e-9
        ratio = ours / official
        ratio = ratio[np.isfinite(ratio) & (ratio > 0)]
        out[name] = {
            "median_sigma_ratio": float(np.median(ratio)),
            "p10_sigma_ratio": float(np.percentile(ratio, 10)),
            "p90_sigma_ratio": float(np.percentile(ratio, 90)),
        }
    return out


def run_config(local, remote, bands, taper, pw):
    """Full chain for one (taper, prewhitening) config -> (periods, Z, Z_err)."""
    kw = {**STFT_BASE, "window_type": taper, "prewhitening": pw}
    fc_l = compute_fc_levels(local, "", "", **kw)
    fc_r = compute_fc_levels(remote, "", "", **kw)
    sample = next(iter(fc_l.values()))
    kw_nom = sample.attrs.get("channel_nomenclature", "default")
    ins, outs, _ = classify_channels(
        [str(c) for c in sample.data_vars], kw_nom)
    per = {}
    for lv in sorted(fc_l):
        p, Z, Ze = estimate_tf(
            fc_l[lv], ins, outs, estimator_engine="Robust",
            remote_reference=True, periods_per_octave=3,
            ref_fc=fc_r.get(lv), band_style="l2u",
            bands_override=bands.get(lv, []))
        per[lv] = {"periods": p, "Z": Z, "Z_err": Ze,
                   "input_channels": ins, "output_channels": outs,
                   "channel_nomenclature": kw_nom}
    periods, Z, Z_err = merge_tf_levels(per)
    return periods, Z[:, :2, :2], Z_err[:, :2, :2]


def plot_variant(taper: str, pw: str, remote_runs: tuple[str, ...] = ("b",),
                 coh_min: float | None = None) -> None:
    """Render one variant against the official EMTF product (§24 protocol):
    the raw overlay (each dataset at its own periods, ours rotated −5.7°)
    and the §15 EMTF-convention AppRes & Phase figure via the plot node.

    ``remote_runs`` selects the remote run(s) — all runs are merged
    physically (RunTS.merge) before alignment, matching the FCU's
    full-remote data selection (§24.3 E1b). ``coh_min`` optionally applies
    weight-mode coherence selection on the exact band table (the head
    remedy).
    """
    import matplotlib
    matplotlib.use("Agg")
    from emptyui_aurora._processing import coherence_sort_fc
    from emptyui_mth5.nomenclature import classify_channels
    from verify_sdg31 import raw_overlay_figure, render_appres_figure

    tag = f"{taper}_{pw.replace(' ', '_') or 'nopw'}"
    if remote_runs != ("b",):
        tag += "_" + "-".join(remote_runs)
    if coh_min is not None:
        tag += f"_sort{coh_min:g}"
    reg = build_default_registry()
    periods_off, zo, _ = read_official_edi()
    bands = exact_official_bands(1.0 / np.asarray(periods_off))
    with tempfile.TemporaryDirectory() as ws:
        ctx = RunContext(workspace=ws, store=DataStore())
        print("loading + calibrating + aligning ...")
        local = load_calibrated(reg, ctx, "SDG31", "b")
        remote = None
        for rr in remote_runs:
            rt = load_calibrated(reg, ctx, "SDH28", rr)
            remote = rt if remote is None else remote.merge(rt, gap_method="slinear")
        al = reg.get("mth5.runs.align").execute(ctx, local, remote)
        L, R = al["data1"], al["data2"]
        from emptyui_aurora._processing import compute_fc_levels
        kw = {**STFT_BASE, "window_type": taper, "prewhitening": pw}
        s0 = next(iter(compute_fc_levels(L, "", "", **kw).values()))
        ins, outs, _ = classify_channels(
            [str(c) for c in s0.data_vars],
            s0.attrs.get("channel_nomenclature", "default"))
        fc_l = compute_fc_levels(L, "", "", **kw)
        fc_r = compute_fc_levels(R, "", "", **kw)
        if coh_min is not None:
            for lv in sorted(fc_l):
                fc_l[lv], fc_r[lv], _ = coherence_sort_fc(
                    fc_l[lv], fc_r.get(lv), ins, outs,
                    coherence_min=coh_min, coherence_max=1.0, mode="weight")
        per = {}
        for lv in sorted(fc_l):
            p, Zl, Zel = estimate_tf(
                fc_l[lv], ins, outs, estimator_engine="Robust",
                remote_reference=True, periods_per_octave=3,
                ref_fc=fc_r.get(lv), band_style="l2u",
                bands_override=bands.get(lv, []))
            per[lv] = {"periods": p, "Z": Zl, "Z_err": Zel,
                       "input_channels": ins, "output_channels": outs,
                       "channel_nomenclature": "default"}
        periods, Z, Z_err = merge_tf_levels(per)
        Z, Z_err = Z[:, :2, :2], Z_err[:, :2, :2]
        sel = (f", full remote ({'+'.join(remote_runs)})" if len(remote_runs) > 1
               else "")
        title = (f"emptyUI ({taper} + {pw or 'no'} PW{sel}, win 128, "
                 f"exact EMTF bands"
                 f"{f', weight sort {coh_min:g}' if coh_min is not None else ''})"
                 f" vs official")
        raw_overlay_figure(
            periods, Z, str(DATA_DIR / f"sdg31_parity_{tag}_vs_official.png"),
            title, periods_off, zo)
        print("raw overlay ->",
              DATA_DIR / f"sdg31_parity_{tag}_vs_official.png")
        render_appres_figure(
            reg, ctx, tag, periods, Z, Z_err,
            f"sdg31_parity_{tag}_appres_phase.png",
            note=f"{taper}+{pw or 'noPW'}"
                 f"{'+'.join(remote_runs) if len(remote_runs) > 1 else ' remote'}"
                 f"{' sort ' + format(coh_min, 'g') if coh_min is not None else ''}")


def main():
    reg = build_default_registry()
    periods_off, zo, var_o = read_official_edi()
    # exact_official_bands takes frequencies; read_official_edi returns periods
    freqs_off = 1.0 / np.asarray(periods_off)
    bands = exact_official_bands(freqs_off)

    with tempfile.TemporaryDirectory() as ws:
        ctx = RunContext(workspace=ws, store=DataStore())
        print("loading + calibrating stations ...")
        local = load_calibrated(reg, ctx, "SDG31", "b")
        remote = load_calibrated(reg, ctx, "SDH28", "b")
        print("aligning runs ...")
        al = reg.get("mth5.runs.align").execute(ctx, local, remote)
        local, remote = al["data1"], al["data2"]

        results = {"protocol": {
            "rotation_deg": ROTATION_DEG,
            "bands": "exact official reconstruction",
            "estimator": "Robust (RME_RR), remote reference",
            "stft_base": STFT_BASE,
        }, "variants": {}}

        print(f"{'taper':>8} {'pw':>16} | {'short med/p90':>13} "
              f"{'tail med/p90':>13} {'pass%':>6} | {'phase med':>9}")
        for taper in TAPERS:
            for pw in PWS:
                label = f"{taper}+{pw or 'nopw'}"
                periods, Z, Z_err = run_config(local, remote, bands, taper, pw)
                Z_rot = rotate_tensor(Z, ROTATION_DEG)
                rows = tensor_compare(periods, Z_rot, periods_off, zo, var_o)
                stats = summarize_split(rows, label)
                from verify_sdg31 import match_periods
                mi = match_periods(periods, periods_off)
                stats["error_bars"] = error_agreement(mi, Z_err, var_o)
                results["variants"][label] = {
                    "kwargs": {"window_type": taper, "prewhitening": pw,
                               **STFT_BASE},
                    "stats": stats,
                    "rows": [_row_dict(r) for r in rows],
                }
                s, t = stats["short_period"], stats["reliable_tail"]
                print(f"{taper:>8} {pw or 'off':>16} | "
                      f"{s['median_abs_log10_rho']:.3f}/{s['p90_abs_log10_rho']:.3f}   "
                      f"{t['median_abs_log10_rho']:.3f}/{t['p90_abs_log10_rho']:.3f}   "
                      f"{100 * min(s['pass_rho'], s['pass_phase']):>5.1f} | "
                      f"{s['median_abs_phase_deg']:>9.2f}")

        out = DATA_DIR / "sdg31_parity_metrics.json"
        out.write_text(json.dumps(results, indent=1))
        print(f"\nmetrics -> {out}")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--plot", nargs="+",
                    help="render comparison figures for one variant: taper, "
                         "pw ('' or 'first difference'), optional remote "
                         "runs (e.g. 'b,c' — merged physically before "
                         "alignment), optional coherence_min (weight-mode "
                         "selection; omit for no sort)")
    args = ap.parse_args()
    if args.plot:
        remote_runs = (tuple(args.plot[2].split(",")) if len(args.plot) > 2
                       else ("b",))
        coh_min = float(args.plot[3]) if len(args.plot) > 3 else None
        plot_variant(args.plot[0], args.plot[1], remote_runs, coh_min)
    else:
        main()
