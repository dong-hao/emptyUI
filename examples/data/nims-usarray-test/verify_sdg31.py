"""SDG31 verification: emptyUI processing vs the official EMTF (SPUD) TF.

Pins the emptyUI MT processing chain against a ground-truth transfer
function (DESIGN.md §16.3):

1. Loads the 4P.SDG31 (and 4P.SDH28 remote reference) MTH5 archives built by
   ``build_mth5.py`` (1 Hz, NIMS/100 m dipole, counts -> mV/km / nT via the
   archived responses — Calibrate step).
2. Runs the emptyUI processing chain headlessly: Extract Run -> Calibrate ->
   Align Runs (+SDH28) -> Compute FC -> [Coherence Sort] -> Estimate TF
   (Robust / RME_RR). Parameters are inferred from the official EDI's exact
   frequency grid (window 128 s, factor 4, 6 decimation levels). Two grid
   modes are compared: the band table reconstructed EXACTLY from the
   official band-center frequencies (the reference), and the §14.9 default
   method ``l2u`` (reference-template + log-uniform window-starvation
   coarsening — the Estimate TF node default) at window 128 and 256.
3. Compares against the official USArray.SDG31.2017.edi (EMTF, robust remote
   reference vs SDH28) at the official periods: apparent resistivity + phase
   for xy/yx, error statistics, and overlay figures in the USArray style
   (DESIGN §15 conventions; official = filled ○/✕, emptyUI = hollow □/◇).

Findings recorded here (2026-09-10):
- The xy-down / yx-up systematic offset (from ~100 s) is a COORDINATE-FRAME
  ROTATION of ~5.5°: the official EDI is in the original site-layout frame
  (compass azimuths 5.7°/95.7°), our data use the geographic FDSN azimuths
  (4.26°/94.26°) — the difference is the IGRF declination (4.26°, stored in
  the archive) plus the 1.4° site-azimuth difference. Rotating our tensor
  into the official frame removes the offset (see metrics: "rotation_fit").
- The T < 15 s end is noise-limited in the 1 Hz archive; window 64 +
  coherence sorting recovers 9-15 s to within ~10-40% (the intermittent
  signal is diluted by 128 s windows), while 7.3 s stays degenerate.
- §14.9 frequency-grid determination (delivered): the ``l2u`` scheme is the
  Estimate TF node default. The comparison is nearest-period MATCHING (one
  row per official period); the overlay figures draw each dataset at its
  OWN evaluated periods and only the matched rows, so a figure can show
  fewer ours points than the grid has. ROTATION-CORRECTED (−5.7°,
  documented frame rotation, applied to every variant): the exact table
  gives 0.019 median / 0.045 p90 / 98% pass; the l2u default gives
  0.019 / 0.089 / 93% at window 128 and 0.020 / 0.071 / 93% at window 256
  (the grid is window-invariant — the template scales ×window/128 — but
  the finer bins at window 256 improve the p90; see DESIGN §14.9 for the
  full history). The metrics JSON "rows" carries the official->ours period
  mapping + per-period deltas; "n_distinct_ours" detects double-matching.

Run:  .venv/bin/python examples/data/nims-usarray-test/verify_sdg31.py
Outputs (next to the EDI): sdg31_plot_<variant>_vs_official.png RAW
comparison figures (no period matching — each dataset at its own evaluated
periods; emptyUI rotated −5.7° to true north),
sdg31_plot_appres_phase.png (EMTF-convention AppRes & Phase figure via the
Plot AppRes & Phase node — the head-to-head counterpart of
SDG31b_H28coh.png) and sdg31_verify_metrics.json.
"""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

import numpy as np
from emptyui.context import RunContext
from emptyui.registry import build_default_registry
from emptyui.store import DataStore
from emptyui_aurora._processing import (
    _l2u_band_table,
    build_tf,
    coherence_sort_fcs,
    compute_fc_levels,
    estimate_tf,
    merge_tf_levels,
)
from emptyui_mth5.nomenclature import classify_channels
from emptyui_mth5.tf import rotate_transfer_function

MU0 = 4.0 * np.pi * 1e-7
DATA_DIR = Path(__file__).resolve().parent
OFFICIAL_EDI = DATA_DIR / "USArray.SDG31.2017.edi"
LOCAL = ("SDG31", "b")
REMOTE = ("SDH28", "b")

#: Inferred from the official EDI frequency grid (window 128 samples at
#: 1 Hz, factor 4, 6 decimation levels, classic EMTF band ladder).
FC_KWARGS = {"num_decimations": 5, "factor": 4, "overlap": 0.5, "window_type": "hamming"}
EST_KWARGS = {"estimator_engine": "Robust", "periods_per_octave": 3,
                  "band_style": "l2u"}

#: Comparison gates (DESIGN §16.3): the well-estimated band [15 s, 18 725 s]
#: must agree within these; the < 15 s end is expected to be noise-limited.
RELIABLE_MIN_S = 15.0
GATE_RHO_DECADES = 0.1   # ~26%
GATE_PHASE_DEG = 10.0

#: Frame rotation applied to compare with the official EMTF product
#: (DESIGN §16.3): USArray data are acquired in the geomagnetic frame; the
#: official product was rotated to true north ("Angle To GG North: 0.00").
#: Standard convention: angle positive = clockwise from true north;
#: magnetic -> true north with declination D => angle = -D (here -5.7 deg).
ROTATION_DEG = -5.7

#: §14.9 l2u threshold: a band is kept only if its total samples
#: (n_windows × band width) exceed this; below it the level's grid is
#: REBUILT uniformly (see _l2u_band_table). Matches the Estimate TF node
#: default.
COARSEN_THRESHOLD_BINS = 40

#: §14.9 adaptive grid: the official USArray head clip (1 Hz NIMS band is
#: noise-limited below ~7 s — Kelbert et al. 2026; the official grid starts
#: at 7.314 s). 0 = start at the template's shortest band (default).
OFFICIAL_T_SHORT = 7.314


# --------------------------------------------------------------------------
# data + processing
# --------------------------------------------------------------------------

def load_calibrated(reg, ctx, station, run):
    """Load MTH5 -> Extract Run -> Calibrate (archived responses, counts ->
    mV/km / nT)."""
    m = reg.get("mth5.io.load").execute(
        ctx, path=str(DATA_DIR / f"4P_{station}.h5"), mode="r")["mth5"]
    data = reg.get("mth5.io.extract_run").execute(
        ctx, m, survey="4P", station=station, run=run,
        selection="run id")["data"]
    return reg.get("mth5.calibrate").execute(ctx, data)["data"]


def exact_official_bands(freqs_off):
    """(f_hi, f_lo) intervals per decimation level rebuilt from the official
    EDI band-center frequencies (6,6,6,6,4,2 per level — classic EMTF
    structure; band edges at the geometric midpoints between centers)."""
    counts = (6, 6, 6, 6, 4, 2)
    out = {}
    pos = 0
    for lv in range(len(counts)):
        cs = freqs_off[pos:pos + counts[lv]]
        pos += counts[lv]
        r = np.sqrt(cs[0] / cs[1])
        edges = [cs[0] * r]
        for i in range(len(cs) - 1):
            edges.append(np.sqrt(cs[i] * cs[i + 1]))
        edges.append(cs[-1] / np.sqrt(cs[-2] / cs[-1]))
        out[lv] = [(edges[i], edges[i + 1]) for i in range(len(edges) - 1)]
    return out


def run_variant(reg, ctx, local, remote, freqs_off, window=128,
                sort=False, sort_mode="drop", bands_mode="exact",
                min_bins_per_band=COARSEN_THRESHOLD_BINS,
                t_short=0.0):
    """Processing chain; returns (periods, Z (n,2,2) V/m per A/m, Z_err, tf).

    ``bands_mode``: "exact" (reconstructed official band table — the
    reference), "l2u" (the §14.9 default frequency-grid method:
    reference-template + log-uniform window-starvation coarsening, exposed
    via ``min_bins_per_band``/``t_short``), or "auto" (None — the
    per-level fallback walk).
    """
    if remote is not None:
        al = reg.get("mth5.runs.align").execute(ctx, local, remote)
        local, remote = al["data1"], al["data2"]
    fc_l = compute_fc_levels(local, "", "", num_samples_window=window,
                             **FC_KWARGS)
    # the ComputeFC node attaches station identity to the fc attrs
    # (survey/station/run + lat/lon/elev); the low-level call does not —
    # replicate it so the TF/plot carry the same provenance
    from emptyui_aurora.process import _station_identity_attrs
    from emptyui_mth5.persist import runts_context

    surv, stat, run_id = runts_context(local, "", "", "")
    identity = _station_identity_attrs(local, surv, stat, run_id)
    for lv, fc in fc_l.items():
        fc.attrs.update(identity)
    fc_r = None
    if remote is not None:
        fc_r = compute_fc_levels(remote, "", "", num_samples_window=window,
                                 **FC_KWARGS)
        if sort:
            sample = next(iter(fc_l.values()))
            ins, outs, _ = classify_channels(
                [str(c) for c in sample.data_vars], None)
            fc_l, fc_r, _ = coherence_sort_fcs(
                fc_l, fc_r, ins, outs, coherence_min=0.25,
                coherence_max=0.98, mode=sort_mode)
    sample = next(iter(fc_l.values()))
    ins, outs, kw = classify_channels(
        [str(c) for c in sample.data_vars], None)
    if bands_mode == "exact":
        bands = exact_official_bands(freqs_off)
    elif bands_mode == "l2u":
        bands = _l2u_band_table(
            fc_l, min_bins_per_band=min_bins_per_band,
            t_short=t_short or None)
        print(f"  l2u grid: { {lv: len(v) for lv, v in sorted(bands.items())} }"
              f" bands/level (min_bins={min_bins_per_band}, "
              f"t_short={t_short})")
    else:  # auto
        bands = None
    per = {}
    for lv in sorted(fc_l):
        ref = fc_r.get(lv) if fc_r else None
        p, Z, Ze = estimate_tf(
            fc_l[lv], ins, outs, estimator_engine="Robust",
            remote_reference=bool(fc_r), periods_per_octave=3,
            ref_fc=ref, band_style="l2u",
            bands_override=(bands.get(lv, []) if bands else None))
        per[lv] = {"periods": p, "Z": Z, "Z_err": Ze,
                   "input_channels": ins, "output_channels": outs,
                   "channel_nomenclature": kw}
    periods, Z, Z_err = merge_tf_levels(per)
    tf = build_tf(
        periods, Z, ins, outs, channel_nomenclature=kw, Z_err=Z_err,
        survey_id=sample.attrs.get("survey_id"),
        station_id=sample.attrs.get("station_id"),
        run_id=sample.attrs.get("run_id"),
        latitude=sample.attrs.get("station_latitude"),
        longitude=sample.attrs.get("station_longitude"),
        elevation=sample.attrs.get("station_elevation"),
        processing_notes=f"estimator=Robust; remote_reference={bool(fc_r)}; "
                         f"window={window}; bands={bands_mode}; sort={sort}",
    )
    return periods, Z[:, :2, :2], Z_err[:, :2, :2], tf


# --------------------------------------------------------------------------
# official EDI
# --------------------------------------------------------------------------

def read_official_edi():
    """Parse the official EDI (SEG 1.0): frequencies, Z, variances.

    Sign convention exp(+i omega t) per the header; Z in mV/km per nT.
    """
    text = OFFICIAL_EDI.read_text()

    def section(marker):
        lines = text.split(marker, 1)[1].splitlines()
        nums = []
        for ln in lines:
            if ln.strip().startswith(">"):
                break
            nums += [float(x) for x in ln.split()]
        return np.array(nums)

    freqs = section(">FREQ //30")
    periods = 1.0 / freqs
    zxy = section(">ZXYR ROT=ZROT  //30") + 1j * section(">ZXYI ROT=ZROT  //30")
    zyx = section(">ZYXR ROT=ZROT  //30") + 1j * section(">ZYXI ROT=ZROT  //30")
    zxx = section(">ZXXR ROT=ZROT  //30") + 1j * section(">ZXXI ROT=ZROT  //30")
    zyy = section(">ZYYR ROT=ZROT  //30") + 1j * section(">ZYYI ROT=ZROT  //30")
    var = {
        "xy": section(">ZXY.VAR ROT=ZROT  //30"),
        "yx": section(">ZYX.VAR ROT=ZROT  //30"),
        "xx": section(">ZXX.VAR ROT=ZROT  //30"),
        "yy": section(">ZYY.VAR ROT=ZROT  //30"),
    }
    return periods, {"zxy": zxy, "zyx": zyx, "zxx": zxx, "zyy": zyy}, var


def _to_si(z):
    """mV/km per nT -> V/m per A/m (E mV/km -> V/m x1e-6; H nT -> A/m x1e-9/mu0)."""
    return z * 1e-6 * MU0 / 1e-9


# --------------------------------------------------------------------------
# comparison
# --------------------------------------------------------------------------

def rho_phi(periods, z):
    rho = np.abs(z) ** 2 / (MU0 * 2 * np.pi / periods)
    phi = np.degrees(np.arctan2(z.imag, z.real))
    return rho, phi


def fold_phase(phi):
    """Off-diagonal phase fold into (0°, 90°] — the EMTF/USArray display
    convention (negative phases +180°; matches plotting._fold_phase_deg)."""
    return np.where(np.asarray(phi) < 0, np.asarray(phi) + 180.0,
                    np.asarray(phi))


def match_periods(ours, official):
    """Nearest log-period index for each official period."""
    return np.array([int(np.argmin(np.abs(np.log(ours / t)))) for t in official])


def tensor_compare(periods, Z, official_periods, zo, var_o=None):
    """Compare our full tensor (n,2,2) with the official one at matched
    periods; returns rows with per-component ρ/φ deltas."""
    idx = match_periods(periods, official_periods)
    rows = []
    for i in range(len(official_periods)):
        z_off = np.array([[zo["zxx"][i], zo["zxy"][i]],
                          [zo["zyx"][i], zo["zyy"][i]]]) * _to_si(1.0)
        z_our = Z[idx[i]]
        row = {"period_official": float(official_periods[i]),
               "period_ours": float(periods[idx[i]]),
               "var_official_xy": float(var_o["xy"][i]) if var_o else None}
        for (r, c), name in (((0, 1), "xy"), ((1, 0), "yx"),
                             ((0, 0), "xx"), ((1, 1), "yy")):
            rho_o, phi_o = rho_phi(np.array([official_periods[i]]),
                                   np.array([z_off[r, c]]))
            rho_m, phi_m = rho_phi(np.array([official_periods[i]]),
                                   np.array([z_our[r, c]]))
            row[f"delta_log10_rho_{name}"] = float(np.log10(rho_m[0] / rho_o[0]))
            # off-diagonal phases use the folded (0-90) display convention —
            # raw YX phases are negative (-100..-170 deg) at this site and
            # would otherwise produce spurious ~180 deg deltas
            ph_o = fold_phase(phi_o) if name in ("xy", "yx") else phi_o
            ph_m = fold_phase(phi_m) if name in ("xy", "yx") else phi_m
            row[f"delta_phi_deg_{name}"] = float(ph_m[0] - ph_o[0])
        rows.append(row)
    return rows


def summarize(rows, label):
    reliable = [r for r in rows if r["period_official"] >= RELIABLE_MIN_S]
    short = [r for r in rows if r["period_official"] < RELIABLE_MIN_S]
    stats = {}
    for name, group in (("reliable", reliable), ("short_period", short)):
        if not group:
            stats[name] = None
            continue
        dr = np.abs([r["delta_log10_rho_xy"] for r in group]
                    + [r["delta_log10_rho_yx"] for r in group])
        dp = np.abs([r["delta_phi_deg_xy"] for r in group]
                    + [r["delta_phi_deg_yx"] for r in group])
        stats[name] = {
            "n": len(group),
            "n_distinct_ours": len({r["period_ours"] for r in group}),
            "median_abs_log10_rho": float(np.median(dr)),
            "p90_abs_log10_rho": float(np.percentile(dr, 90)),
            "max_abs_log10_rho": float(np.max(dr)),
            "median_abs_phase_deg": float(np.median(dp)),
            "p90_abs_phase_deg": float(np.percentile(dp, 90)),
            "max_abs_phase_deg": float(np.max(dp)),
            "pass_rho": float(np.mean(dr <= GATE_RHO_DECADES)),
            "pass_phase": float(np.mean(dp <= GATE_PHASE_DEG)),
        }
    stats["label"] = label
    return stats


def _row_dict(r):
    """Row-level comparison data for the metrics JSON (transparency: the
    official->ours period mapping and per-period deltas)."""
    return {k: float(r[k]) for k in (
        "period_official", "period_ours",
        "delta_log10_rho_xy", "delta_log10_rho_yx",
        "delta_phi_deg_xy", "delta_phi_deg_yx")}


def fit_rotation(official_periods, zo, periods, Z, lo=RELIABLE_MIN_S):
    """Best-fit frame rotation (deg) — diagnostic only; the comparison uses
    the documented ROTATION_DEG. Fits θ with R_std(θ) Z_off R_std(θ)ᵀ ≈
    Z_ours on the reliable band (median over periods)."""
    def rot(theta_deg):
        t = np.radians(theta_deg)
        return np.array([[np.cos(t), np.sin(t)], [-np.sin(t), np.cos(t)]])

    thetas = []
    for i in range(len(official_periods)):
        if official_periods[i] < lo:
            continue
        j = int(np.argmin(np.abs(np.log(periods / official_periods[i]))))
        z_off = np.array([[zo["zxx"][i], zo["zxy"][i]],
                          [zo["zyx"][i], zo["zyy"][i]]]) * _to_si(1.0)
        z_our = Z[j]
        best, bth = None, None
        for th in np.arange(-12.0, 12.01, 0.05):
            d = np.linalg.norm(rot(th) @ z_off @ rot(th).T - z_our)
            if best is None or d < best:
                best, bth = d, th
        thetas.append(bth)
    return float(np.median(thetas))


def rotate_tensor(Z, angle_deg):
    """Z' = R(angle) Z R(angle)ᵀ — STANDARD MT convention (DESIGN §16.3):
    angle positive = clockwise from true north;
    R(angle) = [[cos α, sin α], [−sin α, cos α]].
    Magnetic -> true north with declination D: angle = −D."""
    a = np.radians(angle_deg)
    R = np.array([[np.cos(a), np.sin(a)], [-np.sin(a), np.cos(a)]])
    return np.einsum("ij,njk,lk->nil", R, Z, R)


def raw_overlay_figure(periods, Z, path, title, periods_o, zo,
                       corrected=True):
    """Raw comparison figure — NO nearest-period matching: each dataset is
    plotted at its OWN evaluated periods. Our TF is rotated to true north
    (``corrected``, the documented frame convention) before plotting, so the
    curves are directly comparable to the official product. Official = filled
    ○/✕ (blue/red), emptyUI = hollow □/◇ (same colors) — distinct symbols so
    both datasets are unambiguous."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    Zc = rotate_tensor(Z, ROTATION_DEG) if corrected else Z
    fig, (ax_rho, ax_phi) = plt.subplots(
        2, 1, figsize=(7.0, 8.0), sharex=True,
        gridspec_kw={"height_ratios": [3.2, 1.3], "hspace": 0.18})
    for name, marker_off, marker_my, color in (
            ("xy", "o", "s", "#1f77b4"), ("yx", "x", "D", "#d62728")):
        row = 0 if name == "xy" else 1
        col = 1 if name == "xy" else 0
        rho_o, phi_o = rho_phi(periods_o, np.asarray(zo["z" + name]) * _to_si(1.0))
        rho_m, phi_m = rho_phi(periods, Zc[:, row, col])
        phi_o = fold_phase(phi_o)
        phi_m = fold_phase(phi_m)
        ax_rho.plot(periods_o, rho_o, marker_off, color=color,
                    linestyle="None", markersize=5,
                    label=f"{name.upper()} official (EMTF)")
        ax_rho.plot(periods, rho_m, marker_my, color=color,
                    linestyle="None", markersize=5, markerfacecolor="none",
                    label=f"{name.upper()} emptyUI")
        ax_phi.plot(periods_o, phi_o, marker_off, color=color,
                    linestyle="None", markersize=5,
                    label=f"{name.upper()} official (EMTF)")
        ax_phi.plot(periods, phi_m, marker_my, color=color,
                    linestyle="None", markersize=5, markerfacecolor="none",
                    label=f"{name.upper()} emptyUI")
    ax_rho.set_xscale("log")
    ax_rho.set_yscale("log")
    ax_phi.set_xscale("log")
    ax_rho.set_ylim(5.0, 4.0e4)
    ax_rho.set_ylabel(r"$\rho_a$ ($\Omega$ m)")
    ax_phi.set_ylabel(r"$\phi$ (deg)")
    ax_phi.set_xlabel("Period (s)")
    ax_phi.set_ylim(0, 90)
    ax_phi.set_yticks(np.arange(0, 91, 20))
    ax_rho.set_title(title + (" — rotated −5.7°" if corrected else ""),
                     fontsize=9)
    ax_rho.legend(fontsize=7, loc="lower right", frameon=True)
    ax_phi.legend(fontsize=7, loc="lower right", frameon=True)
    fig.subplots_adjust(left=0.12, right=0.97, top=0.92, bottom=0.09,
                        hspace=0.28)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def render_appres_figure(reg, ctx, label, p, Z, Ze, out_name, note,
                         window=128):
    """    EMTF-convention AppRes & Phase figure via the Plot AppRes & Phase node
    (DESIGN §15: ○ XY / ✕ YX, error bars, log-log ρ, phase 0-90, EMTF-style
    title, ρ_a y-limits pinned to the reference figure's range 5-4e4 Ωm) —
    the head-to-head counterpart of SDG31b_H28coh.png. The TF is rotated to
    TRUE NORTH with the package rotation (mth5.tf.rotate /
    rotate_transfer_function: angle = -5.7 deg, standard convention, positive
    = clockwise from true north) — the official product is in the geographic
    frame ("Angle To GG North: 0.00") while the as-archived data are
    geomagnetic. Degenerate bands (relative error > 200%) are dropped for
    the DISPLAY figure only (the metrics keep them) so the figure stays
    comparable to the reference."""
    tf_rot = rotate_transfer_function(
        build_tf(
            p, Z, ["hx", "hy"], ["ex", "ey"], channel_nomenclature="default",
            Z_err=Ze, survey_id="4P", station_id="SDG31", run_id="b",
            latitude=45.3718, longitude=-98.2931, elevation=394.0,
        ),
        ROTATION_DEG,
    )
    arr = tf_rot.dataset["transfer_function"]
    Zc = arr.sel(output=["ex", "ey"], input=["hx", "hy"]).values
    Zec = tf_rot.dataset["transfer_function_error"].sel(
        output=["ex", "ey"], input=["hx", "hy"]
    ).values
    rel_err = np.abs(Zec) / np.maximum(np.abs(Zc), 1e-30)
    keep = np.max(rel_err, axis=(1, 2)) < 2.0
    n_drop = int(np.sum(~keep))
    if n_drop:
        print(f"  {label}: dropped {n_drop} degenerate band(s) from the "
              f"display figure (relative error > 200%)")
    tf_disp = build_tf(
        p[keep], Zc[keep], ["hx", "hy"], ["ex", "ey"],
        channel_nomenclature="default", Z_err=Zec[keep],
        survey_id="4P", station_id="SDG31", run_id="b",
        latitude=45.3718, longitude=-98.2931, elevation=394.0,
        processing_notes=f"estimator=Robust; remote_reference=True; "
                         f"window={window}; bands={note}; "
                         f"rotated={ROTATION_DEG} deg to true north",
    )
    plot_res = reg.get("mth5.plot.appres_phase").execute(
        ctx, tf_disp, mode="off-diagonal", rho_ylim="5,40000")
    shutil.copy(plot_res["plot"]["path"], DATA_DIR / out_name)
    print("EMTF-convention figure:", DATA_DIR / out_name)


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def main():
    reg = build_default_registry()
    ctx = RunContext(workspace=Path(tempfile.mkdtemp(prefix="sdg31_verify_")),
                     store=DataStore())

    periods_o, zo, var_o = read_official_edi()
    print(f"official: {len(periods_o)} periods, "
          f"{periods_o.min():.1f}-{periods_o.max():.0f} s")

    local = load_calibrated(reg, ctx, *LOCAL)
    remote_cal = load_calibrated(reg, ctx, *REMOTE)
    variants = {
        # reference: the official band table reconstructed exactly
        "remote_ref": {"window": 128, "bands_mode": "exact"},
        # §14.9 default frequency-grid method (Estimate TF node default):
        # reference-template + log-uniform window-starvation coarsening
        "remote_ref_l2u": {
            "window": 128, "bands_mode": "l2u",
            "min_bins_per_band": COARSEN_THRESHOLD_BINS},
        "remote_ref_l2u_256": {
            "window": 256, "bands_mode": "l2u",
            "min_bins_per_band": COARSEN_THRESHOLD_BINS},
        # legacy per-level fallback walk (no container-level table)
        "remote_ref_auto": {"window": 128, "bands_mode": "auto"},
    }
    results = {}
    for label, kwargs in variants.items():
        periods, Z, Z_err, tf = run_variant(
            reg, ctx, local, remote_cal, freqs_off=1.0 / periods_o, **kwargs)
        results[label] = (periods, Z, Z_err, tf)
        print(f"{label}: {len(periods)} periods "
              f"({periods.min():.1f}-{periods.max():.0f} s)")

    # documented rotation is used for the comparison; the empirical fit is
    # kept as a diagnostic only (should agree with ROTATION_DEG)
    prim = "remote_ref"
    prim_periods, prim_Z, _, _ = results[prim]
    theta = fit_rotation(periods_o, zo, prim_periods, prim_Z)
    print(f"diagnostic best-fit frame rotation: {theta:.2f} deg "
          f"(documented: {ROTATION_DEG} deg)")

    all_stats = {}
    all_rows = {}
    for label, (p_v, Z_v, Zerr_v, tf_v) in results.items():
        rows = tensor_compare(p_v, Z_v, periods_o, zo, var_o)
        st = summarize(rows, label)
        all_stats[label] = st
        all_rows[label] = [_row_dict(r) for r in rows]
        # enriched rows for the overlay figure (rho/phi values)
        for r, i in zip(rows, range(len(rows))):
            z_off = np.array([[zo["zxx"][i], zo["zxy"][i]],
                              [zo["zyx"][i], zo["zyy"][i]]]) * _to_si(1.0)
            z_our = Z_v[match_periods(p_v, periods_o)[i]]
            for (r_, c_), name in (((0, 1), "xy"), ((1, 0), "yx")):
                rho_o, phi_o = rho_phi(np.array([r["period_official"]]),
                                       np.array([z_off[r_, c_]]))
                rho_m, phi_m = rho_phi(np.array([r["period_official"]]),
                                       np.array([z_our[r_, c_]]))
                r[f"rho_official_{name}"] = float(rho_o[0])
                r[f"rho_ours_{name}"] = float(rho_m[0])
                r[f"phi_official_{name}"] = float(phi_o[0])
                r[f"phi_ours_{name}"] = float(phi_m[0])
        # RAW comparison figure (no matching — each dataset at its own
        # periods; ours rotated to true north)
        raw_overlay_figure(
            p_v, Z_v, DATA_DIR / f"sdg31_plot_{label}_vs_official.png",
            f"SDG31: emptyUI vs EMTF official — {label}", periods_o, zo)
    # rotation-corrected comparison — the DOCUMENTED -5.7 deg (standard
    # convention, positive = clockwise from true north) applied to the
    # primary (exact-grid) variant AND the §14.9 adaptive variants: the
    # rotation is a FRAME property, not a grid property — every
    # geomagnetic-frame estimate must be rotated before comparison with
    # the true-north official product, regardless of the band table.
    def rotated_compare(label, periods, Z):
        Zc = rotate_tensor(Z, ROTATION_DEG)
        rows = tensor_compare(periods, Zc, periods_o, zo, var_o)
        all_stats[label + "_corrected"] = summarize(
            rows, label + " rotation-corrected")
        all_rows[label + "_corrected"] = [_row_dict(r) for r in rows]
        for r, i in zip(rows, range(len(rows))):
            z_off = np.array([[zo["zxx"][i], zo["zxy"][i]],
                              [zo["zyx"][i], zo["zyy"][i]]]) * _to_si(1.0)
            z_our = Zc[match_periods(periods, periods_o)[i]]
            for (r_, c_), name in (((0, 1), "xy"), ((1, 0), "yx")):
                rho_o, phi_o = rho_phi(np.array([r["period_official"]]),
                                       np.array([z_off[r_, c_]]))
                rho_m, phi_m = rho_phi(np.array([r["period_official"]]),
                                       np.array([z_our[r_, c_]]))
                r[f"rho_official_{name}"] = float(rho_o[0])
                r[f"rho_ours_{name}"] = float(rho_m[0])
                r[f"phi_official_{name}"] = float(phi_o[0])
                r[f"phi_ours_{name}"] = float(phi_m[0])
        return rows

    rotated_compare(prim, prim_periods, prim_Z)
    for label in ("remote_ref_l2u", "remote_ref_l2u_256", "remote_ref_auto"):
        p, Zl, _, _ = results[label]
        rotated_compare(label, p, Zl)

    # EMTF-convention AppRes & Phase figures via the Plot AppRes & Phase node
    # (DESIGN §15: ○ XY / ✕ YX, error bars, log-log ρ, phase 0-90, EMTF-style
    # title, ρ_a y-limits pinned to the reference figure's range 5-4e4 Ωm) —
    # the head-to-head counterpart of SDG31b_H28coh.png. Rendered from the
    # N=128 exact-grid variant (so the evaluated periods match the EMTF grid)
    # and, for independent assessment, from the §14.9 l2u default method (the
    # Estimate TF node default); both rotated to TRUE NORTH. The ~11.6 s RR
    # band is degenerate (the source field is spatially incoherent at 10-15 s
    # — cross-station H coherence ~0.003 — so the RME_RR denominator
    # collapses); it is dropped for the DISPLAY figure only (the metrics keep
    # it) so the figure stays comparable to the reference.
    p0, Z0, Ze0, _tf0 = results["remote_ref"]
    render_appres_figure(reg, ctx, "remote_ref", p0, Z0, Ze0,
                         "sdg31_plot_appres_phase.png", "exact")
    for label, window, note in (
            ("remote_ref_l2u", 128, "l2u (default, window 128)"),
            ("remote_ref_l2u_256", 256, "l2u (window 256)")):
        p_p, Z_p, Ze_p, _tf_p = results[label]
        render_appres_figure(reg, ctx, label, p_p, Z_p, Ze_p,
                             f"sdg31_plot_appres_phase_{label.replace('remote_ref_', '')}.png",
                             note, window=window)

    out = {
        "config": {"fc": FC_KWARGS, "estimate": EST_KWARGS,
                   "gates": {"rho_decades": GATE_RHO_DECADES,
                             "phase_deg": GATE_PHASE_DEG,
                             "reliable_min_s": RELIABLE_MIN_S},
                   "rotation_deg": ROTATION_DEG,
                   "min_bins_per_band": COARSEN_THRESHOLD_BINS,
                   "official_t_short": OFFICIAL_T_SHORT},
        "official": {"n_periods": len(periods_o),
                     "periods_s": [float(p) for p in periods_o],
                     "source": str(OFFICIAL_EDI.name)},
        "rotation_fit_diagnostic_deg": theta,
        "stats": all_stats,
        "rows": all_rows,
    }
    (DATA_DIR / "sdg31_verify_metrics.json").write_text(
        json.dumps(out, indent=2))

    print(f"\n=== verification summary (reliable band >= {RELIABLE_MIN_S:.0f} s) ===")
    for label, st in all_stats.items():
        r = st["reliable"]
        print(f"{label}: n={r['n']} median|dlog10rho|={r['median_abs_log10_rho']:.3f} "
              f"p90={r['p90_abs_log10_rho']:.3f} median|dphi|={r['median_abs_phase_deg']:.2f} deg "
              f"p90={r['p90_abs_phase_deg']:.2f} deg pass_rho={r['pass_rho']:.0%} "
              f"pass_phase={r['pass_phase']:.0%}")
        s = st["short_period"] or {}
        if s:
            print(f"   short-period (<%.0f s): n={s['n']} "
                  f"median|dlog10rho|={s['median_abs_log10_rho']:.3f}"
                  % RELIABLE_MIN_S)
    print("figures + metrics written next to:", OFFICIAL_EDI.name)


if __name__ == "__main__":
    main()
