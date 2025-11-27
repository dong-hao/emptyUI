"""Shared publication-quality TF plot helpers (DESIGN.md §15).

Conventions follow the USArray reference figure (``SDG31b_H28coh.png``):
x = Period (secs) on log10, apparent resistivity on log10 (Ω·m), phase in
linear degrees; off-diagonal phase folded into (0°, 90°); markers/colors
○ XY (blue) / ✕ YX (red); error bars with caps; equal per-decade scale
between the period and ρ_a axes (MT convention); framed legend; site title
with LON/LAT.

This module is plugin-tier (mth5); the core-nodes mirror
(``emptyui-core-nodes/_plotstyle.py``) stays minimal so core never depends
on mth5.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

MU0 = 4.0 * np.pi * 1e-7

#: USArray convention: XY = circle, YX = "x"; shapes + linestyles so
#: components stay distinguishable without color (colorblind-safe).
MARKERS = {"xy": "o", "yx": "x", "xx": "s", "yy": "D"}
COLORS = {"xy": "#1f77b4", "yx": "#d62728", "xx": "#1f77b4", "yy": "#d62728"}
LINESTYLES = {"xy": "-", "yx": "--", "xx": ":", "yy": "-."}

DPI = 150


def apply_style(fig, axes=()):
    """Apply the shared publication style to a figure and its axes."""
    fig.patch.set_facecolor("white")
    for ax in axes:
        ax.tick_params(labelsize=9, which="both")
        ax.grid(True, which="both", alpha=0.25, linewidth=0.5)
        for spine in ax.spines.values():
            spine.set_linewidth(0.8)


def station_title(tf) -> str:
    """USArray-style title: ``Site: <id> [ LON = …; LAT = … ]``."""
    try:
        st = tf.station_metadata
    except Exception:  # noqa: BLE001
        return "Site: unknown"
    sid = getattr(st, "id", None) or "unknown"
    try:
        lat = st.location.latitude
        lon = st.location.longitude
    except Exception:  # noqa: BLE001
        return f"Site: {sid}"
    if lat not in (None, 0.0) or lon not in (None, 0.0):
        return (f"Site: {sid} [ LON = {float(lon):.4f}; "
                f"LAT = {float(lat):.4f} ]")
    return f"Site: {sid}"


def run_notes(tf) -> str:
    """Processing provenance recorded on the TF (run comment), if any."""
    try:
        value = tf.run_metadata.comments.value
    except Exception:  # noqa: BLE001
        return ""
    return value if value else ""


def equal_decade_layout(fig, ax, x_min, x_max, y_min, y_max,
                        inches_per_decade: float = 0.9,
                        extra_height: float = 0.0) -> None:
    """Size the figure so one period decade spans the same physical length as
    one ρ_a decade (MT convention, DESIGN.md §15).

    The axes limits are set explicitly first, then the figure height is
    corrected once (linear in figure height) so the panel's measured
    inches-per-decade matches on both axes.
    """
    xd = max(1.0, float(np.log10(x_max / x_min)))
    yd = max(1.0, float(np.log10(y_max / y_min)))
    fig.set_size_inches(1.3 + inches_per_decade * xd,
                        1.0 + inches_per_decade * yd + extra_height)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    fig.canvas.draw()
    pos = ax.get_position()
    fw, fh = fig.get_size_inches()
    x_ratio = (pos.width * fw) / xd
    y_ratio = (pos.height * fh) / yd
    fig.set_size_inches(fw, fh * (x_ratio / y_ratio))
    fig.canvas.draw()


def _decade_limits(values, lo=None, hi=None):
    """Whole-decade (log10) limits around data values, with sane fallbacks."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v) & (v > 0)]
    if v.size == 0:
        return (lo if lo is not None else 1e-1), (hi if hi is not None else 1e4)
    pmin = np.floor(np.log10(v.min()))
    pmax = np.ceil(np.log10(v.max()))
    if lo is not None:
        pmin = min(pmin, np.log10(lo))
    if hi is not None:
        pmax = max(pmax, np.log10(hi))
    if pmax - pmin < 1.0:
        pmax = pmin + 1.0
    return 10.0 ** pmin, 10.0 ** pmax


def _fold_phase_deg(phase):
    """Fold phases into the first quadrant (0°, 90°] — aurora RhoPlot
    convention: rotate negative phases by +180°; the ylim clips the rare
    values beyond 90°."""
    p = np.asarray(phase, dtype=float)
    return np.where(p < 0, p + 180.0, p)


def impedance_panels(tf, mode: str):
    """Extract (period, components) for the impedance plot.

    Returns a list of dicts: {label, z, z_std, key} per component present in
    the TF. ``mode`` is ``off-diagonal`` (xy/yx) or ``diagonal`` (xx/yy).
    """
    Z = tf.dataset["transfer_function"]
    pairs = (("zxy", tf.ex, tf.hy), ("zyx", tf.ey, tf.hx)) if mode == "off-diagonal" \
        else (("zxx", tf.ex, tf.hx), ("zyy", tf.ey, tf.hy))
    period = np.asarray(Z.coords["period"], dtype=float)
    out = []
    for label, out_ch, in_ch in pairs:
        if out_ch not in Z.coords["output"].values or in_ch not in Z.coords["input"].values:
            continue
        z = Z.sel(output=out_ch, input=in_ch).values.astype(complex)
        try:
            z_std = np.asarray(
                tf.transfer_function_error.sel(output=out_ch, input=in_ch).values,
                dtype=float,
            )
        except Exception:  # noqa: BLE001 — no error field
            z_std = None
        out.append({"label": label, "z": z, "z_std": z_std, "period": period})
    return out


def tipper_panels(tf):
    """Extract (period, Tzx, Tzy) and their stds for the tipper plot."""
    t = tf.tipper
    period = np.asarray(t.coords["period"], dtype=float)
    out = []
    for label, index in (("Tzx", tf.index_tzx), ("Tzy", tf.index_tzy)):
        comp = t.sel(**index).values.astype(complex)
        try:
            terr = tf.tipper_error
            std = np.asarray(terr.sel(**index).values, dtype=float)
        except Exception:  # noqa: BLE001
            std = None
        out.append({"label": label, "z": comp, "z_std": std, "period": period})
    return out


def rho_phase_errors(z, z_std, period, rho):
    """Error bars for ρ_a and φ from the element std (aurora convention:
    real/imag std = std/√2; σ_ρ/ρ = 2·σ_|Z|/|Z|; σ_φ(rad) = σ_|Z|/|Z|)."""
    if z_std is None:
        return None, None
    z_se = np.asarray(z_std, dtype=float) / np.sqrt(2.0)
    az = np.abs(z)
    with np.errstate(divide="ignore", invalid="ignore"):
        rho_se = 2.0 * np.asarray(rho) * z_se / np.maximum(az, 1e-300)
        rho_se = np.where(np.isfinite(rho_se), rho_se, 0.0)
        phi_se = np.degrees(z_se / np.maximum(az, 1e-300))
        phi_se = np.where(np.isfinite(phi_se), phi_se, 0.0)
    return rho_se, phi_se


def plot_impedance_response(tf, mode: str, path):
    """Render the apparent-resistivity / phase figure (2 panels) to ``path``."""
    comps = impedance_panels(tf, mode)
    if not comps:
        raise ValueError("transfer function has no impedance data")
    period = comps[0]["period"]
    rho_all = []
    phase_all = []
    for c in comps:
        z = c["z"]
        rho = (np.abs(z) ** 2) / (MU0 * (2.0 * np.pi / period))
        phase = np.angle(z, deg=True)
        ok = np.isfinite(rho) & (rho > 0) & np.isfinite(phase)
        if not np.any(ok):
            continue
        rho_all.append(rho[ok])
        if mode == "off-diagonal":
            phase_all.append(_fold_phase_deg(phase)[ok])
        else:
            phase_all.append(phase[ok])

    x_min, x_max = _decade_limits(period)
    y_min, y_max = _decade_limits(np.concatenate(rho_all))
    phase_h = 2.6
    fig = plt.figure(figsize=(6.5, 8.0))
    gs = fig.add_gridspec(2, 1, height_ratios=[3.2, phase_h], hspace=0.18)
    ax_rho = fig.add_subplot(gs[0])
    ax_phi = fig.add_subplot(gs[1], sharex=ax_rho)

    for c in comps:
        z = c["z"]
        rho = (np.abs(z) ** 2) / (MU0 * (2.0 * np.pi / period))
        phase = np.angle(z, deg=True)
        ok = np.isfinite(rho) & (rho > 0) & np.isfinite(phase)
        if not np.any(ok):
            continue
        label = key = c["label"][1:]
        label = label.upper()
        rho_se, phi_se = rho_phase_errors(z, c["z_std"], period, rho)
        rho_se_ok = rho_se[ok] if rho_se is not None else None
        ax_rho.errorbar(period[ok], rho[ok], yerr=rho_se_ok,
                        fmt=MARKERS.get(key, "o"), color=COLORS.get(key, "k"),
                        linestyle=LINESTYLES.get(key, "-"), label=label,
                        capsize=2.5, markersize=4, linewidth=1.0, elinewidth=0.8)
        ph = _fold_phase_deg(phase)[ok] if mode == "off-diagonal" else phase[ok]
        phi_se_ok = phi_se[ok] if phi_se is not None else None
        ax_phi.errorbar(period[ok], ph, yerr=phi_se_ok,
                        fmt=MARKERS.get(key, "o"), color=COLORS.get(key, "k"),
                        linestyle=LINESTYLES.get(key, "-"), label=label,
                        capsize=2.5, markersize=4, linewidth=1.0, elinewidth=0.8)

    apply_style(fig, (ax_rho, ax_phi))
    ax_rho.set_xscale("log")
    ax_rho.set_yscale("log")
    ax_phi.set_xscale("log")
    ax_rho.set_ylabel(r"$\rho_a$ ($\Omega$ m)")
    ax_phi.set_ylabel(r"$\phi$ (Degrees)")
    ax_phi.set_xlabel("Period (secs)")
    if mode == "off-diagonal":
        ax_phi.set_ylim(0, 90)
        ax_phi.set_yticks(np.arange(0, 91, 15))
    else:
        ax_phi.set_ylim(-180, 180)
        ax_phi.set_yticks(np.arange(-180, 181, 45))
    for ax in (ax_rho, ax_phi):
        ax.legend(loc="best", frameon=True, framealpha=0.9, fontsize=8,
                  borderpad=0.3)
    fig.suptitle(station_title(tf), fontsize=11, fontweight="bold")
    notes = run_notes(tf)
    if notes:
        fig.text(0.5, 0.005, notes, ha="center", va="bottom", fontsize=7,
                 style="italic", color="0.3")
    equal_decade_layout(fig, ax_rho, x_min, x_max, y_min, y_max,
                        extra_height=phase_h)
    fig.savefig(path, dpi=DPI, bbox_inches="tight",
                metadata={"Software": "emptyUI", "Title": station_title(tf)})
    plt.close(fig)
    return path


def plot_tipper(tf, path):
    """Render the tipper Re/Im figure (2 panels, Tzx and Tzy) to ``path``."""
    panels = tipper_panels(tf)
    if not panels:
        raise ValueError("transfer function has no tipper data")
    period = panels[0]["period"]
    fig, axes = plt.subplots(2, 1, figsize=(6.5, 6.4), sharex=True)
    for ax, p in zip(axes, panels):
        z = p["z"]
        ok = np.isfinite(z.real) & np.isfinite(z.imag)
        if not np.any(ok):
            continue
        z_std = p["z_std"]
        re_se = (z_std / np.sqrt(2.0)) if z_std is not None else None
        for part, data, err, marker, color, ls, label in (
            ("Re", z.real, re_se, "o", "#1f77b4", "-", "Re"),
            ("Im", z.imag, re_se, "x", "#d62728", "--", "Im"),
        ):
            yerr = err[ok] if err is not None else None
            ax.errorbar(period[ok], data[ok], yerr=yerr, fmt=marker,
                        color=color, linestyle=ls, label=label,
                        capsize=2.5, markersize=4, linewidth=1.0, elinewidth=0.8)
        ax.set_xscale("log")
        ax.set_ylabel(f"{p['label']} (dimensionless)")
        ax.set_ylim(-1, 1)
        ax.legend(loc="best", frameon=True, framealpha=0.9, fontsize=8,
                  borderpad=0.3)
    apply_style(fig, axes)
    axes[-1].set_xlabel("Period (secs)")
    fig.suptitle(station_title(tf), fontsize=11, fontweight="bold")
    notes = run_notes(tf)
    if notes:
        fig.text(0.5, 0.005, notes, ha="center", va="bottom", fontsize=7,
                 style="italic", color="0.3")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=DPI, bbox_inches="tight",
                metadata={"Software": "emptyUI", "Title": station_title(tf)})
    plt.close(fig)
    return path
