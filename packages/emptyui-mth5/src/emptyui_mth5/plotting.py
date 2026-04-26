"""Shared publication-quality TF plot helpers (DESIGN.md §10.6).

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
#: marker face per component (USArray reference figure: XY = OPEN circle ○ —
#: white face with the colored edge; other markers filled)
MARKER_FACES = {"xy": "white"}

DPI = 150


def apply_style(fig, axes=()):
    """Apply the shared publication style to a figure and its axes.

    Grids follow the EMTF reference figure: solid major gridlines with
    dotted minor gridlines (the USArray MT figures show the fine minor
    grid on the log axes).
    """
    fig.patch.set_facecolor("white")
    for ax in axes:
        ax.tick_params(labelsize=9, which="both")
        ax.grid(True, which="major", alpha=0.30, linewidth=0.6)
        ax.grid(True, which="minor", alpha=0.20, linewidth=0.5,
                linestyle=":")
        for spine in ax.spines.values():
            spine.set_linewidth(0.8)


def station_title(tf) -> str:
    """EMTF/USArray-style title: ``<station><run> [ LON = …; LAT = … ]``
    (the SDG31b_H28coh.png reference format — station + processing tag,
    no "Site:" prefix; the run tag is appended when the TF carries one)."""
    try:
        st = tf.station_metadata
    except Exception:  # noqa: BLE001
        return "Site: unknown"
    sid = getattr(st, "id", None) or "unknown"
    try:
        rid = tf.run_metadata.id
    except Exception:  # noqa: BLE001
        rid = None
    tag = f"{sid}{rid}" if rid not in (None, "", "0") else str(sid)
    try:
        lat = st.location.latitude
        lon = st.location.longitude
    except Exception:  # noqa: BLE001
        return tag
    if lat not in (None, 0.0) or lon not in (None, 0.0):
        return (f"{tag} [ LON = {float(lon):.4f}; "
                f"LAT = {float(lat):.4f} ]")
    return tag


def run_notes(tf) -> str:
    """Processing provenance recorded on the TF (run comment), if any."""
    try:
        value = tf.run_metadata.comments.value
    except Exception:  # noqa: BLE001
        return ""
    return value if value else ""


def _wrap_notes(fig, text: str, width_px: float, fontsize: float) -> str:
    """Greedy word-wrap measured in display pixels (DESIGN.md §10.6).

    Proportional fonts make character-count wrapping unreliable, so each
    candidate line is measured with the real renderer: a word moves to the
    next line only when the current line would exceed ``width_px`` (the
    axes width). A single word wider than the budget is kept on its own
    line rather than dropped.
    """
    renderer = fig.canvas.get_renderer()
    lines: list[str] = []
    current = ""
    for word in text.split():
        trial = f"{current} {word}" if current else word
        probe = fig.text(0.0, 0.0, trial, fontsize=fontsize)
        w = probe.get_window_extent(renderer).width
        probe.remove()
        if w <= width_px or not current:
            current = trial
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)
    return "\n".join(lines)


def _annotate_notes(fig, ax, notes: str, show: bool = True,
                    fontsize: float = 7.0, gap_px: float = 5.0) -> None:
    """Place the provenance note below the x-axis label of ``ax``.

    Anchored to the *measured* x-label bottom (never overlapping it,
    DESIGN.md §10.6), centered on the axes, and word-wrapped so no line is wider than
    the axes — an unwrapped long note used to blow the figure width up via
    ``bbox_inches="tight"``. Skipped entirely when ``show`` is False (the
    production-images toggle on the plot nodes).
    """
    if not show or not notes:
        return
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    ax_bb = ax.get_window_extent(renderer)
    wrapped = _wrap_notes(fig, notes, ax_bb.width, fontsize)
    probe = fig.text(0.0, 0.0, wrapped, fontsize=fontsize)
    h_px = probe.get_window_extent(renderer).height
    probe.remove()
    fig_w_px = fig.get_size_inches()[0] * fig.dpi
    fig_h_px = fig.get_size_inches()[1] * fig.dpi
    xlabel_bottom = ax.xaxis.label.get_window_extent(renderer).y0
    cx = 0.5 * (ax_bb.x0 + ax_bb.x1) / fig_w_px
    # top of the block sits gap_px below the x-label's bottom edge
    y = (xlabel_bottom - gap_px - h_px) / fig_h_px
    fig.text(cx, y, wrapped, ha="center", va="bottom", fontsize=fontsize,
             style="italic", color="0.3")


def impedance_layout(fig, ax_rho, ax_phi, x_min, x_max, y_min, y_max,
                     phase_inches: float, inches_per_decade: float = 0.9) -> None:
    """Size the impedance figure: equal per-decade scale for the ρ_a panel
    (MT convention, one period decade = one Ω·m decade in physical length)
    and an explicit phase-panel height in inches.

    The phase panel height is independent of the ρ_a decade span: the
    off-diagonal phase panel is deliberately half the diagonal one (the
    off-diagonal range is 0–90° vs 360° for the signed diagonal phases).
    Single-pass by design: margins are measured on the current geometry and
    carried into the resized figure — do not re-run it on its own output.
    """
    xd = max(1.0, float(np.log10(x_max / x_min)))
    yd = max(1.0, float(np.log10(y_max / y_min)))
    fig.set_size_inches(1.3 + inches_per_decade * xd,
                        1.0 + inches_per_decade * yd + phase_inches)
    ax_rho.set_xlim(x_min, x_max)
    ax_rho.set_ylim(y_min, y_max)
    fig.canvas.draw()
    pr = ax_rho.get_position()
    pp = ax_phi.get_position()
    fw, fh = fig.get_size_inches()
    left, right = pr.x0, pr.x1
    top_margin = (1.0 - pr.y1) * fh
    bottom_margin = pp.y0 * fh
    gap = (pr.y0 - pp.y1) * fh
    w_r = (right - left) * fw
    h_r = w_r * yd / xd  # equal per-decade scale
    new_h = top_margin + h_r + gap + phase_inches + bottom_margin
    fig.set_size_inches(fw, new_h)
    ax_rho.set_position([left, (bottom_margin + phase_inches + gap) / new_h,
                         right - left, h_r / new_h])
    ax_phi.set_position([left, bottom_margin / new_h,
                         right - left, phase_inches / new_h])
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


def plot_impedance_response(tf, mode: str, path, rho_ylim=None,
                            show_params: bool = True):
    """Render the apparent-resistivity / phase figure (2 panels) to ``path``.

    ``rho_ylim`` optionally pins the apparent-resistivity y-limits
    (``(min, max)``) instead of the auto whole-decade fit — used to match an
    external reference figure's axis range exactly (DESIGN.md §10.6, the
    SDG31b_H28coh.png head-to-head: 5 - 4e4 Ohm m). ``show_params=False``
    drops the processing-provenance line for production images (§23.3).
    """
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
            # L-M8 display convention: the plot FOLDS the off-diagonal
            # phase to 0-90° (the half-height panel, DESIGN.md §10.6) — the
            # ViewTF table deliberately shows the RAW ±180° values
            # (provenance: what was estimated); the fold is display-only
            phase_all.append(_fold_phase_deg(phase)[ok])
        else:
            phase_all.append(phase[ok])

    x_min, x_max = _decade_limits(period)
    if rho_ylim is not None:
        y_min, y_max = float(rho_ylim[0]), float(rho_ylim[1])
    else:
        if not rho_all:
            # L-M9: every panel's data was all-NaN (a broken/uncalibrated
            # TF) — np.concatenate([]) crashed with a raw ValueError; fail
            # with the actionable message the tipper plot's guard uses
            raise ValueError(
                "transfer function's impedance data is all NaN — cannot "
                "plot; check the calibration/decimation of the source FC")
        y_min, y_max = _decade_limits(np.concatenate(rho_all))
    # Off-diagonal phase spans 0-90° -> half-height panel; diagonal spans
    # the full signed range -> full-height panel (DESIGN.md §10.6).
    phase_h = 1.3 if mode == "off-diagonal" else 2.6
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
        # markers only, no connecting lines (EMTF reference convention);
        # XY renders as an OPEN circle (white face, MARKER_FACES)
        fmt = MARKERS.get(key, "o")
        mfc = MARKER_FACES.get(key, COLORS.get(key, "k"))
        ax_rho.errorbar(period[ok], rho[ok], yerr=rho_se_ok,
                        fmt=fmt, color=COLORS.get(key, "k"),
                        markerfacecolor=mfc,
                        label=label, capsize=2.5, markersize=5,
                        linewidth=1.0, elinewidth=0.8)
        ph = _fold_phase_deg(phase)[ok] if mode == "off-diagonal" else phase[ok]
        phi_se_ok = phi_se[ok] if phi_se is not None else None
        ax_phi.errorbar(period[ok], ph, yerr=phi_se_ok,
                        fmt=fmt, color=COLORS.get(key, "k"),
                        markerfacecolor=mfc,
                        label=label, capsize=2.5, markersize=5,
                        linewidth=1.0, elinewidth=0.8)

    apply_style(fig, (ax_rho, ax_phi))
    ax_rho.set_xscale("log")
    ax_rho.set_yscale("log")
    ax_phi.set_xscale("log")
    ax_rho.set_ylabel(r"$\rho_a$ ($\Omega$ m)")
    ax_phi.set_ylabel(r"$\phi$ (Degrees)")
    ax_phi.set_xlabel("Period (secs)")
    if mode == "off-diagonal":
        ax_phi.set_ylim(0, 90)
        ax_phi.set_yticks(np.arange(0, 91, 20))
    else:
        ax_phi.set_ylim(-180, 180)
        ax_phi.set_yticks(np.arange(-180, 181, 45))
    # boxed marker-only legend in the ρ_a panel (EMTF reference: no legend
    # in the phase panel)
    ax_rho.legend(loc="lower right", frameon=True, framealpha=0.9,
                  fontsize=8, borderpad=0.3, handlelength=0.0,
                  handletextpad=0.5)
    fig.suptitle(station_title(tf), fontsize=11, fontweight="bold")
    impedance_layout(fig, ax_rho, ax_phi, x_min, x_max, y_min, y_max,
                     phase_h)
    _annotate_notes(fig, ax_phi, run_notes(tf), show_params)
    fig.savefig(path, dpi=DPI, bbox_inches="tight",
                metadata={"Software": "emptyUI", "Title": station_title(tf)})
    plt.close(fig)
    return path


def plot_tipper(tf, path, ylim=None, show_params: bool = True):
    """Render the tipper figure — Re/Im panels, Tzx/Tzy series (DESIGN.md §10.6).

    Mirrors the AppRes & Phase grammar exactly (USArray reference figure):
    panels = Re (top) / Im (bottom); series = Tzx = OPEN circle (blue) /
    Tzy = cross (red); symbols only (no connecting lines); markersize 5,
    capsize 2.5, elinewidth 0.8; shared log-period x-limits
    (``_decade_limits``). ``ylim`` = (min, max) pins the y-axis on both
    panels; None keeps the smart default ([-1, 1], auto-expanded when the
    data legitimately exceeds 1 near 2D/3D contrasts). ``show_params=False``
    drops the processing-provenance line for production images (§23.3).
    """
    panels = tipper_panels(tf)
    if not panels:
        raise ValueError("transfer function has no tipper data")
    period = panels[0]["period"]
    # series per component (AppRes grammar: component -> marker/color)
    series = {p["label"].lower(): p for p in panels}  # tzx / tzy
    parts = (("Re", lambda z: z.real), ("Im", lambda z: z.imag))

    fig = plt.figure(figsize=(6.5, 6.4))
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 1], hspace=0.18)
    ax_re = fig.add_subplot(gs[0])
    ax_im = fig.add_subplot(gs[1], sharex=ax_re)
    axes = [ax_re, ax_im]
    for i, (part, getter) in enumerate(parts):
        ax = axes[i]
        for key, p in series.items():
            z = p["z"]
            ok = np.isfinite(z.real) & np.isfinite(z.imag)
            if not np.any(ok):
                continue
            data = getter(z)[ok]
            z_std = p["z_std"]
            err = (z_std / np.sqrt(2.0))[ok] if z_std is not None else None
            std = ("xy" if key == "tzx" else "yx")
            ax.errorbar(
                period[ok], data, yerr=err,
                fmt=MARKERS.get(std, "o"),
                color=COLORS.get(std, "k"),
                markerfacecolor=MARKER_FACES.get(std, COLORS.get(std, "k")),
                label=p["label"].upper(), capsize=2.5, markersize=5,
                linewidth=1.0, elinewidth=0.8)
        ax.set_xscale("log")
        ax.set_ylabel(f"$T_z$ {part} (dimensionless)")
        finite = np.concatenate([
            np.abs(getter(p["z"])[np.isfinite(p["z"])])
            for p in series.values()]) if series else np.array([])
        if ylim is not None:
            ax.set_ylim(float(ylim[0]), float(ylim[1]))
        else:
            bound = max(1.0, 1.1 * float(finite.max())) if finite.size else 1.0
            ax.set_ylim(-bound, bound)
    # boxed marker-only legend in the Re panel (AppRes convention: no
    # legend in the lower panel)
    axes[0].legend(loc="best", frameon=True, framealpha=0.9, fontsize=8,
                   borderpad=0.3, handlelength=0.0, handletextpad=0.5)
    apply_style(fig, axes)
    axes[0].set_xlim(*_decade_limits(period))
    axes[-1].set_xlabel("Period (secs)")
    fig.suptitle(station_title(tf), fontsize=11, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    _annotate_notes(fig, axes[-1], run_notes(tf), show_params)
    fig.savefig(path, dpi=DPI, bbox_inches="tight",
                metadata={"Software": "emptyUI", "Title": station_title(tf)})
    plt.close(fig)
    return path


def _impedance_matrices(tf):
    """Full 2x2 complex impedance per period, ordered (ex,ey) x (hx,hy).

    Raises when any of the four components is missing (the phase tensor is
    only defined for the complete tensor).
    """
    Z = tf.dataset["transfer_function"]
    period = np.asarray(Z.coords["period"], dtype=float)
    outs = (tf.ex, tf.ey)
    ins = (tf.hx, tf.hy)
    for o in outs:
        if o not in Z.coords["output"].values:
            raise ValueError(
                f"transfer function has no '{o}' output — the phase tensor "
                "needs the complete 2x2 impedance")
    for i in ins:
        if i not in Z.coords["input"].values:
            raise ValueError(
                f"transfer function has no '{i}' input — the phase tensor "
                "needs the complete 2x2 impedance")
    z = np.empty((period.size, 2, 2), dtype=complex)
    for a, o in enumerate(outs):
        for b, i in enumerate(ins):
            z[:, a, b] = Z.sel(output=o, input=i).values.astype(complex)
    return period, z


def _ellipse_outline(p, w_px, h_px, angle_deg, px_per_decade, px_per_y, n=72):
    """Ellipse outline in data coords, constructed in *display* space.

    On a log x-axis a data-space ellipse cannot be rotated correctly: the
    data-space path is log-warped (its huge data-space half-width rotates
    into the y-extent — the flattened-sliver bug). Instead sample the
    ellipse in display pixels (x = decade offsets, y linear), rotate
    there, and map back: ``x_data = p·10^(dx/px_per_decade)``.
    Returns (x_data, y_data) polygon vertices.
    """
    t = np.linspace(0.0, 2.0 * np.pi, n)
    xs = 0.5 * w_px * np.cos(t)
    ys = 0.5 * h_px * np.sin(t)
    th = np.radians(angle_deg)
    xr = xs * np.cos(th) - ys * np.sin(th)
    yr = xs * np.sin(th) + ys * np.cos(th)
    return p * 10.0 ** (xr / px_per_decade), yr / px_per_y


def plot_phase_tensor(tf, path, show_params: bool = True):
    """Render the Caldwell-style phase-tensor figure (DESIGN.md §10.6).

    PT ellipses vs period: the image ellipse of Φ = Re(Z)⁻¹·Im(Z) at each
    period (semi-axes scaled by the largest σ1 so the biggest ellipse fits
    the panel; relative sizes and aspect preserved), principal orientation
    from the PT azimuth, fill color = skew (Caldwell β, colorbar) — the
    standard 3D-structure indicator. Ellipses are constructed in display
    space so the log x-axis cannot distort them (_ellipse_outline). No PT
    error bars in this MVP (Z_err → PT covariance propagation is
    nontrivial; §20.2). ``show_params=False`` drops the
    processing-provenance line for production images (§23.3).
    """
    from .phase_tensor import phase_tensor, pt_ellipse_params

    period, z = _impedance_matrices(tf)
    phi = phase_tensor(z)
    az, skew, _ell, s1, s2 = pt_ellipse_params(phi)

    ok = np.isfinite(s1)
    if not np.any(ok):
        raise ValueError(
            "phase tensor is undefined — Re(Z) is singular at every period")
    period_ok = period[ok]
    s1_ok = s1[ok]
    s2_ok = np.abs(s2[ok])  # |σ2| is the true minor semi-axis (σ2 signed)
    skew_ok = skew[ok]
    az_ok = az[ok]

    scale = float(np.nanmax(s1_ok))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("phase tensor is degenerate (all-zero impedance)")
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    apply_style(fig, (ax,))
    cmap = plt.get_cmap("RdYlBu_r")
    vmax = max(5.0, float(np.nanmax(np.abs(skew_ok))))
    norm = plt.Normalize(vmin=-vmax, vmax=vmax)

    x_min, x_max = _decade_limits(period_ok)
    y_span = 2.7
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(-y_span / 2, y_span / 2)
    ax.set_xscale("log")
    ax.set_xlabel("Period (secs)")
    ax.set_ylabel("Phase tensor (normalized)")
    ax.set_yticks([-1, 0, 1])
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    cbar = fig.colorbar(sm, ax=ax, pad=0.02, fraction=0.05)
    cbar.set_label(r"Skew $\beta$ (deg)", fontsize=9)
    ax.set_title(station_title(tf) + " — phase tensor", fontsize=11,
                 fontweight="bold")
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    _annotate_notes(fig, ax, run_notes(tf), show_params)

    # measure the laid-out axes: pixels per decade (log x) and per y-unit
    fig.canvas.draw()
    bbox = ax.get_window_extent()
    px_per_decade = bbox.width / (np.log10(x_max) - np.log10(x_min))
    px_per_y = bbox.height / y_span
    # the largest ellipse spans 80% of the panel height
    unit = 0.8 * bbox.height / scale  # pixels per σ unit

    for p, w, h, angle, sk in zip(period_ok, s1_ok, s2_ok, az_ok, skew_ok):
        xs, ys = _ellipse_outline(
            p, w * unit, h * unit, angle, px_per_decade, px_per_y)
        ax.fill(xs, ys, facecolor=cmap(norm(sk)), edgecolor="0.25",
                linewidth=0.7, alpha=0.85, zorder=2)
    fig.savefig(path, dpi=DPI, bbox_inches="tight",
                metadata={"Software": "emptyUI", "Title": station_title(tf)})
    plt.close(fig)
    return path
