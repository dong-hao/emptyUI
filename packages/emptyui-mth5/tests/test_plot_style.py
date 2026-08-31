"""Publication-quality plot conventions (docs/DESIGN.md §10.6).

Asserts the USArray-style conventions of the TF display figures: log10
period and ρ_a axes, off-diagonal phase folded to (0, 90°), diagonal phase
signed [-180, 180°], tipper Re/Im with ylim [-1, 1], error bars when the TF
carries errors, ○ XY / ✕ YX markers, station id in the title, and the equal
per-decade scale (MT convention).
"""

from __future__ import annotations

import logging
from pathlib import Path
from unittest.mock import patch

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest
from emptyui.context import RunContext
from emptyui.store import DataStore
from emptyui_aurora._processing import build_tf
from emptyui_mth5 import plotting
from emptyui_mth5.plots import PlotTipper


def _impedance_tf(with_errors=True):
    periods = np.array([2.0, 4.0, 8.0, 16.0])
    Z = np.zeros((4, 3, 2), dtype=complex)
    Z[:, 0, 1] = 3 - 1j  # zxy (ex x by)
    Z[:, 1, 0] = 2 + 1j  # zyx (ey x bx)
    err = np.zeros((4, 3, 2), dtype=float)
    if with_errors:
        err[:, 0, 1] = 0.3
        err[:, 1, 0] = 0.2
    return build_tf(
        periods, Z, ["bx", "by"], ["e1", "e2", "bz"],
        channel_nomenclature="lemi12", Z_err=err,
        survey_id="bgs", station_id="NO46", run_id="sr1_0001",
        latitude=56.72, longitude=-2.85, elevation=120.0,
        processing_notes="estimator=Robust; levels=3",
    )


def _render(tf, mode, tmp_path: Path):
    """Render keeping the figure open so artists can be inspected."""
    path = tmp_path / "fig.png"
    with patch.object(plotting.plt, "close", lambda *a, **k: None):
        plotting.plot_impedance_response(tf, mode, path)
    fig = plt.gcf()
    return fig, path


def _diagonal_tf():
    periods = np.array([2.0, 4.0, 8.0, 16.0])
    Z = np.zeros((4, 3, 2), dtype=complex)
    Z[:, 0, 0] = 1 + 0j  # xx
    Z[:, 1, 1] = 2 + 0j  # yy
    return build_tf(periods, Z, ["bx", "by"], ["e1", "e2", "bz"],
                    channel_nomenclature="lemi12", station_id="NO46",
                    survey_id="bgs", latitude=56.72, longitude=-2.85)


def _tipper_tf():
    periods = np.array([2.0, 4.0, 8.0])
    Z = np.zeros((3, 3, 2), dtype=complex)
    Z[:, 0, 0] = 1 + 0j
    Z[:, 1, 1] = 2 + 0j
    Z[:, 2, 0] = 0.3 + 0.1j  # Tzx
    Z[:, 2, 1] = -0.2 + 0.05j  # Tzy
    err = np.zeros((3, 3, 2), dtype=float)
    err[:, 2, 0] = 0.01
    err[:, 2, 1] = 0.01
    return build_tf(periods, Z, ["bx", "by"], ["e1", "e2", "bz"],
                    channel_nomenclature="lemi12", Z_err=err,
                    station_id="NO46", survey_id="bgs",
                    latitude=56.72, longitude=-2.85)


def test_appres_phase_offdiagonal_conventions(tmp_path: Path):
    tf = _impedance_tf()
    fig, path = _render(tf, "off-diagonal", tmp_path)
    try:
        assert path.read_bytes().startswith(b"\x89PNG")
        ax_rho, ax_phi = fig.axes
        assert ax_rho.get_xscale() == "log" and ax_rho.get_yscale() == "log"
        assert ax_phi.get_xscale() == "log"
        assert ax_phi.get_ylim() == (0, 90)
        markers = {
            line.get_marker() for line in ax_rho.get_lines()
            if line.get_marker() in ("o", "x", "s", "D")
        }
        assert markers == {"o", "x"}  # XY = circle, YX = x (USArray)
        colors = {line.get_color() for line in ax_rho.get_lines()}
        assert len(colors) == 2  # blue/red families
        assert ax_rho.get_ylabel() == r"$\rho_a$ ($\Omega$ m)"
        assert ax_phi.get_ylabel() == r"$\phi$ (Degrees)"
        assert ax_phi.get_xlabel() == "Period (secs)"
        assert "NO46" in fig._suptitle.get_text()
        assert "LON" in fig._suptitle.get_text()
    finally:
        plt.close(fig)


def test_appres_phase_error_bars_present(tmp_path: Path):
    tf = _impedance_tf(with_errors=True)
    fig, _ = _render(tf, "off-diagonal", tmp_path)
    try:
        ax_rho = fig.axes[0]
        import matplotlib.container

        assert any(
            isinstance(c, matplotlib.container.ErrorbarContainer)
            for c in ax_rho.containers
        )
        assert tf.impedance_error is not None
    finally:
        plt.close(fig)


def test_appres_phase_diagonal_signed(tmp_path: Path):
    fig, _ = _render(_diagonal_tf(), "diagonal", tmp_path)
    try:
        ax_phi = fig.axes[1]
        assert ax_phi.get_ylim() == (-180, 180)
        assert fig._suptitle.get_text().startswith("NO46")
    finally:
        plt.close(fig)


def _phase_series(ax_phi, name):
    """Drawn phase y-values of the series labelled ``name`` (XY / YX)."""
    for container in ax_phi.containers:
        line = container.lines[0]
        if name in (container.get_label(), line.get_label()):
            return np.asarray(line.get_ydata(), dtype=float)
    raise AssertionError(f"no {name!r} series drawn on the phase panel")


def test_fold_phase_deg_is_a_full_mod180():
    """The ±180° fold is complete in both directions (docs/issues.md H5).

    The old one-sided ``p < 0 → p + 180`` pass left anything at/above 180°
    untouched, so wrapped phases were clipped away by the (0°, 90°) axis.
    """
    from emptyui_mth5.plotting import _fold_phase_deg

    raw = np.array([-135.0, -18.4, 0.0, 45.0, 90.0, 161.6, 180.0, 190.0,
                    375.0, np.nan])
    got = _fold_phase_deg(raw)
    np.testing.assert_allclose(
        got[:9], [45.0, 161.6, 0.0, 45.0, 90.0, 161.6, 0.0, 10.0, 15.0],
        atol=1e-9)
    assert np.isnan(got[9])


def test_in_quadrant_phase_draws_every_point(tmp_path: Path, caplog):
    """A healthy off-diagonal TF fills the phase panel with no warning."""
    periods = np.array([2.0, 4.0, 8.0, 16.0])
    Z = np.zeros((4, 3, 2), dtype=complex)
    Z[:, 0, 1] = 3 + 1j    # xy: +18.4° — first quadrant
    Z[:, 1, 0] = 2 + 1j    # yx: +26.6° — first quadrant
    tf = build_tf(periods, Z, ["bx", "by"], ["e1", "e2", "bz"],
                  channel_nomenclature="lemi12",
                  survey_id="bgs", station_id="NO46", run_id="sr1_0001")
    with caplog.at_level(logging.WARNING):
        fig, _ = _render(tf, "off-diagonal", tmp_path)
    try:
        ax_phi = fig.axes[1]
        for name in ("XY", "YX"):
            series = _phase_series(ax_phi, name)
            assert len(series) == len(periods)
            assert np.all((series >= 0.0) & (series <= 90.0)), (name, series)
        assert not [r for r in caplog.records
                    if "phase panel" in r.getMessage()]
    finally:
        plt.close(fig)


def test_out_of_quadrant_phase_is_reported_not_silent(tmp_path: Path,
                                                       caplog):
    """Values ±180° cannot place in (0°, 90°) are counted and warned about.

    The shipped fixture's ``Zxy = 3 - 1j`` folds to 161.6° — outside the
    display quadrant, so matplotlib drops those markers: previously they
    vanished silently and the panel simply looked empty (docs/issues.md H5).
    """
    tf = _impedance_tf()
    with caplog.at_level(logging.WARNING, logger="emptyui"):
        fig, _ = _render(tf, "off-diagonal", tmp_path)
    try:
        ax_phi = fig.axes[1]
        xy = _phase_series(ax_phi, "XY")
        yx = _phase_series(ax_phi, "YX")
        # YX (26.6°) is inside the panel and stays drawn
        assert np.all((yx >= 0.0) & (yx <= 90.0))
        # XY folds to 161.6° — outside the 0–90° box, so it is clipped, and
        # the clip is now loud instead of silent
        assert np.all((xy > 90.0) & (xy < 180.0)), xy
        messages = [r.getMessage() for r in caplog.records
                     if "phase panel" in r.getMessage()]
        assert len(messages) == 1, messages
        assert "XY" in messages[0]
        assert "4 point(s)" in messages[0]
        assert "161.6" in messages[0]
    finally:
        plt.close(fig)


def test_equal_per_decade_scale(tmp_path: Path):
    tf = _impedance_tf()
    fig, _ = _render(tf, "off-diagonal", tmp_path)
    try:
        ax = fig.axes[0]
        fig.canvas.draw()
        pos = ax.get_position()
        fw, fh = fig.get_size_inches()
        xd = np.log10(ax.get_xlim()[1]) - np.log10(ax.get_xlim()[0])
        yd = np.log10(ax.get_ylim()[1]) - np.log10(ax.get_ylim()[0])
        x_per_decade = (pos.width * fw) / xd
        y_per_decade = (pos.height * fh) / yd
        assert abs(x_per_decade - y_per_decade) / x_per_decade < 0.05
    finally:
        plt.close(fig)


def test_rho_ylim_pins_axis(tmp_path: Path):
    """Explicit ρ_a y-limits pin the axis (reference-figure matching,
    docs/DESIGN.md §10.6 / SDG31b_H28coh.png head-to-head)."""
    tf = _impedance_tf()
    path = tmp_path / "fig.png"
    with patch.object(plotting.plt, "close", lambda *a, **k: None):
        plotting.plot_impedance_response(tf, "off-diagonal", path,
                                         rho_ylim=(5.0, 40000.0))
    fig = plt.gcf()
    try:
        ax_rho = fig.axes[0]
        assert ax_rho.get_ylim() == (5.0, 40000.0)
    finally:
        plt.close(fig)


def test_offdiagonal_phase_panel_half_height(tmp_path: Path):
    """The off-diagonal phase panel is about half the diagonal one (0-90°
    vs the full signed range) — docs/DESIGN.md §10.6 / feedback 2026-09-06."""
    fig_off, _ = _render(_impedance_tf(), "off-diagonal", tmp_path)
    ax_off = fig_off.axes[1]
    fh_off = fig_off.get_size_inches()[1]
    h_off = ax_off.get_position().height * fh_off
    plt.close(fig_off)

    fig_diag, _ = _render(_diagonal_tf(), "diagonal", tmp_path)
    ax_diag = fig_diag.axes[1]
    fh_diag = fig_diag.get_size_inches()[1]
    h_diag = ax_diag.get_position().height * fh_diag
    plt.close(fig_diag)

    assert 0.4 <= h_off / h_diag <= 0.6


def test_tipper_reim_conventions(tmp_path: Path):
    ctx = RunContext(workspace=tmp_path, store=DataStore())
    tf = _tipper_tf()
    out = PlotTipper().execute(ctx, tf)
    path = Path(out["plot"]["path"])
    assert path.read_bytes().startswith(b"\x89PNG")

    path2 = tmp_path / "tip.png"
    with patch.object(plotting.plt, "close", lambda *a, **k: None):
        plotting.plot_tipper(tf, path2)
    fig = plt.gcf()
    try:
        ax1, ax2 = fig.axes
        for ax in (ax1, ax2):
            assert ax.get_ylim() == (-1, 1)
            # symbols only (no connecting lines) + shared component grammar:
            # Tzx = circle (blue), Tzy = x (red) — mirror of AppRes
            lines = [l for l in ax.get_lines()
                     if l.get_marker() in ("o", "x", "s", "D")]
            markers = {l.get_marker() for l in lines}
            assert markers == {"o", "x"}
            labels = {c.get_label() for c in ax.containers}
            assert labels == {"TZX", "TZY"}
            assert all(l.get_linestyle() in ("None", "", "none")
                       for l in lines if l.get_marker() != "_")
            assert ax.get_ylabel().endswith("(dimensionless)")
            # shared log-period x-limits (AppRes convention)
            assert ax.get_xscale() == "log"
        assert ax1.get_ylabel().startswith("$T_z$ Re")
        assert ax2.get_ylabel().startswith("$T_z$ Im")
        # Tzx renders as an OPEN circle (white face) — USArray reference
        tzx_line = next(c[0] for c in ax1.containers if c.get_label() == "TZX")
        assert tzx_line.get_markerfacecolor() == "white"
        assert "NO46" in fig._suptitle.get_text()
    finally:
        plt.close(fig)


def test_tipper_ylim_auto_expands_beyond_one(tmp_path: Path):
    """|T| > 1 is legitimate near 2D/3D conductivity contrasts — the y-axis
    must expand instead of clipping the display (default stays [-1, 1])."""
    periods = np.array([2.0, 4.0, 8.0])
    Z = np.zeros((3, 3, 2), dtype=complex)
    Z[:, 2, 0] = 1.4 + 0.2j  # Tzx exceeding 1
    Z[:, 2, 1] = -0.2 + 0.05j
    tf = build_tf(periods, Z, ["bx", "by"], ["e1", "e2", "bz"],
                  channel_nomenclature="lemi12", station_id="NO46",
                  survey_id="bgs")
    path = tmp_path / "tip.png"
    with patch.object(plotting.plt, "close", lambda *a, **k: None):
        plotting.plot_tipper(tf, path)
    fig = plt.gcf()
    try:
        ax1, _ = fig.axes
        lo, hi = ax1.get_ylim()
        assert lo == pytest.approx(-1.1 * 1.4, abs=0.15) and hi == pytest.approx(1.1 * 1.4, abs=0.15)
    finally:
        plt.close(fig)


def test_plot_nodes_render_via_workspace(tmp_path: Path):
    """The plot nodes write into the workspace even when it is a str path
    (the executor passes a str) — RunContext normalizes to Path."""
    ctx = RunContext(workspace=str(tmp_path), store=DataStore())
    tf = _tipper_tf()
    out = PlotTipper().execute(ctx, tf)
    assert Path(out["plot"]["path"]).exists()


def test_high_dpi_saved(tmp_path: Path):
    tf = _impedance_tf()
    fig, path = _render(tf, "off-diagonal", tmp_path)
    plt.close(fig)
    from PIL import Image
    with Image.open(path) as im:
        assert round(im.info.get("dpi", (0, 0))[0]) == plotting.DPI


def test_appres_xy_is_open_circle(tmp_path: Path):
    """XY renders as an OPEN circle (white face, colored edge) — USArray
    reference figure convention (review 2026-09-12)."""
    tf = _impedance_tf()
    fig, _path = _render(tf, "off-diagonal", tmp_path)
    try:
        ax_rho = fig.axes[0]
        xy_line = next(c[0] for c in ax_rho.containers if c.get_label() == "XY")
        assert xy_line.get_markerfacecolor() == "white"
        yx = next(c for c in ax_rho.containers if c.get_label() == "YX")
        assert yx[0].get_marker() == "x"  # cross has no face
    finally:
        plt.close(fig)


def test_tipper_ylim_param_pins_axis(tmp_path: Path):
    """An explicit tipper_ylim range pins both Re/Im panels exactly."""
    ctx = RunContext(workspace=tmp_path, store=DataStore())
    out = PlotTipper().execute(ctx, _tipper_tf(), tipper_ylim="-2,2")
    path = Path(out["plot"]["path"])
    assert path.read_bytes().startswith(b"\x89PNG")
    path2 = tmp_path / "tip2.png"
    with patch.object(plotting.plt, "close", lambda *a, **k: None):
        plotting.plot_tipper(_tipper_tf(), path2, ylim=(-2.0, 2.0))
    fig = plt.gcf()
    try:
        for ax in fig.axes[:2]:
            assert ax.get_ylim() == (-2.0, 2.0)
    finally:
        plt.close(fig)


def test_tipper_ylim_param_validation(tmp_path):
    ctx = RunContext(workspace=tmp_path, store=DataStore())
    for bad in ("2,-2", "1", "x", "1,2,3"):
        with pytest.raises(ValueError, match="tipper_ylim"):
            PlotTipper().execute(ctx, _tipper_tf(), tipper_ylim=bad)


_LONG_NOTE = (
    "estimator=RobustStaged; remote=False; band_style=octave; "
    "periods_per_octave=4; min_bins_per_band=7; t_short=0.0001; "
    "decimation_levels=8; window=128 samples (parkes); levels=3; "
    "coherence_threshold=0.6; nomenclature=lemi12; reference=SDG31b"
)


def _long_note_tf():
    periods = np.array([2.0, 4.0, 8.0, 16.0])
    Z = np.zeros((4, 3, 2), dtype=complex)
    Z[:, 0, 1] = 3 - 1j
    Z[:, 1, 0] = 2 + 1j
    return build_tf(periods, Z, ["bx", "by"], ["e1", "e2", "bz"],
                    channel_nomenclature="lemi12",
                    survey_id="bgs", station_id="NO46", run_id="sr1_0001",
                    latitude=56.72, longitude=-2.85,
                    processing_notes=_LONG_NOTE)


def _long_note_tipper_tf():
    periods = np.array([2.0, 4.0, 8.0])
    Z = np.zeros((3, 3, 2), dtype=complex)
    Z[:, 0, 0] = 1 + 0j
    Z[:, 1, 1] = 2 + 0j
    Z[:, 2, 0] = 0.3 + 0.1j
    Z[:, 2, 1] = -0.2 + 0.05j
    return build_tf(periods, Z, ["bx", "by"], ["e1", "e2", "bz"],
                    channel_nomenclature="lemi12",
                    survey_id="bgs", station_id="NO46", run_id="sr1_0001",
                    latitude=56.72, longitude=-2.85,
                    processing_notes=_LONG_NOTE)


def _note_text(fig):
    return next((t for t in fig.texts if "estimator" in t.get_text()), None)


def test_print_params_toggle(tmp_path: Path):
    """The provenance line is printed by default and dropped on demand."""
    for fn, tf in ((plotting.plot_impedance_response, _long_note_tf()),
                   (plotting.plot_tipper, _long_note_tipper_tf()),
                   (plotting.plot_phase_tensor, _long_note_tf())):
        path = tmp_path / f"on_{fn.__name__}.png"
        with patch.object(plotting.plt, "close", lambda *a, **k: None):
            fn(tf, path) if fn is not plotting.plot_impedance_response \
                else fn(tf, "off-diagonal", path)
        fig = plt.gcf()
        try:
            assert _note_text(fig) is not None
        finally:
            plt.close(fig)
        path = tmp_path / f"off_{fn.__name__}.png"
        with patch.object(plotting.plt, "close", lambda *a, **k: None):
            fn(tf, path, show_params=False) \
                if fn is not plotting.plot_impedance_response \
                else fn(tf, "off-diagonal", path, show_params=False)
        fig = plt.gcf()
        try:
            assert _note_text(fig) is None
        finally:
            plt.close(fig)


def test_notes_wrapped_to_axes_width(tmp_path: Path):
    """A long provenance line wraps; no line is wider than the axes."""
    tf = _long_note_tf()
    path = tmp_path / "fig.png"
    with patch.object(plotting.plt, "close", lambda *a, **k: None):
        plotting.plot_impedance_response(tf, "off-diagonal", path)
    fig = plt.gcf()
    try:
        note = _note_text(fig)
        assert note is not None
        assert "\n" in note.get_text()
        renderer = fig.canvas.get_renderer()
        ax_w = fig.axes[1].get_window_extent(renderer).width
        for line in note.get_text().split("\n"):
            probe = fig.text(0, 0, line, fontsize=note.get_fontsize())
            assert probe.get_window_extent(renderer).width <= ax_w + 1.0
            probe.remove()
        # wrapped note no longer inflates the saved canvas (history §23.3: the old
        # single-line note rendered ~1100px wide on a ~310px figure)
        import io

        from PIL import Image
        buf = io.BytesIO(path.read_bytes())
        assert Image.open(buf).size[0] < 800
    finally:
        plt.close(fig)


def test_notes_below_xlabel(tmp_path: Path):
    """The note sits below the x-axis label, never overlapping it."""
    tf = _long_note_tf()
    path = tmp_path / "fig.png"
    with patch.object(plotting.plt, "close", lambda *a, **k: None):
        plotting.plot_impedance_response(tf, "off-diagonal", path)
    fig = plt.gcf()
    try:
        note = _note_text(fig)
        renderer = fig.canvas.get_renderer()
        xlabel_bb = fig.axes[1].xaxis.label.get_window_extent(renderer)
        note_bb = note.get_window_extent(renderer)
        # anchored 5px below the x-label — never overlapping (history §23.3); when
        # the wrapped block needs more room than the canvas has,
        # bbox_inches="tight" extends the saved image downward to include it
        assert note_bb.y1 <= xlabel_bb.y0 + 0.6
    finally:
        plt.close(fig)


def test_plot_impedance_response_all_nan_impedance(tmp_path):
    """L-M9: all-NaN impedance must fail with the actionable message (the
    tipper plot's guard pattern), not a raw concatenate crash."""
    periods = np.array([2.0, 4.0, 8.0, 16.0])
    Z = np.full((4, 3, 2), np.nan + 0j, dtype=complex)
    tf = build_tf(
        periods, Z, ["bx", "by"], ["e1", "e2", "bz"],
        channel_nomenclature="lemi12", survey_id="bgs", station_id="NO46",
        run_id="sr1_0001",
    )
    with pytest.raises(ValueError, match="all NaN"):
        plotting.plot_impedance_response(tf, "off-diagonal", tmp_path / "fig.png")


# --- L7: print_params manifest/method default agreement -------------------

def test_print_params_manifest_default_matches_the_method():
    """L7: the executor injects the manifest default, so runs were
    correct — but a direct node.execute() call (as tests and embeddings
    do) silently flipped the documented-off provenance line on PlotTipper
    and PlotPhaseTensor (the method default was True)."""
    import inspect

    from emptyui_mth5.plots import PlotAppResPhase, PlotPhaseTensor, PlotTipper

    for node_cls in (PlotAppResPhase, PlotTipper, PlotPhaseTensor):
        node = node_cls()
        manifest_default = next(
            p.default for p in node.params if p.name == "print_params")
        method_default = inspect.signature(
            node.execute).parameters["print_params"].default
        assert method_default == manifest_default, (
            f"{node_cls.__name__}: manifest {manifest_default!r} != "
            f"method {method_default!r}")
