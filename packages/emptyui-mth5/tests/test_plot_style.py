"""Publication-quality plot conventions (DESIGN.md §15).

Asserts the USArray-style conventions of the TF display figures: log10
period and ρ_a axes, off-diagonal phase folded to (0, 90°), diagonal phase
signed [-180, 180°], tipper Re/Im with ylim [-1, 1], error bars when the TF
carries errors, ○ XY / ✕ YX markers, station id in the title, and the equal
per-decade scale (MT convention).
"""

from __future__ import annotations

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
        assert fig._suptitle.get_text().startswith("Site: NO46")
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


def test_offdiagonal_phase_panel_half_height(tmp_path: Path):
    """The off-diagonal phase panel is about half the diagonal one (0-90°
    vs the full signed range) — DESIGN.md §15 / feedback 2026-09-06."""
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
            markers = {
                line.get_marker() for line in ax.get_lines()
                if line.get_marker() in ("o", "x", "s", "D")
            }
            assert markers == {"o", "x"}  # Re = circle, Im = x
            assert ax.get_ylabel().endswith("(dimensionless)")
        assert ax1.get_ylabel().startswith("Tzx")
        assert ax2.get_ylabel().startswith("Tzy")
        assert "NO46" in fig._suptitle.get_text()
    finally:
        plt.close(fig)


def test_plot_nodes_render_via_workspace(tmp_path: Path):
    """The plot nodes write into the workspace even when it is a str path
    (the executor passes a str) — RunContext normalizes to Path."""
    ctx = RunContext(workspace=str(tmp_path), store=DataStore())
    tf = _tipper_tf()
    out = PlotTipper().execute(ctx, tf)
    assert Path(out["plot"]["path"]).exists()


def test_plot_tipper_requires_tipper(tmp_path: Path):
    """A 2x2 (impedance-only) TF has no tipper -> clear error."""
    ctx = RunContext(workspace=tmp_path, store=DataStore())
    periods = np.array([2.0, 4.0])
    Z = np.zeros((2, 2, 2), dtype=complex)
    Z[:, 0, 0] = 1 + 0j
    Z[:, 1, 1] = 2 + 0j
    tf = build_tf(periods, Z, ["bx", "by"], ["e1", "e2"],
                  channel_nomenclature="lemi12")
    assert not tf.has_tipper()
    with pytest.raises(ValueError, match="no tipper"):
        PlotTipper().execute(ctx, tf)


def test_high_dpi_saved(tmp_path: Path):
    tf = _impedance_tf()
    fig, path = _render(tf, "off-diagonal", tmp_path)
    plt.close(fig)
    from PIL import Image
    with Image.open(path) as im:
        assert round(im.info.get("dpi", (0, 0))[0]) == plotting.DPI
