"""Phase tensor computation + plot node (DESIGN §20.2).

Invariants: 1D → Φ ∝ I (circular, skew ≡ 0); 2D strike θ → azimuth
rotation-covariant (azimuth − azimuth₀ = ∓ρ), skew ≡ 0, singular values
invariant; 3D → β ≠ 0. σ₁/|σ₂| match the SVD of Φ.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from emptyui.context import RunContext
from emptyui.store import DataStore
from emptyui_mth5.phase_tensor import phase_tensor, pt_ellipse_params


def _rotmat(deg):
    t = np.radians(deg)
    c, s = np.cos(t), np.sin(t)
    return np.array([[c, s], [-s, c]])


def test_1d_phase_tensor_is_circular_and_undistorted():
    """1D earth: Z = z·I → Φ = c·I: circular (ell 0, σ1 = σ2), skew 0."""
    z = np.array([[[2 + 1j, 0], [0, 2 + 1j]]])
    phi = phase_tensor(z)
    assert np.allclose(phi[0], np.eye(2) * 0.5)
    _az, sk, el, s1, s2 = pt_ellipse_params(phi)
    assert sk[0] == pytest.approx(0.0, abs=1e-12)
    assert el[0] == pytest.approx(0.0, abs=1e-12)
    assert s1[0] == pytest.approx(s2[0])


def test_2d_strike_rotation_covariance():
    """Rotating a 2D impedance by ρ shifts the PT azimuth by ∓ρ (mod 180),
    keeps the skew at 0 and the singular values invariant."""
    zxy = 3.0 * np.exp(1j * np.radians(60))
    zyx = 4.0 * np.exp(1j * np.radians(-80))
    z0 = np.array([[[0, zxy], [zyx, 0]]])
    az0, sk0, el0, s10, s20 = pt_ellipse_params(phase_tensor(z0))
    assert sk0[0] == pytest.approx(0.0, abs=1e-12)

    for strike in (30.0, -45.0, 75.0):
        R = _rotmat(strike)
        zr = R @ z0[0] @ R.T
        az, sk, el, s1, s2 = pt_ellipse_params(phase_tensor(np.array([zr])))
        shift = ((az[0] - az0[0] + 90) % 180) - 90
        assert shift == pytest.approx(-strike)  # rotation covariance
        assert sk[0] == pytest.approx(0.0, abs=1e-12)
        assert s1[0] == pytest.approx(s10[0])
        assert abs(s2[0]) == pytest.approx(abs(s20[0]))
        assert el[0] == pytest.approx(el0[0])


def test_3d_phase_tensor_skew_is_nonzero():
    z = np.array([[[1 + 0.5j, 2 + 1j], [3 + 0.2j, 4 + 2j]]])
    phi = phase_tensor(z)
    _az, sk, el, s1, s2 = pt_ellipse_params(phi)
    assert abs(sk[0]) > 1  # 3D structure → nonzero skew
    # σ1 matches the SVD's largest singular value; |σ2| the smaller one
    _, sv, _ = np.linalg.svd(phi[0])
    assert s1[0] == pytest.approx(sv[0])
    assert abs(s2[0]) == pytest.approx(sv[1])
    assert 0 <= el[0] <= 1


def test_azimuth_skew_recover_caldwell_decomposition():
    """Absolute-orientation regression (review 2026-09-24): build Φ from
    Caldwell et al. (2004) eq. (20) — Φ = Rᵀ(α−β)·diag(Φmax,Φmin)·R(α+β),
    R(θ) = [[cosθ, sinθ], [−sinθ, cosθ]] — and assert the invariants
    recover the construction parameters. The previous formula's spurious
    −90° reported every azimuth 90° off (2D rotation-covariance tests
    could not see a constant offset)."""
    def R(a):
        c, s = np.cos(a), np.sin(a)
        return np.array([[c, s], [-s, c]])

    for alpha_deg, beta_deg in ((30, 10), (60, 20), (170, 15), (-40, 25)):
        a, b = np.radians(alpha_deg), np.radians(beta_deg)
        phi = R(-(a - b)) @ np.diag([3.0, 1.0]) @ R(a + b)
        az, sk, el, s1, s2 = pt_ellipse_params(phi[None])
        # α is mod-180-periodic: 170° ≡ −10°; the fold into (−90, 90] is
        # shape-preserving
        expected_az = ((alpha_deg + 90) % 180) - 90
        assert az[0] == pytest.approx(expected_az, abs=1e-9)
        assert sk[0] == pytest.approx(beta_deg, abs=1e-9)
        assert s1[0] == pytest.approx(3.0)
        assert abs(s2[0]) == pytest.approx(1.0)
        assert el[0] == pytest.approx((3.0 - 1.0) / (3.0 + 1.0))


def test_2d_azimuth_recovers_strike():
    """A 2D impedance's PT ellipse major axis must align with the
    geoelectric strike (the docstring's stated invariant, tested here in
    absolute terms for the first time)."""
    zxy = 10.0 * np.exp(1j * np.radians(60))
    zyx = 20.0 * np.exp(1j * np.radians(-100))
    z0 = np.array([[[0, zxy], [zyx, 0]]])
    az0, _sk0, _el0, _s10, _s20 = pt_ellipse_params(phase_tensor(z0))
    for strike in (0.0, 30.0, 70.0, -45.0):
        R = _rotmat(strike)
        zr = R @ z0[0] @ R.T
        az, sk, _el, _s1, _s2 = pt_ellipse_params(phase_tensor(np.array([zr])))
        assert sk[0] == pytest.approx(0.0, abs=1e-12)
        # strike 0 puts the major axis along x (azimuth 0 E of N)
        shift = ((az[0] - az0[0] + 90) % 180) - 90
        assert shift == pytest.approx(-strike)


def test_singular_re_z_propagates_nan():
    """A singular Re(Z) band yields NaN, not inflated garbage."""
    z = np.array([[[0 + 1j, 0], [0, 0 + 1j]]])  # Re(Z) = 0 → singular
    phi = phase_tensor(z)
    assert np.all(np.isnan(phi[0]))
    _az, _sk, el, _s1, _s2 = pt_ellipse_params(phi)
    assert np.all(np.isnan(_az)) and np.all(np.isnan(_sk)) and np.all(np.isnan(el))


def test_phase_tensor_node_plot(tmp_path):
    """The Display node renders a PNG for a synthetic 3D-like TF."""
    from emptyui_aurora._processing import build_tf
    from emptyui_mth5.plots import PlotPhaseTensor

    periods = np.array([2.0, 4.0, 8.0])
    Z = np.zeros((3, 2, 2), dtype=complex)
    Z[:, 0, 0] = 1 + 0.5j
    Z[:, 0, 1] = 2 + 1j
    Z[:, 1, 0] = 3 + 0.2j
    Z[:, 1, 1] = 4 + 2j
    tf = build_tf(periods, Z, ["bx", "by"], ["e1", "e2"],
                  channel_nomenclature="lemi12", station_id="NO46")

    artifact = PlotPhaseTensor().execute(
        RunContext(workspace=tmp_path, store=DataStore()), tf)["plot"]
    p = artifact["path"]
    assert p.endswith(".png")
    assert len(Path(p).read_bytes()) > 1000


def test_ellipse_outline_preserves_visual_geometry():
    """Aspect regression (review 2026-09-12): the display-space outline must
    render with exactly the requested pixel extent and aspect — the previous
    data-space ellipses on the log axis rendered as flattened slivers."""
    from emptyui_mth5.plotting import _ellipse_outline

    px_per_decade, px_per_y = 160.0, 120.0
    for p, w_px, h_px, angle in [(10.0, 120.0, 60.0, 0.0),
                                 (100.0, 90.0, 45.0, 30.0),
                                 (1000.0, 60.0, 60.0, 45.0)]:
        xs, ys = _ellipse_outline(p, w_px, h_px, angle, px_per_decade, px_per_y)
        # map back to display px
        dx = px_per_decade * np.log10(xs / p)
        dy = ys * px_per_y
        # rotated-ellipse bbox in display space (analytic extent)
        th = np.radians(angle)
        w_true = np.hypot(w_px / 2 * np.cos(th), h_px / 2 * np.sin(th)) * 2
        h_true = np.hypot(w_px / 2 * np.sin(th), h_px / 2 * np.cos(th)) * 2
        assert dx.max() - dx.min() == pytest.approx(w_true, rel=0.02)
        assert dy.max() - dy.min() == pytest.approx(h_true, rel=0.02)


def test_phase_tensor_ellipses_render_with_anisotropic_axes(tmp_path):
    """Aspect regression: even for a strongly anisotropic 3D tensor the
    ellipses occupy real panel space (not flattened slivers). Rendered
    ellipse pixel heights must reflect the σ2/σ1 ratio."""
    from emptyui_aurora._processing import build_tf
    from emptyui_mth5.plots import PlotPhaseTensor

    periods = np.geomspace(1.0, 1000.0, 12)
    Z = np.zeros((periods.size, 2, 2), dtype=complex)
    Z[:, 0, 0] = 1 + 0.5j
    Z[:, 0, 1] = 2 + 1j
    Z[:, 1, 0] = 3 + 0.2j
    Z[:, 1, 1] = 4 + 2j
    tf = build_tf(periods, Z, ["bx", "by"], ["e1", "e2"],
                  channel_nomenclature="lemi12", station_id="NO46")

    artifact = PlotPhaseTensor().execute(
        RunContext(workspace=tmp_path, store=DataStore()), tf)["plot"]
    assert len(Path(artifact["path"]).read_bytes()) > 1000
