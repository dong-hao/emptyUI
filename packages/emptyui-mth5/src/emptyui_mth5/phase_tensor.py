"""Phase tensor computation (DESIGN.md §10.6) — Caldwell et al. (2004).

The phase tensor Φ = X⁻¹·Y of a 2×2 impedance Z = X + iY is dimensionless
and galvanic-distortion free. Its polar form Φ = R(α1)·diag(σ1,σ2)·R(α2)
(the SVD with rotation matrices) gives the PT ellipse: semi-axes σ1 ≥ σ2,
principal azimuth θ = (α1 − α2)/2, skew β = (α1 + α2)/2, and ellipticity
λ = (σ1 − σ2)/(σ1 + σ2). Invariants (under test): 1D → Φ ∝ I (circular,
θ ≡ 0, β ≡ 0); 2D strike θ → PT azimuth recovers the strike, β ≡ 0.
"""

from __future__ import annotations

import numpy as np


def phase_tensor(z: np.ndarray) -> np.ndarray:
    """Φ = Re(Z)⁻¹·Im(Z) for impedance arrays (..., 2, 2) complex.

    Uses the pseudo-inverse with a condition guard: a singular Re(Z)
    (e.g. a degenerate band) yields NaN rather than inflated garbage.
    """
    z = np.asarray(z, dtype=complex)
    phi = np.empty(z.shape, dtype=float)
    flat = z.reshape(-1, 2, 2)
    out = phi.reshape(-1, 2, 2)
    for k in range(flat.shape[0]):
        xk = flat[k].real
        if np.linalg.matrix_rank(xk) < 2 or not np.all(np.isfinite(xk)):
            out[k] = np.nan
            continue
        out[k] = np.linalg.pinv(xk, rcond=1e-12) @ flat[k].imag
    return phi


def pt_ellipse_params(phi: np.ndarray):
    """(azimuth_deg, skew_deg, ellipticity, sigma1, sigma2) per matrix.

    Caldwell et al. (2004) invariants (sign-safe, vectorized): with
    p1 = (φxx+φyy)/2, p2 = (φxx−φyy)/2, p3 = (φxy+φyx)/2, p4 = (φxy−φyx)/2 —
    the rotation-invariant pairs are (p1, p4) and (p2, p3) (they rotate by
    2× the coordinate angle under Φ' = RΦ Rᵀ):

    - σ1 = √(p1²+p4²) + √(p2²+p3²)  (major semi-axis, ≥ 0)
    - σ2 = √(p1²+p4²) − √(p2²+p3²)  (signed minor value; |σ2| = minor axis)
    - principal azimuth θ = ½·atan2(p3, p2) (the ellipse major-axis
      direction; mod-180-periodic, folded into (−90°, 90°]). Rotation-
      covariant: rotating the impedance by ρ shifts θ by ∓ρ (the sign
      follows the rotation convention of the data) — verified on 2D
      synthetic strikes (absolute azimuth recovers the strike).
    - skew β = ½·atan(p4/p1) = ½·asin((φyx−φxy)/(σ1+σ2)) — Caldwell's
      β ∈ (−45°, 45°) by construction (the arctan fold IS the paper's
      branch convention, eq. A10); 2D → 0 (φxy = φyx), 1D → 0.

    NaN matrices propagate. ellipticity = (σ1−|σ2|)/(σ1+|σ2|) ∈ [0, 1].
    """
    phi = np.asarray(phi, dtype=float)
    p1 = (phi[..., 0, 0] + phi[..., 1, 1]) / 2.0
    p2 = (phi[..., 0, 0] - phi[..., 1, 1]) / 2.0
    p3 = (phi[..., 0, 1] + phi[..., 1, 0]) / 2.0
    p4 = (phi[..., 0, 1] - phi[..., 1, 0]) / 2.0
    a = np.hypot(p1, p4)
    b = np.hypot(p2, p3)
    sigma1 = a + b
    sigma2 = a - b  # signed (Caldwell); |sigma2| = minor semi-axis
    az = np.degrees(np.arctan2(p3, p2)) / 2.0
    az = ((az + 90.0) % 180.0) - 90.0  # fold into (−90, 90]
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = np.where(a > 0, p4 / p1, np.nan)  # p1=0, p4≠0 → ±inf → ±45°
        skew = np.degrees(np.arctan(ratio)) / 2.0
    denom = sigma1 + np.abs(sigma2)
    ell = np.where(denom > 0, (sigma1 - np.abs(sigma2)) / np.where(denom == 0, 1, denom),
                   np.nan)
    return az, skew, ell, sigma1, sigma2
