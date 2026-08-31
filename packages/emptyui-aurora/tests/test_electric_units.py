"""Unit-aware electric scaling (docs/DESIGN.md §10.5): Compute FC converts E to V/m
according to the payload's unit attrs — the review found the electric path
scaled unconditionally x1e-6 (mV/km assumed), silently mis-scaling payloads
already in V/m and double-scaling when e_dipole_length_m was set on mV/km
data. Unitless payloads keep the legacy behavior (compat).

Mirrors test_magnetic_units.py (the magnetic path was fixed the same way in
history §16.3).
"""

from __future__ import annotations

import logging

import numpy as np
import pytest
from _synthetic import compute_fc_seconds
from emptyui_aurora._processing import compute_fc_levels
from test_magnetic_units import N_TO_AM, _runts


def _field_vm(rng, n=16384, sr=8.0, z=(1.0, 3.0, -1.0, 4.0)):
    """H in nT, E already in V/m (e.g. calibrated data)."""
    t = np.arange(n) / sr
    hx = 50.0 * np.sin(2 * np.pi * 0.5 * t) + 0.5 * rng.standard_normal(n)
    hy = 40.0 * np.sin(2 * np.pi * 1.3 * t) + 0.5 * rng.standard_normal(n)
    ha = np.vstack([hx, hy]) * N_TO_AM
    e_am = np.array([[z[0], z[1]], [z[2], z[3]]]) @ ha
    return hx, hy, e_am[0], e_am[1]  # E in V/m


def _recovered(runts, **fc_kw):
    levels = compute_fc_levels(
        runts, ["hx", "hy"], ["ex", "ey"],
        num_decimations=1, factor=4, num_samples_window=256,
        overlap=0.5, window_type="boxcar", channel_nomenclature="default",
        **fc_kw,
    )
    from emptyui_aurora._processing import estimate_tf_levels, merge_tf_levels

    per = estimate_tf_levels(
        levels, ["hx", "hy"], ["ex", "ey"], estimator_engine="OLS",
        periods_per_octave=4, channel_nomenclature="default",
    )
    return merge_tf_levels(per)


def _assert_z(P, Z, true=(1.0, 3.0, -1.0, 4.0), rel=0.01):
    ok = np.abs(Z).reshape(len(P), -1).max(axis=1) > 1e-9
    for i in range(2):
        for j in range(2):
            assert np.median(Z[ok, i, j].real) == pytest.approx(
                true[i * 2 + j], rel=rel
            ), (i, j)


def test_vm_payload_passes_through(tmp_path):
    """E already in V/m (calibrated data) must NOT be scaled — previously
    silently mis-scaled by 1e6."""
    rng = np.random.default_rng(41)
    hx, hy, ex, ey = _field_vm(rng)
    runts = _runts(hx, hy, ex, ey, units_e="V/m")
    P, Z, _ = _recovered(runts)
    _assert_z(P, Z)
    assert "passthrough (V/m)" in levels(runts)["electric_scaling"]


def levels(runts):
    return compute_fc_levels(
        runts, ["hx", "hy"], ["ex", "ey"],
        num_decimations=0, num_samples_window=256, overlap=0.5,
        window_type="boxcar", channel_nomenclature="default",
    )[0].attrs


def test_mv_km_payload_still_scaled(tmp_path):
    """mV/km attrs -> x1e-6 (the legacy correct path, label recorded)."""
    rng = np.random.default_rng(42)
    hx, hy, _, _ = _field_vm(rng)
    e_am = np.vstack([hx, hy]) * N_TO_AM
    z = np.array([[1.0, 3.0], [-1.0, 4.0]]) @ e_am
    runts = _runts(hx, hy, z[0] * 1e6, z[1] * 1e6)  # E in mV/km
    P, Z, _ = _recovered(runts)
    _assert_z(P, Z)
    assert "mV/km -> V/m" in levels(runts)["electric_scaling"]


def test_mv_per_meter_scaled(tmp_path):
    """mV/m attrs -> x1e-3."""
    rng = np.random.default_rng(43)
    hx, hy, ex, ey = _field_vm(rng)
    runts = _runts(hx, hy, ex * 1e3, ey * 1e3, units_e="mV/m")
    P, Z, _ = _recovered(runts)
    _assert_z(P, Z)
    assert "mV/m -> V/m" in levels(runts)["electric_scaling"]


def test_raw_mv_potential_without_dipole_errors(tmp_path):
    """milliVolt attrs without a dipole length are a hard, actionable error —
    previously silently mis-scaled by 1e6."""
    rng = np.random.default_rng(44)
    hx, hy, ex, ey = _field_vm(rng)
    runts = _runts(hx, hy, ex * 1e3, ey * 1e3, units_e="milliVolt")
    with pytest.raises(ValueError, match="e_dipole_length_m"):
        compute_fc_seconds(runts, ["hx", "hy"], ["ex", "ey"],
                   window_length_s=256, overlap=0.5, window_type="boxcar",
                   channel_nomenclature="default")


def test_dipole_on_mvkm_data_warns(tmp_path, caplog):
    """e_dipole_length_m + mV/km attrs contradict: warn (the combination
    double-scales by dipole/1000) but proceed with the explicit choice."""
    rng = np.random.default_rng(45)
    hx, hy, ex, ey = _field_vm(rng)
    runts = _runts(hx, hy, ex * 1e6, ey * 1e6)  # E in mV/km
    with caplog.at_level(logging.WARNING, logger="emptyui"):
        fc = compute_fc_seconds(runts, ["hx", "hy"], ["ex", "ey"],
                        window_length_s=256, overlap=0.5,
                        window_type="boxcar", e_dipole_length_m=100.0,
                        channel_nomenclature="default")
    assert any("double-scale" in r.message for r in caplog.records)
    assert fc.attrs["electric_scaling"] == "mV potential -> V/m (dipole)"


def test_dipole_on_raw_mv_data_is_correct(tmp_path):
    """The intended dipole path: raw mV potential attrs + dipole length."""
    rng = np.random.default_rng(46)
    hx, hy, ex, ey = _field_vm(rng)
    # electrode potential in mV over a 100 m dipole: V/m = mV * 1e-3 / 100
    runts = _runts(hx, hy, ex * 1e3 * 100.0, ey * 1e3 * 100.0,
                   units_e="milliVolt")
    P, Z, _ = _recovered(runts, e_dipole_length_m=100.0)
    _assert_z(P, Z)


def test_unitless_keeps_legacy_behavior(tmp_path, caplog):
    """No unit attrs -> x1e-6 (compat: synthetic fixtures, core payloads);
    the legacy branch warns (E1) — a stripped V/m payload would be 1e6 off
    and the operator must see the assumption, not infer it."""
    rng = np.random.default_rng(47)
    hx, hy, ex, ey = _field_vm(rng)
    runts = _runts(hx, hy, ex * 1e6, ey * 1e6, units_e="")
    with caplog.at_level(logging.WARNING, logger="emptyui"):
        P, Z, _ = _recovered(runts)
    _assert_z(P, Z)
    assert "legacy (unitless, mV/km assumed)" in levels(runts)["electric_scaling"]
    assert any("NO unit attrs" in r.message for r in caplog.records)


def test_unrecognized_unit_warns_and_passes_through(tmp_path, caplog):
    """Unrecognized non-empty units: warning + passthrough — never silently
    scale what cannot be interpreted."""
    rng = np.random.default_rng(48)
    hx, hy, ex, ey = _field_vm(rng)
    runts = _runts(hx, hy, ex, ey, units_e="furlongs per fortnight")
    with caplog.at_level(logging.WARNING, logger="emptyui"):
        P, Z, _ = _recovered(runts)
    assert any("unrecognized unit" in r.message for r in caplog.records)
    _assert_z(P, Z)  # E was already V/m — passthrough recovers it


def test_canonical_unit_registry():
    """mt_metadata's registry canonicalizes abbreviations."""
    from emptyui_aurora._processing import _canonical_unit

    assert _canonical_unit("mV/km") == "milliVolt per kilometer"
    assert _canonical_unit("V/m") == "Volt per meter"
    assert _canonical_unit("") is None
    assert _canonical_unit(None) is None
    # unknown strings fall back to lowercased text
    assert _canonical_unit("Wombat") == "wombat"
