"""FC-level frequency-domain calibration (DESIGN §24.12).

Compute FC divides the Fourier coefficients by the chained complex response
of the payload's *applied* filters (aurora semantics between STFT and
estimation): ``applied=True`` = the response is part of the recorded data
and must be removed; a prior time-domain Calibrate flips the flags, so the
two paths compose without double application. The chains travel with the
payload (core TimeSeries.calibration / RunTS dataset attrs + object
registry) and survive the core-node detour and Align Runs.
"""

from __future__ import annotations

import numpy as np
import pytest

from test_magnetic_units import _field, _runts


def _fap(name="t_fap", amp=2.0, phase_deg=90.0):
    """A deterministic FAP filter: response = amp * exp(i*phase) everywhere
    inside its table (phases passed in radians — mt_metadata's degrees
    heuristic misreads small-phase tables)."""
    from mt_metadata.timeseries.filters import FrequencyResponseTableFilter

    fap = FrequencyResponseTableFilter(
        frequencies=np.array([1e-3, 1e3]),
        amplitudes=np.array([amp, amp]),
        phases=np.deg2rad(np.array([phase_deg, phase_deg])),
        units_in="nanoTesla", units_out="nanoTesla", gain=1.0,
    )
    fap.name = name
    return fap


def _attach(runts, chains):
    """Attach {channel: [(fap, applied), ...]} as RunTS refs + registry."""
    registry = {}
    ds = runts.dataset
    for comp, pairs in chains.items():
        refs = []
        for obj, applied in pairs:
            registry[obj.name] = obj
            refs.append({"applied_filter": {"name": obj.name,
                                            "applied": applied}})
        ds[comp].attrs["filters"] = refs
    runts.filters = registry
    return runts


def _ratio(calibrated, raw, comp):
    """FC ratio calibrated/raw, dropping the DC bin and NaN rows."""
    a = calibrated[comp].values
    b = raw[comp].values
    with np.errstate(invalid="ignore", divide="ignore"):
        r = a / b
    return r[1:, :]


class _Ctx:
    def progress(self, *a, **k):
        pass


def test_compute_fc_divides_by_applied_response():
    """FCs are divided by amp*exp(i*phase) of the applied chain — for a
    flat amp=2/phase=90 filter the calibrated FC is raw/(2j)."""
    from emptyui_aurora._processing import compute_fc_levels

    rng_seed = 11
    rng = np.random.default_rng(rng_seed)
    runts = _runts(*_field(rng))
    cal = _attach(runts, {"hx": [(_fap(), True)]})
    fc_cal = compute_fc_levels(cal, "hx,hy", "ex,ey", num_decimations=0,
                               num_samples_window=256, overlap=0.5,
                               window_type="boxcar")
    raw = _runts(*_field(np.random.default_rng(rng_seed)))
    fc_raw = compute_fc_levels(raw, "hx,hy", "ex,ey", num_decimations=0,
                               num_samples_window=256, overlap=0.5,
                               window_type="boxcar")
    r = _ratio(fc_cal[0], fc_raw[0], "hx")
    expected = 1.0 / (2.0 * np.exp(1j * np.pi / 2))
    assert np.allclose(r, expected, atol=1e-9)
    # other channels untouched
    assert np.allclose(_ratio(fc_cal[0], fc_raw[0], "ex"), 1.0, atol=1e-12)
    # the calibration is recorded on the fc attrs
    assert "hx=t_fap" in fc_cal[0].attrs["calibration_applied"]


def test_compute_fc_skips_unapplied_filters():
    """applied=False filters are not divided out (they are not part of the
    recorded data)."""
    from emptyui_aurora._processing import compute_fc_levels

    rng_seed = 12
    rng = np.random.default_rng(rng_seed)
    runts = _runts(*_field(rng))
    cal = _attach(runts, {"hx": [(_fap(), False)]})
    fc_cal = compute_fc_levels(cal, "hx,hy", "ex,ey", num_decimations=0,
                               num_samples_window=256, overlap=0.5,
                               window_type="boxcar")
    raw = _runts(*_field(np.random.default_rng(rng_seed)))
    fc_raw = compute_fc_levels(raw, "hx,hy", "ex,ey", num_decimations=0,
                               num_samples_window=256, overlap=0.5,
                               window_type="boxcar")
    assert np.allclose(_ratio(fc_cal[0], fc_raw[0], "hx"), 1.0, atol=1e-12)
    assert fc_cal[0].attrs["calibration_applied"] == "none"


def test_compute_fc_chains_two_filters():
    """A magnetic chain (box x coil) divides by the product response."""
    from emptyui_aurora._processing import compute_fc_levels

    rng_seed = 13
    rng = np.random.default_rng(rng_seed)
    runts = _runts(*_field(rng))
    cal = _attach(runts, {"hy": [(_fap("box", 1.5, 20.0), True),
                                 (_fap("coil", 3.0, 30.0), True)]})
    fc_cal = compute_fc_levels(cal, "hx,hy", "ex,ey", num_decimations=0,
                               num_samples_window=256, overlap=0.5,
                               window_type="boxcar")
    raw = _runts(*_field(np.random.default_rng(rng_seed)))
    fc_raw = compute_fc_levels(raw, "hx,hy", "ex,ey", num_decimations=0,
                               num_samples_window=256, overlap=0.5,
                               window_type="boxcar")
    r = _ratio(fc_cal[0], fc_raw[0], "hy")
    expected = 1.0 / (1.5 * 3.0 * np.exp(1j * np.radians(50.0)))
    assert np.allclose(r, expected, atol=1e-9)
    assert "hy=box+coil" in fc_cal[0].attrs["calibration_applied"]


def test_compute_fc_zero_dc_response_sets_nan():
    """A response that vanishes at DC (induction-coil linear extension):
    the DC divisor becomes 1 (aurora) and the DC column is NaN."""
    from emptyui_aurora._processing import compute_fc_levels

    rng = np.random.default_rng(14)
    runts = _runts(*_field(rng))
    cal = _attach(runts, {"hy": [(_fap("coil0", amp=1.0, phase_deg=90.0),
                                  True)]})
    fc = compute_fc_levels(cal, "hx,hy", "ex,ey", num_decimations=0,
                           num_samples_window=256, overlap=0.5,
                           window_type="boxcar")
    # amp=1/phase=90 does not vanish at DC via extrapolation... use a coil-
    # shaped table: build a linear-in-f response whose DC extrapolation is 0
    from mt_metadata.timeseries.filters import FrequencyResponseTableFilter

    coil = FrequencyResponseTableFilter(
        frequencies=np.array([1e-3, 1e3]),
        amplitudes=np.array([1e-3, 1e3]),  # amplitude == frequency -> 0 at DC
        phases=np.deg2rad(np.array([90.0, 90.0])),
        units_in="nanoTesla", units_out="nanoTesla", gain=1.0)
    coil.name = "coil_lin"
    runts2 = _attach(_runts(*_field(rng)), {"hy": [(coil, True)]})
    fc2 = compute_fc_levels(runts2, "hx,hy", "ex,ey", num_decimations=0,
                            num_samples_window=256, overlap=0.5,
                            window_type="boxcar")
    assert np.all(np.isnan(fc2[0]["hy"].values[:, 0]))
    assert not np.any(np.isnan(fc[0]["hy"].values))


def test_core_payload_calibration_survives_filters():
    """coerce_run carries the chains into the core TimeSeries; the core
    Detrend preserves them; Compute FC applies them from the core payload."""
    from emptyui_core_nodes.types import channel_calibration, coerce_run
    from emptyui_core_nodes.filters import Detrend
    from emptyui_aurora._processing import compute_fc_levels

    rng_seed = 15
    rng = np.random.default_rng(rng_seed)
    runts = _attach(_runts(*_field(rng)), {"hx": [(_fap(), True)]})
    core = coerce_run(runts)
    assert channel_calibration(core)["hx"][0][0].name == "t_fap"
    assert channel_calibration(core)["hx"][0][1] is True
    core2 = Detrend().execute(_Ctx(), core, method="mean")["data"]
    assert channel_calibration(core2) == channel_calibration(core)
    fc_cal = compute_fc_levels(core2, "hx,hy", "ex,ey", num_decimations=0,
                               num_samples_window=256, overlap=0.5,
                               window_type="boxcar")
    raw = _runts(*_field(np.random.default_rng(rng_seed)))
    fc_raw = compute_fc_levels(raw, "hx,hy", "ex,ey", num_decimations=0,
                               num_samples_window=256, overlap=0.5,
                               window_type="boxcar")
    r = _ratio(fc_cal[0], fc_raw[0], "hx")
    assert np.allclose(r, 1.0 / (2.0j), atol=1e-9)


def test_channel_calibration_drops_unresolvable_refs(caplog):
    """Refs without a registered object are dropped with a warning — a
    partial chain must never silently mis-calibrate."""
    from emptyui_core_nodes.types import channel_calibration

    runts = _runts(*_field(np.random.default_rng(16)))
    runts.dataset["hx"].attrs["filters"] = [
        {"applied_filter": {"name": "missing_filter", "applied": True}}]
    with caplog.at_level("WARNING", logger="emptyui"):
        cal = channel_calibration(runts)
    assert cal == {}
    assert any("missing_filter" in r.message for r in caplog.records)
