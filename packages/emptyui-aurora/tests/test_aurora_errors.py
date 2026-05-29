"""Edge-case / error-surface tests for the aurora processing core.

Grouped (and marked ``errors``) so the happy-path suite can run fast with
``-m "not errors"``. Helpers are imported from the feature-test modules to
stay DRY (same directory, on sys.path via pytest rootdir insertion).
"""

from __future__ import annotations

import numpy as np
import pytest
from _synthetic import (
    compute_fc_seconds,
    synthetic_fc,
    synthetic_runts,
    synthetic_source,
)
from emptyui_aurora._processing import (
    _coherence_weight,
    coherence_sort_fc,
    compute_fc_levels,
    estimate_tf,
)
from test_processing import _runts_with_channels, _synthetic_runts
from test_remote_reference import _ctx, _Run, _synthetic_stations

pytestmark = pytest.mark.errors


# --- channel resolution / ComputeFC input edges (from test_processing.py) -

def test_compute_fc_explicit_missing_output_dropped_with_warning():
    """An explicitly requested output channel missing from the data is dropped
    (best-effort tipper extension), not an error."""
    rng = np.random.default_rng(7)
    n = 4096
    runts = _runts_with_channels({
        "bx": rng.standard_normal(n), "by": rng.standard_normal(n),
        "e1": rng.standard_normal(n), "e2": rng.standard_normal(n),
    })
    fc = compute_fc_seconds(runts, "bx,by", "e1,e2,bz",
                    window_length_s=128, overlap=0.5, window_type="boxcar")
    assert set(fc.data_vars) == {"bx", "by", "e1", "e2"}
    assert fc.attrs["output_channels"] == "e1,e2"


def test_compute_fc_explicit_missing_input_errors():
    """A missing INPUT channel stays a hard error."""
    rng = np.random.default_rng(8)
    n = 4096
    runts = _runts_with_channels({
        "bx": rng.standard_normal(n), "by": rng.standard_normal(n),
        "e1": rng.standard_normal(n), "e2": rng.standard_normal(n),
    })
    with pytest.raises(ValueError, match="input"):
        compute_fc_seconds(runts, "hx,hy", "e1,e2",
                   window_length_s=128, overlap=0.5, window_type="boxcar")


def test_compute_fc_rejects_nan_input():
    """A NaN anywhere in the input must fail at the Compute FC entry point,
    not propagate silently through the STFT (scipy propagates NaN into every
    downstream sample)."""
    src = synthetic_source(3, n=1024, kind="white")
    runts = synthetic_runts(src, magnetic=("bx", "by"),
                            electric=("e1", "e2"), z=(1.0, 2.0, 3.0, 4.0))
    runts.dataset["e1"].values[10] = np.nan
    with pytest.raises(ValueError, match="NaN/Inf"):
        compute_fc_seconds(runts, ["bx", "by"], ["e1", "e2"], 256.0, 0.5, "boxcar")
    with pytest.raises(ValueError, match="NaN/Inf"):
        compute_fc_levels(runts, ["bx", "by"], ["e1", "e2"])


def test_estimate_tf_accepts_mth5_read_fc_layout():
    """FCs read back from an mth5 archive have a (time, frequency) layout and
    no attrs; the band loop must transpose them and still estimate."""
    import xarray as xr

    sr = 8.0
    n = 8192
    t = np.arange(n) / sr
    rng = np.random.default_rng(3)
    hx = rng.standard_normal(n)
    hy = rng.standard_normal(n)
    ex = 1.0 * hx + 3.0 * hy + 0.1 * rng.standard_normal(n)
    ey = -1.0 * hx + 4.0 * hy + 0.1 * rng.standard_normal(n)

    class FakeRunTS:
        sample_rate = sr

        def __init__(self):
            self.dataset = xr.Dataset(
                {"ex": ("time", ex), "ey": ("time", ey),
                 "hx": ("time", hx), "hy": ("time", hy)},
                coords={"time": t},
            )

    fc = compute_fc_seconds(FakeRunTS(), ["hx", "hy"], ["ex", "ey"],
                    window_length_s=128, overlap=0.5, window_type="boxcar")
    # mth5 stores (time, frequency); emulate the read-back layout + bare attrs
    mth5_layout = fc.transpose("time", "frequency").copy()
    mth5_layout.attrs = {}
    periods, Z, _ = estimate_tf(mth5_layout, ["hx", "hy"], ["ex", "ey"],
                                estimator_engine="Least Squares", periods_per_octave=8)
    assert periods.size > 1
    assert np.all(np.isfinite(Z))
    assert np.any(np.abs(Z) > 0)


# --- estimator error surfaces (from test_processing.py) ------------------

def test_ls_remote_reference_rejected():
    """There is no least-squares + remote-reference path by design."""
    runts = _synthetic_runts()
    fc = compute_fc_seconds(runts, ["hx", "hy"], ["ex", "ey"],
                    window_length_s=256, overlap=0.5, window_type="boxcar")
    with pytest.raises(ValueError):
        estimate_tf(fc, ["hx", "hy"], ["ex", "ey"],
                    estimator_engine="Least Squares", remote_reference=True)


def test_remote_reference_grid_mismatch_raises_clear_error():
    """A remote-reference fc with a different window (frequency grid) must fail
    with a clear message instead of aurora's bare Exception."""
    runts = _synthetic_runts()
    fc = compute_fc_seconds(runts, ["hx", "hy"], ["ex", "ey"],
                    window_length_s=256, overlap=0.5, window_type="boxcar")
    ref = compute_fc_seconds(runts, ["hx", "hy"], ["ex", "ey"],
                     window_length_s=512, overlap=0.5, window_type="boxcar")
    with pytest.raises(ValueError, match="frequency grid"):
        estimate_tf(fc, ["hx", "hy"], ["ex", "ey"],
                    estimator_engine="Robust", remote_reference=True,
                    ref_fc=ref, periods_per_octave=8)


# --- remote-reference / alignment error surfaces -------------------------

def test_remote_reference_unaligned_time_grid_errors(tmp_path):
    """FCs with identical frequencies but different window (time) grids must
    fail with a clear error — the runs were not aligned."""
    local, _ = _synthetic_stations()
    shifted = _Run({
        "bx": local.dataset["bx"].values[:-320],
        "by": local.dataset["by"].values[:-320],
        "e1": local.dataset["e1"].values[:-320],
        "e2": local.dataset["e2"].values[:-320],
    }, local.sample_rate, start_datetime="2024-01-01T00:00:40")
    fc = compute_fc_levels(local, ["bx", "by"], ["e1", "e2"],
                           num_samples_window=256, overlap=0.5,
                           window_type="boxcar")[0]
    ref = compute_fc_levels(shifted, ["bx", "by"], ["e1", "e2"],
                            num_samples_window=256, overlap=0.5,
                            window_type="boxcar")[0]
    with pytest.raises(ValueError, match="time grid"):
        estimate_tf(fc, ["bx", "by"], ["e1", "e2"],
                    estimator_engine="Robust", remote_reference=True,
                    ref_fc=ref, periods_per_octave=4)


def test_remote_reference_partial_ref_container_errors(tmp_path):
    """A remote fc container missing a decimation level must error, not
    silently degrade that level to a self-reference (≈RME)."""
    local, remote = _synthetic_stations()
    ins, outs = ["bx", "by"], ["e1", "e2"]
    # align first so the time grids match (same as the end-to-end test); the
    # point here is the *level* mismatch, not misalignment
    from emptyui_mth5.runs import AlignRuns

    aligned = AlignRuns().execute(_ctx(tmp_path), local, remote)
    a, b = aligned["data1"], aligned["data2"]
    fc_kwargs = {"num_decimations": 1, "num_samples_window": 256,
                 "overlap": 0.5, "window_type": "boxcar", "magnetic_unit": "h"}
    levels_a = compute_fc_levels(a, ins, outs, **fc_kwargs)
    levels_b = compute_fc_levels(b, ins, "", **fc_kwargs)
    del levels_b[1]  # simulate a Compute FC with fewer decimation levels
    with pytest.raises(ValueError, match="no decimation level 1"):
        from emptyui_aurora._processing import estimate_tf_levels
        estimate_tf_levels(levels_a, ins, outs, estimator_engine="Robust",
                           remote_reference=True, ref_levels=levels_b)
    # nothing connected at all keeps the documented self-reference behavior
    from emptyui_aurora._processing import estimate_tf_levels
    per = estimate_tf_levels(levels_a, ins, outs, estimator_engine="Robust",
                             remote_reference=True, ref_levels=None)
    assert np.all(np.isfinite(per[0]["Z"]))


def test_coherence_sort_partial_ref_container_errors(tmp_path):
    """Same rule for the coherence-sort two-FC mode."""
    from emptyui_aurora._processing import coherence_sort_fcs

    local, _ = _synthetic_stations()
    ins, outs = ["bx", "by"], ["e1", "e2"]
    fc_kwargs = {"num_decimations": 1, "num_samples_window": 256,
                 "overlap": 0.5, "window_type": "boxcar", "magnetic_unit": "h"}
    levels_a = compute_fc_levels(local, ins, outs, **fc_kwargs)
    # dummy remote: duplicated data -> identical grids per level
    levels_b = compute_fc_levels(local, ins, "", **fc_kwargs)
    del levels_b[1]
    with pytest.raises(ValueError, match="no decimation level 1"):
        coherence_sort_fcs(levels_a, levels_b, ins, outs)


def test_align_runs_no_overlap_errors(tmp_path):
    a = _Run({"bx": np.zeros(64)}, 8.0, start_datetime="2024-01-01T00:00:00")
    b = _Run({"bx": np.zeros(64)}, 8.0, start_datetime="2024-01-02T00:00:00")
    with pytest.raises(ValueError, match="do not overlap"):
        from emptyui_mth5.runs import AlignRuns
        AlignRuns().execute(_ctx(tmp_path), a, b)


def test_align_runs_sample_rate_mismatch_errors(tmp_path):
    a = _Run({"bx": np.zeros(64)}, 8.0, start_datetime="2024-01-01T00:00:00")
    b = _Run({"bx": np.zeros(64)}, 1.0, start_datetime="2024-01-01T00:00:00")
    with pytest.raises(ValueError, match="sample rates"):
        from emptyui_mth5.runs import AlignRuns
        AlignRuns().execute(_ctx(tmp_path), a, b)


# --- coherence sort error / boundary surfaces ----------------------------

def test_grid_mismatch_coherence_sort_errors():
    """The coherence-sort two-FC mode validates the shared grid itself
    (pairing='strict' — the legacy identical-grid contract; the default
    'intersect' mode pairs by window time with row drops instead)."""
    local = synthetic_fc(synthetic_runts(synthetic_source(9), seed=9),
                         ["hx", "hy"], ["ex", "ey"])
    remote = local.isel(time=slice(None, -1))
    with pytest.raises(ValueError, match="different time grid"):
        coherence_sort_fc(local, remote, ["hx", "hy"], ["ex", "ey"],
                          pairing="strict")


def test_estimate_tf_skips_nonfinite_rows():
    """estimate_tf handles NaN FC rows (generic robustness, DESIGN.md §10.3)."""
    runts = synthetic_runts(synthetic_source(12), seed=12)
    fc = synthetic_fc(runts, ["hx", "hy"], ["ex", "ey"])
    fc = fc.copy(deep=True)
    fc["ex"].values[:, 0:5] = np.nan
    _, Z, _ = estimate_tf(fc, ["hx", "hy"], ["ex", "ey"],
                          periods_per_octave=8)
    assert np.all(np.isfinite(Z))


def test_coherence_weight_bounds():
    """Weights honour the bounds and mode; off mode is always 1."""
    assert _coherence_weight(0.5, 0.25, 0.98, "weight") == 1.0
    assert _coherence_weight(0.5, 0.25, 0.98, "drop") == 1.0
    assert _coherence_weight(0.5, 0.25, 0.98, "off") == 1.0
    assert _coherence_weight(0.10, 0.25, 0.98, "drop") == 0.0
    assert _coherence_weight(0.99, 0.25, 0.98, "drop") == 0.0
    assert 0.0 < _coherence_weight(0.22, 0.25, 0.98, "weight") < 1.0
    assert 0.0 < _coherence_weight(0.99, 0.25, 0.98, "weight") < 1.0
    # far outside -> fully down-weighted (coh² is capped at 1, so the upper
    # ramp only reaches 0 for max + ramp >= 1)
    assert _coherence_weight(0.0, 0.25, 0.98, "weight") == 0.0
    assert 0.0 < _coherence_weight(1.0, 0.25, 0.98, "weight") < 1.0


def test_merge_tf_levels_without_z_err_merges_as_zeros():
    """A hand-built level payload lacking Z_err merges with zero errors
    instead of crashing on scalar indexing (2026-09-11 review fix)."""
    from emptyui_aurora._processing import merge_tf_levels

    per = {
        0: {
            "periods": np.array([1.0, 10.0]),
            "Z": np.array([[[1, 0], [0, 1]], [[2, 0], [0, 2]]], dtype=complex),
        },
    }
    P, _Z, Z_err = merge_tf_levels(per)
    assert P.shape == (2,)
    assert np.all(Z_err == 0)

    # scalar Z_err broadcasts instead of crashing
    per[0]["Z_err"] = 0.5
    _, _, Z_err = merge_tf_levels(per)
    assert np.all(Z_err == 0.5)


# --- review item B13 / M11: overlap bounds are validated up front ---------

def test_compute_fc_overlap_out_of_bounds_rejected():
    import numpy as np
    from emptyui_aurora._processing import compute_fc_levels
    from test_magnetic_units import _field, _runts

    r = _runts(*_field(np.random.default_rng(3), n=512, sr=8.0))
    for bad in (1.0, 1.5, -0.1):
        with pytest.raises(ValueError, match="overlap must be in"):
            compute_fc_levels(r, ["hx", "hy"], ["ex", "ey"],
                              num_decimations=0, num_samples_window=64,
                              overlap=bad, window_type="boxcar")
    # the manifest bound (0.99) still passes
    levels = compute_fc_levels(r, ["hx", "hy"], ["ex", "ey"],
                               num_decimations=0, num_samples_window=64,
                               overlap=0.99, window_type="boxcar")
    assert 0 in levels
