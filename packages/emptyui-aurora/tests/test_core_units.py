"""Units survive the core-node detour, and raw counts are rejected.

Regression for the NIMS/SDG31 UI workflow (2026-09-15): the natural chain
Merge Runs -> Detrend (core) -> Align -> Compute FC used to strip the
per-channel unit attrs at the core Detrend node, so Compute FC ran blind
(electric counts got the legacy x1e-6, magnetic counts stayed untouched)
and the apparent resistivity landed anywhere between 1e-6 and 1e8 ohm-m
depending on the magnetic-unit param — all wrong. Now:

- coerce_run carries the RunTS dataset attrs' units into the core
  TimeSeries, the core filter nodes preserve them, and Align re-stamps
  them on its RunTS outputs;
- Compute FC hard-rejects raw instrument counts ("digital counts") with
  an actionable message instead of warn+passthrough (FDSN-built mth5
  archives store raw counts — a Calibrate step is mandatory, §16.4).
"""

from __future__ import annotations

import logging

import numpy as np
import pytest

from test_magnetic_units import MU0, N_TO_AM, _field, _runts


def _core_series(runts):
    from emptyui_core_nodes.types import coerce_run

    return coerce_run(runts)


class _Ctx:
    def progress(self, *a, **k):
        pass


def test_coerce_run_extracts_units():
    runts = _runts(*np.random.default_rng(1).normal(size=(4, 2048)))
    core = _core_series(runts)
    assert core.units["hx"] == "nanoTesla"
    assert core.units["ex"] == "milliVolt per kilometer"


def test_detrend_and_bandpass_preserve_units():
    from emptyui_core_nodes.filters import Bandpass, Detrend

    rng = np.random.default_rng(2)
    runts = _runts(*_field(rng))
    core = _core_series(runts)
    d = Detrend()
    out = d.execute(_Ctx(), core, method="mean")["data"]
    assert out.units == core.units
    b = Bandpass()
    out = b.execute(_Ctx(), core, high_freq=3.0)["filtered"]
    assert out.units == core.units


def test_align_restamps_units_from_core_series(tmp_path):
    """Align consumes a core TimeSeries (post-Detrend) and its RunTS output
    must carry the units so Compute FC sees them."""
    from emptyui.context import RunContext
    from emptyui.store import DataStore
    from emptyui_mth5.runs import AlignRuns

    rng = np.random.default_rng(3)
    a = _core_series(_runts(*_field(rng, n=4096)))
    b = _core_series(_runts(*_field(rng, n=4096)))
    node = AlignRuns()
    out = node.execute(RunContext(workspace=tmp_path, store=DataStore()), a, b)
    for key in ("data1", "data2"):
        ds = out[key].dataset
        assert ds["hx"].attrs.get("units") == "nanoTesla"
        assert ds["ex"].attrs.get("units") == "milliVolt per kilometer"


def test_compute_fc_reads_core_series_units(tmp_path):
    """A calibrated run through a core Detrend keeps the right scaling."""
    from emptyui_aurora._processing import compute_fc

    rng = np.random.default_rng(4)
    core = _core_series(_runts(*_field(rng)))
    fc = compute_fc(core, "hx,hy", "ex,ey", 128.0, 0.5, "boxcar")
    assert fc.attrs["electric_scaling"] == "mV/km -> V/m"


def test_compute_fc_rejects_counts(tmp_path):
    """Raw counts -> hard error naming the channel and the fix."""
    from emptyui_aurora._processing import compute_fc

    rng = np.random.default_rng(5)
    runts = _runts(*_field(rng), units_h="digital counts",
                   units_e="digital counts")
    core = _core_series(runts)
    with pytest.raises(ValueError, match="raw instrument counts.*Calibrate"):
        compute_fc(core, "hx,hy", "ex,ey", 128.0, 0.5, "boxcar")
    # the RunTS payload path rejects too (units intact, no core node)
    with pytest.raises(ValueError, match="raw instrument counts"):
        compute_fc(runts, "hx,hy", "ex,ey", 128.0, 0.5, "boxcar")


def test_compute_fc_rejects_counts_explicit_b(tmp_path):
    """Explicit magnetic_unit='b' does not bypass the counts rejection."""
    from emptyui_aurora._processing import compute_fc

    rng = np.random.default_rng(6)
    runts = _runts(*_field(rng), units_h="digital counts")
    with pytest.raises(ValueError, match="raw instrument counts"):
        compute_fc(runts, "hx,hy", "ex,ey", 128.0, 0.5, "boxcar",
                   magnetic_unit="b")


def test_counts_impedance_would_be_meaningless(tmp_path):
    """Documents the failure mode the rejection prevents: counts through
    the legacy path produce a rho that is wrong by orders of magnitude."""
    from emptyui_aurora._processing import compute_fc_levels

    rng = np.random.default_rng(7)
    runts = _runts(*_field(rng), units_h="digital counts",
                   units_e="digital counts")
    with pytest.raises(ValueError, match="raw instrument counts"):
        compute_fc_levels(runts, "hx,hy", "ex,ey", num_decimations=2,
                          factor=4, num_samples_window=128)
