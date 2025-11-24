"""Tests for the Calibrate node (instrument response removal)."""

from __future__ import annotations

import pathlib
from pathlib import Path

import numpy as np
import pytest
from emptyui.context import RunContext
from emptyui.store import DataStore
from emptyui_core_nodes.types import TimeSeries
from emptyui_mth5.calibrate import Calibrate

SYNTH = pathlib.Path("/tmp/opencode/synth/test12rr.h5")


def test_calibrate_explicit_coefficient_filter(tmp_path: Path):
    """A constant (coefficient) response is divided out of the signal."""
    sr = 8.0
    n = 1024
    t = np.arange(n) / sr
    x = 3.0 * np.sin(2 * np.pi * 1.0 * t)  # integer # cycles -> clean rfft
    ts = TimeSeries(channels={"ex": x, "hx": np.sin(2 * np.pi * 0.5 * t)}, sample_rate=sr)

    from mt_metadata.timeseries.filters import CoefficientFilter

    ctx = RunContext(workspace=tmp_path, store=DataStore())
    out = Calibrate().execute(ctx, ts, calibration=CoefficientFilter(gain=2.0))["data"]

    # measured = 2.0 * true -> removing the response recovers true = measured/2
    np.testing.assert_allclose(out.channels["ex"], x / 2.0, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(out.channels["hx"], ts.channels["hx"] / 2.0, rtol=1e-6, atol=1e-6)
    assert out.sample_rate == sr


def test_calibrate_requires_something(tmp_path: Path):
    """A core TimeSeries with no calibration input cannot be calibrated."""
    ts = TimeSeries(channels={"ex": np.zeros(64)}, sample_rate=8.0)
    ctx = RunContext(workspace=tmp_path, store=DataStore())
    with pytest.raises(TypeError):
        Calibrate().execute(ctx, ts, calibration=None)


@pytest.mark.skipif(not SYNTH.exists(), reason="test12rr not on disk")
def test_calibrate_mth5_runts_path(tmp_path: Path):
    """RunTS.calibrate() path runs without error on an mth5 RunTS."""
    import warnings

    warnings.filterwarnings("ignore")
    from mth5.mth5 import MTH5

    m = MTH5()
    m.open_mth5(str(SYNTH), "r")
    st = m.survey_group.stations_group.get_station("test1")
    runts = st.get_run(st.run_summary.iloc[0]["id"]).to_runts()
    m.close_mth5()

    ctx = RunContext(workspace=tmp_path, store=DataStore())
    out = Calibrate().execute(ctx, runts)["data"]
    # returns a RunTS with the same channel set
    assert set(out.channels) == set(runts.channels)
