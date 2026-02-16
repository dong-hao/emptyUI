"""Edge-case / error-surface tests for the bundled core nodes.

Grouped (and marked ``errors``) so the happy-path suite can run fast with
``-m "not errors"``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from emptyui.context import RunContext
from emptyui.store import DataStore
from emptyui_core_nodes.filters import Bandpass, Detrend
from emptyui_core_nodes.io import LoadCsv
from emptyui_core_nodes.types import TimeSeries, ensure_finite

pytestmark = pytest.mark.errors


@pytest.fixture()
def ctx(tmp_path: Path):
    return RunContext(workspace=tmp_path, store=DataStore())


def test_time_only_csv_clean_error(ctx, tmp_path: Path):
    """A file with only a time column must raise a clean ValueError, not
    KeyError (regression: channels[next(iter(channels))])."""
    p = tmp_path / "only_t.csv"
    np.savetxt(p, np.arange(10.0), delimiter=",", header="time",
               comments="", fmt="%.3f")
    with pytest.raises(ValueError, match="no channel columns"):
        LoadCsv().execute(ctx, path=str(p))


def test_non_monotonic_time_column_errors(ctx, tmp_path: Path):
    """A predominantly non-monotonic time column (clock jumping backwards)
    cannot yield a sample rate -> clear error. (Single stray samples don't
    fail — the median is deliberately robust.)"""
    p = tmp_path / "bad_t.csv"
    t = np.arange(10.0, 0.0, -1.0)  # descending
    np.savetxt(p, np.column_stack([t, t * 2]), delimiter=",",
               header="time,ex", comments="", fmt="%.3f")
    with pytest.raises(ValueError, match="monotonically"):
        LoadCsv().execute(ctx, path=str(p))


def test_ensure_finite_names_channels_and_sample():
    with pytest.raises(ValueError, match="ey .*sample 3"):
        ensure_finite({"ex": np.zeros(8), "ey": np.array([0, 1, 2, np.nan, 4.0])},
                      "Compute FC")


def test_bandpass_rejects_nan_input(ctx):
    ts = TimeSeries(channels={"ex": np.array([0.0, 1.0, np.nan, 2.0] * 32)},
                    sample_rate=8.0)
    with pytest.raises(ValueError, match="NaN/Inf"):
        Bandpass().execute(ctx, ts, low_freq=0.1)


def test_detrend_rejects_nan_input(ctx):
    ts = TimeSeries(channels={"ex": np.array([np.inf] + [0.0] * 63)},
                    sample_rate=8.0)
    with pytest.raises(ValueError, match="NaN/Inf"):
        Detrend().execute(ctx, ts, method="mean")


def test_detrend_none_passthrough_allows_nan(ctx):
    """method='none' is a passthrough — no guard, data returned as-is."""
    x = np.array([np.nan] + [0.0] * 63)
    ts = TimeSeries(channels={"ex": x}, sample_rate=8.0)
    out = Detrend().execute(ctx, ts, method="none")["data"]
    assert np.isnan(out.channels["ex"][0])


# --- run collections are rejected by singular/linear ts tooling ----------


def _collection():
    from types import SimpleNamespace

    return SimpleNamespace(runs=[object()], extents=[{"run": "r1"}])


def test_detrend_rejects_run_collection(tmp_path):
    from emptyui_core_nodes.filters import Detrend

    ctx = RunContext(workspace=tmp_path, store=DataStore())
    try:
        Detrend().execute(ctx, _collection())
    except ValueError as exc:
        assert "run collection" in str(exc)
    else:
        raise AssertionError("collection input must be rejected")


def test_bandpass_rejects_run_collection(tmp_path):
    from emptyui_core_nodes.filters import Bandpass

    ctx = RunContext(workspace=tmp_path, store=DataStore())
    try:
        Bandpass().execute(ctx, _collection())
    except ValueError as exc:
        assert "run collection" in str(exc)
    else:
        raise AssertionError("collection input must be rejected")


def test_plot_time_series_rejects_run_collection(tmp_path):
    from emptyui_core_nodes.plots import PlotTimeSeriesNode

    ctx = RunContext(workspace=tmp_path, store=DataStore())
    try:
        PlotTimeSeriesNode().execute(ctx, _collection())
    except ValueError as exc:
        assert "sample-rate" in str(exc)
    else:
        raise AssertionError("collection input must be rejected")
