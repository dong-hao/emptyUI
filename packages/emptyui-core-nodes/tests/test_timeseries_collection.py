"""Plot Time Series on run collections (backlog #21 Tier 1): sample-rate
family viewing — all runs of the selected family as segments on the shared
absolute time axis."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from emptyui.context import RunContext
from emptyui.store import DataStore
from emptyui_core_nodes.plots import PlotTimeSeriesNode


def _collection(rates, starts, comps=("hx", "hy", "ex", "ey")):
    """A collection (duck-typed: runs + extents — core stays mth5-free)."""
    import pandas as pd
    import xarray as xr
    from types import SimpleNamespace

    runs, extents = [], []
    for rid, sr, start in zip(("r1", "r2", "r3"), rates, starts):
        n = 64
        t = pd.Timestamp(start).to_datetime64() + (
            np.arange(n) / sr * 1e9).astype("timedelta64[ns]")
        rng = np.random.default_rng(abs(hash(rid)) % 2**32)
        data_vars = {c: ("time", rng.standard_normal(n)) for c in comps}
        ds = xr.Dataset(data_vars, coords={"time": t})
        runs.append(SimpleNamespace(dataset=ds, sample_rate=sr,
                                    start=t[0], filters={}))
        extents.append({"survey": "4P", "station": "SDX", "run": rid,
                        "start": str(pd.Timestamp(start)),
                        "end": str(pd.Timestamp(start) + pd.Timedelta(seconds=n / sr)),
                        "sample_rate": sr})
    return SimpleNamespace(runs=runs, extents=extents)


def test_auto_plots_the_lowest_rate_family(tmp_path):
    """auto = the lowest rate family — all its runs, one figure."""
    coll = _collection([15.0, 15.0, 150.0],
                       ["2024-01-01T00:00:00", "2024-01-01T01:00:00",
                        "2024-01-01T02:00:00"])
    ctx = RunContext(workspace=tmp_path, store=DataStore())
    res = PlotTimeSeriesNode().execute(ctx, coll)
    assert Path(res["plot"]["path"]).stat().st_size > 1000
    # the explicit family selects only that family's runs
    res2 = PlotTimeSeriesNode().execute(ctx, coll, sample_rate="150",
                                        channels="hx")
    assert Path(res2["plot"]["path"]).stat().st_size > 1000


def test_unknown_family_rejects_listing_rates(tmp_path):
    coll = _collection([15.0, 150.0], ["2024-01-01T00:00:00"] * 2)
    ctx = RunContext(workspace=tmp_path, store=DataStore())
    with pytest.raises(ValueError, match="available rates"):
        PlotTimeSeriesNode().execute(ctx, coll, sample_rate="999")


def test_window_crops_runs_and_skips_outside(tmp_path, caplog):
    """The time bar's window crops each run (clamped); runs entirely
    outside are skipped with one aggregated warning."""
    import logging

    coll = _collection([15.0, 15.0],
                       ["2024-01-01T00:00:00", "2024-01-01T05:00:00"])
    ctx = RunContext(workspace=tmp_path, store=DataStore())
    with caplog.at_level(logging.WARNING, logger="emptyui"):
        res = PlotTimeSeriesNode().execute(
            ctx, coll, channels="hx",
            start="2024-01-01T00:00:01", end="2024-01-01T00:00:03")
    assert Path(res["plot"]["path"]).stat().st_size > 1000
    assert any("fall outside the selected window" in r.message
               for r in caplog.records)


def test_runs_outside_window_only_rejects(tmp_path):
    coll = _collection([15.0], ["2024-01-01T00:00:00"])
    ctx = RunContext(workspace=tmp_path, store=DataStore())
    with pytest.raises(ValueError, match="falls within"):
        PlotTimeSeriesNode().execute(
            ctx, coll, channels="hx",
            start="2024-01-02T00:00:00", end="2024-01-02T01:00:00")
