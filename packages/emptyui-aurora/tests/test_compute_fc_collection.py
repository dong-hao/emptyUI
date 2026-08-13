"""Compute FC on run collections ('time range' route) — DESIGN §9.3.

The orchestration anchor: the collection route is per-member cascades
merged along time — the same engine the retired Build FC node drove
(their parity was pinned while both existed); these tests pin that the
route's orchestration adds nothing on top of the library calls.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from emptyui.context import RunContext
from emptyui.store import DataStore

pytestmark = pytest.mark.filterwarnings("ignore")

from _archive import multi_run_archive


def _two_run_archive(tmp_path: Path):
    return multi_run_archive(
        tmp_path / "two.h5",
        rids=["sr2400_0001", "sr2400_0002"],
        starts=["2024-01-01T00:00:00", "2024-01-01T00:01:36"])


KW = {"decimation_levels": 0, "factor": 4, "window_samples": 16, "overlap": 0.5,
      "window_type": "boxcar", "clock_zero": "epoch", "persist": False}


def test_collection_route_is_memberwise_cascades(tmp_path):
    """Extract(range) + Compute FC == manual per-member compute_fc_levels +
    merge_fc_containers — the route's orchestration (per-family trim,
    identity, run-id stamping aside) adds no physics."""
    from emptyui_aurora._processing import compute_fc_levels, merge_fc_containers
    from emptyui_aurora.process import ComputeFC
    from emptyui_mth5.runs import ExtractRun

    archive = _two_run_archive(tmp_path)
    ctx = RunContext(workspace=tmp_path, store=DataStore())
    W = {"start": "2024-01-01 00:00:00", "end": "2024-01-01 00:02:08"}

    coll = ExtractRun().execute(
        ctx, archive, selection="time range", **W)["data"]
    fc_a = ComputeFC().execute(ctx, coll, **KW)["fc"]

    manual = []
    for member in coll.runs:
        manual.append(compute_fc_levels(
            member, "", "", num_decimations=KW["decimation_levels"],
            factor=KW["factor"], num_samples_window=KW["window_samples"],
            overlap=KW["overlap"], window_type=KW["window_type"],
            clock_zero=KW["clock_zero"]))
    fc_b = merge_fc_containers(manual)

    assert sorted(fc_a) == sorted(fc_b)
    for lv in fc_a:
        assert np.array_equal(fc_a[lv].time.values, fc_b[lv].time.values)
        for var in ("ex", "ey"):
            assert np.allclose(fc_a[lv][var].values, fc_b[lv][var].values,
                               rtol=1e-12, equal_nan=True)


def test_collection_run_id_coords(tmp_path):
    """Per-window run identity follows the contributing run."""
    from emptyui_aurora.process import ComputeFC
    from emptyui_mth5.runs import ExtractRun

    archive = _two_run_archive(tmp_path)
    ctx = RunContext(workspace=tmp_path, store=DataStore())
    coll = ExtractRun().execute(
        ctx, archive, selection="time range",
        start="2024-01-01 00:00:00", end="2024-01-01 00:02:08")["data"]
    fc = ComputeFC().execute(ctx, coll, **KW)["fc"]
    ids = {str(x) for x in fc[0].run_id.values.tolist()}
    assert ids == {"sr2400_0001", "sr2400_0002"}


def test_none_input_names_the_remedy(tmp_path):
    """A None input yields an actionable hint, not a confusing crash."""
    from emptyui_aurora.process import ComputeFC

    ctx = RunContext(workspace=tmp_path, store=DataStore())
    with pytest.raises(ValueError, match="no time series input"):
        ComputeFC().execute(ctx, None)


def test_empty_collection_rejected(tmp_path):
    from emptyui_aurora.process import ComputeFC
    from emptyui_mth5.runs import RunCollection

    ctx = RunContext(workspace=tmp_path, store=DataStore())
    with pytest.raises(ValueError, match="empty"):
        ComputeFC().execute(ctx, RunCollection([], []))


def test_clock_zero_warning_for_multi_run(caplog, tmp_path):
    """Multiple runs without anchoring: warn naming the remedy."""
    from types import SimpleNamespace

    import xarray as xr
    from emptyui_aurora.process import ComputeFC
    from emptyui_mth5.runs import RunCollection

    def member():
        return xr.Dataset({"hx": ("time", np.zeros(64, dtype=float)),
                           "hy": ("time", np.zeros(64, dtype=float))})

    coll = RunCollection(
        runs=[SimpleNamespace(dataset=member(), sample_rate=1.0,
                              start=None, filters={}),
              SimpleNamespace(dataset=member(), sample_rate=1.0,
                              start=None, filters={})],
        extents=[{"survey": "S", "station": "B", "run": "a",
                  "start": "2024-01-01T00:00:00",
                  "end": "2024-01-01T00:01:04", "sample_rate": 1.0},
                 {"survey": "S", "station": "B", "run": "b",
                  "start": "2024-01-01T00:02:00",
                  "end": "2024-01-01T00:03:04", "sample_rate": 1.0}])
    ctx = RunContext(workspace=tmp_path, store=DataStore())
    ComputeFC().execute(ctx, coll, input_channels="hx,hy",
                        output_channels="", decimation_levels=0,
                        window_samples=16, overlap=0.5, clock_zero="",
                        persist=False)
    assert "clock_zero='epoch'" in caplog.text
