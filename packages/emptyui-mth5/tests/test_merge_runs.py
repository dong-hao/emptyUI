"""Tests for the rebuilt Merge Runs node (DESIGN §24.5 P1): params-based
run list, physical gap-aware concatenation. Error surfaces live in
test_errors.py."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from emptyui.context import RunContext
from emptyui.store import DataStore
from emptyui_mth5.merge import MergeRuns, merge_runts

pytestmark = pytest.mark.filterwarnings("ignore")


def _synthetic_runts(start: str = "2024-01-01T00:00:00", sr: float = 1.0,
                     n: int = 256, comps=("bx", "by", "e1", "e2")):
    """A real mth5 RunTS with LEMI-style channels and an absolute start."""
    from mt_timeseries import ChannelTS, RunTS

    rng = np.random.default_rng(1)
    ch_list = []
    for comp in comps:
        c = ChannelTS("magnetic" if comp.startswith("b") else "electric",
                      data=rng.standard_normal(n), start=start,
                      sample_rate=sr)
        c.component = comp
        ch_list.append(c)
    runts = RunTS(array_list=ch_list)
    runts.run_metadata.id = "orig"
    return runts


def test_merge_runts_gap_and_length():
    """Two runs 64 s apart concatenate with the 64-s gap bridged by
    interpolation: merged length = 64 + 64 + 64 (gap) with 64-s runs."""
    sr = 1.0
    r1 = _synthetic_runts(start="2024-01-01T00:00:00", sr=sr, n=64)
    r2 = _synthetic_runts(start="2024-01-01T00:02:08", sr=sr, n=64)  # 64 s gap
    merged = merge_runts([r1, r2], gap_method="slinear", new_run_id="bc")
    n = merged.dataset.sizes["time"]
    assert n == 192
    assert float(merged.sample_rate) == sr
    assert merged.run_metadata.id == "bc"


def test_merge_runts_contiguous():
    """Back-to-back runs (no gap) concatenate exactly."""
    r1 = _synthetic_runts(start="2024-01-01T00:00:00", n=64)
    r2 = _synthetic_runts(start="2024-01-01T00:01:04", n=64)
    merged = merge_runts([r1, r2])
    assert merged.dataset.sizes["time"] == 128


def _two_run_archive(tmp_path: Path):
    """A tiny mth5 archive with two 1 Hz runs 64 s apart (32-s runs, no
    overlap; the merge bridges a 32-s gap)."""
    from mth5.mth5 import MTH5

    path = tmp_path / "two_runs.h5"
    m = MTH5()
    m.open_mth5(str(path), "w")
    sg = m.add_survey("4P")
    stg = sg.stations_group.add_station("SDX")
    for i, rid in enumerate(("a", "b")):
        start = "2024-01-01T00:00:00" if i == 0 else "2024-01-01T00:01:04"
        runts = _synthetic_runts(start=start, n=32)
        runts.run_metadata.id = rid
        rg = stg.add_run(rid)
        rg.from_runts(runts)
    stg.write_metadata()
    sg.update_metadata()
    m.close_mth5()
    archive = MTH5()
    archive.open_mth5(str(path), "r")
    return archive


def test_merge_runs_node_two_runs(tmp_path: Path):
    """The node resolves ids like Extract Run and concatenates the listed
    runs in acquisition order."""
    archive = _two_run_archive(tmp_path)
    ctx = RunContext(workspace=tmp_path, store=DataStore())
    res = MergeRuns().execute(ctx, archive, survey="4P", station="SDX",
                              runs="a,b")
    merged = res["merged_run"]
    assert merged.dataset.sizes["time"] == 96
    assert merged.run_metadata.id == "merged"
    # empty runs param = every run in archive order
    res2 = MergeRuns().execute(ctx, archive, runs="")
    assert res2["merged_run"].dataset.sizes["time"] == 96
    archive.close_mth5()


def test_merge_runs_node_missing_run(tmp_path: Path):
    archive = _two_run_archive(tmp_path)
    ctx = RunContext(workspace=tmp_path, store=DataStore())
    with pytest.raises(ValueError, match="not in station archive"):
        MergeRuns().execute(ctx, archive, runs="a,z")
    archive.close_mth5()


def test_merge_runs_node_single_run_passthrough(tmp_path: Path):
    """A single listed run is still extracted and re-id'd (no error)."""
    archive = _two_run_archive(tmp_path)
    ctx = RunContext(workspace=tmp_path, store=DataStore())
    res = MergeRuns().execute(ctx, archive, runs="b", new_run_id="only")
    assert res["merged_run"].run_metadata.id == "only"
    assert res["merged_run"].dataset.sizes["time"] == 32
    archive.close_mth5()
