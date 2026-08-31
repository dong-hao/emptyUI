"""Time-selector hosts (docs/DESIGN.md §8.3 phase 2): Extract Run (archive extent),
Align Runs (window within the overlap), Merge Channels (window on the run).
Guards: clamp + warn (never reject), explicit bounded slot > 1 s."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from emptyui.context import RunContext
from emptyui.store import DataStore
from emptyui_mth5.runs import AlignRuns, ExtractRun, MergeChannels


def _ctx(tmp_path):
    return RunContext(workspace=tmp_path, store=DataStore())


def _runts(start="2023-06-01T12:00:00", n=600, sr=10.0, comps=("ex", "ey", "hx", "hy")):
    # (RunTS has no n_samples — count via the dataset time axis)
    """mth5 RunTS, 60 s at 10 Hz, GPS start."""
    from mth5.timeseries import RunTS

    pieces = []
    for c in comps:
        from mt_timeseries import ChannelTS

        ch = ChannelTS("electric" if c[0] == "e" else "magnetic",
                       data=np.arange(n, dtype=float), sample_rate=sr,
                       start=start)
        ch.component = c
        pieces.append(ch)
    return RunTS(array_list=pieces)


# ---------------------------------------------------------------- Extract Run
def test_extract_run_slot_crops(archive, tmp_path):
    ctx = _ctx(tmp_path)
    res = ExtractRun().execute(ctx, archive, selection="run id", start="1980-01-01 00:00:30",
                               end="1980-01-01 00:00:40")
    runts = res["data"]
    assert len(runts.dataset.time) == 11  # 1 Hz run, 10 s inclusive-exclusive crop
    assert str(runts.start)[:19] == "1980-01-01T00:00:30"


def test_extract_run_out_of_range_clamped(archive, tmp_path, caplog):
    import logging

    with caplog.at_level(logging.WARNING, logger="emptyui"):
        res = ExtractRun().execute(
            _ctx(tmp_path), archive, selection="run id",
            start="1979-01-01 00:00:00",
            end="2999-01-01 00:00:00")
    row = archive.run_summary.iloc[0]
    full = (pd.Timestamp(row["end"]) - pd.Timestamp(row["start"])).total_seconds()
    assert len(res["data"].dataset.time) == pytest.approx(full, rel=0.01)
    assert any("clamped" in r.message for r in caplog.records)


def test_extract_run_minimum_slot(archive, tmp_path):
    with pytest.raises(ValueError, match="minimum slot"):
        ExtractRun().execute(_ctx(tmp_path), archive, selection="run id",
                             start="1980-01-01 00:00:30",
                             end="1980-01-01 00:00:31")


def test_extract_run_no_slot_guard_when_unbounded(archive, tmp_path):
    res = ExtractRun().execute(_ctx(tmp_path), archive, selection="run id")
    row = archive.run_summary.iloc[0]
    full = (pd.Timestamp(row["end"]) - pd.Timestamp(row["start"])).total_seconds()
    assert len(res["data"].dataset.time) == pytest.approx(full, rel=0.01)


# ----------------------------------------------------------------- Align Runs
def test_align_runs_window_within_overlap(tmp_path):
    ctx = _ctx(tmp_path)
    d1 = _runts(start="2023-06-01T12:00:00", n=600)
    d2 = _runts(start="2023-06-01T12:00:10", n=600)  # overlap: 50 s
    res = AlignRuns().execute(ctx, d1, d2, start="5", end="25")
    r1, r2 = res["data1"], res["data2"]
    assert len(r1.dataset.time) == 200  # 20 s window at 10 Hz
    assert len(r2.dataset.time) == 200
    # identical interval, anchored 5 s into the overlap (= 12:00:15)
    assert str(r1.start)[:19] == "2023-06-01T12:00:15"
    assert str(r2.start)[:19] == "2023-06-01T12:00:15"


def test_align_runs_iso_window(tmp_path):
    ctx = _ctx(tmp_path)
    d1 = _runts(start="2023-06-01T12:00:00", n=600)
    d2 = _runts(start="2023-06-01T12:00:10", n=600)
    res = AlignRuns().execute(ctx, d1, d2,
                              start="2023-06-01 12:00:20",
                              end="2023-06-01 12:00:30")
    assert len(res["data1"].dataset.time) == 100
    assert str(res["data1"].start)[:19] == "2023-06-01T12:00:20"


def test_align_runs_window_clamped_and_guarded(tmp_path):
    ctx = _ctx(tmp_path)
    d1 = _runts(start="2023-06-01T12:00:00", n=600)
    d2 = _runts(start="2023-06-01T12:00:10", n=600)
    # window beyond the overlap clamps into it (warn, never reject)
    res = AlignRuns().execute(ctx, d1, d2, start="0", end="9999")
    assert len(res["data1"].dataset.time) == 500  # full 50 s overlap
    with pytest.raises(ValueError, match="minimum slot"):
        AlignRuns().execute(ctx, d1, d2, start="0", end="0.5")
    # ISO bounds without GPS time are rejected (relative seconds work)
    rel1 = _runts(start="")
    with pytest.raises(ValueError, match="no absolute start"):
        AlignRuns().execute(ctx, rel1, _runts(start=""), start="2023-06-01",
                            end="")
    ok = AlignRuns().execute(ctx, rel1, _runts(start=""), start="2", end="20")
    assert len(ok["data1"].dataset.time) == 180


# ------------------------------------------------------------- Merge Channels
def test_merge_channels_window(tmp_path):
    from emptyui_mth5.runs import SplitChannels

    ctx = _ctx(tmp_path)
    split = SplitChannels().execute(ctx, _runts(comps=("e1", "e2", "bx", "by", "bz")))
    res = MergeChannels().execute(ctx, **split, start="10", end="30")
    runts = res["data"]
    assert len(runts.dataset.time) == 200
    assert set(runts.channels) == {"e1", "e2", "bx", "by", "bz"}
    with pytest.raises(ValueError, match="minimum slot"):
        MergeChannels().execute(ctx, **split, start="0", end="0.5")
    # unbounded stays full-length
    full = MergeChannels().execute(ctx, **split)["data"]
    assert len(full.dataset.time) == 600
