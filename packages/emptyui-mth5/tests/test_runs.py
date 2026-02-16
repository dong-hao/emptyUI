"""Extract Run node: id resolution + fallbacks. Error surfaces live in
test_errors.py."""

from __future__ import annotations

from pathlib import Path

from emptyui.context import RunContext
from emptyui.store import DataStore
from emptyui_mth5.runs import (
    ExtractRun,
    _resolve_run,
    _resolve_station,
    _resolve_survey,
)


def test_empty_params_fall_back_to_first_run(archive, tmp_path: Path):
    """With empty survey/station/run, Extract Run resolves the first available
    survey/station/run instead of erroring (regression: `surveys_list` no
    longer exists on the survey group)."""
    ctx = RunContext(workspace=tmp_path, store=DataStore())
    res = ExtractRun().execute(ctx, archive)
    runts = res["data"]
    assert set(runts.channels) >= {"ex", "ey", "hx", "hy"}


def test_explicit_ids(archive):
    s = _resolve_survey(archive, "")
    st = _resolve_station(s, s.stations_group.groups_list[0])
    r = _resolve_run(st, st.run_summary.iloc[0]["id"])
    assert r.metadata.id


def _lemi_runts(comps=("bx", "by", "bz", "e1", "e2")):
    """Build an mth5 RunTS with LEMI-style channel names."""
    import numpy as np
    from mt_timeseries import ChannelTS, RunTS

    ch_list = []
    for comp in comps:
        typ = "magnetic" if comp[0] in ("b", "h") else "electric"
        c = ChannelTS(typ)
        c.channel_metadata.units = "nT" if typ == "magnetic" else "mV/km"
        c.sample_rate = 1.0
        c.ts = np.arange(64, dtype=float)
        c.component = comp
        ch_list.append(c)
    return RunTS(array_list=ch_list)


def test_split_channels_lemi12_names(tmp_path):
    """LEMI channel names (e1/e2/bx/by/bz) must split onto the standard
    ex/ey/hx/hy/hz ports — the old getattr(ex) raised NameError."""
    from emptyui_mth5.runs import SplitChannels

    runts = _lemi_runts()
    ctx = RunContext(workspace=tmp_path, store=DataStore())
    res = SplitChannels().execute(ctx, runts)
    # ex/ey ports carry the e1/e2 channels; hx/hy/hz carry bx/by/bz
    assert res["ex"].component == "e1"
    assert res["ey"].component == "e2"
    assert res["hx"].component == "bx"
    assert res["hy"].component == "by"
    assert res["hz"].component == "bz"


def test_split_channels_lemi34_four_channel(tmp_path):
    """A 4-E-channel instrument: with lemi34 selected, e3/e4 map to ex/ey."""
    from emptyui_mth5.runs import SplitChannels

    runts = _lemi_runts(("bx", "by", "bz", "e3", "e4"))
    ctx = RunContext(workspace=tmp_path, store=DataStore())
    res = SplitChannels().execute(ctx, runts, channel_nomenclature="lemi34")
    assert res["ex"].component == "e3"
    assert res["ey"].component == "e4"
    assert res["hx"].component == "bx"


def test_split_merge_channels_roundtrip(tmp_path):
    """Split (LEMI names) -> Merge must rebuild a run with the actual channel
    names intact (regression: RunTS(dataset=...) was an invalid constructor)."""
    from emptyui_mth5.runs import MergeChannels, SplitChannels

    runts = _lemi_runts()
    ctx = RunContext(workspace=tmp_path, store=DataStore())
    split = SplitChannels().execute(ctx, runts)
    merged = MergeChannels().execute(
        ctx, ex=split["ex"], ey=split["ey"], hx=split["hx"], hy=split["hy"],
        hz=split["hz"],
    )
    run = merged["data"]
    assert set(run.channels) == {"e1", "e2", "bx", "by", "bz"}


def test_align_runs_mixed_naive_aware_starts(tmp_path):
    """Regression: Align Runs raised a raw TypeError comparing a naive
    (LoadCsv) start against a tz-aware (mth5 RunTS) start. Naive stamps are
    assumed UTC and normalized; the overlap is still found."""

    import numpy as np
    import pandas as pd
    import xarray as xr
    from emptyui_mth5.runs import AlignRuns

    sr, n = 8.0, 2048

    class _P:
        """Duck-typed ts.run payload (mth5-branch interpretation)."""

        sample_rate = sr
        start_datetime = ""

        def __init__(self, start):
            t = np.arange(n) / sr
            self.dataset = xr.Dataset(
                {"bx": ("time", np.sin(2 * np.pi * 0.5 * t) + 1.0),
                 "by": ("time", np.sin(2 * np.pi * 1.3 * t) + 1.0),
                 "e1": ("time", np.sin(2 * np.pi * 0.5 * t) + 2.0),
                 "e2": ("time", np.sin(2 * np.pi * 1.3 * t) + 2.0)},
                coords={"time": t},
            )
            self.start = start

    # local: naive ISO stamp (LoadCsv style); remote: tz-aware (mth5 GPS)
    local = _P("2024-01-01T00:00:00")
    remote = _P(pd.Timestamp("2024-01-01T00:04:00", tz="UTC"))
    ctx = RunContext(workspace=tmp_path, store=DataStore())
    out = AlignRuns().execute(ctx, local, remote)
    a, b = out["data1"], out["data2"]
    assert pd.Timestamp(str(a.start)) == pd.Timestamp("2024-01-01T00:04:00", tz="UTC")
    assert pd.Timestamp(str(b.start)) == pd.Timestamp(str(a.start))
    assert len(a.dataset.data_vars["bx"].values) == n - 240 * sr


def test_core_timeseries_placeholder_epoch_and_utc(tmp_path, caplog):
    """The core-TimeSeries branch of _run_time_info applies the same
    placeholder-epoch guard as the mth5 branch (digitized paper records are
    stamped 1970/1980 by the digitizing tool), and naive stamps are
    normalized to aware UTC."""
    import logging

    import numpy as np
    import pandas as pd
    from emptyui_core_nodes.types import TimeSeries
    from emptyui_mth5.runs import _run_time_info

    ctx_ts = TimeSeries(channels={"ex": np.zeros(16)}, sample_rate=8.0,
                        start_datetime="1970-01-01T00:00:00")
    with caplog.at_level(logging.WARNING, logger="emptyui"):
        _ch, _sr, a, _rel = _run_time_info(ctx_ts)
    assert a is None
    assert any("placeholder epoch" in r.message for r in caplog.records)

    ctx_ts.start_datetime = "2024-01-01T00:00:00"
    _ch, _sr, a, _rel = _run_time_info(ctx_ts)
    assert a == pd.Timestamp("2024-01-01T00:00:00", tz="UTC")


# --- 'time range' mode (inclusive burst-mode selection) ------------------


def _three_run_archive(tmp_path: Path):
    from _archive import multi_run_archive

    return multi_run_archive(
        tmp_path / "three.h5",
        rids=["r1", "r2", "r3"],
        starts=["2024-01-01T00:00:00", "2024-01-01T00:02:00",
                "2024-01-01T01:00:00"])


def test_range_mode_extracts_intersecting_runs(tmp_path):
    """'time range' keeps the runs intersecting the UTC window (inclusive),
    each cropped to the window; only the 'runs' output is populated."""
    archive = _three_run_archive(tmp_path)
    ctx = RunContext(workspace=tmp_path, store=DataStore())
    res = ExtractRun().execute(
        ctx, archive, selection="time range",
        start="2024-01-01 00:01:00", end="2024-01-01 00:03:00")
    coll = res["runs"]
    assert res.get("data") is None
    assert [e["run"] for e in coll.extents] == ["r2"]
    e = coll.extents[0]
    assert e["start"] == "2024-01-01T00:02:00+00:00"
    assert e["end"] == "2024-01-01T00:02:31+00:00"
    assert coll.runs[0].dataset.sizes["time"] == 32
    assert coll.runs[0].sample_rate == 1.0


def test_range_mode_inclusive_window_crops(tmp_path):
    """A window slicing INTO a run keeps that run, cropped to the overlap."""
    archive = _three_run_archive(tmp_path)
    ctx = RunContext(workspace=tmp_path, store=DataStore())
    res = ExtractRun().execute(
        ctx, archive, selection="time range",
        start="2024-01-01 00:00:10", end="2024-01-01 00:00:20")
    coll = res["runs"]
    assert [e["run"] for e in coll.extents] == ["r1"]
    assert coll.runs[0].dataset.sizes["time"] == 11  # slice endpoints inclusive


def test_range_mode_empty_window_error(tmp_path):
    """No runs in the window -> a loud error naming the archive span."""
    archive = _three_run_archive(tmp_path)
    ctx = RunContext(workspace=tmp_path, store=DataStore())
    try:
        ExtractRun().execute(
            ctx, archive, selection="time range",
            start="2024-01-01 02:00:00", end="2024-01-01 03:00:00")
    except ValueError as exc:
        assert "no runs fall within" in str(exc)
        assert "archive spans" in str(exc)
    else:
        raise AssertionError("empty range selection must raise")


def test_range_mode_sample_rates_filter(tmp_path):
    from _archive import multi_run_archive

    archive = multi_run_archive(
        tmp_path / "rates.h5", rids=["a1", "b1"],
        starts=["2024-01-01T00:00:00", "2024-01-01T00:00:00"],
        rates=[1.0, 2.0])
    ctx = RunContext(workspace=tmp_path, store=DataStore())
    res = ExtractRun().execute(
        ctx, archive, selection="time range", sample_rates="1",
        start="2024-01-01 00:00:00", end="2024-01-01 00:01:00")
    assert [e["run"] for e in res["runs"].extents] == ["a1"]
    res2 = ExtractRun().execute(
        ctx, archive, selection="time range", sample_rates="2",
        start="2024-01-01 00:00:00", end="2024-01-01 00:01:00")
    assert [e["run"] for e in res2["runs"].extents] == ["b1"]


def test_run_id_mode_no_overlap_error(tmp_path):
    """A bounded window missing the resolved run names the run and hints
    at the longest runs (the cryptic 0 s clamp is gone)."""
    archive = _three_run_archive(tmp_path)
    ctx = RunContext(workspace=tmp_path, store=DataStore())
    try:
        ExtractRun().execute(
            ctx, archive, run="r1",
            start="2024-01-01 02:00:00", end="2024-01-01 03:00:00")
    except ValueError as exc:
        msg = str(exc)
        assert "does not overlap run 'r1'" in msg
        assert "holds 3 runs" in msg
        assert "longest:" in msg
    else:
        raise AssertionError("out-of-overlap window must raise")


def test_run_id_mode_default_unchanged(archive, tmp_path: Path):
    """'run id' (default) keeps the legacy single-run surface: 'data'
    populated, no crop without bounds."""
    ctx = RunContext(workspace=tmp_path, store=DataStore())
    res = ExtractRun().execute(ctx, archive, run="")
    assert res["data"] is not None
    assert "runs" not in res
