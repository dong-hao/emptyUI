"""Extract Run node: id resolution + fallbacks. Error surfaces live in
test_errors.py."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
from emptyui.context import RunContext
from emptyui.store import DataStore
from emptyui_mth5.runs import (
    AlignRuns,
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
    res = ExtractRun().execute(ctx, archive, selection="run id")
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
    coll = res["data"]
    assert res["data"] is not None
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
    coll = res["data"]
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
    assert [e["run"] for e in res["data"].extents] == ["a1"]
    res2 = ExtractRun().execute(
        ctx, archive, selection="time range", sample_rates="2",
        start="2024-01-01 00:00:00", end="2024-01-01 00:01:00")
    assert [e["run"] for e in res2["data"].extents] == ["b1"]


def test_run_id_mode_no_overlap_error(tmp_path):
    """A bounded window missing the resolved run names the run and hints
    at the longest runs (the cryptic 0 s clamp is gone)."""
    archive = _three_run_archive(tmp_path)
    ctx = RunContext(workspace=tmp_path, store=DataStore())
    try:
        ExtractRun().execute(
            ctx, archive, selection="run id", run="r1",
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
    res = ExtractRun().execute(ctx, archive, selection="run id", run="")
    assert res["data"] is not None
    assert "runs" not in res


def test_first_run_fallback_warns_on_multi_run_station(archive, tmp_path: Path,
                                                       caplog):
    """The silent first-row fallback is a burst-archive footgun (the first
    summary row is a short slot, not the continuous stream) — with more
    than one run it warns naming the matched run and the longest runs
    (§9.3); single-run stations stay silent."""
    import logging
    import sys
    from pathlib import Path as _Path

    sys.path.insert(0, str(_Path(__file__).parent))
    from _archive import multi_run_archive

    ctx = RunContext(workspace=tmp_path, store=DataStore())
    with caplog.at_level(logging.WARNING, logger="emptyui"):
        # multi-run station: the fallback warns loudly
        multi_run_archive(tmp_path / "multi.h5",
                          rids=["r1", "r2", "r3"],
                          starts=["2024-01-01T00:00:00",
                                  "2024-01-02T00:00:00",
                                  "2024-01-03T00:00:00"])
        from mth5.mth5 import MTH5

        a1 = MTH5()
        a1.open_mth5(tmp_path / "multi.h5", "r", locking=False)
        res = ExtractRun().execute(ctx, a1, selection="run id", run="")
        matched = res["data"].run_metadata.id
        assert any(matched in r.message and "longest:" in r.message
                   for r in caplog.records)
        caplog.clear()
        # single-run station: the fallback is unambiguous — no warning
        multi_run_archive(tmp_path / "single.h5", rids=["r1"],
                          starts=["2024-01-01T00:00:00"])
        a2 = MTH5()
        a2.open_mth5(tmp_path / "single.h5", "r", locking=False)
        ExtractRun().execute(ctx, a2, selection="run id", run="")
        assert not any("matched the first summary row" in r.message
                       for r in caplog.records)


def test_fast_path_payload_is_real_runts(archive, tmp_path: Path):
    """Run-id extraction rides the fast channel read and wraps a real RunTS
    carrying the archive's identity + filter registry (§9.3) — Calibrate/
    Split/Merge Channels and Compute FC behave identically downstream."""
    from emptyui_mth5.runs import _resolve_station, _resolve_survey

    ctx = RunContext(workspace=tmp_path, store=DataStore())
    res = ExtractRun().execute(ctx, archive, selection="run id", run="")
    d = res["data"]
    assert type(d).__name__ == "_CachedRunTS"
    st = _resolve_station(_resolve_survey(archive, ""), "")
    assert d.run_metadata.id == st.run_summary.iloc[0]["id"]
    assert d.station_metadata.id
    assert d.survey_metadata.id
    # the registry is attached: channel_calibration resolves per channel
    from emptyui_core_nodes.types import channel_calibration

    cal = channel_calibration(d)
    assert cal, "wrapped RunTS lost the filter registry"


def test_load_uses_uniform_locking_policy(tmp_path, monkeypatch):
    """UNIFORM locking policy (DESIGN §9.3): every emptyUI open disables
    HDF5 file locking. Default-locking handles live for the server
    lifetime (cached payloads) and pin the file; worse, HDF5 rejects MIXED
    locking configurations across concurrent handles in one process
    ("file locking flag values don't match") — the loader-creates +
    Load-MTH5-reads flow in one graph."""
    from _archive import multi_run_archive

    multi_run_archive(tmp_path / "lk.h5", rids=["r1"],
                      starts=["2024-01-01T00:00:00"])
    from emptyui_mth5.io import LoadMth5
    from mth5 import mth5 as m5

    captured = {}
    orig = m5.MTH5.open_mth5

    def spy(self, fn, mode="a", **kw):
        captured.clear()
        captured.update(mode=mode, **kw)
        return orig(self, fn, mode, **kw)

    monkeypatch.setattr(m5.MTH5, "open_mth5", spy)
    ctx = RunContext(workspace=tmp_path, store=DataStore())
    LoadMth5().execute(ctx, str(tmp_path / "lk.h5"), mode="r")
    assert captured["locking"] is False
    LoadMth5().execute(ctx, str(tmp_path / "lk.h5"), mode="r+")
    assert captured["locking"] is False


# --- Tier 3: singular-semantics nodes reject run collections loudly ------


def _coll_stub():
    from types import SimpleNamespace

    return SimpleNamespace(runs=[object()], extents=[{"run": "r1"}])


def test_split_channels_rejects_collection(tmp_path):
    from emptyui_mth5.runs import SplitChannels

    ctx = RunContext(workspace=tmp_path, store=DataStore())
    try:
        SplitChannels().execute(ctx, _coll_stub())
    except ValueError as exc:
        assert "requires a single run" in str(exc)
        assert "'run id'" in str(exc)
    else:
        raise AssertionError("collection input must be rejected")


def _coll_member(sample_rate: float, start: str, n: int, seed: int = 0):
    """A duck collection member (the Extract Run 'time range' shape):
    SimpleNamespace(dataset, sample_rate, start) with a real time axis."""
    import numpy as np
    import xarray as xr

    rng = np.random.default_rng(seed)
    t = pd.date_range(start, periods=n,
                      freq=pd.Timedelta(seconds=1.0 / sample_rate))
    return SimpleNamespace(
        dataset=xr.Dataset(
            {"bx": ("time", rng.standard_normal(n)),
             "by": ("time", rng.standard_normal(n)),
             "ex": ("time", rng.standard_normal(n)),
             "ey": ("time", rng.standard_normal(n))},
            coords={"time": t},
        ),
        sample_rate=sample_rate,
        start=t[0],
    )


def _naive(ts):
    """Normalize a stamp for comparison (aware/naive mix in pandas; MTime
    objects need str())."""
    t = pd.Timestamp(str(ts))
    return t.tz_convert("UTC").tz_localize(None) if t.tzinfo is not None else t


def _assert_extent(row, member):
    """The extent row reflects the member's actual cropped span."""
    t = member.dataset["time"].values
    assert _naive(row["start"]) == _naive(t[0])
    assert _naive(row["end"]) >= _naive(t[-1])


def _coll_row(sample_rate: float, start: str, n: int, seed: int = 0,
              station: str = "", run: str = ""):
    """(member, extent row) with CONSISTENT stamps: the extent row mirrors
    the member's actual dataset span (inclusive last sample)."""
    m = _coll_member(sample_rate, start, n, seed)
    t = m.dataset["time"].values
    row = {"survey": "S", "station": station, "run": run,
           "start": pd.Timestamp(t[0]).isoformat(),
           "end": pd.Timestamp(t[-1]).isoformat(),
           "sample_rate": sample_rate}
    return m, row


def _coll(rows, station):
    from emptyui_mth5.runs import RunCollection

    return RunCollection(runs=[m for m, _ in rows],
                         extents=[e for _, e in rows])


def test_align_runs_collections_member_crop(tmp_path):
    """Collection inputs (Extract Run 'time range' payloads) align
    member-wise: every member is cropped to the common overlap of the two
    collections' union spans, extent rows are updated, and the output kind
    is preserved (collections in -> collections out).

    Spans in seconds (8 Hz): a1 [0, 9.875], a2 [10, 19.875], b1 [5, 24.875]
    -> common overlap [5, 19.875]."""
    from emptyui_mth5.runs import AlignRuns, RunCollection

    a1, e1 = _coll_row(8.0, "2024-01-01T00:00:00", 80, station="A", run="a1")
    a2, e2 = _coll_row(8.0, "2024-01-01T00:00:10", 80, seed=1, station="A",
                       run="a2")
    b1, eb = _coll_row(8.0, "2024-01-01T00:00:05", 160, seed=2, station="B",
                       run="b1")
    coll_a = RunCollection(runs=[a1, a2], extents=[e1, e2])
    coll_b = RunCollection(runs=[b1], extents=[eb])
    ctx = RunContext(workspace=tmp_path, store=DataStore())
    out = AlignRuns().execute(ctx, coll_a, coll_b)
    oa, ob = out["data1"], out["data2"]
    assert isinstance(oa, RunCollection) and isinstance(ob, RunCollection)
    # A: a1 cropped to [5, 9.875] s, a2 cropped to [10, 19.875] s — both kept
    assert [e["run"] for e in oa.extents] == ["a1", "a2"]
    assert _naive(oa.extents[0]["start"]) == pd.Timestamp("2024-01-01T00:00:05")
    assert _naive(oa.extents[0]["end"]) == pd.Timestamp("2024-01-01T00:00:09.875")
    assert _naive(oa.extents[1]["start"]) == pd.Timestamp("2024-01-01T00:00:10")
    for e, m in zip(oa.extents, oa.runs):
        _assert_extent(e, m)
    # B: b1 cropped to the same overlap [5, 19.875] s
    assert [e["run"] for e in ob.extents] == ["b1"]
    assert _naive(ob.extents[0]["start"]) == _naive(oa.extents[0]["start"])
    assert _naive(ob.extents[0]["end"]) == _naive(oa.extents[1]["end"])


def test_align_runs_collections_drop_outside_members(tmp_path, caplog):
    """Members entirely outside the common overlap are DROPPED with a named
    warning (they cannot contribute pairable FC windows)."""
    import logging

    from emptyui_mth5.runs import AlignRuns, RunCollection

    a1, e1 = _coll_row(8.0, "2024-01-01T00:00:00", 80, station="A", run="a1")
    a2, e2 = _coll_row(8.0, "2024-01-01T00:00:10", 80, seed=1, station="A",
                       run="a2")
    b1, eb = _coll_row(8.0, "2024-01-01T00:00:10", 80, seed=2, station="B",
                       run="b1")
    coll_a = RunCollection(runs=[a1, a2], extents=[e1, e2])
    coll_b = RunCollection(runs=[b1], extents=[eb])
    ctx = RunContext(workspace=tmp_path, store=DataStore())
    with caplog.at_level(logging.WARNING, logger="emptyui"):
        out = AlignRuns().execute(ctx, coll_a, coll_b)
    oa, ob = out["data1"], out["data2"]
    # a1 lies entirely before the overlap -> dropped from A; a2 kept
    assert [e["run"] for e in oa.extents] == ["a2"]
    assert [e["run"] for e in ob.extents] == ["b1"]
    assert any("dropping input 1 run 'a1'" in r.message and
               "outside the common overlap" in r.message
               for r in caplog.records)


def test_align_runs_collections_no_overlap_error(tmp_path):
    from emptyui_mth5.runs import AlignRuns, RunCollection

    a1, e1 = _coll_row(8.0, "2024-01-01T00:00:00", 80, station="A", run="a1")
    b1, eb = _coll_row(8.0, "2024-01-01T00:01:00", 80, seed=2, station="B",
                       run="b1")
    coll_a = RunCollection(runs=[a1], extents=[e1])
    coll_b = RunCollection(runs=[b1], extents=[eb])
    ctx = RunContext(workspace=tmp_path, store=DataStore())
    with pytest.raises(ValueError, match="do not overlap"):
        AlignRuns().execute(ctx, coll_a, coll_b)


def test_align_runs_collections_window_params(tmp_path):
    """The time-selector window (history §23.4) narrows the overlap member-wise."""
    from emptyui_mth5.runs import AlignRuns, RunCollection

    a1, e1 = _coll_row(8.0, "2024-01-01T00:00:00", 160, station="A", run="a1")
    b1, eb = _coll_row(8.0, "2024-01-01T00:00:00", 160, seed=2, station="B",
                       run="b1")
    coll_a = RunCollection(runs=[a1], extents=[e1])
    coll_b = RunCollection(runs=[b1], extents=[eb])
    ctx = RunContext(workspace=tmp_path, store=DataStore())
    out = AlignRuns().execute(ctx, coll_a, coll_b,
                              start="5", end="10")
    oa, ob = out["data1"], out["data2"]
    assert _naive(oa.extents[0]["start"]) == pd.Timestamp("2024-01-01T00:00:05")
    assert _naive(oa.extents[0]["end"]) == pd.Timestamp("2024-01-01T00:00:10")
    assert _naive(ob.extents[0]["start"]) == _naive(oa.extents[0]["start"])
    assert _naive(ob.extents[0]["end"]) == _naive(oa.extents[0]["end"])


def test_align_runs_single_x_collection(tmp_path):
    """Mixed wiring: the single side stays a single RunTS output, the
    collection side stays a collection — both cropped to the common overlap
    (the single acts as a 1-member collection internally)."""
    from emptyui_mth5.runs import AlignRuns, RunCollection

    single = _coll_member(8.0, "2024-01-01T00:00:02", 80)  # [2, 11.875] s
    b1, eb = _coll_row(8.0, "2024-01-01T00:00:05", 80, seed=2, station="B",
                       run="b1")
    coll_b = RunCollection(runs=[b1], extents=[eb])
    ctx = RunContext(workspace=tmp_path, store=DataStore())
    out = AlignRuns().execute(ctx, single, coll_b)
    oa, ob = out["data1"], out["data2"]
    # single side: a RunTS cropped to the overlap [5, 12) s (the single's
    # span end is exclusive: start + n/sr)
    from mt_timeseries import RunTS
    assert isinstance(oa, RunTS)
    assert _naive(str(oa.start)) == pd.Timestamp("2024-01-01T00:00:05")
    # collection side cropped to the same overlap
    assert isinstance(ob, RunCollection)
    assert _naive(ob.extents[0]["start"]) == _naive(oa.start)
    assert _naive(ob.extents[0]["end"]) == pd.Timestamp("2024-01-01T00:00:12")
    assert len(oa.dataset.data_vars["bx"].values) == 7 * 8


def test_align_runs_collections_rate_mismatch_warns(tmp_path, caplog):
    """Differing rate SETS across the two collections warn (alignment is
    rate-agnostic; the FC pairing refuses real mismatches downstream)."""
    import logging

    from emptyui_mth5.runs import AlignRuns, RunCollection

    a1, e1 = _coll_row(8.0, "2024-01-01T00:00:00", 80, station="A", run="a1")
    b1, eb = _coll_row(16.0, "2024-01-01T00:00:00", 160, seed=2, station="B",
                       run="b1")
    coll_a = RunCollection(runs=[a1], extents=[e1])
    coll_b = RunCollection(runs=[b1], extents=[eb])
    ctx = RunContext(workspace=tmp_path, store=DataStore())
    with caplog.at_level(logging.WARNING, logger="emptyui"):
        out = AlignRuns().execute(ctx, coll_a, coll_b)
    assert any("different sample-rate sets" in r.message
               for r in caplog.records)
    # the alignment still proceeds (crop is rate-agnostic)
    assert len(out["data1"].extents) == 1
    assert len(out["data2"].extents) == 1


def test_calibrate_degenerate_collection_still_rejected(tmp_path):
    """Calibrate maps collections through their own responses (Tier 2);
    a degenerate collection (non-interpretable member) still rejects."""
    from emptyui_mth5.calibrate import Calibrate

    ctx = RunContext(workspace=tmp_path, store=DataStore())
    try:
        Calibrate().execute(ctx, _coll_stub())
    except ValueError as exc:
        assert "run collection" in str(exc)
    else:
        raise AssertionError("degenerate collection must be rejected")


def test_load_mth5_while_loader_handle_held(tmp_path):
    """The exact reported flow: a loader-created archive with its handle
    still cached, then Load MTH5 in the SAME process — uniform locking
    makes the concurrent opens compatible."""
    from _archive import multi_run_archive

    m = multi_run_archive(tmp_path / "cc.h5", rids=["r1"],
                          starts=["2024-01-01T00:00:00"])
    # m holds an OPEN read handle, like a cached loader output
    from emptyui_mth5.io import LoadMth5

    ctx = RunContext(workspace=tmp_path, store=DataStore())
    res = LoadMth5().execute(ctx, str(tmp_path / "cc.h5"), mode="r")
    assert res["mth5"] is not None
    m.close_mth5()
    res["mth5"].close_mth5()


# --- Save MTH5: close-in-place vs Save As ---------------------------------


def test_save_mth5_save_as_copies_open_archive(tmp_path):
    """A different path copies the archive (Save As); the handle stays open
    and usable, and the copy is self-contained/readable."""
    from _archive import multi_run_archive
    from emptyui_mth5.io import SaveMth5

    m = multi_run_archive(tmp_path / "src.h5", rids=["r1"],
                          starts=["2024-01-01T00:00:00"])
    ctx = RunContext(workspace=tmp_path, store=DataStore())
    dest = tmp_path / "out" / "copy.h5"
    SaveMth5().execute(ctx, m, path=str(dest))
    assert dest.exists()
    # the handle remains open and bound to its original path
    assert len(m.run_summary) == 1
    # the copy is readable and equivalent
    import h5py

    with h5py.File(dest, "r", locking=False) as h5:
        names = []
        h5.visit(names.append)
        assert any(n.endswith("/r1/hx") for n in names)
    m.close_mth5()


def test_save_mth5_overwrites_existing_dest_with_warning(tmp_path):
    from _archive import multi_run_archive
    from emptyui_mth5.io import SaveMth5

    m = multi_run_archive(tmp_path / "src.h5", rids=["r1"],
                          starts=["2024-01-01T00:00:00"])
    ctx = RunContext(workspace=tmp_path, store=DataStore())
    dest = tmp_path / "copy.h5"
    dest.write_text("stale content")
    SaveMth5().execute(ctx, m, path=str(dest))
    assert dest.read_bytes()[:4] == (tmp_path / "src.h5").read_bytes()[:4]
    m.close_mth5()


def test_save_mth5_same_file_spelled_differently_closes_in_place(tmp_path):
    """The same-file check must be path-normalization-proof (§7.1): a
    relative spelling of the archive's own path used to take the Save-As
    branch, unlink the live archive, and crash — the file was GONE."""
    from _archive import multi_run_archive
    from emptyui_mth5.io import SaveMth5

    src = tmp_path / "src.h5"
    m = multi_run_archive(src, rids=["r1"], starts=["2024-01-01T00:00:00"])
    ctx = RunContext(workspace=tmp_path, store=DataStore())
    import os

    old_cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        SaveMth5().execute(ctx, m, path="./src.h5")  # same file, relative
    finally:
        os.chdir(old_cwd)
    # the archive survived (close-in-place, not Save-As)
    assert src.exists()
    m.close_mth5()


def test_save_mth5_failed_copy_preserves_destination(tmp_path, monkeypatch):
    """Save As onto an existing destination must not destroy it when the
    copy fails (temp file + os.replace, §7.1 degrade-never-destroy)."""
    import shutil

    from _archive import multi_run_archive
    from emptyui_mth5.io import SaveMth5

    m = multi_run_archive(tmp_path / "src.h5", rids=["r1"],
                          starts=["2024-01-01T00:00:00"])
    ctx = RunContext(workspace=tmp_path, store=DataStore())
    dest = tmp_path / "copy.h5"
    dest.write_bytes(b"precious user archive")

    def broken_copy2(src, dst):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(shutil, "copy2", broken_copy2)
    try:
        SaveMth5().execute(ctx, m, path=str(dest))
    except OSError as exc:
        assert "No space left" in str(exc)
    else:
        raise AssertionError("the failed copy should surface")
    assert dest.read_bytes() == b"precious user archive"
    m.close_mth5()


def test_h5clear_refuses_live_in_process_writer(tmp_path, monkeypatch):
    """The stale-flag recovery must not h5clear an archive THIS process
    holds open for write (loader outputs live for the server lifetime):
    the pre-check raises with the remedy instead of clearing under a live
    writer (§9.3; cross-process staleness is unchanged — the registry dies
    with the process, so a crashed server's flag is still auto-cleared)."""
    import subprocess

    import h5py
    from emptyui_mth5._compat import (
        ensure_h5_openable,
        is_write_open_in_process,
        register_write_open,
        release_write_open,
    )

    p = tmp_path / "live.h5"
    with h5py.File(p, "w", locking=False):
        pass
    real_file = h5py.File
    register_write_open(p)
    try:
        assert is_write_open_in_process(p)
        # force the "already open" error path: the file is fine, so simulate
        # the flag by patching the probe to raise
        def busy_file(*args, **kwargs):
            raise OSError("file is already open for write (may use h5clear)")

        monkeypatch.setattr(h5py, "File", busy_file)
        try:
            ensure_h5_openable(p, log=None)
        except OSError as exc:
            assert "currently open" in str(exc)
            assert "h5clear is refused" in str(exc)
        else:
            raise AssertionError("a live writer must refuse, not clear")
    finally:
        release_write_open(p)
    assert not is_write_open_in_process(p)
    # released: the pre-check passes (the file opens cleanly) — the real
    # h5py.File back, or the still-raising mock would be misread as a
    # stale flag and "cleared" (silently by a real h5clear on macOS,
    # loudly on runners without the HDF5 tools)
    monkeypatch.setattr(h5py, "File", real_file)
    ensure_h5_openable(p, log=None)
    # and the stale-flag path (no in-process holder) still clears when
    # h5clear is available
    import shutil

    def busy_file2(*args, **kwargs):
        raise OSError("file is already open for write (may use h5clear)")

    monkeypatch.setattr(h5py, "File", busy_file2)
    monkeypatch.setattr(subprocess, "run",
                        lambda *a, **k: subprocess.CompletedProcess(a[0], 0))
    # hermetic: runners without the HDF5 tools must reach the mocked
    # `h5clear -s` too — fake the lookup
    monkeypatch.setattr(shutil, "which",
                        lambda cmd: "/usr/bin/fake-h5clear"
                        if cmd == "h5clear" else None)
    ensure_h5_openable(p, log=None)  # cleared via the mocked h5clear


def test_split_merge_roundtrip_carries_survey(tmp_path):
    """Split → Merge rebuilds the survey identity (§3): mt_timeseries
    merges the channels' station/run ids but never the survey — a merged
    output used to persist survey "0" (compounded by runts_context's
    default-id rule). Split's outputs now carry the survey so Merge can
    rebuild it."""
    import numpy as np
    from emptyui_mth5.runs import MergeChannels, SplitChannels
    from mt_timeseries import ChannelTS, RunTS

    ch_list = []
    for comp in ("bx", "by", "bz", "e1", "e2"):
        typ = "magnetic" if comp[0] in ("b", "h") else "electric"
        c = ChannelTS(typ)
        c.channel_metadata.units = "nT" if typ == "magnetic" else "mV/km"
        c.sample_rate = 1.0
        c.ts = np.arange(64, dtype=float)
        c.component = comp
        ch_list.append(c)
    runts = RunTS(array_list=ch_list, station_metadata={"id": "SDX"},
                  run_metadata={"id": "r1"}, survey_metadata={"id": "4P"})
    ctx = RunContext(workspace=tmp_path, store=DataStore())
    split = SplitChannels().execute(ctx, runts)
    merged = MergeChannels().execute(
        ctx, ex=split["ex"], ey=split["ey"], hx=split["hx"], hy=split["hy"],
        hz=split["hz"])["data"]
    # the survey identity survives (the fix); NB station/run do NOT travel
    # through channel objects in this mt_timeseries version — the
    # ChannelTS-from-DataArray construction resets them to the defaults.
    # That gap is handled at the identity layer: runts_context ignores the
    # default "0" ids, so the params/fallback win (test_persist.py).
    assert merged.survey_metadata.id == "4P"


def test_slice_window_handles_open_bounds():
    """L-M2: an end-only or start-only bound crops ONE side (the old
    start-only guard silently dropped an end-only bound in the no-extent
    fallback); both-None returns unchanged."""
    import numpy as np
    import pandas as pd
    import xarray as xr
    from emptyui_mth5.runs import _slice_window

    t = pd.date_range("2024-01-01", periods=8, freq="s")
    da = xr.DataArray(np.arange(8.0), coords={"time": t}, dims=("time",))
    data_vars = {"ex": da}

    end_only = _slice_window(data_vars, None, pd.Timestamp("2024-01-01 00:00:04", tz="UTC"))
    assert end_only["ex"].sizes["time"] == 5  # through the 4th second

    start_only = _slice_window(data_vars, pd.Timestamp("2024-01-01 00:00:05", tz="UTC"), None)
    assert start_only["ex"].sizes["time"] == 3

    assert _slice_window(data_vars, None, None) is data_vars


def test_align_runs_declares_collection_ok_inputs():
    """M24: Align Runs aligns run COLLECTIONS member-wise (the backend has
    for a while — test_align_runs_collections_*), but its inputs lacked
    `collection_ok`, so the frontend's wiring check refused to connect
    Extract Run's 'time range' output (a ts.runs payload may only be
    wired into collection_ok inputs, DESIGN §4) — blocking the
    burst-mode remote-reference route in the UI."""
    from emptyui_mth5.runs import AlignRuns

    node = AlignRuns()
    assert node.inputs[0].name == "data1" and node.inputs[0].collection_ok
    assert node.inputs[1].name == "data2" and node.inputs[1].collection_ok


def test_read_run_channels_matches_slow_path(archive):
    """The fast channel read is faithful to get_channel().to_xarray():
    values, dtype, the time axis, and every attr the duck payload path
    consumes (filters for the response registry, units/type for channel
    classification, sample_rate, time_period.start for the time-base
    cross-check). docs/issues.md L21."""
    import numpy as np
    from emptyui_mth5.runs import read_run_channels

    survey_obj = _resolve_survey(archive, "")
    station_obj = _resolve_station(survey_obj, "")
    run_obj = _resolve_run(station_obj, "")
    slow = {name: run_obj.get_channel(name).to_xarray()
            for name in run_obj.groups_list}
    fast, sr, start = read_run_channels(run_obj)
    assert set(fast) == set(slow)
    first = run_obj.groups_list[0]
    assert sr == float(slow[first].attrs["sample_rate"])
    assert start is not None
    for name, da in fast.items():
        ref = slow[name]
        assert da.dtype == ref.dtype, name
        np.testing.assert_array_equal(da.values, ref.values)
        np.testing.assert_array_equal(da.time.values, ref.time.values)
        for key in ("filters", "units", "type", "sample_rate",
                    "time_period.start", "component"):
            assert da.attrs.get(key) == ref.attrs.get(key), (name, key)


def test_full_extraction_avoids_metadata_storms(archive, tmp_path,
                                                monkeypatch):
    """Full-collection extraction from a burst archive must not pay the
    per-channel metadata storm: RunGroup.get_channel is never called and
    the station run summary is built at most once (docs/issues.md L21 — the
    day_12h profile: 347k HDF5 attr reads + 146 run_summary walks for
    145 runs made a 15.6 s extraction)."""
    from mth5.groups import run as run_mod
    from mth5.groups import station as station_mod

    def _forbidden(self, *args, **kwargs):
        raise AssertionError(
            "get_channel called during extraction — the fast path must "
            "read channel datasets directly")

    monkeypatch.setattr(run_mod.RunGroup, "get_channel", _forbidden)

    orig = station_mod.StationGroup.run_summary
    count = {"n": 0}

    @property
    def counted(self):
        count["n"] += 1
        return orig.fget(self)

    monkeypatch.setattr(station_mod.StationGroup, "run_summary", counted)

    ctx = RunContext(workspace=tmp_path, store=DataStore())
    res = ExtractRun().execute(ctx, archive, selection="time range",
                               start="1980-01-01", end="1990-01-01")
    coll = res["data"]
    assert len(coll.runs) >= 1
    assert count["n"] <= 1  # once per execute, not once per run


def test_align_collections_never_rebuild_runts(monkeypatch):
    """Collection members stay duck payloads: aligning slices each
    member's dataset instead of rebuilding a RunTS (whose mt_metadata
    validation storm cost 84 s of the 98 s align on a one-day RR corpus —
    5993 ChannelTS + 1196 RunTS constructions for 598+598 members;
    docs/issues.md L22). The crop must be data-identical to the rebuild, and
    the payload attrs downstream consumers read (sample_rate, filters)
    ride along."""
    import mt_timeseries
    import numpy as np

    def _forbidden(*args, **kwargs):
        raise AssertionError(
            "ChannelTS constructed during collection alignment — the "
            "fast path must slice the member's dataset")

    monkeypatch.setattr(mt_timeseries, "ChannelTS", _forbidden)

    from emptyui_mth5.runs import RunCollection

    a1, e1 = _coll_row(8.0, "2024-01-01T00:00:00", 80, station="A", run="a1")
    a1.filters = {"remote_clb": object()}  # the response registry rides through
    b1, eb = _coll_row(8.0, "2024-01-01T00:00:00", 80, seed=2, station="B",
                       run="b1")
    ctx = RunContext(workspace=Path("/tmp/opencode/l22"), store=DataStore())
    out = AlignRuns().execute(ctx, RunCollection(runs=[a1], extents=[e1]),
                              RunCollection(runs=[b1], extents=[eb]))
    oa, ob = out["data1"], out["data2"]
    assert isinstance(oa, RunCollection) and isinstance(ob, RunCollection)
    for m_in, m_out, e in ((a1, oa.runs[0], oa.extents[0]),
                           (b1, ob.runs[0], ob.extents[0])):
        assert m_out.sample_rate == m_in.sample_rate
        assert getattr(m_out, "filters", None) == getattr(
            m_in, "filters", None)
        np.testing.assert_array_equal(
            np.asarray(m_out.dataset.ex), np.asarray(m_in.dataset.ex))
        np.testing.assert_array_equal(
            m_out.dataset["time"].values, m_in.dataset["time"].values)
        _assert_extent(e, m_out)
