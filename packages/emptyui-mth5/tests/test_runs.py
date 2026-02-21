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
    from mth5 import mth5 as m5

    from emptyui_mth5.io import LoadMth5

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


def test_align_runs_rejects_collections(tmp_path):
    from emptyui_mth5.runs import AlignRuns

    ctx = RunContext(workspace=tmp_path, store=DataStore())
    try:
        AlignRuns().execute(ctx, _coll_stub(), _coll_stub())
    except ValueError as exc:
        assert "time series 1 is a run collection" in str(exc)
        assert "pairing='intersect'" in str(exc)
    else:
        raise AssertionError("collection input must be rejected")


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
