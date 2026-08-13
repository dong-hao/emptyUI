"""Session QC on the uniform Compute FC route (§9.4) — the tests ported
from the retired Build FC node onto Extract Run 'time range' → Compute
FC (issues.md: the Build FC retirement).

Covers what the collection route previously did NOT have: session tiling
(`session_hours`), contiguous-span regrouping (`regroup_contiguous`),
the per-window session-id attribution (start-based, §10.4), the
no-fabrication span guard, the single-run wrap, and the payload dipole
attr (L-A4's new home). The span/tiling helpers get direct unit tests.
"""

from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from emptyui.context import RunContext
from emptyui.store import DataStore
from emptyui_aurora.process import (
    ComputeFC,
    SelectRuns,
    _concat_span_channels,
    _contiguous_spans,
    _tile_sessions,
)
from emptyui_mth5.runs import ExtractRun, _dipole_length_m

pytestmark = pytest.mark.filterwarnings("ignore")



def _multi_run_archive(tmp_path: Path, n: int = 4, contiguous: bool = True,
                       noisy_idx: int | None = None,
                       run_samples: int | None = None):
    """n runs of 32 s at 1 Hz (``run_samples`` samples each when given),
    back to back (or gapped); optionally one run carries a large in-band
    sine on the electric channels. (The retired Build FC node's fixture,
    kept for the uniform-route tests.)"""
    from mt_timeseries import ChannelTS, RunTS
    from mth5.mth5 import MTH5

    rng = np.random.default_rng(7)
    path = tmp_path / "multi.h5"
    m = MTH5()
    m.open_mth5(str(path), "w")
    sg = m.add_survey("4P")
    stg = sg.stations_group.add_station("SDX")
    n_smp = run_samples if run_samples is not None else 32
    for i in range(n):
        start_s = i * 32 if contiguous else i * 96
        start = np.datetime64("2024-01-01T00:00:00") + np.timedelta64(
            start_s, "s")
        chs = []
        for comp in ("hx", "hy"):
            c = ChannelTS("magnetic", data=rng.standard_normal(n_smp),
                          start=str(start), sample_rate=1.0)
            c.component = comp
            chs.append(c)
        for comp in ("ex", "ey"):
            x = rng.standard_normal(n_smp)
            if i == noisy_idx:
                x = x * 10.0
            c = ChannelTS("electric", data=x, start=str(start),
                          sample_rate=1.0)
            c.component = comp
            chs.append(c)
        runts = RunTS(array_list=chs)
        runts.run_metadata.id = f"sr2400_{i + 1:04d}"
        rg = stg.add_run(runts.run_metadata.id)
        rg.from_runts(runts)
    stg.write_metadata()
    sg.update_metadata()
    m.close_mth5()
    archive = MTH5()
    archive.open_mth5(str(path), "r")
    return archive


def _extract(archive):
    return ExtractRun().execute(
        RunContext(workspace=None, store=DataStore()), mth5=archive,
        selection="time range")["data"]


def _compute(ctx, coll, output_channels="ex,ey", **kw):
    kw.setdefault("persist", False)
    return ComputeFC().execute(
        ctx, data=coll, input_channels="hx,hy", output_channels=output_channels,
        decimation_levels=0, window_samples=16, overlap=0.5,
        window_type="boxcar", **kw)


# --- helpers -----------------------------------------------------------------


def test_tile_sessions_boundaries():
    """A 32-sample run (last sample at t+31) tiles into 2 16-s sessions;
    the slice bound is trimmed half a sample short of the next session's
    start (no doubled sample)."""
    start = pd.Timestamp("2024-01-01T00:00:00").tz_localize("UTC")
    end = start + pd.Timedelta(seconds=31)
    units = _tile_sessions([("r1", 1.0, start, end)], 16.0)
    assert [u[0] for u in units] == ["r1_s0001", "r1_s0002"]
    assert units[0][5] == start
    # hi is half a sample before the next unit's start
    assert units[0][6] == units[1][2] - pd.to_timedelta(0.5, unit="s")
    assert units[1][3] == end + pd.to_timedelta(1.0, unit="s")


def test_tile_sessions_odd_tail():
    start = pd.Timestamp("2024-01-01T00:00:00").tz_localize("UTC")
    end = start + pd.Timedelta(seconds=19)  # 20 samples, last at t+19
    units = _tile_sessions([("r1", 1.0, start, end)], 16.0)
    assert len(units) == 2
    # the odd tail: [16, 20) — 4 s from the 4 remaining samples
    assert (units[1][3] - units[1][2]).total_seconds() == pytest.approx(4.0)


def test_contiguous_spans_joins_adjacent_splits_gaps():
    t0 = pd.Timestamp("2024-01-01T00:00:00").tz_localize("UTC")
    units = [
        ("a", 1.0, t0, t0 + pd.Timedelta(seconds=32), "a", 0, None, None),
        ("b", 1.0, t0 + pd.Timedelta(seconds=32),
         t0 + pd.Timedelta(seconds=64), "b", 1, None, None),
        ("g", 1.0, t0 + pd.Timedelta(seconds=96),
         t0 + pd.Timedelta(seconds=128), "g", 2, None, None),
    ]
    assert _contiguous_spans(units) == [["a", "b"], ["g"]]


def test_contiguous_spans_never_joins_overlapping_units():
    """The guard beyond the original: aligned collections crop every member
    to the same overlap — joining them would concatenate parallel views of
    the same samples."""
    t0 = pd.Timestamp("2024-01-01T00:00:00").tz_localize("UTC")
    units = [
        ("a", 1.0, t0, t0 + pd.Timedelta(seconds=64), "a", 0, None, None),
        ("b", 1.0, t0, t0 + pd.Timedelta(seconds=64), "b", 1, None, None),
    ]
    assert _contiguous_spans(units) == [["a"], ["b"]]


def test_contiguous_spans_deepest_rate_first_and_none_singletons():
    t0 = pd.Timestamp("2024-01-01T00:00:00").tz_localize("UTC")
    units = [
        ("slow1", 1.0, t0, t0 + pd.Timedelta(seconds=32), "slow1", 0, None, None),
        ("fast1", 8.0, t0, t0 + pd.Timedelta(seconds=4), "fast1", 1, None, None),
        ("nop", 1.0, None, None, "nop", 2, None, None),
    ]
    spans = _contiguous_spans(units)
    assert spans[0] == ["fast1"]  # deepest rate first
    assert ["nop"] in spans


def test_concat_span_channels_pads_aux_drops_real(caplog):
    import xarray as xr

    da = lambda v: xr.DataArray(np.asarray(v, dtype=float), dims=("time",))
    parts = [("a", {"hx": da([1, 1, 1, 1]), "ex": da([2, 2, 2, 2]),
                    "saturated": da([0, 0, 0, 1])}),
             ("b", {"hx": da([3, 3, 3, 3]), "ex": da([4, 4, 4, 4])})]
    with caplog.at_level(logging.WARNING, logger="emptyui"):
        out = _concat_span_channels(parts, "span", logging.getLogger("emptyui"))
    assert set(out) == {"hx", "ex", "saturated"}
    assert out["saturated"].values.tolist() == [0, 0, 0, 1, 0, 0, 0, 0]
    # a fabricated channel must not carry the template's attrs (filters refs)
    assert "filters" not in out["saturated"].attrs
    # a missing REAL channel is dropped loudly
    parts2 = [("a", {"hx": da([1])}), ("b", {"ex": da([2])})]
    with caplog.at_level(logging.WARNING, logger="emptyui"):
        out2 = _concat_span_channels(parts2, "span", logging.getLogger("emptyui"))
    assert "hx" not in out2 and "ex" not in out2
    assert any("lack channel" in r.message for r in caplog.records)


# --- route-level ports --------------------------------------------------------


def test_collection_regroups_contiguous_runs(tmp_path):
    """Two back-to-back runs cascade as ONE span with regroup_contiguous,
    but each window keeps its own run's id (Select Runs groups by it)."""
    archive = _multi_run_archive(tmp_path, n=2, contiguous=True)
    ctx = RunContext(workspace=tmp_path, store=DataStore())
    res = _compute(ctx, _extract(archive), regroup_contiguous=True)
    runs = set(map(str, np.unique(res["fc"][0]["run_id"].values)))
    assert runs == {"sr2400_0001", "sr2400_0002"}
    # the cascade identity stays per-span in the attrs
    assert res["fc"][0].attrs["run_id"] == "sr1_span0001"
    archive.close_mth5()


def test_collection_regroup_keeps_gaps_separate(tmp_path):
    archive = _multi_run_archive(tmp_path, n=2, contiguous=False)
    ctx = RunContext(workspace=tmp_path, store=DataStore())
    res = _compute(ctx, _extract(archive), regroup_contiguous=True)
    runs = set(map(str, np.unique(res["fc"][0]["run_id"].values)))
    assert runs == {"sr2400_0001", "sr2400_0002"}
    archive.close_mth5()


def test_collection_select_runs_drops_noisy_session(tmp_path):
    """The inflated run is dropped from the merged span while clean runs
    keep their ids — session QC needs no pre-cascade read anywhere."""
    archive = _multi_run_archive(tmp_path, n=6, contiguous=True, noisy_idx=2)
    ctx = RunContext(workspace=tmp_path, store=DataStore())
    fc = _compute(ctx, _extract(archive), regroup_contiguous=True)["fc"]
    res = SelectRuns().execute(ctx, fc, deviation_max=2.0,
                               dead_max=1e9, use_prior=False)
    runs = set(map(str, np.unique(res["fc"][0]["run_id"].values)))
    assert "sr2400_0003" not in runs
    assert runs == {f"sr2400_{i + 1:04d}" for i in range(6)} - {"sr2400_0003"}
    archive.close_mth5()


def test_collection_select_runs_overlap_guard(tmp_path):
    """Windows of the PREVIOUS run that overlap a rejected one picked up
    its samples through the cascade — the conservative guard drops them
    too (start-time attribution already handles right straddlers)."""
    archive = _multi_run_archive(tmp_path, n=6, contiguous=True, noisy_idx=2)
    ctx = RunContext(workspace=tmp_path, store=DataStore())
    fc = _compute(ctx, _extract(archive), regroup_contiguous=True)["fc"]
    res = SelectRuns().execute(ctx, fc, deviation_max=2.0,
                               dead_max=1e9, use_prior=False)
    out = res["fc"][0]
    runs = np.asarray(out["run_id"].values)
    times = np.asarray(out["time"].values).astype("datetime64[ns]")
    t0 = np.datetime64("2024-01-01T00:00:00", "ns")
    rel = (times - t0) / np.timedelta64(1, "s")
    m2 = runs == "sr2400_0002"
    assert rel[m2].max() <= 56.0
    assert m2.sum() == 3  # starts 32/40/48 survive; start 56 dropped
    assert ((rel > 56.0) & (rel < 72.0)).sum() == 0
    archive.close_mth5()


def _back_to_back_archive(tmp_path, run_samples: int, noisy_idx=None):
    """n=2 runs of ``run_samples`` samples at 1 Hz, truly back-to-back
    (spacing == length — the shared fixture spaces by 32 s, which would
    overlap longer runs)."""
    from mt_timeseries import ChannelTS, RunTS
    from mth5.mth5 import MTH5

    rng = np.random.default_rng(5)
    path = tmp_path / "b2b.h5"
    m = MTH5()
    m.open_mth5(str(path), "w")
    sg = m.add_survey("4P")
    stg = sg.stations_group.add_station("SDX")
    for i in range(2):
        start = np.datetime64("2024-01-01T00:00:00") + np.timedelta64(
            i * run_samples, "s")
        chs = []
        for comp in ("hx", "hy"):
            c = ChannelTS("magnetic", data=rng.standard_normal(run_samples),
                          start=str(start), sample_rate=1.0)
            c.component = comp
            chs.append(c)
        for comp in ("ex", "ey"):
            x = rng.standard_normal(run_samples)
            if i == noisy_idx:
                x = x * 10.0
            c = ChannelTS("electric", data=x, start=str(start),
                          sample_rate=1.0)
            c.component = comp
            chs.append(c)
        runts = RunTS(array_list=chs)
        runts.run_metadata.id = f"sr2400_{i + 1:04d}"
        rg = stg.add_run(runts.run_metadata.id)
        rg.from_runts(runts)
    stg.write_metadata()
    sg.update_metadata()
    m.close_mth5()
    archive = MTH5()
    archive.open_mth5(str(path), "r")
    return archive


def test_collection_select_runs_start_attribution(tmp_path):
    """Attribution follows the window START (§10.4): a window whose START
    falls inside a rejected run is dropped even when its CENTER lies in
    the kept run."""
    # 36-s runs put the boundary OFF the 8-s window grid: the window
    # starting at 32 s (inside the rejected run 0001 [0, 36)) is centered
    # at 40 s (inside kept run 0002 [36, 72))
    archive = _back_to_back_archive(tmp_path, 36, noisy_idx=0)
    ctx = RunContext(workspace=tmp_path, store=DataStore())
    fc = _compute(ctx, _extract(archive), regroup_contiguous=True)["fc"]
    res = SelectRuns().execute(ctx, fc, deviation_max=2.0,
                               dead_max=1e9, use_prior=False)
    out = res["fc"][0]
    runs = np.asarray(out["run_id"].values)
    times = np.asarray(out["time"].values).astype("datetime64[ns]")
    t0 = np.datetime64("2024-01-01T00:00:00", "ns")
    rel = (times - t0) / np.timedelta64(1, "s")
    m2 = runs == "sr2400_0002"
    assert rel[m2].min() >= 48.0
    archive.close_mth5()


def test_collection_session_labels(tmp_path):
    """With session_hours, every window carries '{run}_sNNNN' — and with
    regroup_contiguous a run's own sessions merge back into one cascade
    span (they are contiguous) while the ids stay session-granular."""
    archive = _multi_run_archive(tmp_path, n=1, contiguous=True)
    ctx = RunContext(workspace=tmp_path, store=DataStore())
    res = _compute(ctx, _extract(archive), regroup_contiguous=True,
                   session_hours=16.0 / 3600.0)
    runs = set(map(str, np.unique(res["fc"][0]["run_id"].values)))
    assert runs == {"sr2400_0001_s0001", "sr2400_0001_s0002"}
    archive.close_mth5()


def test_collection_session_selection_feeds_select_runs(tmp_path):
    """The session labels make run-level QC work on the uniform route:
    Select Runs scores per session end-to-end and publishes its table."""
    archive = _multi_run_archive(tmp_path, n=1, contiguous=True)
    ctx = RunContext(workspace=tmp_path, store=DataStore())
    fc = _compute(ctx, _extract(archive), regroup_contiguous=True,
                  session_hours=16.0 / 3600.0)["fc"]
    out = SelectRuns().execute(ctx, fc)
    assert out["table"] is not None
    runs = set(map(str, np.asarray(out["fc"][min(out["fc"])]["run_id"].values)))
    assert runs.issubset({"sr2400_0001_s0001", "sr2400_0001_s0002"})
    archive.close_mth5()


def test_single_run_session_hours_wraps_collection(tmp_path):
    """§9.4 on the classic single-run flow: a lone ts.run is treated as a
    one-run collection, so session tiling works without Extract Run's
    range mode."""
    archive = _multi_run_archive(tmp_path, n=1, contiguous=True)
    ctx = RunContext(workspace=tmp_path, store=DataStore())
    single = ExtractRun().execute(
        ctx, mth5=archive, selection="run id", run="sr2400_0001")["data"]
    res = _compute(ctx, single, session_hours=16.0 / 3600.0)
    runs = set(map(str, np.unique(res["fc"][0]["run_id"].values)))
    assert runs == {"sr2400_0001_s0001", "sr2400_0001_s0002"}
    archive.close_mth5()


def test_span_does_not_fabricate_missing_channels(tmp_path, caplog):
    """Span assembly pads ONLY the QC aux mask: a run missing a REAL
    channel is dropped from the cascaded span with a loud warning."""
    from mt_timeseries import ChannelTS, RunTS
    from mth5.mth5 import MTH5

    path = tmp_path / "missing_ch.h5"
    m = MTH5()
    m.open_mth5(str(path), "w")
    sg = m.add_survey("4P")
    stg = sg.stations_group.add_station("SDX")
    rng = np.random.default_rng(11)
    for i in range(2):
        start = np.datetime64("2024-01-01T00:00:00") + np.timedelta64(
            i * 32, "s")
        chs = []
        for comp in ("hx", "hy"):
            c = ChannelTS("magnetic", data=rng.standard_normal(32),
                          start=str(start), sample_rate=1.0)
            c.component = comp
            chs.append(c)
        for comp in ("ex", "ey", "bz") if i == 0 else ("ex", "ey"):
            c = ChannelTS(
                "magnetic" if comp == "bz" else "electric",
                data=rng.standard_normal(32), start=str(start),
                sample_rate=1.0)
            c.component = comp
            chs.append(c)
        runts = RunTS(array_list=chs)
        runts.run_metadata.id = f"sr2400_{i + 1:04d}"
        rg = stg.add_run(runts.run_metadata.id)
        rg.from_runts(runts)
    stg.write_metadata()
    sg.update_metadata()
    m.close_mth5()
    archive = MTH5()
    archive.open_mth5(str(path), "r")

    ctx = RunContext(workspace=tmp_path, store=DataStore())
    with caplog.at_level(logging.WARNING, logger="emptyui"):
        res = _compute(ctx, _extract(archive), output_channels="",
                       regroup_contiguous=True)
    assert "bz" not in res["fc"][0].data_vars
    assert any("bz" in r.message for r in caplog.records)
    archive.close_mth5()


# --- L-A4 dipole: the attr now rides the payload ------------------------------


def test_dipole_length_m_reads_e_channel_metadata():
    """First positive electric-channel dipole length; 0.0 when none."""
    fake = SimpleNamespace(
        groups_list=["hx", "ex", "ey"],
        get_channel=lambda name: SimpleNamespace(
            metadata=SimpleNamespace(dipole_length=(
                100.0 if name == "ey" else 0.0))))
    assert _dipole_length_m(fake) == 100.0
    empty = SimpleNamespace(
        groups_list=["hx"],
        get_channel=lambda name: SimpleNamespace(
            metadata=SimpleNamespace(dipole_length=None)))
    assert _dipole_length_m(empty) == 0.0


def test_collection_dipole_from_payload_attr(tmp_path):
    """Compute FC falls back to the payload's e_dipole_length_m attr (the
    L-A4 promise, now stamped by Extract Run) when the param is unset."""
    import warnings

    warnings.filterwarnings("ignore")
    from mt_metadata.timeseries import Electric
    from mt_metadata.timeseries import Run as RunMD
    from mt_timeseries import ChannelTS, RunTS
    from mth5.mth5 import MTH5

    rng = np.random.default_rng(1)
    chs = []
    for comp in ("hx", "hy", "ex", "ey"):
        cls = "magnetic" if comp.startswith("h") else "electric"
        c = ChannelTS(cls, data=rng.standard_normal(64), start="2024-01-01",
                      sample_rate=1.0)
        c.component = comp
        if cls == "electric":
            emd = Electric()
            emd.dipole_length = 100.0
            c.channel_metadata = emd
        chs.append(c)
    runts = RunTS(array_list=chs)
    runts.run_metadata = RunMD()
    runts.run_metadata.id = "a"

    path = tmp_path / "dip.h5"
    m = MTH5()
    m.open_mth5(str(path), "w")
    sg = m.add_survey("4P")
    stg = sg.stations_group.add_station("SDX")
    rg = stg.add_run("a")
    rg.from_runts(runts)
    stg.write_metadata()
    sg.update_metadata()
    m.close_mth5()
    archive = MTH5()
    archive.open_mth5(str(path), "r", locking=False)

    ctx = RunContext(workspace=tmp_path, store=DataStore())
    fc = _compute(ctx, _extract(archive), output_channels="",
                  clock_zero="epoch", e_dipole_length_m=None)["fc"]
    assert fc and all(
        float(lv.attrs["e_dipole_length_m"]) > 0 for lv in fc.values())
    archive.close_mth5()
