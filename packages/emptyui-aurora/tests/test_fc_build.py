"""Build FC Database node tests (history §24.8): per-run cascades merged
along time — the burst-mode / many-run per-run cascade builder."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from emptyui.context import RunContext
from emptyui.store import DataStore
from emptyui_aurora.process import BuildFC

pytestmark = pytest.mark.filterwarnings("ignore")


def _two_run_archive(tmp_path: Path):
    """A tiny mth5 archive with two 1 Hz runs 64 s apart (standard names,
    32-s runs, no overlap)."""
    import warnings

    warnings.filterwarnings("ignore")
    from mt_timeseries import ChannelTS, RunTS
    from mth5.mth5 import MTH5

    rng = np.random.default_rng(1)

    def _runts(start, rid):
        chs = []
        for comp in ("hx", "hy"):
            c = ChannelTS("magnetic", data=rng.standard_normal(32),
                          start=start, sample_rate=1.0)
            c.component = comp
            chs.append(c)
        for comp in ("ex", "ey"):
            c = ChannelTS("electric", data=rng.standard_normal(32),
                          start=start, sample_rate=1.0)
            c.component = comp
            chs.append(c)
        runts = RunTS(array_list=chs)
        runts.run_metadata.id = rid
        return runts

    path = tmp_path / "two_runs.h5"
    m = MTH5()
    m.open_mth5(str(path), "w")
    sg = m.add_survey("4P")
    stg = sg.stations_group.add_station("SDX")
    for i, rid in enumerate(("sr2400_0001", "sr2400_0002")):
        start = "2024-01-01T00:00:00" if i == 0 else "2024-01-01T00:01:36"
        rg = stg.add_run(rid)
        rg.from_runts(_runts(start, rid))
    stg.write_metadata()
    sg.update_metadata()
    m.close_mth5()
    archive = MTH5()
    archive.open_mth5(str(path), "r")
    return archive


def test_build_fc_default_clock_zero_is_epoch():
    from emptyui_aurora.process import BuildFC as B

    d = {p.name: p.default for p in B.params}
    assert d["clock_zero"] == "epoch"
    assert d["persist"] is False


def test_build_fc_merges_runs_along_time(tmp_path):
    """Two gapped runs -> one merged fc container spanning both; the run
    identity of the LAST contributing run must not shadow the station."""
    archive = _two_run_archive(tmp_path)
    ctx = RunContext(workspace=tmp_path, store=DataStore())
    res = BuildFC().execute(
        ctx, archive, input_channels="hx,hy", output_channels="ex,ey",
        decimation_levels=0, window_samples=16, overlap=0.5,
        window_type="boxcar")
    fc = res["fc"]
    assert sorted(fc)  # at least level 0
    t = fc[0]["time"].values
    span_s = (t.max() - t.min()) / np.timedelta64(1, "s")
    # windows on both sides of the 32-s gap: span > one run's length
    assert span_s > 32.0
    # identity attrs present (station truth for Estimate TF / export)
    sample = fc[min(fc)]
    assert sample.attrs.get("station_id") == "SDX"
    assert sample.attrs.get("survey_id") == "4P"
    archive.close_mth5()


def test_build_fc_epoch_anchored_grid(tmp_path):
    """The epoch anchor makes the merged window grid absolute: window
    starts sit on t0 + k*step for the anchored levels."""
    archive = _two_run_archive(tmp_path)
    ctx = RunContext(workspace=tmp_path, store=DataStore())
    res = BuildFC().execute(
        ctx, archive, input_channels="hx,hy", output_channels="ex,ey",
        decimation_levels=0, window_samples=16, overlap=0.5,
        window_type="boxcar", clock_zero="epoch")
    t = res["fc"][0]["time"].values.astype("datetime64[ns]")
    step = np.timedelta64(int(16 * 0.5 * 1e9), "ns")  # 8 s stride at 1 Hz
    starts = (t - np.datetime64("1970-01-01T00:00:00", "ns"))
    assert np.all(starts % step == np.timedelta64(0, "ns"))
    archive.close_mth5()


def test_build_fc_missing_run_error(tmp_path):
    archive = _two_run_archive(tmp_path)
    ctx = RunContext(workspace=tmp_path, store=DataStore())
    with pytest.raises(ValueError, match="not in station archive"):
        BuildFC().execute(ctx, archive, runs="sr2400_0009")
    archive.close_mth5()


def test_build_fc_feeds_estimate_tf(tmp_path):
    """The merged container flows into the strict-grid Estimate TF and
    recovers the synthetic impedance (the burst-mode data path)."""
    archive = _two_run_archive(tmp_path)
    ctx = RunContext(workspace=tmp_path, store=DataStore())
    res = BuildFC().execute(
        ctx, archive, input_channels="hx,hy", output_channels="ex,ey",
        decimation_levels=0, window_samples=16, overlap=0.5,
        window_type="boxcar")
    from emptyui_aurora._processing import estimate_tf

    fc = res["fc"][0]
    ins = [c.strip() for c in fc.attrs["input_channels"].split(",")]
    outs = [c.strip() for c in fc.attrs["output_channels"].split(",")]
    p, Z, _err = estimate_tf(fc, ins, outs, estimator_engine="OLS")
    assert p.size >= 1 and np.all(np.isfinite(Z))
    archive.close_mth5()


def _two_family_archive(tmp_path: Path):
    """A burst-like archive with two rate families: sr2400_0001 (4 s @
    2400 Hz) and sr150_0001 (16 s @ 150 Hz)."""
    import warnings

    warnings.filterwarnings("ignore")
    from mt_timeseries import ChannelTS, RunTS
    from mth5.mth5 import MTH5

    rng = np.random.default_rng(3)

    def _runts(start, rate, n, rid):
        chs = []
        for comp in ("hx", "hy"):
            c = ChannelTS("magnetic", data=rng.standard_normal(n),
                          start=start, sample_rate=float(rate))
            c.component = comp
            chs.append(c)
        for comp in ("ex", "ey"):
            c = ChannelTS("electric", data=rng.standard_normal(n),
                          start=start, sample_rate=float(rate))
            c.component = comp
            chs.append(c)
        runts = RunTS(array_list=chs)
        runts.run_metadata.id = rid
        return runts

    path = tmp_path / "two_families.h5"
    m = MTH5()
    m.open_mth5(str(path), "w")
    sg = m.add_survey("4P")
    stg = sg.stations_group.add_station("SDX")
    for rid, rate, n in (("sr2400_0001", 2400.0, 9600),
                         ("sr150_0001", 150.0, 2400)):
        rg = stg.add_run(rid)
        rg.from_runts(_runts("2024-01-01T00:00:00", rate, n, rid))
    stg.write_metadata()
    sg.update_metadata()
    m.close_mth5()
    archive = MTH5()
    archive.open_mth5(str(path), "r")
    return archive


def test_build_fc_per_family_decimation_trim(tmp_path):
    """The higher-rate family decimates only until its rate would reach
    the next family's native rate: 2400 Hz -> 600 (one step), while the
    150 Hz family carries the full cascade — same-rate levels from
    different families would double-cover the same windows."""
    archive = _two_family_archive(tmp_path)
    ctx = RunContext(workspace=tmp_path, store=DataStore())
    node = BuildFC()
    res = node.execute(ctx, archive, decimation_levels=2, factor=4,
                       input_channels="hx,hy", output_channels="ex,ey",
                       window_samples=256, overlap=0.5,
                       window_type="boxcar", clock_zero="epoch")
    rates = sorted((float(res["fc"][lv].attrs["sample_rate"])
                    for lv in res["fc"]), reverse=True)
    assert rates == [2400.0, 600.0, 150.0, 37.5]
    # the 150 Hz level holds only the sr150 run's windows (no double cover
    # from the 2400 family's suppressed L2)
    single = node.execute(ctx, archive, runs="sr150_0001",
                          decimation_levels=2, factor=4,
                          input_channels="hx,hy", output_channels="ex,ey",
                          window_samples=256, overlap=0.5,
                          window_type="boxcar", clock_zero="epoch")
    lvl150 = next(lv for lv in single["fc"]
                  if float(single["fc"][lv].attrs["sample_rate"]) == 150.0)
    lvl150_full = next(lv for lv in res["fc"]
                       if float(res["fc"][lv].attrs["sample_rate"]) == 150.0)
    assert (res["fc"][lvl150_full].sizes["time"]
            == single["fc"][lvl150].sizes["time"])
    archive.close_mth5()


def test_build_fc_carries_run_id_coordinate(tmp_path):
    """§24.13: each merged FC window knows its run — Select Runs groups by
    the per-window 'run_id' coordinate (merge attrs alone can't)."""
    archive = _two_run_archive(tmp_path)
    ctx = RunContext(workspace=tmp_path, store=DataStore())
    res = BuildFC().execute(
        ctx, archive, input_channels="hx,hy", output_channels="ex,ey",
        decimation_levels=0, window_samples=16, overlap=0.5,
        window_type="boxcar")
    runs = res["fc"][0]["run_id"].values
    assert set(map(str, np.unique(runs))) == {"sr2400_0001", "sr2400_0002"}
    # windows are time-sorted across runs: first windows belong to run 1
    assert str(runs[0]) == "sr2400_0001"
    assert str(runs[-1]) == "sr2400_0002"
    archive.close_mth5()


def _multi_run_archive(tmp_path: Path, n: int = 4, contiguous: bool = True,
                       noisy_idx: int | None = None,
                       run_samples: int | None = None):
    """n runs of 32 s at 1 Hz (``run_samples`` samples each when given),
    back to back (or gapped); optionally one run carries a large in-band
    sine on the electric channels."""
    import warnings

    warnings.filterwarnings("ignore")
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
                # broadband inflated noise (the plant signature): the
                # median-over-bins scorer must catch it, unlike a single
                # narrowband tone
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


def test_build_fc_regroups_contiguous_runs(tmp_path):
    """Two back-to-back runs cascade as ONE span with regroup_contiguous,
    but each window keeps its own session's id (Select Runs groups by it);
    gapped runs stay separate."""
    archive = _multi_run_archive(tmp_path, n=2, contiguous=True)
    ctx = RunContext(workspace=tmp_path, store=DataStore())
    res = BuildFC().execute(
        ctx, archive, input_channels="hx,hy", output_channels="ex,ey",
        decimation_levels=0, window_samples=16, overlap=0.5,
        window_type="boxcar", regroup_contiguous=True)
    runs = set(map(str, np.unique(res["fc"][0]["run_id"].values)))
    assert runs == {"sr2400_0001", "sr2400_0002"}
    # the cascade identity stays per-span in the attrs
    assert res["fc"][0].attrs["run_id"] == "sr1_span0001"
    archive.close_mth5()


def test_build_fc_regroup_keeps_gaps_separate(tmp_path):
    archive = _multi_run_archive(tmp_path, n=2, contiguous=False)
    ctx = RunContext(workspace=tmp_path, store=DataStore())
    res = BuildFC().execute(
        ctx, archive, input_channels="hx,hy", output_channels="ex,ey",
        decimation_levels=0, window_samples=16, overlap=0.5,
        window_type="boxcar", regroup_contiguous=True)
    runs = set(map(str, np.unique(res["fc"][0]["run_id"].values)))
    assert runs == {"sr2400_0001", "sr2400_0002"}
    archive.close_mth5()


def test_build_fc_select_runs_drops_noisy_session(tmp_path):
    """Hour-level QC now lives in Select Runs (§24.13): the inflated run is
    dropped from the merged span while clean runs keep their ids — no
    pre-cascade read of the time series anywhere."""
    from emptyui_aurora.process import SelectRuns

    archive = _multi_run_archive(tmp_path, n=6, contiguous=True, noisy_idx=2)
    ctx = RunContext(workspace=tmp_path, store=DataStore())
    fc = BuildFC().execute(
        ctx, archive, input_channels="hx,hy", output_channels="ex,ey",
        decimation_levels=0, window_samples=16, overlap=0.5,
        window_type="boxcar", regroup_contiguous=True)["fc"]
    res = SelectRuns().execute(ctx, fc, deviation_max=2.0,
                               dead_max=1e9, use_prior=False)
    runs = set(map(str, np.unique(res["fc"][0]["run_id"].values)))
    assert "sr2400_0003" not in runs
    assert runs == {f"sr2400_{i + 1:04d}" for i in range(6)} - {"sr2400_0003"}
    archive.close_mth5()


def test_build_fc_select_runs_drops_overlap_neighbors(tmp_path):
    """Windows of the PREVIOUS session that overlap a rejected one picked up
    its samples through the cascade — the conservative guard drops them too
    (start-time attribution already handles right straddlers)."""
    from emptyui_aurora.process import SelectRuns

    archive = _multi_run_archive(tmp_path, n=6, contiguous=True, noisy_idx=2)
    ctx = RunContext(workspace=tmp_path, store=DataStore())
    fc = BuildFC().execute(
        ctx, archive, input_channels="hx,hy", output_channels="ex,ey",
        decimation_levels=0, window_samples=16, overlap=0.5,
        window_type="boxcar", regroup_contiguous=True)["fc"]
    res = SelectRuns().execute(ctx, fc, deviation_max=2.0,
                               dead_max=1e9, use_prior=False)
    out = res["fc"][0]
    runs = np.asarray(out["run_id"].values)
    times = np.asarray(out["time"].values).astype("datetime64[ns]")
    # the fc time coordinate is the window CENTER; attribution follows the
    # window START (§10.4). Runs are 32 s: run 0003 spans [64, 96) — windows
    # starting there have centers 72/80/.... Run 0002 spans [32, 64): its
    # windows' starts are 32/40/48/56 (centers 40/48/56/64) — the one
    # starting at 56 overlaps 0003 and the conservative guard drops it.
    t0 = np.datetime64("2024-01-01T00:00:00", "ns")
    rel = (times - t0) / np.timedelta64(1, "s")
    m2 = runs == "sr2400_0002"
    assert rel[m2].max() <= 56.0
    assert m2.sum() == 3  # starts 32/40/48 survive; start 56 dropped
    # no surviving window sits between run 0002's last kept center (56) and
    # run 0003's first center (72) — the start-56 window was guard-dropped,
    # the start-64 window is 0002's (center 64, dropped by the guard)
    assert ((rel > 56.0) & (rel < 72.0)).sum() == 0
    archive.close_mth5()


def test_build_fc_select_runs_start_attribution(tmp_path):
    """Attribution follows the window START (§10.4): a window whose START
    falls inside a rejected session is dropped even when its CENTER lies in
    the kept session — the center-based attribution leaked one window per
    session boundary into the kept session (the overlap guard only cuts
    left neighbors)."""
    from emptyui_aurora.process import SelectRuns

    # 36-s runs put the boundary OFF the 8-s window grid: the window
    # starting at 32 s (inside the rejected run 0001 [0, 36)) is centered
    # at 40 s (inside kept run 0002 [36, 72))
    archive = _multi_run_archive(tmp_path, n=2, contiguous=True,
                                 noisy_idx=0, run_samples=36)
    ctx = RunContext(workspace=tmp_path, store=DataStore())
    fc = BuildFC().execute(
        ctx, archive, input_channels="hx,hy", output_channels="ex,ey",
        decimation_levels=0, window_samples=16, overlap=0.5,
        window_type="boxcar", regroup_contiguous=True)["fc"]
    res = SelectRuns().execute(ctx, fc, deviation_max=2.0,
                               dead_max=1e9, use_prior=False)
    out = res["fc"][0]
    runs = np.asarray(out["run_id"].values)
    times = np.asarray(out["time"].values).astype("datetime64[ns]")
    t0 = np.datetime64("2024-01-01T00:00:00", "ns")
    rel = (times - t0) / np.timedelta64(1, "s")
    m2 = runs == "sr2400_0002"
    # run 0002's surviving windows start at 40/48/... (centers 48/56/...)
    # — the center-40 window (start 32, inside the rejected run) is gone
    assert rel[m2].min() >= 48.0
    archive.close_mth5()


# --- low-list: clock_zero validation (typos fail loudly; future anchors warn) ---

def test_clock_zero_typo_raises_clean_error(tmp_path):
    from emptyui_aurora._processing import compute_fc_levels
    from test_magnetic_units import _field, _runts

    r = _runts(*_field(np.random.default_rng(3), n=512, sr=8.0))
    for bad in ("epch", "2026-13-40"):
        with pytest.raises(ValueError, match="not a valid anchor"):
            compute_fc_levels(r, ["hx", "hy"], ["ex", "ey"],
                              num_decimations=0, num_samples_window=64,
                              overlap=0.5, window_type="boxcar",
                              clock_zero=bad)


def test_clock_zero_in_the_future_warns(caplog):
    """An anchor LATER than the data silently degraded to run-start
    anchoring (k0 <= 0) — exactly the misalignment the feature prevents;
    it must warn."""
    from emptyui_aurora._processing import compute_fc_levels
    from test_magnetic_units import _field, _runts

    r = _runts(*_field(np.random.default_rng(3), n=512, sr=8.0))
    with caplog.at_level("WARNING", logger="emptyui"):
        levels = compute_fc_levels(
            r, ["hx", "hy"], ["ex", "ey"],
            num_decimations=0, num_samples_window=64,
            overlap=0.5, window_type="boxcar",
            clock_zero="2099-01-01T00:00:00",
            start_datetime="2024-01-01T00:00:00")
    assert any("AFTER the record start" in rec.message for rec in caplog.records)
    assert 0 in levels


def test_fc_time_coord_is_naive_utc():
    """Regression: Compute FC's time coord must be NAIVE UTC (the repo-wide
    convention). A real mth5 RunTS start carries a UTC offset — the built
    coord was a tz-aware DatetimeIndex whose xarray ``.values`` exposes an
    OBJECT array of ``datetime.datetime``: remote-reference pairing's
    ``_time_key`` fell into its float branch and crashed the estimate
    ('float() argument must be a string or a real number'), and every
    ``np.issubdtype(dtype, datetime64)`` check downstream took the wrong
    branch. The ``start_datetime`` attr keeps the offset for provenance."""
    import pandas as pd
    from emptyui_aurora._processing import compute_fc_levels
    from test_magnetic_units import _field, _runts

    r = _runts(*_field(np.random.default_rng(3), n=512, sr=8.0))
    r.start = pd.Timestamp("2024-01-01T00:00:00+00:00")
    r.start_datetime = "2024-01-01T00:00:00+00:00"
    levels = compute_fc_levels(r, ["hx", "hy"], ["ex", "ey"],
                               num_decimations=0, num_samples_window=64,
                               overlap=0.5, window_type="boxcar")
    t = levels[0]["time"].values
    assert np.issubdtype(t.dtype, np.datetime64)  # not object-of-datetimes
    assert levels[0].attrs["start_datetime"] == "2024-01-01T00:00:00+00:00"


def test_build_fc_window_selection(tmp_path):
    """The UTC window keeps only runs whose time period intersects it —
    whole runs, never cropped (slot-aware selection: the slot grid in the
    loader's slot-schedule table maps UTC windows to slots)."""
    archive = _two_run_archive(tmp_path)
    ctx = RunContext(workspace=tmp_path, store=DataStore())
    res = BuildFC().execute(
        ctx, archive, input_channels="hx,hy", output_channels="ex,ey",
        decimation_levels=0, window_samples=16, overlap=0.5,
        window_type="boxcar", clock_zero="epoch",
        start="2024-01-01T00:01:00", end="2024-01-01T00:02:00")
    fc = res["fc"]
    # only sr2400_0002 (00:01:36..00:02:08) intersects the window
    t = fc[0]["time"].values.astype("datetime64[ns]")
    assert t.min() >= np.datetime64("2024-01-01T00:01:36", "ns")
    assert t.max() < np.datetime64("2024-01-01T00:02:09", "ns")
    archive.close_mth5()


def test_build_fc_window_outside_raises(tmp_path):
    archive = _two_run_archive(tmp_path)
    ctx = RunContext(workspace=tmp_path, store=DataStore())
    with pytest.raises(ValueError, match="no runs fall within"):
        BuildFC().execute(
            ctx, archive, input_channels="hx,hy", output_channels="ex,ey",
            start="2024-06-01T00:00:00", end="2024-06-01T01:00:00")
    archive.close_mth5()


def test_build_fc_session_labels(tmp_path):
    """Universal session labels (§9.3): session_hours tiles every run into
    '{run id}_sNNNN' units; with regroup_contiguous a run's own sessions
    merge back into one cascade span (they are contiguous) while gapped
    runs stay separate — and each FC window carries its session id."""
    archive = _two_run_archive(tmp_path)  # two 32-s runs, 64 s apart
    ctx = RunContext(workspace=tmp_path, store=DataStore())
    res = BuildFC().execute(
        ctx, archive, input_channels="hx,hy", output_channels="ex,ey",
        decimation_levels=0, window_samples=16, overlap=0.5,
        window_type="boxcar", clock_zero="epoch",
        regroup_contiguous=True, session_hours=16.0 / 3600.0)
    fc = res["fc"]
    sample = fc[min(fc)]
    runs = np.asarray(sample["run_id"].values)
    # two sessions per run, session labels at window granularity
    assert sorted(set(runs)) == ["sr2400_0001_s0001", "sr2400_0001_s0002",
                                 "sr2400_0002_s0001", "sr2400_0002_s0002"]
    # windows sorted in time follow the session order
    t = sample["time"].values.astype("datetime64[ns]")
    order = np.argsort(t)
    seq = [runs[i] for i in order]
    assert seq == sorted(seq, key=lambda r: (r.split("_s")[0],
                                             int(r.split("_s")[1])))
    archive.close_mth5()


def test_build_fc_session_selection_feeds_select_runs(tmp_path):
    """The session labels make run-level QC work on ordinary (non-burst)
    archives: Select Runs scores and drops per session."""
    from emptyui_aurora.process import (
        CoherenceSort,  # noqa: F401
        SelectRuns,
    )

    archive = _two_run_archive(tmp_path)
    ctx = RunContext(workspace=tmp_path, store=DataStore())
    res = BuildFC().execute(
        ctx, archive, input_channels="hx,hy", output_channels="ex,ey",
        decimation_levels=0, window_samples=16, overlap=0.5,
        window_type="boxcar", clock_zero="epoch",
        regroup_contiguous=True, session_hours=16.0 / 3600.0)
    out = SelectRuns().execute(ctx, res["fc"])
    # the selector runs end-to-end on session labels: the diagnostics
    # artifact is published and the surviving windows carry session ids
    assert out["table"] is not None
    runs = np.asarray(out["fc"][min(out["fc"])]["run_id"].values)
    assert set(runs).issubset({
        "sr2400_0001_s0001", "sr2400_0001_s0002",
        "sr2400_0002_s0001", "sr2400_0002_s0002"})
    archive.close_mth5()


def test_compute_fc_stamps_run_id(tmp_path):
    """The classic single-run flow (Extract -> Compute FC) carries the
    per-window 'run_id' coordinate too — session QC works without routing
    through Build FC (whole runs = all-or-nothing QC)."""
    from emptyui.context import RunContext
    from emptyui.store import DataStore
    from emptyui_aurora.process import ComputeFC
    from mt_timeseries import ChannelTS, RunTS

    rng = np.random.default_rng(7)
    chs = []
    for comp in ("hx", "hy"):
        c = ChannelTS("magnetic", data=rng.standard_normal(256),
                      start="2024-01-01T00:00:00", sample_rate=1.0)
        c.component = comp
        chs.append(c)
    runts = RunTS(array_list=chs)
    runts.run_metadata.id = "b"
    ctx = RunContext(workspace=tmp_path, store=DataStore())
    res = ComputeFC().execute(ctx, runts, input_channels="hx,hy",
                              output_channels="hx,hy", decimation_levels=0,
                              window_samples=16, overlap=0.5,
                              window_type="boxcar", run="b")
    sample = res["fc"][min(res["fc"])]
    assert "run_id" in sample.coords
    assert set(np.asarray(sample["run_id"].values)) == {"b"}


def test_build_fc_default_decimation_levels(tmp_path):
    """The 2dfa4ea flip (Compute FC 4 decimation levels) flipped Build FC's
    execute signature but left its Param at 0 — the frontend seeds Param
    defaults, so a fresh Build FC node computed native-rate-only cascades
    and the per-family decimation trim was inert. Param and signature must
    agree, and a fresh default node yields 4 levels."""
    import inspect

    from emptyui_aurora.process import BuildFC

    param = next(p for p in BuildFC().manifest()["params"]
                 if p["name"] == "decimation_levels")
    assert param["default"] == 4
    sig = inspect.signature(BuildFC.execute).parameters["decimation_levels"]
    assert sig.default == 4

    # end-to-end: a fresh default node (no params) produces 4 levels
    # (the deepest level needs > window samples after 3 decimations)
    archive = _multi_run_archive(tmp_path, n=1, contiguous=True,
                                 run_samples=256 * 256)
    ctx = RunContext(workspace=tmp_path, store=DataStore())
    res = BuildFC().execute(ctx, archive, input_channels="hx,hy",
                            output_channels="ex,ey",
                            regroup_contiguous=False)
    assert sorted(res["fc"]) == [0, 1, 2, 3]
    archive.close_mth5()


def test_build_fc_span_does_not_fabricate_missing_channels(tmp_path, caplog):
    """Span assembly pads ONLY the QC aux mask: a run missing a REAL channel
    is dropped from the cascaded span with a loud warning — the
    union-of-channels padding applied to every channel and fabricated
    all-zero series silently corrupted the tipper."""
    import logging

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
        res = BuildFC().execute(
            ctx, archive, input_channels="hx,hy", output_channels="",
            decimation_levels=0, window_samples=16, overlap=0.5,
            window_type="boxcar", regroup_contiguous=True)
    assert "bz" not in res["fc"][0].data_vars
    assert any("bz" in r.message for r in caplog.records)
    archive.close_mth5()
