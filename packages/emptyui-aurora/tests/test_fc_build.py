"""Build FC Database node tests (DESIGN §24.8): per-run cascades merged
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
    from mth5.mth5 import MTH5
    from mt_timeseries import ChannelTS, RunTS

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
    sample = fc[sorted(fc)[0]]
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
    from mth5.mth5 import MTH5
    from mt_timeseries import ChannelTS, RunTS

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
                       noisy_idx: int | None = None):
    """n runs of 32 s at 1 Hz, back to back (or gapped); optionally one run
    carries a large in-band sine on the electric channels."""
    import warnings

    warnings.filterwarnings("ignore")
    from mth5.mth5 import MTH5
    from mt_timeseries import ChannelTS, RunTS

    rng = np.random.default_rng(7)
    path = tmp_path / "multi.h5"
    m = MTH5()
    m.open_mth5(str(path), "w")
    sg = m.add_survey("4P")
    stg = sg.stations_group.add_station("SDX")
    for i in range(n):
        start_s = i * 32 if contiguous else i * 96
        start = np.datetime64("2024-01-01T00:00:00") + np.timedelta64(
            start_s, "s")
        chs = []
        for comp in ("hx", "hy"):
            c = ChannelTS("magnetic", data=rng.standard_normal(32),
                          start=str(start), sample_rate=1.0)
            c.component = comp
            chs.append(c)
        for comp in ("ex", "ey"):
            x = rng.standard_normal(32)
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
    """Two back-to-back runs cascade as ONE span with regroup_contiguous;
    gapped runs stay separate."""
    archive = _multi_run_archive(tmp_path, n=2, contiguous=True)
    ctx = RunContext(workspace=tmp_path, store=DataStore())
    res = BuildFC().execute(
        ctx, archive, input_channels="hx,hy", output_channels="ex,ey",
        decimation_levels=0, window_samples=16, overlap=0.5,
        window_type="boxcar", regroup_contiguous=True)
    runs = set(map(str, np.unique(res["fc"][0]["run_id"].values)))
    assert runs == {"sr1_span0001"}
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


def test_build_fc_rejects_noisy_run(tmp_path):
    """The pre-cascade scorer skips the run with a large in-band sine on
    the electric channels; clean runs pass and keep their ids."""
    archive = _multi_run_archive(tmp_path, n=6, contiguous=False,
                                 noisy_idx=2)
    ctx = RunContext(workspace=tmp_path, store=DataStore())
    res = BuildFC().execute(
        ctx, archive, input_channels="hx,hy", output_channels="ex,ey",
        decimation_levels=0, window_samples=16, overlap=0.5,
        window_type="boxcar", reject_noisy_runs=True, use_prior=False)
    runs = set(map(str, np.unique(res["fc"][0]["run_id"].values)))
    assert "sr2400_0003" not in runs
    assert runs == {f"sr2400_{i + 1:04d}" for i in range(6)} - {"sr2400_0003"}
    archive.close_mth5()


def test_build_fc_reject_and_regroup(tmp_path):
    """Kept contiguous runs merge into a span around the rejected one."""
    archive = _multi_run_archive(tmp_path, n=6, contiguous=True, noisy_idx=2)
    ctx = RunContext(workspace=tmp_path, store=DataStore())
    res = BuildFC().execute(
        ctx, archive, input_channels="hx,hy", output_channels="ex,ey",
        decimation_levels=0, window_samples=16, overlap=0.5,
        window_type="boxcar", reject_noisy_runs=True, use_prior=False,
        regroup_contiguous=True)
    runs = set(map(str, np.unique(res["fc"][0]["run_id"].values)))
    assert runs == {"sr1_span0001", "sr1_span0002"}
    archive.close_mth5()
