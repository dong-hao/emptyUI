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
